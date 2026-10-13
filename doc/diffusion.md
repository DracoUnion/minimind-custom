# model-diffusion — Block Diffusion 生成（Encoder-Decoder）

> 目录：`model-diffusion/`，文件：`model_minimind.py`, `config_minimind.py`
> 定位：完全重写为 **Encoder(自回归) + Decoder(双向)** 的 **Block Diffusion** 生成模型，参照 HuggingFace `transformers` 的 `DiffusionGemma` 架构。
> Encoder 把 prefix 编码进**共享 KV cache**，Decoder 用**双向注意力**处理 canvas tokens 并**只读**该 cache；训练即「把序列尾部长度的 token 当作去噪目标进行双向预测」。

## 1. 架构总览（对比自回归 LM）

```text
传统自回归:  生成 t_{L+1} 前要看到 t_1..t_L，逐 token 前进
Block Diffusion: 一次生成一整段 canvas，Decoder 对 canvas 所有位置**同时**做双向预测
                                   │
              prefix ──► Encoder ──► 共享 KV cache ──┐
                                                     │ (只读)
canvas 初始 ──► Decoder(双向自注意力) ─────► logits ──┘
                ▲                                   │
                │          Self-Conditioning        │ (多步去噪)
                └────── 上一轮 logits ── soft-embedded ──► 注入下一轮 decoder 输入
```

## 2. 类结构与模块

```text
MiniMindForCausalLM (PreTrainedModel, GenerationMixin)
 └─ model: MiniMindModel
     ├─ embed_tokens: ScaledWordEmbedding  (共享, 乘 hidden_size^0.5)
     ├─ layers: ModuleList[MiniMindBlock]  (Encoder 与 Decoder 共享同一权重)
     ├─ encoder: MiniMindEncoderModel (embed_tokens, layers, norm)
     └─ decoder: MiniMindDecoderModel (embed_tokens, layers, norm, self_conditioning)
 └─ lm_head (权重绑定 embed_tokens)
```

| 类 | 对应 DiffusionGemma | 职责 |
|---|---|---|
| `ScaledWordEmbedding` | `DiffusionGemmaTextScaledWordEmbedding` | 嵌入 × √hidden |
| `Attention`(`is_causal` 参数) | `DiffusionGemmaEncoder/DecoderTextAttention` | encoder 因果写 cache / decoder 双向只读 cache |
| `MiniMindBlock` | `DiffusionGemmaText4MLP` 等 | Pre-Norm 块（共享权重） |
| `SelfConditioning` | `DiffusionGemmaSelfConditioning` | 上一轮 logits → soft 信号注入 |
| `MiniMindEncoderModel` | `DiffusionGemmaEncoderTextModel` | 编码 prefix 写共享 KV cache |
| `MiniMindDecoderModel` | `DiffusionGemmaDecoderModel` | 双向处理 canvas，只读 cache |
| `MiniMindModel` | `DiffusionGemmaModel` | 组合 encoder + decoder |
| `MiniMindForCausalLM` | `DiffusionGemmaForBlockDiffusion` | lm_head + 去噪采样 |

## 3. 计算逻辑

### 3.1 Attention 的双模式

```text
Encoder (is_causal=True):                     Decoder (is_causal=False):
  xq,xk,xv = 投影 → q/k norm → RoPE          同左
  cache: xk = cat([cache_k, xk])              cache: xk = cat([cache_k, xk])   # 拼接 prefix KV
  past_kv = (xk, xv) if use_cache             past_kv = None                    # 只读, 不写
  scores = qkᵀ/√d                             scores = qkᵀ/√d
  +triu(-∞) 因果掩码                          (无因果掩码 → 双向)
  (flash path 仅当 is_causal 且无 cache → encoder 预填充 SDPA)
```

### 3.2 Encoder 前向

```text
hidden = dropout(embed_tokens(input_ids))         # prefix
pos = (freqs_cos[start:start+L], freqs_sin[...])  # RoPE
逐层: hidden = layer(hidden, pos, ..., is_causal=True, use_cache=True)
返回 (encoder_last_hidden, past_key_values, aux_loss)   # past_kv = [(k,v),...]
```

### 3.3 Decoder 前向（含 Self-Conditioning）

```text
inputs_embeds = dropout(embed_tokens(decoder_input_ids))    # canvas

# Self-Conditioning: 把上一轮 logits 转成 soft-embedding
if sc_logits 提供:
    soft_emb = (softmax(sc_logits, fp32) @ embed_tokens.weight) * embed_tokens.embed_scale
    (可选 self_conditioning_mask 按样本置零)
else:
    soft_emb = zeros
inputs_embeds = self_conditioning(inputs_embeds, soft_emb)
# self_conditioning: pre_norm(sc) → gate/up/down(SwiGLU) → add 到 emb → post_norm(无 scale)

# canvas 的位置紧接 prefix 之后: cache_seq_len .. cache_seq_len + canvas_len
pos = (freqs_cos[decoder_position_ids], freqs_sin[decoder_position_ids])   # RoPE 带 batch 索引

逐层: hidden = layer(hidden, pos, decoder_attention_mask,  # past_key_value = encoder cache (只读)
                     use_cache=False, is_causal=False)  # 双向
hidden = norm(hidden)
```

### 3.4 训练数据流与损失 (`MiniMindForCausalLM.forward`)

```text
input_ids (b, seq)  +  labels
   │  自动切分（若未给 decoder_input_ids）:
   │     prefix_ids   = input_ids[:, :-canvas_len]     → encoder
   │     decoder_ids  = input_ids[:, -canvas_len:]     → decoder (canvas)
   ▼
logits = lm_head(decoder_last_hidden)                  # (b, canvas_len, vocab)
loss:
   ① diffusion loss: CE(logits, canvas_labels)          # 所有 canvas 位置, ignore -100
   ② (可选 use_prefix_ar_loss) CE(lm_head(encoder_last_hidden)[:, :-1], prefix_labels[:, 1:])
返回 MiniMindDiffusionOutputWithPast(loss, aux_loss, logits, past_key_values, hidden_states, encoder_last_hidden_state)
```

> 纯 decoder / 无前缀模式：`input_ids=None + decoder_input_ids=...`（如无 prefix 的纯去噪或分步推理）。

### 3.5 去噪生成 (`generate`)

```text
1) canvas ← randint(0, vocab) 均匀初始化
2) 迭代 num_diffusion_steps 轮:
    outputs = forward(input_ids=prefix, decoder_input_ids=canvas,
                      self_conditioning_logits=prev_logits)
    最后一轮:  canvas = argmax(logits)
    其余轮:    canvas = 采样(logits, temperature/top_p/top_k)
               (除首末轮, 以概率 self_conditioning_prob 保留 prev_logits = 当前 logits)
3) 返回 cat([prefix, canvas])
```

## 4. 配置 (config_minimind.py)

列新增字段：

| 字段 | 默认 | 说明 |
|---|---|---|
| `canvas_length` | 64 | 训练时从序列尾部切出的去噪目标长度 |
| `num_diffusion_steps` | 16 | 去噪步数（采样） |
| `use_prefix_ar_loss` | True | 是否叠加 prefix 自回归损失 |
| `self_conditioning_prob` | 0.9 | 采样时用 self-conditioning 的概率 |
| `final_logit_softcapping` | None | 非 None 时对 logits 做 tanh softcap |

（其余 `hidden_size / num_hidden_layers / heads / dropout / flash_attn / use_moe / MoE*` 与基线一致。）

## 5. 训练/推理注意事项

- **权重共享**：encoder 与 decoder 的 `layers` 是同一 `ModuleList`、`embed_tokens` 同一张表、`lm_head` 与 embed 绑定——与 DiffusionGemma 的 `_tied_weights_keys` 语义一致。
- **RoPE 位置**：decoder 的 canvas 位置从 prefix 长度起算，`freqs_cos[decoder_position_ids]` 按 batch 索引（3D），Attention 内部自动选 `unsqueeze_dim=2`。
- **attention_mask**：自动切分时 prefix mask 与 canvas 全 1 mask 拼接成 decoder 掩码（覆盖 prefix+canvas）。
- **兼容训练脚本**：`model(input_ids, labels)` → `res.loss + res.aux_loss` 可直接复用现有 trainer。
- **推理复杂度**：每步去噪需完整 encoder+decoder 前向（O(L²)），总成本 = 步数 × 单步成本；`steps` 太小质量下降，太大耗时，`num_diffusion_steps` 需按任务权衡。