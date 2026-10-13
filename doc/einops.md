# model-einops — 基线标准 Transformer

> 目录：`model-einops/`，文件：`model_minimind.py`, `config_minimind.py`
> 定位：全系列的基础骨架。用标准的 **GQA 多头注意力 + SwiGLU + MoE(可选) + RoPE** 构成预归一化 Transformer 解码器。
> 其余 `model-*` 都在此结构上替换 Token Mixer 或叠加记忆模块。

## 1. 架构总览

```text
输入 tokens
   │
   ▼
┌───────────────┐
│  embed_tokens │  nn.Embedding(vocab, hidden)  ── 与 lm_head 绑定
└───────────────┘
   │
   ▼
  dropout
   │
   ▼
┌────────────────────────────────────────────────────────────┐
│  MiniMindBlock × num_hidden_layers                         │
│                                                            │
│  ┌─────────────────────────────────────────────┐           │
│  │ x ──input_layernorm──→ Attention(标准 GQA)  │  ──┐       │
│  │ 残差: h = x + attn                            │   ▼       │
│  │ x ──post_attention_layernorm──→ MLP(FFN/MoE) │  ──┐       │
│  │ 残差: h = x + mlp                            │   ▼       │
│  └─────────────────────────────────────────────┘           │
└────────────────────────────────────────────────────────────┘
   │
   ▼
┌───────────┐     ┌──────────┐
│  norm     │ ──► │  lm_head │ ──► logits
│  (RMSNorm)│     └──────────┘
└───────────┘      (权重与 embed 绑定)
```

## 2. 配置 (config_minimind.py)

| 字段 | 默认值 | 说明 |
|---|---|---|
| `hidden_size` | 768 | 隐藏维度 |
| `num_hidden_layers` | 8 | 层数 |
| `num_attention_heads` | 8 | Q 头数 |
| `num_key_value_heads` | 4 | KV 头数（GQA） |
| `head_dim` | hidden/heads | 每头维度 |
| `intermediate_size` | ⌈hidden·π/64⌉·64 | FFN 中间维度 |
| `max_position_embeddings` | 32768 | RoPE 预计算长度 |
| `rms_norm_eps` | 1e-6 | RMSNorm 常数 |
| `rope_theta` | 1e6 | RoPE 基数 |
| `inference_rope_scaling` | False | 置 True 启用 YaRN |
| `dropout` | 0.0 | 注意力/残差 dropout |
| `flash_attn` | True | 是否用 SDPA |
| `use_moe` | False | 是否启用 MoE |
| `num_experts` / `num_experts_per_tok` / `moe_intermediate_size` | 4 / 1 / FFN 同 | MoE 专家数等 |

## 3. 核心模块计算逻辑

### 3.1 Attention.forward

输入 `x (b, s, h)`，`position_embeddings=(cos, sin)`：

```text
1) 投影: xq = q_proj(x), xk = k_proj(x), xv = v_proj(x)
   reshape → (b, s, n_heads, d) / (b, s, n_kv_heads, d)
2) 归一化: xq = q_norm(xq),  xk = k_norm(xk)        # Query/Key Norm
3) RoPE:   xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
4) KV Cache: 若有 past_key_value，则 xk = cat([k_old, xk], dim=s)
           保存 past_kv = (xk, xv) if use_cache
5) GQA:  repeat_kv(xk, n_rep), repeat_kv(xv, n_rep)
   → 全部视图变 (b, n_heads, s, d)
6) 注意力:
   if flash and s>1 and no cache and mask全1:
       out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
   else:
       scores = (xq @ xkᵀ) / sqrt(d)
       scores[:, :, :, -s:] += triu(-∞)              # 因果掩码
       scores += (1 - attention_mask) * -1e9          # 自定义掩码
       out = softmax(scores, dim=-1) @ xv
7) reshape → (b, s, n_heads*d) → o_proj → resid_dropout
```

**复杂度**：O(L²·d)；推理借助 KV Cache 逐步 O(1) 生成。

### 3.2 FeedForward

```text
FFN(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )
```

### 3.3 MOEFeedForward

```text
scores = softmax(gate(x))                        # (b*s, E)
sc_topk, exp_topk = topk(scores, k=experts_per_tok)
(可选 norm_topk_prob: 对 sc_topk 归一化)
对每个专家 e:
    hid_idcs = 选中 e 的 token
    y[hid_idcs] += weights * experts[e](x[hid_idcs])
   训练时未激活专家喂入虚拟梯度 (0 * Σparams) 防饿死
aux_loss = (load · scores.mean) * num_experts * coef   # 负载均衡
```

## 4. 模型与生成

- `MiniMindModel.forward(input_ids, attention_mask, past_key_values, use_cache)`
  - 返回 `(hidden_states, presents, aux_loss)`。
  - `start_pos` 从 `past_key_values[0][0].shape[1]` 得到，RoPE 切片 `freqs[start:start+s]`。
- `MiniMindForCausalLM.forward(input_ids, labels, logits_to_keep)`
  - `logits = lm_head(hidden[:, -logits_to_keep:])`
  - `loss = CE(logits[..., :-1], labels[..., 1:])`（忽略 -100）。
  - 返回 `MoeCausalLMOutputWithPast(loss, aux_loss, logits, past_key_values, hidden_states)`。
- `_generate(...)`：自回归循环，支持 temperature / top_k / top_p / repetition_penalty / KV cache。

## 5. 参数量参考

hidden=768, layers=8, heads=8, kv_heads=4, vocab=6400 → 约 100M 级别（含非 MoE）。