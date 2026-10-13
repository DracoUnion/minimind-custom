# model-vq — Transformer-VQ 量化注意力

> 目录：`model-vq/`，文件：`model_minimind.py`, `config_minimind.py`
> 定位：把注意力的 **Key 向量量化成短码**，用有限码本 + 码本聚合缓存建模**任意长上下文**，复杂度近线性。
> 核心：`LearnableVQ`（EMA 码本）+ `VQAttention`（窗口注意力 + 码本聚合注意力）。

## 1. 架构总览

```text
input_ids → embed_tokens → dropout
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│  MiniMindBlock × num_hidden_layers                           │
│  每块:                                                       │
│   ┌────────────────────────────────────────┐                 │
│   │ x ─input_layernorm→ VQAttention        │ ─+residual      │
│   │ x ─post_attn_layernorm→ MLP(FFN/MoE)   │ ─+residual      │
│   └────────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
norm → lm_head → logits
```

训练长序列时按 `block_len` 分块顺序处理，块间传递（detach 后）的 VQ 状态；推理时由调用方持有每层 state dict 回传。

## 2. 配置 (config_minimind.py)

| 字段 | 默认 | 说明 |
|---|---|---|
| `n_code` | 256 | 码本大小（短码数 / 聚合缓存行数） |
| `mem_len` | 256 | 未量化滑窗 KV 缓存长度 |
| `block_len` | 256 | 分块长度 |
| `agg_cache` | True | 是否纳入码本聚合缓存 |
| `c_beta` | 0.25 | commitment loss 权重 |
| `c_gamma` | 0.99 | 码本 EMA 衰减 |

## 3. 核心模块计算逻辑

### 3.1 可学习 VQ 量化器 `LearnableVQ`

码本不是自由参数，而是由两个 running 统计量导出：

```text
码本 c = c_sum / c_count            # get_codebook()
   · c_sum (H, S, D) 累加和,  c_count (H, S) 计数
训练时 EMA 在线更新:
   c_sum  ← γ·c_sum + (1-γ)·Σ_批次归入该码字的向量
   c_count← γ·c_count + (1-γ)·Σ_批次计数
```

量化过程：

```text
z, errs² = get_shortcodes(k, c)     # argmin ||k − c||² → 短码下标 + 距离
cz = get_codewords(z, c)            # 取回码字
k_hat = sg(cz) + st(k)              # ST: 前向用码字, 梯度直通回原始 k
l_commit = mean(errs²)              # commitment loss (训练时)
```

`_sg` = detach，`_st(x) = x − sg(x)`（straight-through estimator）。

### 3.2 量化键注意力的「窗口」分支

```text
q, k, v = 投影 → (b, H, s, D)；k 量化得 k_hat (b,H,s,D), z (b,H,s)
拼接窗口: recent_k_hat = [state.k_hat, k_hat]   # (b,H,M+s,D)
          recent_v     = [state.v, v]
          recent_z     = [state.z, z]           # (b,H,M+s)
RoPE: 对 q 用当前位置, 对窗口键用 (offset-M .. offset+s-1) 绝对位置
scores_w = q_rot·k_rotᵀ · scale
掩码: causal (j − mem_len ≤ i) ∧ 缓存有效位 (j ≥ invalid) [∨ 纯滑窗 j ≥ i]
```

### 3.3 码本聚合缓存（长程记忆）

```text
码本分数 (位置无关, 用未旋转 q):  scores_c = q·cᵀ · scale   # (b,H,s,S)
加 log 偏置: scores_c −= log(count(lower))                   # 计数多 → 权重大
合并 softmax: 用共享 max 归一化, 拼接 exp 权重
out = Σ_w a_w·recent_v + Σ_c a_c·(upper/lower)   # 聚合缓存 value 均值
```

滑出窗口的位置（`recent[..., :-M]`）并入码本缓存：

```text
lower = lower + one_hot(z)              # 码字累计使用次数
upper = 加权平均合并被驱逐位置的 v       # (b,H,S,D)
窗口裁剪为最近 M 个位置, pos_offset += L
```

### 3.4 状态与推理

每层 `attn_state` 为一个 dict：

```text
pos_offset | z | k_hat | v | lower | upper
```

- `MiniMindModel.forward` 缺省用 `initial_state(batch)` 初始化；`position_ids` 由 `pos_offset` 推进。
- 长序列训练走 `_forward_blocks`：逐 block 前向，块间状态 `_detach_state` 切断梯度。
- `prepare_inputs_for_generation` 把 `past_key_values` 原样透传为 `attn_state`，并按 `cache_position` 只喂新 token。

## 4. 复杂度

- 窗口分支 O(M) / token（M 固定）。
- 码本聚合分支 O(S) / token（S 固定）。
- 总量 O(L·(M+S))，**近似线性**，可支持远超 `mem_len` 的上下文而无需全量 KV 缓存。

## 5. 特性对比

| 特性 | 标准注意力 | VQ 注意力 |
|---|---|---|
| Key 存储 | 全量 KV | 量化短码 + 码本统计量 |
| 长程记忆 | O(L²) | 码本聚合 O(L·S) |
| 额外损失 | — | commitment loss (`c_beta`) |
| 位置编码 | RoPE 全量 | 窗口内 RoPE；码本市位置无关 |