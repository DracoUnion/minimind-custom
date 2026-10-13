# model-mamba — Mamba2 混合注意力

> 目录：`model-mamba/`，文件：`model_minimind.py`, `config_minimind.py`
> 定位：在 MiniMind 骨架中把部分注意力层替换为 **Mamba2 选择性状态空间模型（SSM）**，形成「3 线性 + 1 全注意力」的混合结构（Nemotron H 范式）。
> 核心：`MambaMixer`（分块并行扫描 `_chunk_scan` + 递归 `_recurrent_update`）。

## 1. 架构总览

```text
input_ids → embed_tokens → dropout
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│  MiniMindBlock × num_hidden_layers                           │
│  layer_types[i]:  "linear_attention" → MambaMixer            │
│                   "full_attention"   → Attention(标准 GQA)   │
│  每块:                                                       │
│   ┌────────────────────────────────────────┐                 │
│   │ x ─input_layernorm→ [Mamba | Attention]│ ─+residual      │
│   │ x ─post_attn_layernorm→ MLP(FFN/MoE)   │ ─+residual      │
│   └────────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
norm → lm_head → logits
```

`layer_types` 默认按 `["linear"]*3 + ["full"]` 循环铺满所有层，即每 4 层 1 个全注意力。

## 2. 配置 (config_minimind.py)

| 字段 | 默认 | 说明 |
|---|---|---|
| `layer_types` | 3 linear + 1 full 循环 | 每层 mixer 类型 |
| `mamba_num_heads` | 8 | SSM 头数 |
| `mamba_head_dim` | hidden/heads | 每头特征维 |
| `mamba_intermediate_size` | heads*head_dim | 中间维 |
| `ssm_state_size` | 16 | 状态维 n |
| `conv_kernel` | 4 | 因果 conv1d 核大小 |
| `chunk_size` | 64 | 分块扫描块大小 (≥ n) |
| `n_groups` | 1 | B/C 组数 |
| `time_step_min / max` | 1e-3 / 0.1 | dt 限幅 |
| `use_conv_bias` | False | conv/投影 bias |

## 3. MambaMixer 计算逻辑

输入 `x (b, L, hidden)`，输出 `(out, past_kv)`，`past_kv = (conv_state, ssm_state)`。

### 3.1 投影与门控

```text
projected = in_proj(x)                        # 一次投影产出三部分
gate | conv_bc | dt = split(projected, [intermediate, conv_dim, num_heads])
dt += dt_bias  (每头)                          # 时间步
conv_dim = intermediate + 2*n_groups*ssm_state_size   # C、B 各一份
```

### 3.2 因果卷积（短程局部混合）

```text
conv_out = conv1d(conv_bc, kernel=K, groups=conv_dim)  # 因果: padding=K-1 后裁尾
   · 预填充 L>1:  F.conv1d(..., padding=K-1)[.., :L]
   · 生成 L==1:  增量卷积 _causal_conv1d_update（滑窗缓存 conv_state）
```

conv 输出拆为 `x | C | B`：

```text
x  → (b, L, num_heads, head_dim)
C  → (b, L, n_groups, n)     # 输出投影矩阵
B  → (b, L, n_groups, n)     # 输入投影矩阵
A  = -exp(A_log)             # 每头负衰减率
```

### 3.3 SSM 扫描

#### 状态方程（连续→离散）

```text
dA = exp(dt * A)              # dt = softplus(dt) 限幅 [min, max]
x_ssm = dt * B ⊗ x            # 输入加权 (b, L, H, P, N)
状态: s_t = dA * s_{t-1} + x_ssm
输出: y_t = C · s_t + D * x_t  # D 直接跳连
```

#### `_chunk_scan`（预填充 / 训练，分块并行）

```text
把序列切成 nc 个 chunk（补齐 pad）
每个 chunk 内:
    dBx = dt * B * x                          # (B, nc, chunk, H, N, P)
    L_cum = cumsum(log dA)                     # 块内累计 log 衰减
    decay(l, s) = exp(L_l - L_s), 仅 s ≤ l     # 因果对角衰减核 O(chunk²)
    y_intra[l] = Σ_s decay(l,s) * C[l]·dBx[s]  # 块内贡献
块之间:
    running = 上一块末位状态 (B, H, P, N)
    y_off[l] = exp(L_l) * C[l]·running         # 前块贡献
    末位状态更新: running = exp(L_chunk)*running + Σ_s exp(L_c-L_s)·dBx[s]
输出 = y_intra + y_off; 返回 final_state
```

#### `_recurrent_update`（生成 L==1）

```text
dA = exp(dt*A);  dBx = dt * B ⊗ x
new_state = state * dA + dBx
y = C · new_state + D * x
```

### 3.4 输出门控

```text
out = _GatedRMSNorm(out, gate)    # RMSNorm(x) * gate（Zamba2 风格逐头门控）
out = out_proj(out)
```

## 4. 状态缓存约定

- 全注意力层：`(k, v)`，随序列增长。
- Mamba 层：`(conv_state (b, conv_dim, K-1), ssm_state (b, num_heads, head_dim, ssm_state_size))`。
- `MiniMindModel.forward` 用任一**全注意力层**的 `(k, v)` 序列维推断 `start_pos`（Mamba 状态不含位置信息）。

## 5. 参数量参考

SSM 每头参数约 `3*head_dim*n`（A/D/dt 各一层级）+ conv + 投影，远小于同宽度全注意力的 Q/K/V/O 投影，因此混合模型比纯注意力更省参、推理更快。