# model-qsa — 门控 Delta 线性注意力 (QSA)

> 目录：`model-qsa/`，文件：`model_minimind.py`, `config_minimind.py`
> 定位：用 **门控 Delta 网络（Gated Delta Network）** 作为线性注意力，与标准注意力按「3 线性 : 1 全」间隔混合（Qwen3-Next 风格）。
> 核心：`GatedDeltaNet` + `torch_recurrent_gated_delta_rule`。

## 1. 架构总览

```text
input_ids → embed_tokens → dropout
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│  MiniMindBlock × num_hidden_layers                           │
│  layer_types[i]:  "linear_attention" → GatedDeltaNet         │
│                   "full_attention"   → Attention(标准 GQA)   │
│  每块:                                                       │
│   ┌────────────────────────────────────────┐                 │
│   │ x ─input_layernorm→ [Linear | Full]attn│ ─+residual      │
│   │ x ─post_attn_layernorm→ MLP(FFN/MoE)   │ ─+residual      │
│   └────────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
norm → lm_head → logits
```

`layer_types` 默认：位置 `(i+1) % interval == 0` 为 `full_attention`，其余为 `linear_attention`（interval=4 → 3 线性 1 全）。

## 2. 配置 (config_minimind.py)

| 字段 | 默认 | 说明 |
|---|---|---|
| `linear_num_value_heads` | 4 | value 头数 |
| `linear_num_key_heads` | 4 | key 头数 |
| `linear_key_head_dim` | 128 | key 每头维 |
| `linear_value_head_dim` | 128 | value 每头维 |
| `linear_conv_kernel_dim` | 4 | 因果 conv 核 |
| `linear_full_attn_layer` | 4 | （与 interval 同义） |
| `linear_attn_layer_interval` | 4 | 全注意力层间隔 |
| `layer_types` | 自动生成 | 每层类型 |

## 3. GatedDeltaNet 计算逻辑

输入 `x (b, L, hidden)`，状态 `past=(kv_state, conv_state)`。

### 3.1 投影

```text
mixed_qkv = in_proj_qkv(x)      # → [key_dim, key_dim, value_dim]
z   = in_proj_z(x)              # 门控向量 → (b,L,H_v,D_v)
b   = in_proj_b(x)              # beta 投影 → (b,L,H_v)
a   = in_proj_a(x)              # 门控衰减系数
```

### 3.2 因果卷积（局部混合 + 状态缓存）

```text
mixed_qkv_t = causal_conv1d(mixed_qkv)   # 分组深度卷积 K=4
   · 首次: 全序列卷积, 缓存末尾 (K-1) 个输入
   · 增量: 拼接历史缓存做零填充卷积, 原地滑动 conv_state
query, key, value = split(mixed_qkv_t)   # 每个拆成多头的 (b,L,H,D)
```

### 3.3 门控参数

```text
beta = sigmoid(b)
g    = -exp(A_log) * softplus(a + dt_bias)   # 每头衰减门 (≤ 0)
若 H_v != H_k: query/key repeat_interleave 对齐 value 头
```

### 3.4 门控 Delta 递归（`torch_recurrent_gated_delta_rule`）

状态 `S (b, H_v, D_k, D_v)`（key 维 × value 维矩阵），每一步：

```text
l2 归一化 query、key 后再 / sqrt(D_k)
S_t = S_{t-1} · exp(g_t) + beta_t · (v_t − S_{t-1}·k_t) ⊗ k_t
o_t = S_t · q_t
```

含义：key 与已累积记忆的差异 `(v − S·k)` 作为 Delta 项，用 `beta` 缩放写入状态矩阵；**每次正确的 key 访问后持续抹除 / 覆盖**对应记忆，从而实现比固定衰减更强的「写入」能力。

### 3.5 输出

```text
out = RMSNormGated(out, z)      # GatedRMSNorm: RMSNorm(x) * silu(z)
out = out_proj(out)
```

## 4. 状态缓存约定

- 线性层：`(kv_state (b,H_v,D_k,D_v), conv_state (b,conv_dim,K-1))`，定长、不含位置。
- 全注意力层：`(k, v)`。
- `MiniMindModel.forward` 遍历层，从首个 `full_attention` 层的 `(k,v)` 序列维推断 `start_pos`。

## 5. 复杂度与特点

- 单 token 状态更新 O(H_v·D_k·D_v)，递归生成 O(1) / 步；长序列训练可用并行扫描替代递归。
- 相比纯固定衰减（RWKV/Mamba）的线性注意力，Delta 规则显式建模「写入冲突消除」，长程记忆更精准。
- key/value 头维度独立配置（默认 128），与全注意力层的 `head_dim` 无关。