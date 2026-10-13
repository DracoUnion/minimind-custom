# model-rwkv — RWKV5 WKV 线性注意力

> 目录：`model-rwkv/`，文件：`model_minimind.py`, `config_minimind.py`
> 定位：用 **RWKV5 的 WKV 扫描**替换注意力层。默认全层 RWKV，也可与标准注意力混合（`layer_types`）。
> 核心：`RwkvAttention` + 数值稳定的 `rwkv_wkv_scan`（log 域递推）。

## 1. 架构总览

```text
input_ids → embed_tokens → dropout
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│  MiniMindBlock × num_hidden_layers                           │
│  layer_types[i]:  "rwkv" → RwkvAttention                     │
│                   "full_attention" → Attention(标准 GQA)     │
│  每块:                                                       │
│   ┌────────────────────────────────────────┐                 │
│   │ x ─input_layernorm→ [RWKV | Attention] │ ─+residual      │
│   │ x ─post_attn_layernorm→ MLP(FFN/MoE)   │ ─+residual      │
│   └────────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
norm → lm_head → logits
```

## 2. 配置 (config_minimind.py)

| 字段 | 默认 | 说明 |
|---|---|---|
| `layer_types` | 全 "rwkv" | 每层 mixer 类型（可混入 "full_attention"） |
| `attention_hidden_size` | hidden_size | WKV 注意力维（key/value 维） |
| `num_attention_heads` 等 | 与基线同 | 全注意力层用 |
| `rms_norm_eps`, `rope_*` | 与基线同 | 全注意力层用 |

## 3. RwkvAttention 计算逻辑

输入 `x (b, L, hidden)`；状态 `past_key_value = [num, den, max, last_hidden]`。

### 3.1 time-mix 门控（时间混合）

```text
shifted = time_shift(hidden)   # 上移一行: [0; h[:-1]]
key      = hidden * time_mix_key      + shifted * (1 - time_mix_key)
value    = hidden * time_mix_value    + shifted * (1 - time_mix_value)
receptance= hidden*time_mix_receptance+ shifted*(1 - time_mix_receptance)
key      = key_proj(key)
value    = value_proj(value)
receptance = sigmoid(receptance_proj(receptance))
```

### 3.2 WKV 扫描（数值稳定 log 域）

```text
输入: time_decay (负), time_first, key, value; 状态 num/den/max (各 b,hidden)
time_decay = -exp(time_decay)                 # 保证负
每个时间步 t:
    k_t = key_t + time_first                  # 时间偏置
    m_out = max(max_state, k_t)
    e1 = exp(max_state - m_out); e2 = exp(k_t - m_out)
    out_t = (e1*num + e2*v_t) / (e1*den + e2) # 当前 token 输出
    # 状态更新
    m_st = max(max_state + time_decay, key_t)
    e1 = exp(max_state + time_decay - m_st); e2 = exp(key_t - m_st)
    num = e1*num + e2*v_t
    den = e1*den + e2
    max = m_st
```

WKV 公式（非 log 域形式）：

```text
w = exp(time_decay)
out_t = (Σ_{t'<t} w^{t-1-t'} · exp(key_t')·v_t' + exp(key_t + time_first)·v_t)
      / (Σ_{t'<t} w^{t-1-t'} · exp(key_t') + exp(key_t + time_first))
```

### 3.3 输出

```text
output = output_proj(receptance * rk)
```

### 3.4 参数初始化（`_init_rwkv_weights`）

```text
time_mix 系数随层索引做幂次衰减（低层高混合、高层低混合）
time_decay = -5 + 8*(h/attn_hid)^(0.7 + 1.3*ratio)
time_first = log(0.3) + zigzag(h)
```

## 4. 状态缓存约定

RWKV 层缓存为 `(num_state, den_state, max_state, last_hidden)`：

| 分量 | 形状 | 含义 |
|---|---|---|
| `num` | `(b, hidden)` | 累加分子 Σ exp(key)·v |
| `den` | `(b, hidden)` | 累加分母 Σ exp(key) |
| `max` | `(b, hidden)` | log 域最大值（防溢出） |
| `last_hidden` | `(b, hidden)` | 上一时间步 hidden（供 L==1 时 time-mix 用） |

全注意力层缓存为 `(k, v)`。`MiniMindModel.forward` 从任一全注意力层的 `(k,v)` 序列维推断 `start_pos`。

## 5. 复杂度

WKV 每步 O(hidden)，总 O(L·hidden)，与序列长度线性；时间衰减由 `time_decay` 参数学习，无需 RoPE。