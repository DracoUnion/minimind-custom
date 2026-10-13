# model-engram — n-gram 哈希外部记忆

> 目录：`model-engram/`，文件：`model_minimind.py`, `config_minimind.py`
> 定位：在标准 MiniMind 的若干层**注意力之前**注入一张由 **n-gram 滚动哈希**索引的**可学习外部记忆表**——用 finetune 数据把高频 n-gram 的知识固化到权重量里，缓解小模型的幻觉。
> 核心：`_ngram_hashes`（加权异或滚动哈希）+ `MultiHeadEmbedding`（偏移分表的哈希嵌入）+ `Engram`（门控 + 短卷积）。

## 1. 架构总览

```text
input_ids → embed_tokens → dropout
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│  MiniMindBlock × num_hidden_layers                           │
│  对 engram_layer_ids 命中的层:                               │
│                                                                │
│   ┌────────────────────────────────────────────────────┐      │
│   │ ① Engram:                                           │      │
│   │    input_ids ─→ n-gram hashes ─→ 嵌入表(偏移分头)    │      │
│   │                → key/query 门控 → 残差注入 hidden    │      │
│   │    hidden = hidden + engram(hidden, input_ids)      │      │
│   │ ② x ─input_layernorm→ Attention → ─+residual        │      │
│   │ ③ x ─post_attn_layernorm→ MLP → ─+residual          │      │
│   └────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
norm → lm_head → logits
```

## 2. 配置 (config_minimind.py)

| 字段 | 默认 | 说明 |
|---|---|---|
| `use_engram` | False | 总开关 |
| `engram_layer_ids` | `[1, N-2]` | 插入层索引 |
| `engram_max_ngram_size` | 3 | 最大 n-gram 阶数 N |
| `engram_n_embed_per_ngram` | 512 | 每 n-gram 的嵌入维（查表后拼总维） |
| `engram_n_head_per_ngram` | 8 | 每 n-gram 的哈希头数 |
| `engram_vocab_size` | `[512, 1024][:N-1]` | 每阶哈希槽数（向下取质数） |
| `engram_kernel_size` | 4 | 短卷积核 |
| `engram_seed` | 0 | 乘子随机种子 |
| `engram_pad_id` | eos | 边界 n-gram 的左填充 token |

## 3. 核心计算逻辑

### 3.1 n-gram 滚动哈希（`_ngram_hashes`）

```text
输入 input_ids [B, T], 构建 shifts[k] = 左移 k 位(前 k 位用 pad_id 填充)
每阶 n、每头 j:
    mix = shifts[0]*multipliers[0]                       # 加权异或混合前 n 个位置
    mix = xor(mix, shifts[1]*m[1]) ... until n 个
    hash_id[n,j] = mix mod moduli[(n-2)*H + j]           # 每个 (n, head) 唯一质数模数
输出 [B, T, (N-1)*H] 的 int64 哈希 id
```

特点：**同一位置同一 n-gram 上下文恒定同一哈希**，因此每个哈希槽（码字）学到的 embedding 能复用到任何出现该 n-gram 的位置。

### 3.2 哈希嵌入（`MultiHeadEmbedding`）

```text
单张 Embedding 表, total = Σ_moduli 个词
offsets[ head ] = 各头槽数前缀和
emb = embedding(input_ids + offsets)     # 每头在表内用偏移切出自己的语义空间
```

`hash_id + offsets` 后查表 → 每头取 `D_head = engram_n_embed_per_ngram // H` 维，拼接得到 `[B, L, (N-1)*n_embed]`。

### 3.3 Engram 门控注入

```text
embeddings = multi_head_embedding(hashes).flatten()      # [B,L,engram_hidden]

key   = norm_key(key_proj(embeddings))                   # 记忆侧
query = norm_query(hidden_states)                        # 主干侧
gate  = sigmoid( |(key·query)/sqrt(D)|^0.5 · sign )      # 相似度门控, ≥0
value = gate * value_proj(embeddings)                    # [B,L,D]

conv_in = pad_left(value, (K-1)*dilation)   # 因果短卷积, dilation=max_ngram
conv_out = short_conv(conv_in)[.., :L]
out = value + conv_out                                    # 记忆 + 位置局部平滑
```

最后 `hidden = hidden + engram_out`——以残差形式叠加到主干，注意力之前。

### 3.4 模型侧对齐（长上下文）

`MiniMindModel.forward` 接收 `full_input_ids`（完整生成序列），从 `start_pos` 切出当前窗口喂给 Engram：

```text
engram_ids = full_input_ids[:, start_pos:]   # 无 full_ids 时退化为当前 input_ids
逐层: layer(hidden, pos_emb, ..., input_ids=engram_ids)
```

这样 n-gram 左移位能回看缓存中的历史 token。

## 4. 关键设计权衡

- **记忆容量**：质数模数 × 嵌入维决定外存大小；`engram_vocab_size` 可调大以更贴近 Engram 论文章量。
- **插入位置**：默认首尾两层（`[1, N-2]`），避免过于扰动内部表示。
- **与注意力互补**：哈希索引只依赖离散 token 序列，物理分辨率固定；查询门控 `query·key` 决定当前 token 是否「回忆」该 n-gram 记忆。
- **无 extra loss**：记忆表直接经主干梯度端到端学习，无辅助目标。