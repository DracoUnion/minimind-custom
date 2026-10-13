# MiniMind 模型架构文档

本目录记录了 `model-*` 系列模型的结构与模块计算逻辑。系列以同一套 **MiniMind 基础 Transformer**（`model-einops`）为骨架，每个子目录替换或增强其中一个核心部分，形成不同的 Token-Mixer / 记忆增强 / 生成范式，全部共享相同的接口约定。

```
基准: model-einops  (标准 GQA 注意力 + MoE)
        │
        ├─ model-mamba  ── 用 Mamba2 选择性 SSM 替换部分注意力层 (混合)
        ├─ model-rwkv   ── 用 RWKV5 WKV 线性注意力替换部分注意力层 (混合)
        ├─ model-qsa    ── 用门控 Delta 网络(线性注意力)替换部分注意力层 (混合)
        ├─ model-vq     ── 用 VQ 量化键 + 码本聚合缓存替换注意力层 (长上下文)
        ├─ model-engram ── 在注意力前叠加 n-gram 哈希外部记忆 (从 token 序列查索引表)
        └─ model-diffusion ── 彻底重构为 Encoder(自回归) + Decoder(双向) 的 block diffusion
```

## 一、模型矩阵

| 子目录 | Token Mixer | 位置编码 | 状态缓存 | 复杂度/步 | 核心论文 |
|---|---|---|---|---|---|
| `model-einops` | 标准 GQA 注意力 | RoPE | `(k, v)` KV Cache | O(L²) | Attention Is All You Need |
| `model-mamba` | Mamba2 SSM + 部分全注意力 | RoPE (仅全注意力层) | `(conv_state, ssm_state)` | O(L) | Mamba-2, Nemotron H |
| `model-rwkv` | RWKV5 WKV + 部分全注意力 | 无 (WKV 自带时间衰减) | `(num, den, max, last_hidden)` | O(L) | RWKV-5 |
| `model-qsa` | 门控 Delta 网络 + 部分全注意力 | RoPE (仅全注意力层) | `(kv_state, conv_state)` | O(L) | Qwen3.5 门控 Delta |
| `model-vq` | VQ 量化键注意力 | RoPE (窗口 + 码本市) | 每层 state dict + 码本聚合缓存 | O(L·S) 近似 | Transformer-VQ |
| `model-engram` | 标准注意力 + n-gram 外部记忆 | RoPE | `(k, v)` KV Cache | O(L²) + 查表 O(1) | Engram |
| `model-diffusion` | Encoder 因果 / Decoder 双向 | RoPE | 共享 KV Cache (decoder 只读) | O(L²) 多步去噪 | DiffusionGemma |

> S = 码本大小 (`n_code`), L = 序列长度。

## 二、公共组件

以下模块在 7 个实现中**几乎逐字相同**（各目录下 `model_minimind.py` 顶部均有）：

### 2.1 RMSNorm

```text
x ──→ x² ──mean(-1)──→ +eps ──rsqrt──→ * x ──→ * weight ──→ out
```

公式：`RMSNorm(x) = x / sqrt(mean(x²) + eps) * γ`，fp32 计算、回灌原 dtype。

### 2.2 RoPE（预计算频率）

`precompute_freqs_cis(dim, end, rope_base, rope_scaling)` 预计算 `(end, head_dim)` 的余弦/正弦表并注册为 buffer（`persistent=False`）。支持 YaRN 长度外推（`rope_scaling.json`）。

```text
freqs[i] = 1 / base^(2i/dim)
freqs_cos[m, d] = cos(m * freqs[d])   (维度复制一份对齐 head_dim)
```

### 2.3 旋转应用与 GQA 重复

```text
rotate_half(x) = cat([-x[后一半], x[前一半]])
RoPE(x) = x * cos + rotate_half(x) * sin
repeat_kv(x, r) → (b, s, (h*r), d)     # GQA: 每个 KV 头复制 r 次对齐 Q 头
```

### 2.4 SwiGLU 前馈

```text
FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
```

### 2.5 MoE 前馈（可选）

```text
scores = softmax(gate(x))                     # (b*s, E)
topk → top-k 专家索引
y = Σ_i expert_i(x) * weight_i
aux_loss = load_balance(scores, topk)         # 负载均衡辅助损失 (自回归训练时叠加)
```

### 2.6 统一模型外壳

除 `model-diffusion` 外，其余模型外壳一致：

```text
input_ids ──embed_tokens(绑定 lm_head)──→ dropout → [MiniMindBlock × N] → norm → hidden
hidden ──lm_head──→ logits      (可选 labels → 移位交叉熵 loss)
```

`MiniMindForCausalLM` 继承 `PreTrainedModel, GenerationMixin`，自带 `_generate` 采样循环（temperature / top_k / top_p / repetition_penalty / KV cache）。

## 三、各模型文档

- [model-einops — 基线标准 Transformer](einops.md)
- [model-mamba — Mamba2 混合注意力](mamba.md)
- [model-rwkv — RWKV5 WKV 线性注意力](rwkv.md)
- [model-qsa — 门控 Delta 线性注意力 (QSA)](qsa.md)
- [model-vq — Transformer-VQ 量化注意力](vq.md)
- [model-engram — n-gram 哈希外部记忆](engram.md)
- [model-diffusion — Block Diffusion 生成（Encoder-Decoder）](diffusion.md)

## 四、快速定位

| 想了解 | 看哪个文件 |
|---|---|
| 层类型如何按 `layer_types` 分配 mixer | `mamba.md`, `rwkv.md`, `qsa.md` |
| 线性注意力的状态缓存长什么样 | `mamba.md` §4, `rwkv.md` §4, `qsa.md` §4 |
| 状态空间 `_chunk_scan` 分块并行扫描 | `mamba.md` §3.3 |
| WKV 数值稳定 log 域扫描 | `rwkv.md` §3.2 |
| 门控 Delta 递归更新规则 | `qsa.md` §3.2 |
| VQ 码本 EMA 更新与聚合缓存 | `vq.md` §3.2, §3.3 |
| n-gram 滚动哈希如何查表 | `engram.md` §3.2 |
| Encoder/Decoder 共享 KV cache | `diffusion.md` §3 |
| Self-conditioning 注入 | `diffusion.md` §3.3 |
| 训练 / 推理如何切分 prefix 与 canvas | `diffusion.md` §3.4 |