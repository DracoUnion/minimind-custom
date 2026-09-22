"""
MiniMind 语言模型实现（Transformer-VQ 版本）

在 MiniMind 标准 Transformer 的基础上，把自注意力替换为 Transformer-VQ 的
VQ Attention（《Transformer-VQ: Linear-Time Transformers via Vector
Quantization》）。

核心机制:
1. 键先经可学习 VQ 码本量化成短码 shortcode，再用码字参与注意力。
2. 注意力分数由两部分组成，二者合并做一次 softmax:
   - 滑动窗口 (recent / XL cache): 最近 mem_len 个位置的量化键 k_hat，
     与当前 block 一起做因果窗口注意力，位置信息由 RoPE 提供。
   - 码本聚合缓存 (aggregated cache): 把已经滑出窗口的历史位置按 shortcode
     归并，只保留每个码字对应的 value 均值与使用计数，注意力代价降为 O(L·S)。
3. 码本 (c_sum / c_count) 用 EMA 上线更新，无需额外优化器分组；
   量化误差作为 commitment loss 回传。

为复用 MiniMind 的接口，保留以下结构并与原实现一致:
- Q/K/V 投影、Query/Key RMSNorm、RoPE、GQA
- SwiGLU / MoE FFN、RMSNorm、嵌入与 lm_head 权重绑定

与原始 Transformer-VQ 的差异:
- 用 RoPE 提供窗口内的相对位置，替代原来的正弦 XL 相对偏置。
- 码本用 EMA buffer 直接更新，替代原来的 surrogate codebook loss。
- K/V 使用 MiniMind 的独立投影，而非融合的 kvg 投影。
"""
import math, torch, torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from .config_minimind import MiniMindConfig

# 用于屏蔽的极大负数（代替 -inf，避免 NaN）
MASK_INFTY_APPROX = 1e30


# ==============================================================================
# VQ 辅助函数与量化器
# ==============================================================================

def _sg(x):
    """stop-gradient: 前向不变，反向不传播梯度。"""
    return x.detach()


def _st(x):
    """
    straight-through estimator: 前向恒等，反向零梯度。
    等价于 x - stop_gradient(x)。
    """
    return x - _sg(x)


def get_shortcodes(vecs, codebook):
    """
    VQ 编码：为每个键向量查找最近的码字。

    参数:
        vecs: [B, H, L, K] 待量化向量（最后一个维度为特征维）
        codebook: [H, S, K] 码本（每个头一份）

    返回:
        z: [B, H, L] int32 短码下标
        errs2: [B, H, L] 到最近码字的平方欧氏距离
    """
    B, H, L, K = vecs.shape
    # ||a - b||^2 = ||a||^2 - 2 a·b + ||b||^2
    vecs_sq = vecs.square().sum(dim=-1, keepdim=True)            # [B, H, L, 1]
    cb_sq = codebook.square().sum(dim=-1)                        # [H, S]
    dots = torch.einsum("bhlk,hsk->bhls", vecs, codebook)        # [B, H, L, S]
    diffs2 = vecs_sq - 2.0 * dots + cb_sq.unsqueeze(0).unsqueeze(2)
    z = torch.argmin(diffs2, dim=-1)                             # [B, H, L]
    errs2 = torch.gather(diffs2, dim=-1, index=z.unsqueeze(-1)).squeeze(-1)
    return z.to(torch.int32), F.relu(errs2)                       # 数值噪声可能产生微小负值


def get_codewords(shortcodes, codebook):
    """按短码下标从码本取回码字。shortcodes [B,H,L] -> [B,H,L,K]"""
    B = shortcodes.shape[0]
    codebook = codebook.unsqueeze(0).expand(B, -1, -1, -1)        # [B, H, S, K]
    idx = shortcodes.long().unsqueeze(-1).expand(-1, -1, -1, codebook.size(-1))
    return torch.gather(codebook, dim=2, index=idx)


class LearnableVQ(nn.Module):
    """
    可学习的向量量化器。

    码本本身不是自由参数，而是由两个 running 统计量导出:
        c = c_sum / c_count
    训练时用 EMA 在线更新这两个统计量，从而平滑地跟踪键的分布。
    """

    def __init__(self, n_head, n_code, d_k, gamma=0.99, dtype=torch.float32):
        super().__init__()
        self.n_head, self.n_code, self.d_k, self.gamma = n_head, n_code, d_k, gamma
        # 用 buffer 保存，随模型迁移设备、随 state_dict 保存，但不参与反向传播
        self.register_buffer("c_sum", torch.randn(n_head, n_code, d_k, dtype=dtype) / math.sqrt(d_k))
        self.register_buffer("c_count", torch.ones(n_head, n_code, dtype=dtype))

    def get_codebook(self):
        """导出码本 c = c_sum / c_count（不参与梯度）。"""
        return self.c_sum / torch.clamp(self.c_count.unsqueeze(-1), min=0.01)

    @torch.no_grad()
    def _ema_update(self, z, vecs, loss_mask=None):
        """用当前 batch 的量化统计更新码本 running 统计量。"""
        S = self.n_code
        r = F.one_hot(z.long(), num_classes=S).to(vecs.dtype)      # [B, H, L, S]
        if loss_mask is not None:
            r = r * loss_mask.unsqueeze(1).unsqueeze(-1)           # [B, 1, L, 1]
        # 当前 batch 的码字累加和与计数
        batch_sum = torch.einsum("bhls,bhld->hsd", r, vecs)        # [H, S, d]
        batch_cnt = r.sum(dim=(0, 2))                             # [H, S]
        g = self.gamma
        self.c_sum.mul_(g).add_(batch_sum, alpha=1 - g)
        self.c_count.mul_(g).add_(batch_cnt, alpha=1 - g)

    def forward(self, k, loss_mask=None):
        """
        量化键 k 并返回量化结果。

        参数:
            k: [B, H, L, K] 未旋转的键
            loss_mask: [B, L] 0/1 掩码，用于忽略 padding 位置的统计
        返回 dict: quantized / shortcodes / l_commit / errs2
        """
        orig_dtype = k.dtype
        k_hp = k.float()
        c = self.get_codebook()
        z, errs2 = get_shortcodes(vecs=k_hp, codebook=c)
        cz = get_codewords(shortcodes=z, codebook=c).to(orig_dtype)
        # ST: 前向用码字，反向梯度直接传给原始键
        k_hat = _sg(cz) + _st(k)

        if self.training:
            self._ema_update(z, k_hp, loss_mask)
            if loss_mask is not None:
                l_commit = (loss_mask.unsqueeze(1) * errs2.to(orig_dtype)).sum(dim=-1).mean()
            else:
                l_commit = errs2.to(orig_dtype).mean()
        else:
            l_commit = k.new_zeros(())
        return dict(quantized=k_hat, shortcodes=z, l_commit=l_commit)


# ==============================================================================
# 基础组件
# ==============================================================================

class RMSNorm(torch.nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization) 根均方层归一化

    output = x / sqrt(mean(x^2) + eps) * gamma
    参考论文: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA 中的 Key/Value 头重复。"""
    if n_rep == 1:
        return x
    return repeat(x, 'b s h d -> b s (h r) d', r=n_rep)


# ==============================================================================
# VQ 注意力层
# ==============================================================================

class VQAttention(nn.Module):
    """
    带 VQ 压缩键的注意力层。

    状态 (attn_state) 内容:
        pos_offset : 已经处理过的 token 总数（标量）
        z          : 窗口内各位置的 shortcode        [B, H, M]
        k_hat      : 窗口内各位置的量化键（未旋转）   [B, H, M, D]
        v          : 窗口内各位置的 value             [B, H, M, D]
        lower      : 每个码字累计的使用次数           [B, H, S]
        upper      : 每个码字累计的 value 之和        [B, H, S, D]
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.n_head = config.num_attention_heads
        self.n_kv_head = config.num_key_value_heads
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(config.hidden_size, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # 可学习 VQ 量化器（键的码本）
        self.quantizer = LearnableVQ(
            n_head=self.n_head, n_code=config.n_code,
            d_k=self.head_dim, gamma=config.c_gamma,
        )

    # ------------------------------------------------------------------
    # 递归状态
    # ------------------------------------------------------------------
    def initial_state(self, batch_size, device=None, dtype=torch.float32):
        """构造初始注意力状态（空窗口 + 空码本缓存）。"""
        M, S, H, D = self.config.mem_len, self.config.n_code, self.n_head, self.head_dim
        if device is None:
            device = torch.device("cpu")
        prefix = [batch_size, H]
        return dict(
            pos_offset=0,
            z=torch.full((*prefix, M), S, dtype=torch.int32, device=device),
            k_hat=torch.zeros((*prefix, M, D), dtype=dtype, device=device),
            v=torch.zeros((*prefix, M, D), dtype=dtype, device=device),
            lower=torch.zeros((*prefix, S), dtype=dtype, device=device),
            upper=torch.zeros((*prefix, S, D), dtype=dtype, device=device),
        )

    # ------------------------------------------------------------------
    # 位置编码
    # ------------------------------------------------------------------
    def _rope(self, x, positions):
        """
        对 [B, H, T, D] 的 x 施加 RoPE，positions 为 [T] 的绝对位置。
        """
        d = self.head_dim
        inv = 1.0 / (self.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=positions.device) / d))
        freqs = torch.outer(positions.float(), inv)                  # [T, d//2]
        emb = torch.cat([freqs, freqs], dim=-1)                      # [T, d]
        cos = emb.cos().to(x.dtype).unsqueeze(0).unsqueeze(0)   # [1,1,T,D]
        sin = emb.sin().to(x.dtype).unsqueeze(0).unsqueeze(0)
        def rotate_half(t):
            return torch.cat((-t[..., t.shape[-1] // 2:], t[..., : t.shape[-1] // 2]), dim=-1)
        return x * cos + rotate_half(x) * sin

    # ------------------------------------------------------------------
    # 掩码
    # ------------------------------------------------------------------
    @staticmethod
    def get_causal_mask(block_len, mem_len, invalid_len, with_locality):
        """
        构造窗口注意力的布尔掩码 [L, M+L]。

        参数:
            invalid_len: 初始尚未填充的有效缓存长度之外的位置数
            with_locality: 是否额外限制只看窗口内 (j >= i)，即纯滑动窗口
        """
        i = torch.arange(block_len, dtype=torch.int32).unsqueeze(-1)          # [L,1]
        j = torch.arange(mem_len + block_len, dtype=torch.int32).unsqueeze(0)  # [1,M+L]
        alloc_mask = j >= invalid_len.unsqueeze(0)                            # 缓存中有效位置
        causal_mask = j - mem_len <= i                                       # 因果
        keep = alloc_mask & causal_mask
        if with_locality:
            keep = keep & (j >= i)
        return keep

    @staticmethod
    def get_agg_biases(lower):
        """把码字使用计数转成 log 空间偏置；计数为 0 的码字被屏蔽。"""
        return torch.where(
            lower == 0,
            torch.full_like(lower, MASK_INFTY_APPROX),
            -torch.log(torch.clamp(lower, min=1.0)),
        )

    # ------------------------------------------------------------------
    # 注意力核心
    # ------------------------------------------------------------------
    def _attn(self, q, k_hat, v, z, position_ids, state):
        """
        参数:
            q      : [B, H, L, D] query（未旋转；窗口分支内部再施加 RoPE）
            k_hat  : [B, H, L, D] 量化后的键（未旋转）
            v      : [B, H, L, D] value
            z      : [B, H, L]    短码
            position_ids: [L] 当前 block 的绝对位置
        返回:
            out: [B, H, L, D] 注意力输出
            recent_z / recent_k_hat / recent_v: 拼接后的窗口内容（供状态更新）
        """
        B, H, L, D = q.shape
        M, S = self.config.mem_len, self.config.n_code
        scale = self.head_dim ** -0.5

        # --- 拼接窗口: 历史缓存 + 当前 block ---
        recent_k_hat = torch.cat([state["k_hat"], k_hat], dim=-2)   # [B,H,M+L,D]
        recent_v = torch.cat([state["v"], v], dim=-2)               # [B,H,M+L,D]
        recent_z = torch.cat([state["z"], z], dim=-1)               # [B,H,M+L]
        # 窗口内各键的绝对位置: 历史为 pos_offset-M..pos_offset-1，当前为 position_ids
        offset = state["pos_offset"]
        if M > 0:
            hist_pos = torch.arange(offset - M, offset, device=position_ids.device)
            window_pos = torch.cat([hist_pos, position_ids])         # [M+L]
        else:
            window_pos = position_ids

        # 窗口分数需要位置信息，对 query 与窗口键施加 RoPE
        q_rot = self._rope(q, position_ids)                          # [B,H,L,D]
        k_rot = self._rope(recent_k_hat, window_pos)                 # [B,H,M+L,D]
        scores_w = torch.einsum("bhld,bhwd->bhlw", q_rot, k_rot) * scale

        # 因果 + 窗口掩码
        invalid_len = torch.as_tensor(max(M - offset, 0), device=q.device, dtype=torch.int32)
        keep = self.get_causal_mask(
            block_len=L, mem_len=M, invalid_len=invalid_len,
            with_locality=not self.config.agg_cache,
        ).to(q.device)                                               # [L, M+L]
        scores_w = scores_w.masked_fill(
            ~keep.unsqueeze(0).unsqueeze(0),
            -MASK_INFTY_APPROX,
        )

        # --- 码本聚合分数（位置无关，用未旋转的 query） ---
        c = self.quantizer.get_codebook().to(q.dtype)                # [H,S,D]
        scores_c = torch.einsum("bhld,hsd->bhls", q, c) * scale      # [B,H,L,S]
        if self.config.agg_cache:
            # 加 log(count) 偏置，使计数多的码字获得更大的权重
            scores_c = scores_c - self.get_agg_biases(state["lower"].to(q.dtype)).unsqueeze(-2)
        else:
            scores_c = scores_c.masked_fill(
                torch.ones_like(scores_c, dtype=torch.bool), -MASK_INFTY_APPROX
            )

        # --- 合并 softmax ---
        max_scores = _sg(torch.maximum(
            scores_w.max(dim=-1)[0], scores_c.max(dim=-1)[0],
        ))                                                           # [B,H,L]
        a_w = torch.exp(scores_w - max_scores.unsqueeze(-1))
        a_c = torch.exp(scores_c - max_scores.unsqueeze(-1))
        denom = a_w.sum(dim=-1) + a_c.sum(dim=-1)                    # [B,H,L]

        out = torch.einsum("bhlw,bhwd->bhld", a_w / denom.unsqueeze(-1), recent_v)
        if self.config.agg_cache:
            # 码本缓存存的是 value 之和，除以计数得到均值
            code_v = state["upper"] / torch.clamp(state["lower"], min=1.0).unsqueeze(-1)
            out = out + torch.einsum("bhls,bhsd->bhld", a_c / denom.unsqueeze(-1), code_v.to(q.dtype))
        return out, recent_z, recent_k_hat, recent_v

    # ------------------------------------------------------------------
    # 状态更新
    # ------------------------------------------------------------------
    def update_state(self, recent_z, recent_k_hat, recent_v, state):
        """
        用当前窗口内容更新状态:
        - 滑出窗口的位置（recent[..., :-M]）并入码本聚合缓存
        - 窗口只保留最近 M 个位置
        """
        M, S = self.config.mem_len, self.config.n_code
        L = recent_z.shape[-1] - M

        if self.config.agg_cache and L > 0:
            evict_z = recent_z[..., :L]                              # [B,H,L]
            evict_v = recent_v[..., :L, :]                           # [B,H,L,D]
            # 无效位置（初始窗口未填满时的 padding）的 shortcode 为 S，不应并入缓存
            valid = (evict_z != S)                                   # [B,H,L]
            delta = F.one_hot(evict_z.clamp(max=S - 1).long(), num_classes=S).to(recent_v.dtype)
            delta = delta * valid.unsqueeze(-1).to(delta.dtype)      # [B,H,L,S]
            new_lower = state["lower"] + delta.sum(dim=-2)           # [B,H,S]
            # 数值稳定的加权平均: new_upper = f1*upper + Σ f2*v
            f1 = state["lower"] / torch.clamp(new_lower, min=1.0)
            f2 = delta / torch.clamp(new_lower.unsqueeze(-2), min=1.0)
            new_upper = f1.unsqueeze(-1) * state["upper"] \
                + torch.einsum("bhls,bhld->bhsd", f2, evict_v)
        else:
            new_lower, new_upper = state["lower"], state["upper"]

        return dict(
            pos_offset=state["pos_offset"] + L,
            z=recent_z[..., recent_z.shape[-1] - M:].contiguous() if M > 0 else recent_z[..., :0].contiguous(),
            k_hat=recent_k_hat[..., recent_k_hat.shape[-2] - M:, :].contiguous() if M > 0 else recent_k_hat[..., :0, :].contiguous(),
            v=recent_v[..., recent_v.shape[-2] - M:, :].contiguous() if M > 0 else recent_v[..., :0, :].contiguous(),
            lower=new_lower,
            upper=new_upper,
        )

    @staticmethod
    def _detach_state(state):
        return {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in state.items()}

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(self, x, position_ids, state, use_cache=True):
        B, L, _ = x.shape

        # 1. Q/K/V 投影 + 归一化
        q = self.q_norm(rearrange(self.q_proj(x), 'b s (h d) -> b s h d', h=self.n_head))
        k = self.k_norm(rearrange(self.k_proj(x), 'b s (h d) -> b s h d', h=self.n_kv_head))
        v = rearrange(self.v_proj(x), 'b s (h d) -> b s h d', h=self.n_kv_head)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # 转成 [B, H, S, D]
        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')

        # 2. 键量化（量化未旋转的键，保证码本与位置无关、可聚合）
        vq_out = self.quantizer(k, loss_mask=None)
        k_hat, z = vq_out["quantized"], vq_out["shortcodes"]

        # 3. VQ 注意力
        out, recent_z, recent_k_hat, recent_v = self._attn(q, k_hat, v, z, position_ids, state)
        out = self.attn_dropout(out)

        # 4. 更新状态
        if use_cache:
            new_state = self.update_state(recent_z, recent_k_hat, recent_v, state)
            if not self.training:
                new_state = self._detach_state(new_state)
        else:
            new_state = state

        out = rearrange(out, 'b h s d -> b s (h d)')
        out = self.resid_dropout(self.o_proj(out))
        return out, new_state, vq_out


# ==============================================================================
# 前馈网络
# ==============================================================================

class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
    参考: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    """MoE 混合专家前馈网络（router + top-k 专家 + 负载均衡辅助损失）。"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x = rearrange(x, 'b s d -> (b s) d')

        scores = F.softmax(self.gate(x), dim=-1)
        sc_topk, exp_topk = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1)
        if self.config.norm_topk_prob:
            sc_topk /= (sc_topk.sum(dim=-1, keepdim=True) + 1e-20)

        y = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            if (exp_topk == i).any():
                hid_idcs, hid_ranks = torch.where(exp_topk == i)
                hidden_state = self.experts[i](x[hid_idcs])
                hidden_state *= sc_topk[hid_idcs, hid_ranks].unsqueeze(-1)
                y[hid_idcs] += hidden_state
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(exp_topk, self.config.num_experts).float().mean((0, 1))
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return rearrange(y, '(b s) d -> b s d', b=batch_size, s=seq_len)


# ==============================================================================
# Transformer 块和完整模型
# ==============================================================================

class MiniMindBlock(nn.Module):
    """
    MiniMind Transformer 块 (Pre-Norm)

    input → RMSNorm → VQAttention → +residual →
    RMSNorm → MLP (FFN/MoE) → +residual → output
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = VQAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_ids, attn_state, use_cache=True):
        residual = hidden_states
        attn_out, new_attn_state, vq_out = self.self_attn(
            self.input_layernorm(hidden_states), position_ids, attn_state, use_cache,
        )
        hidden_states = attn_out + residual

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + residual
        return hidden_states, new_attn_state, vq_out


class MiniMindModel(nn.Module):
    """
    MiniMind 基础模型（不含语言模型头），注意力替换为 VQ Attention。

    训练/长序列:
        若设置了 config.block_len 且序列长于该值，会把序列按 block 分块顺序处理，
        块间传递（detach 后的）VQ 缓存。这样长程上下文通过码本聚合缓存以 O(L·S)
        的代价被建模。
    推理:
        单次前向一个 block（或一个 token），state 由调用方持有并回传。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def initial_state(self, batch_size, device=None, dtype=torch.float32):
        return [l.self_attn.initial_state(batch_size, device=device, dtype=dtype) for l in self.layers]

    def _forward_blocks(self, hidden_states, position_ids, attn_state, use_cache, block_len):
        """
        把序列按 block_len 分块顺序前向，块间传递 VQ 缓存。

        注意: 即便 use_cache=False（不把缓存返回给调用方），块与块之间也必须
        传递缓存，分块处理才有意义。块间传递的缓存会被 detach，与
        Transformer-VQ 的 grad_thru_cache=False 行为一致。
        """
        outs, l_commit = [], hidden_states.new_zeros(())
        cur_state = attn_state
        for start in range(0, hidden_states.shape[1], block_len):
            end = min(start + block_len, hidden_states.shape[1])
            h, pos = hidden_states[:, start:end], position_ids[start:end]
            block_states = []
            for layer, layer_state in zip(self.layers, cur_state):
                h, state_out, vq_out = layer(h, pos, layer_state, use_cache=True)
                block_states.append(layer.self_attn._detach_state(state_out))
                l_commit = l_commit + vq_out["l_commit"]
            outs.append(h)
            cur_state = block_states
        # 若调用方不需要缓存，则返回原始（未推进的）状态
        return torch.cat(outs, dim=1), (cur_state if use_cache else attn_state), l_commit

    def forward(self, input_ids, position_ids=None, attention_mask=None, past_key_values=None,
                use_cache=False, attn_state=None, block_len=None, **kwargs):
        batch_size, seq_length = input_ids.shape

        if attn_state is None:
            attn_state = self.initial_state(batch_size, device=input_ids.device)
        if position_ids is None or position_ids.dim() > 1:
            # HF 的 generate 会传入 [B, S] 的 position_ids；同一 batch 内各序列
            # 位置一致，这里取第一行还原成 [S]，缺失时按缓存偏移生成。
            offset = attn_state[0]["pos_offset"] if attn_state else 0
            if position_ids is None:
                position_ids = torch.arange(offset, offset + seq_length, device=input_ids.device)
            else:
                position_ids = position_ids[0]

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        block_len = block_len or self.config.block_len
        if block_len and seq_length > block_len:
            hidden_states, new_attn_state, l_commit = self._forward_blocks(
                hidden_states, position_ids, attn_state, use_cache, block_len,
            )
        else:
            new_attn_state, l_commit = [], hidden_states.new_zeros(())
            for layer, layer_state in zip(self.layers, attn_state):
                hidden_states, state_out, vq_out = layer(hidden_states, position_ids, layer_state, use_cache)
                new_attn_state.append(state_out)
                l_commit = l_commit + vq_out["l_commit"]

        hidden_states = self.norm(hidden_states)

        # MoE 负载均衡损失
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(()),
        )
        # VQ commitment loss 计入 aux_loss（与 MiniMind 训练脚本的 loss 结构一致）
        aux_loss = aux_loss + self.config.c_beta * l_commit
        return hidden_states, new_attn_state, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """MiniMind 因果语言模型（VQ Attention 版本）"""
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重绑定: 输入嵌入与输出投影共享权重
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep=0, labels=None, position_ids=None, attn_state=None, **kwargs):
        hidden_states, new_attn_state, aux_loss = self.model(
            input_ids, position_ids=position_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache, attn_state=attn_state, **kwargs,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(rearrange(x, 'b s v -> (b s) v'), rearrange(y, 'b s -> (b s)'), ignore_index=-100)

        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits,
                                         past_key_values=new_attn_state, hidden_states=hidden_states)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      inputs_embeds=None, cache_position=None, **kwargs):
        """
        HF generate 的输入准备。

        VQ 注意力的缓存不是一个 [B,H,S,D] 张量，而是每层一个状态 dict，
        因此需要把 past_key_values 原样透传给 forward 的 attn_state，
        并按 cache_position 只喂入未处理过的 token。
        """
        if past_key_values is not None and cache_position is not None:
            input_ids = input_ids[:, cache_position]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attn_state=past_key_values,
            use_cache=True,
        )

    # 采样工具
    @staticmethod
    def _sample_next(logits, temperature, top_k, top_p, do_sample, repetition_penalty, input_ids):
        logits = logits / temperature
        if repetition_penalty != 1.0:
            for i in range(input_ids.shape[0]):
                logits[i, torch.unique(input_ids[i])] /= repetition_penalty
        if top_k > 0:
            logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cum_probs > top_p
            mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
            logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
        if do_sample:
            return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 生成方法参考: https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def _generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85,
                  top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1,
                  do_sample=True, repetition_penalty=1.0, **kwargs):
        """
        自回归文本生成（使用 VQ 注意力缓存）。

        prompt 一次性预填充，随后逐 token 解码；VQ 缓存中保留最近 mem_len 个位置，
        更早的历史按 shortcode 归并进码本聚合缓存。
        """
        if max_new_tokens is None:
            max_new_tokens = 8192
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        attn_state = kwargs.pop("attn_state", None)

        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())

        # 预填充
        out = self.forward(input_ids, attention_mask, attn_state=attn_state, use_cache=use_cache, **kwargs)
        attn_state = out.past_key_values
        next_token = self._sample_next(out.logits[:, -1, :], temperature, top_k, top_p,
                                       do_sample, repetition_penalty, input_ids)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if streamer: streamer.put(next_token.cpu())

        # 逐步解码
        for _ in range(max_new_tokens - 1):
            out = self.forward(next_token, attention_mask, attn_state=attn_state, use_cache=use_cache, **kwargs)
            attn_state = out.past_key_values
            next_token = self._sample_next(out.logits[:, -1, :], temperature, top_k, top_p,
                                           do_sample, repetition_penalty, input_ids)
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1),
                                         next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        if streamer: streamer.end()
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': attn_state}
        return input_ids

    @torch.inference_mode()
    def chat(model, tok, ques, history=[], **kw):
        iids = tok.apply_chat_template(
            history + [{'role': 'user', 'content': ques}],
            add_generation_prompt=1,
        )
        oids = model.generate(
            inputs=torch.tensor([iids]).to(model.device),
            **(model.generation_config.to_dict() | kw),
        )
        oids = oids[0][len(iids):].tolist()
        if oids[-1] == tok.eos_token_id:
            oids = oids[:-1]
        return tok.decode(oids)
