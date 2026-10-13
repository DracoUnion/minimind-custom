"""
MiniMind Block Diffusion 语言模型实现

基于 DiffusionGemma 的 encoder-decoder diffusion 架构改造而成，支持:
- Encoder: 编码 prefix (自回归注意力, 写入共享 KV cache)
- Decoder: 处理 canvas tokens (双向注意力, 只读 encoder 的 KV cache)
- Self-Conditioning: 用上一步去噪的 logits 生成 soft embedding 注入 decoder
- RMSNorm / RoPE / GQA / SwiGLU / MoE / YaRN

参考: transformers/models/diffusion_gemma/modeling_diffusion_gemma.py

训练约定:
    model(input_ids, labels) 时, 输入序列尾部 `config.canvas_length` 个 token 作为
    canvas(去噪目标), 前面部分作为 prefix(由 encoder 编码)。loss 默认在 canvas 上
    计算双向预测交叉熵; 若 `config.use_prefix_ar_loss=True`, 额外叠加 prefix 的
    自回归损失 (类似 DiffusionGemma 的 encoder AR loss)。
"""
import math, torch, torch.nn.functional as F
from torch import nn
from dataclasses import dataclass
from einops import rearrange, repeat
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from .config_minimind import MiniMindConfig


# ==============================================================================
# 基础组件
# ==============================================================================

class RMSNorm(torch.nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization) 根均方层归一化

    相比 LayerNorm，RMSNorm 去除了均值计算，只使用 RMS 进行缩放:
    output = x / RMS(x) * gamma
    其中 RMS(x) = sqrt(mean(x^2) + eps)

    支持 `with_scale=False` (无学习缩放参数), 用于 self-conditioning 的输出归一化。
    参考论文: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-5, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            # 可学习的缩放参数，初始化为全1
            self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        """
        计算 RMS 归一化
        rsqrt = 1/sqrt，比先 sqrt 再倒数更高效
        mean(-1, keepdim=True) 表示在最后一个维度（特征维度）上求均值
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先转为 float32 保证数值稳定性，计算完再转回原类型
        normed = self.norm(x.float())
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(x)


class ScaledWordEmbedding(nn.Embedding):
    """
    带缩放因子的词嵌入 (对应 DiffusionGemma 的 DiffusionGemmaTextScaledWordEmbedding)。
    embedding 输出乘以 `hidden_size ** 0.5`, 使得 soft-embedding (softmax(logits) @ weight)
    与 token embedding 处于同一量级, self-conditioning 才能正确叠加。
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """
    预计算旋转位置编码 (RoPE) 的频率复数指数

    RoPE 原理: 通过旋转矩阵将位置信息编码到Query和Key中
    对于位置 m 的向量 x，旋转后的表示为:
    [x_0, x_1, x_2, x_3, ...] @ 旋转矩阵
    = [x_0*cos(mθ)-x_1*sin(mθ), x_0*sin(mθ)+x_1*cos(mθ), ...]

    参数说明:
        dim: 每个注意力头的维度 (head_dim)
        end: 预计算的最大序列长度
        rope_base: 旋转基数，控制旋转速度 (默认1e6，较大值意味着更慢的位置编码变化)
        rope_scaling: 长度外推配置 (YaRN)

    返回:
        freqs_cos: 余弦值，形状 (end, dim//2 * 2)
        freqs_sin: 正弦值，形状 (end, dim//2 * 2)
    """
    # 计算基础频率: 1 / (base^(2i/dim))，其中 i 是维度索引
    # 只取偶数索引对应的频率 [: (dim // 2)]
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    # YaRN (Yet another RoPE extension) 长度外推
    # 当需要处理的序列长度超过训练时的长度时，通过调整频率实现长度外推
    if rope_scaling is not None:
        # 从配置中提取参数
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),  # 原始训练长度
            rope_scaling.get("factor", 16),  # 扩展因子
            rope_scaling.get("beta_fast", 32.0),  # 高频边界
            rope_scaling.get("beta_slow", 1.0),  # 低频边界
            rope_scaling.get("attention_factor", 1.0)  # 注意力温度系数
        )
        # 只有当需要的长度超过原始长度时才应用 YaRN
        if end / orig_max > 1.0:
            # 计算频率维度的逆映射，用于确定高低频边界
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # 计算低频和高频的边界索引
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # 创建线性 ramp 函数，用于混合原始频率和缩放后的频率
            # gamma(i) = clamp((i - low) / (high - low), 0, 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # 应用 YaRN 公式: f'(i) = f(i) * ((1-γ) + γ/factor)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 生成位置索引 (0 到 end-1)
    t = torch.arange(end, device=freqs.device)
    # 外积: 每个位置 m 与每个频率 θ 相乘，得到 m * θ
    freqs = torch.outer(t, freqs).float()

    # 计算余弦和正弦，并复制一份 (因为相邻两个维度共享相同的频率)
    # cat 后形状为 (end, dim)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    应用旋转位置编码到 Query 和 Key

    参数:
        q: Query 张量，形状 (..., seq_len, head_dim)
        k: Key 张量，形状 (..., seq_len, head_dim)
        cos: 预计算的余弦值，形状 (seq_len, head_dim)
        sin: 预计算的正弦值，形状 (seq_len, head_dim)
        unsqueeze_dim: 需要在哪个维度上扩展 cos/sin 以便广播

    返回:
        应用旋转编码后的 q 和 k
    """
    def rotate_half(x):
        """
        将输入的后一半维度旋转到前面，前一半旋转到后面
        例如: [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
        """
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # RoPE 公式:
    # q' = q * cos + rotate_half(q) * sin
    # k' = k * cos + rotate_half(k) * sin
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA (Grouped Query Attention) 中的 Key/Value 重复

    在 GQA 中，Query 头数 > Key/Value 头数
    例如: Query 有 8 个头，Key/Value 只有 2 个头
    这时需要把每个 Key/Value 复制 4 次，与 Query 对应

    参数:
        x: 输入张量，形状 (batch, seq_len, num_kv_heads, head_dim)
        n_rep: 每个 KV head 需要重复的次数 = num_query_heads // num_kv_heads

    返回:
        重复后的张量，形状 (batch, seq_len, num_query_heads, head_dim)
    """
    if n_rep == 1:
        return x
    # 使用 einops 进行更清晰的维度操作
    # 将 h 维度扩展为 (h * r)，即每个 head 复制 r 次
    return repeat(x, 'b s h d -> b s (h r) d', r=n_rep)


# ==============================================================================
# 注意力层
# ==============================================================================

class Attention(nn.Module):
    """
    多头自注意力层 (支持 GQA)

    该层同时用于 Encoder 和 Decoder, 通过 `is_causal` 区分两种模式:
    - Encoder (is_causal=True):  自回归因果注意力, 写入 KV cache
    - Decoder (is_causal=False): 双向注意力, 读取 encoder 的 KV cache (prefix),
      将其与当前 canvas 的 KV 拼接后计算注意力, 不更新 cache

    架构:
    1. 线性投影: Q, K, V 分别投影
    2. RMSNorm: 对 Q 和 K 进行归一化 (Query/Key Norm)
    3. RoPE: 应用旋转位置编码
    4. KV Cache: encoder 写入 / decoder 只读
    5. 注意力计算: Flash Attention 或手动实现
    6. 输出投影
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # KV 头数，如果不指定则等于 Query 头数 (标准 MHA)
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads  # Query 头数
        self.n_local_kv_heads = self.num_key_value_heads  # KV 头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个 KV head 需要重复的次数
        self.head_dim = config.head_dim  # 每个头的维度
        self.proj_size = self.n_local_heads * self.head_dim
        self.kv_proj_size = self.n_local_kv_heads * self.head_dim

        # 线性投影层 (无偏置)
        # Q 投影到 (num_heads * head_dim)
        self.q_proj = nn.Linear(config.hidden_size, self.proj_size, bias=False)
        # K, V 投影到 (num_kv_heads * head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_proj_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_proj_size, bias=False)
        # 输出投影
        self.o_proj = nn.Linear(self.proj_size, config.hidden_size, bias=False)

        # Query/Key 归一化 (有助于稳定训练)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # 是否使用 Flash Attention (PyTorch 2.0+ 原生支持)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, attention_mask=None, past_key_value=None,
                use_cache=False, is_causal=True):
        """
        前向传播

        参数:
            x: 输入，形状 (batch_size, seq_len, hidden_size)
            position_embeddings: (cos, sin) 位置编码
            attention_mask: 注意力掩码
            past_key_value: (k, v) KV 缓存
                - Encoder 模式: 已有 prefix KV, 新的 KV 拼接到尾部, 并随 use_cache 返回
                - Decoder 模式: encoder 写入的 prefix KV, 只读拼接, 不返回
            use_cache: 是否返回更新后的 K, V 缓存 (仅 encoder)
            is_causal: 是否使用因果掩码 (decoder 为 False, 即双向)

        返回:
            output: 注意力输出
            past_kv: 更新后的 K, V 缓存 (decoder 返回 None)
        """
        bsz, seq_len, _ = x.shape

        # 1. 线性投影并 reshape 为多头形式
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # (batch, seq, hidden) -> (batch, seq, num_heads, head_dim)
        xq = rearrange(xq, 'b s (h d) -> b s h d', h=self.n_local_heads)
        xk = rearrange(xk, 'b s (h d) -> b s h d', h=self.n_local_kv_heads)
        xv = rearrange(xv, 'b s (h d) -> b s h d', h=self.n_local_kv_heads)

        # 2. 应用 Query/Key 归一化
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # 3. 应用旋转位置编码 (RoPE)
        cos, sin = position_embeddings
        # encoder 的 cos/sin 是 2D (seq, head_dim); decoder 的 cos/sin 是带 batch 维的
        # 3D (batch, seq, head_dim), 需要不同的 unsqueeze 维度以正确广播
        unsqueeze_dim = 2 if cos.ndim == 3 else 1
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, unsqueeze_dim=unsqueeze_dim)

        # 4. KV Cache
        #    encoder: 拼接到已有 cache 尾部 (自回归)
        #    decoder: 拼接 encoder 的 prefix KV + 当前 canvas KV (双向, 只读)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # decoder 不写 cache (只读 encoder 的 prefix KV)
        past_kv = (xk, xv) if use_cache else None

        # 5. 调整维度顺序以便计算注意力
        # (batch, seq, head, dim) -> (batch, head, seq, dim)
        xq = rearrange(xq, 'b s h d -> b h s d')
        xk = rearrange(repeat_kv(xk, self.n_rep), 'b s h d -> b h s d')
        xv = rearrange(repeat_kv(xv, self.n_rep), 'b s h d -> b h s d')

        # 6. 注意力计算
        #    Flash 路径: 仅用于无 cache 且因果 (encoder 预填充) 的情形,
        #    decoder 有 prefix cache 且双向, 走手动路径
        if self.flash and is_causal and (seq_len > 1) and (past_key_value is None) and \
                (attention_mask is None or torch.all(attention_mask == 1)):
            # 使用 Flash Attention (内存高效、计算更快)
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 手动实现注意力
            # Q @ K^T / sqrt(d_k)
            scores = (xq @ rearrange(xk, 'b h s d -> b h d s')) / math.sqrt(self.head_dim)
            # 因果掩码 (仅 encoder): 每个位置只能看到自己和之前的位置
            # triu(1) 创建上三角矩阵，对角线以上全为 -inf
            if is_causal:
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # 应用额外的 attention_mask (如果有)
            # (b, s) => (b, 1, 1, s), s 覆盖 prefix+canvas 总长度 (decoder 时)
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask.to(scores.dtype)
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            # softmax 归一化后与 V 相乘
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        # 7. 重新组合多头并输出投影
        # (batch, head, seq, dim) -> (batch, seq, head*dim)
        output = rearrange(output, 'b h s d -> b s (h d)')
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


# ==============================================================================
# 前馈网络
# ==============================================================================

class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络

    架构: gate_proj(x) ⊙ up_proj(x) -> down_proj
    其中 gate_proj 使用 SiLU/Swish 激活:
    SwiGLU(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

    相比标准 FFN (ReLU/GELU):
    - SwiGLU 引入门控机制，选择性传递信息
    - 实验表明在语言模型上效果更好
    参考: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        # 门控投影 (带激活函数)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # 上投影
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # 下投影
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        # 激活函数 (通常是 SiLU/Swish)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU: gate(x) 经过激活后与 up(x) 逐元素相乘，再投影回 hidden_size
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    """
    MoE (Mixture of Experts) 混合专家前馈网络

    架构:
    1. Router: 决定每个 token 使用哪些专家
    2. Experts: 多个并行的 FFN 专家
    3. Top-K 路由: 每个 token 只激活 K 个专家
    4. 辅助损失: 负载均衡损失，防止路由崩溃

    优势:
    - 总参数量大，但每个 token 只激活部分参数 (稀疏激活)
    - 可以扩展到非常大的模型规模
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 路由门控网络: 输入 hidden_size，输出 num_experts (每个专家的权重)
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # 专家网络列表，每个专家是一个 FFN
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        MoE 前向传播

        流程:
        1. 展平 token: (batch, seq, hidden) -> (batch*seq, hidden)
        2. 计算路由分数: softmax(gate(x))
        3. Top-K 选择: 每个 token 选择 K 个专家
        4. 专家计算: 被选中的专家处理对应的 token
        5. 加权聚合: 按路由权重聚合专家输出
        6. 计算辅助损失: 负载均衡损失
        """
        batch_size, seq_len, hidden_dim = x.shape
        # 展平为 (batch*seq, hidden)，方便并行处理所有 token
        x = rearrange(x, 'b s d -> (b s) d')

        # 1. 计算路由分数，形状 (batch*seq, num_experts)
        scores = F.softmax(self.gate(x), dim=-1)

        # 2. Top-K 路由: 选择得分最高的 K 个专家
        # sc_topk: (batch*seq, K)exp_topk: (batch*seq, K)
        sc_topk, exp_topk = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1)

        # 可选: 归一化 Top-K 概率
        if self.config.norm_topk_prob:
            sc_topk /= (sc_topk.sum(dim=-1, keepdim=True) + 1e-20)

        # 3. 初始化输出张量
        y = torch.zeros_like(x)

        # 4. 遍历所有专家，处理被分配给该专家的 token
        for i, expert in enumerate(self.experts):
            # mask: 哪些 token 选择了专家 i，形状 (batch*seq, K)
            if (exp_topk == i).any():
                # 找出选择了专家 i 的 token 索引
                # hid_idcs 是调用专家 i 的向量序号，hid_ranks 是该专家对于对应向量的排名
                # 形状均为 (n_select)
                hid_idcs, hid_ranks = torch.where(exp_topk == i)
                # 把属于该专家的向量传入该专家
                # hidden_state 形状为 (n_select, dim)
                hidden_state = self.experts[i](x[hid_idcs])
                # 获取该专家权重，最后加一维便于对齐
                weights = sc_topk[hid_idcs, hid_ranks].unsqueeze(-1)
                # 乘上权重
                hidden_state *= weights
                # 然后将当前专家的输出填回到结果数组中
                y[hid_idcs] += hidden_state
            elif self.training:
                # 训练时，如果没有 token 选择这个专家，添加一个虚拟梯度
                # 确保该专家能收到梯度更新，防止某些专家完全不被训练
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # 5. 计算负载均衡辅助损失 (Load Balancing Loss)
        # 目标: 让各个专家处理的 token 数量尽量均衡
        if self.training and self.config.router_aux_loss_coef > 0:
            # load: 每个专家实际处理的 token 比例，形状 (num_experts,)
            load = F.one_hot(exp_topk, self.config.num_experts).float().mean((0, 1))
            # aux_loss = sum(load * scores.mean) * num_experts * coef
            # 当某个专家处理太多 token 时，loss 会增大
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        # 6. 恢复形状
        return rearrange(y, '(b s) d -> b s d', b=batch_size, s=seq_len)


# ==============================================================================
# Transformer 块和完整模型
# ==============================================================================

class MiniMindBlock(nn.Module):
    """
    MiniMind Transformer 块

    架构 (Pre-Norm):
    input → RMSNorm → Attention → +residual →
    RMSNorm → MLP (FFN/MoE) → +residual → output

    Encoder 与 Decoder 共享同一组层的权重, 通过 `is_causal` 在 forward 时切换
    因果 / 双向注意力。
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        # Attention 前的 LayerNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # MLP 前的 LayerNorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 根据配置选择标准 FFN 或 MoE
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, use_cache=False, is_causal=True):
        # 1. 自注意力子层 (带残差连接)
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # Pre-Norm
            position_embeddings,
            attention_mask,
            past_key_value,
            use_cache,
            is_causal,
        )
        hidden_states += residual  # 残差连接

        # 2. MLP 子层 (带残差连接)
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))  # Pre-Norm
        hidden_states += residual  # 残差连接

        return hidden_states, present_key_value


class SelfConditioning(nn.Module):
    """
    Self-conditioning 模块 (对应 DiffusionGemma 的 DiffusionGemmaSelfConditioning)。

    将上一轮去噪输出的 logits 转换出的 soft-embedding 处理成一个
    self-conditioning 信号, 叠加到 decoder 的输入 embedding 上。

    使用 SwiGLU 门控结构, 带 pre/post RMSNorm。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.pre_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        # post_norm 不学习缩放 (参考 DiffusionGemma: with_scale=False)
        self.post_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps, with_scale=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, inputs_embeds, self_conditioning_signal):
        """
        参数:
            inputs_embeds: 当前 canvas 的 token embedding
                (batch_size, canvas_length, hidden_size)
            self_conditioning_signal: 上一轮 logits 的 soft-embedding
                (batch_size, canvas_length, hidden_size), 形状与 inputs_embeds 相同

        返回:
            叠加了 self-conditioning 信号后的 embedding, 形状不变
        """
        normed = self.pre_norm(self_conditioning_signal)
        sc_signal = self.down_proj(self.act_fn(self.gate_proj(normed)) * self.up_proj(normed))
        combined = inputs_embeds + sc_signal
        return self.post_norm(combined)


class MiniMindEncoderModel(nn.Module):
    """
    MiniMind 编码器模型 (对应 DiffusionGemma 的 DiffusionGemmaEncoderTextModel)

    以自回归 (因果) 注意力编码 prefix tokens, 并把每层的 K/V 写入共享 KV cache,
    供 decoder 读取。Encoder 与 Decoder 共享同一组层权重。
    """
    def __init__(self, config: MiniMindConfig, embed_tokens=None, layers=None):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        # 词嵌入 (与 decoder / lm_head 共享权重)
        self.embed_tokens = embed_tokens or ScaledWordEmbedding(
            config.vocab_size, config.hidden_size, embed_scale=config.hidden_size ** 0.5
        )
        self.dropout = nn.Dropout(config.dropout)

        # Transformer 块堆叠 (与 decoder 共享同一份 ModuleList)
        self.layers = layers or nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])

        # 最终的 LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 位置编码并注册为 buffer (不参与训练，持久化保存)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        """
        前向传播: 编码 prefix, 写 KV cache

        参数:
            input_ids: prefix token IDs, 形状 (batch, seq_len)
            attention_mask: 注意力掩码
            past_key_values: 缓存的 K, V 列表
            use_cache: 是否返回更新后的 KV cache (decoder 需要)

        返回:
            hidden_states: prefix 的最终隐藏状态 (用于可选的 encoder AR loss)
            presents: 更新后的 KV cache 列表, 每层为 (k, v)
            aux_loss: MoE 辅助损失 (如果有)
        """
        batch_size, seq_length = input_ids.shape

        # 初始化 KV Cache 列表 (如果未提供)
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算起始位置 (用于 KV Cache / RoPE 索引)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Token 嵌入
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 获取当前序列的位置编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 逐层前向传播 (encoder: 因果注意力)
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                is_causal=True,
            )
            presents.append(present)

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # 计算所有层的 MoE 辅助损失之和
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())

        return hidden_states, presents, aux_loss


class MiniMindDecoderModel(nn.Module):
    """
    MiniMind 解码器模型 (对应 DiffusionGemma 的 DiffusionGemmaDecoderModel)

    处理 canvas tokens: 双向自注意力 + 交叉注意力 (只读 encoder 的 KV cache)。
    Decoder 不更新 KV cache, 一次处理整个 canvas (block diffusion)。

    Encoder 与 Decoder 共享同一组层权重; decoder 额外拥有 self-conditioning 模块。
    """
    def __init__(self, config: MiniMindConfig, embed_tokens=None, layers=None):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        # 词嵌入 (与 encoder / lm_head 共享权重)
        self.embed_tokens = embed_tokens or ScaledWordEmbedding(
            config.vocab_size, config.hidden_size, embed_scale=config.hidden_size ** 0.5
        )
        self.dropout = nn.Dropout(config.dropout)

        # Transformer 块堆叠 (与 encoder 共享同一份 ModuleList)
        self.layers = layers or nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])

        # 最终的 LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Self-conditioning 模块
        self.self_conditioning = SelfConditioning(config)

        # 预计算 RoPE 位置编码并注册为 buffer (不参与训练，持久化保存)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, decoder_input_ids, past_key_values=None,
                self_conditioning_logits=None, self_conditioning_mask=None,
                decoder_attention_mask=None, decoder_position_ids=None):
        """
        前向传播: 处理 canvas, 双向注意力 + 交叉注意力 (只读 encoder KV cache)

        参数:
            decoder_input_ids: canvas token IDs, 形状 (batch, canvas_length)
            past_key_values: encoder 写入的共享 KV cache (list of (k, v), 每层一个)
                - 若为 None, 则退化为无 prefix 的纯双向注意力 (纯去噪)
            self_conditioning_logits: 上一轮去噪的 logits
                (batch, canvas_length, vocab_size), 用于计算 self-conditioning embedding
            self_conditioning_mask: (batch,) bool mask, False 表示该样本不使用 self-conditioning
            decoder_attention_mask: 覆盖 prefix+canvas 总长度的掩码 (batch, prefix_len+canvas_len)
            decoder_position_ids: canvas 的位置 id (默认从 prefix 长度起算)

        返回:
            hidden_states: canvas 的最终隐藏状态, 形状 (batch, canvas_length, hidden_size)
        """
        inputs_embeds = self.dropout(self.embed_tokens(decoder_input_ids))

        # 处理 self-conditioning:
        # 如果传入了上一轮 logits, 用 softmax 后的概率对词表加权得到 soft-embedding
        if self_conditioning_logits is not None:
            # logits 转 soft-embedding: softmax(logits) @ embed_weight
            soft_embeddings = torch.matmul(
                self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(self.embed_tokens.weight.dtype),
                self.embed_tokens.weight,
            ) * self.embed_tokens.embed_scale
            # 按样本决定是否启用 self-conditioning
            if self_conditioning_mask is not None:
                soft_embeddings = soft_embeddings * self_conditioning_mask.to(soft_embeddings.dtype)[:, None, None]
        else:
            # 第一轮去噪没有前一轮 logits, self-conditioning 信号置零
            soft_embeddings = torch.zeros_like(inputs_embeds)

        # 将 self-conditioning 信号注入 decoder 输入 embedding
        inputs_embeds = self.self_conditioning(inputs_embeds, soft_embeddings)

        # Canvas 的位置在 prefix 之后, 即 [prefix_len, prefix_len+canvas_len)
        if past_key_values is not None and past_key_values[0] is not None:
            cache_seq_length = past_key_values[0][0].shape[1]
        else:
            cache_seq_length = 0

        canvas_length = inputs_embeds.shape[1]
        if decoder_position_ids is None:
            decoder_position_ids = torch.arange(
                cache_seq_length,
                cache_seq_length + canvas_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )
            decoder_position_ids = decoder_position_ids.unsqueeze(0)

        # 获取 canvas 的位置编码 (从 prefix 长度起算)
        position_embeddings = (
            self.freqs_cos[decoder_position_ids],
            self.freqs_sin[decoder_position_ids],
        )

        # 逐层前向传播 (decoder: 双向注意力, 只读 encoder KV cache)
        hidden_states = inputs_embeds
        for layer, past_key_value in zip(self.layers, past_key_values if past_key_values is not None else [None] * len(self.layers)):
            hidden_states, _ = layer(
                hidden_states,
                position_embeddings,
                attention_mask=decoder_attention_mask,
                past_key_value=past_key_value,
                use_cache=False,  # decoder 不写 cache
                is_causal=False,  # 双向
            )

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        return hidden_states


@dataclass
class MiniMindDiffusionOutputWithPast(MoeCausalLMOutputWithPast):
    """
    MiniMind diffusion 模型的输出。

    继承 MoeCausalLMOutputWithPast 以兼容训练脚本 (res.loss / res.aux_loss),
    额外携带 encoder 的最终隐藏状态 (用于可选的 prefix 自回归损失)。
    """
    encoder_last_hidden_state: torch.FloatTensor | None = None


class MiniMindModel(nn.Module):
    """
    MiniMind diffusion 模型 (对应 DiffusionGemma 的 DiffusionGemmaModel)

    结构:
    - Encoder: 编码 prefix → 共享 KV cache
    - Decoder: 双向处理 canvas → 交叉注意力到 encoder KV cache
    - Encoder 与 Decoder 共享同一组 Transformer 层权重
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        # Encoder 与 Decoder 共享的词嵌入
        self.embed_tokens = ScaledWordEmbedding(
            config.vocab_size, config.hidden_size, embed_scale=config.hidden_size ** 0.5
        )
        # Encoder 与 Decoder 共享的 Transformer 层
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])

        self.encoder = MiniMindEncoderModel(config, self.embed_tokens, self.layers)
        self.decoder = MiniMindDecoderModel(config, self.embed_tokens, self.layers)

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                decoder_input_ids=None, self_conditioning_logits=None,
                self_conditioning_mask=None, decoder_attention_mask=None,
                decoder_position_ids=None, use_cache=True, **kwargs):
        """
        前向传播

        参数:
            input_ids: prefix token IDs, 形状 (batch, prefix_len)
                (解码器需要; 若为 None, 则必须提供 past_key_values)
            attention_mask: prefix 的注意力掩码
            past_key_values: 共享 KV cache (encoder 写入, decoder 只读)
            decoder_input_ids: canvas token IDs, 形状 (batch, canvas_length)
            self_conditioning_logits / mask: 上一轮 logits 及逐样本开关
            decoder_attention_mask: 覆盖 prefix+canvas 的掩码
            decoder_position_ids: canvas 位置 id
            use_cache: 是否由 encoder 写 cache (默认 True)

        返回:
            last_hidden_state: canvas 的最终隐藏状态
            past_key_values: encoder 写好的共享 KV cache
            encoder_last_hidden_state: prefix 的最终隐藏状态 (用于可选 AR loss)
            aux_loss: MoE 辅助损失
        """
        kwargs.pop("use_cache", None)

        # 1. Encoder: 把 prefix 编码进共享 KV cache (decoder 只读)
        encoder_last_hidden_state = None
        aux_loss = None
        if input_ids is not None:
            encoder_last_hidden_state, past_key_values, aux_loss = self.encoder(
                input_ids, attention_mask, past_key_values, use_cache=True
            )
        elif past_key_values is None:
            # 允许无 prefix 的纯 decoder diffusion: decoder 使用双向 self-attention
            # 此时没有 encoder KV cache, past_key_values 保持 None
            aux_loss = torch.zeros((), device=self.embed_tokens.weight.device)

        # 2. Decoder: 处理 canvas (双向 + 交叉注意力)
        # 如果未显式提供 canvas, 随机采样初始化 (block diffusion 的第一步)
        if decoder_input_ids is None:
            decoder_input_ids = torch.randint(
                low=0,
                high=self.config.vocab_size,
                size=(input_ids.shape[0], self.config.canvas_length),
                device=self.encoder.device,
            )

        last_hidden_state = self.decoder(
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
        )

        return MiniMindDiffusionOutputWithPast(
            loss=None,
            aux_loss=aux_loss,
            logits=None,
            past_key_values=past_key_values,
            hidden_states=last_hidden_state,
            encoder_last_hidden_state=encoder_last_hidden_state,
        )


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniMind Block Diffusion 语言模型 (对应 DiffusionGemmaForBlockDiffusion)

    继承:
    - PreTrainedModel: HuggingFace Transformers 基类，支持 save_pretrained/load_pretrained
    - GenerationMixin: 支持 generate() 方法进行去噪生成

    训练时直接调用 `model(input_ids, labels=labels)`:
    - input_ids 尾部 `canvas_length` 个 token 作为 canvas (去噪目标)
    - 前面部分作为 prefix 交给 encoder
    - loss 在 canvas 上计算双向预测交叉熵; 可选叠加 prefix 的自回归损失
    """
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)

        # 基础模型 (encoder + decoder)
        self.model = MiniMindModel(self.config)

        # 语言模型头: 将隐藏状态映射到词表
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 权重绑定: 输入嵌入和输出投影共享权重 (soft-embedding 也基于该权重)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None,
                decoder_input_ids=None, self_conditioning_logits=None,
                self_conditioning_mask=None, decoder_attention_mask=None,
                decoder_position_ids=None, **kwargs):
        """
        前向传播

        参数:
            input_ids: 输入 token IDs (batch, seq_len)。
                默认把尾部 `canvas_length` 个 token 作为 canvas, 前面作为 prefix。
                若显式传入 `decoder_input_ids`, 则 input_ids 仅作为 prefix。
            attention_mask: prefix 的注意力掩码
            labels: 目标 token IDs, 形状与 input_ids 相同 (batch, seq_len)。
                只有 canvas 部分 (尾部 canvas_length 个) 参与 diffusion loss;
                prefix 部分可选参与自回归 loss。
            decoder_input_ids: canvas token IDs (batch, canvas_length)。
                不传时由 input_ids 自动切分。
            self_conditioning_logits / mask: 上一轮去噪 logits 及逐样本开关
            decoder_attention_mask: 覆盖 prefix+canvas 的掩码
            decoder_position_ids: canvas 位置 id

        返回:
            MiniMindDiffusionOutputWithPast, 包含 loss, aux_loss, logits, hidden_states
        """
        # 切分 prefix / canvas
        if input_ids is not None:
            seq_len = input_ids.shape[1]
            canvas_len = min(self.config.canvas_length, seq_len)
            prefix_len = seq_len - canvas_len
        else:
            # 纯去噪模式: 不编码 prefix, decoder 直接处理 canvas
            canvas_len = decoder_input_ids.shape[1] if decoder_input_ids is not None else self.config.canvas_length
            prefix_len = 0
        auto_canvas = decoder_input_ids is None

        if auto_canvas and input_ids is not None:
            # 自动切分: prefix + canvas
            prefix_ids = input_ids[:, :prefix_len] if prefix_len > 0 else None
            decoder_input_ids = input_ids[:, -canvas_len:]
            # 切分 mask: prefix 部分给 encoder, canvas 部分 (全1) 拼接到 decoder mask
            if attention_mask is not None:
                prefix_mask = attention_mask[:, :prefix_len] if prefix_len > 0 else None
                canvas_mask = attention_mask.new_ones((attention_mask.shape[0], canvas_len))
                if decoder_attention_mask is None:
                    decoder_attention_mask = torch.cat([attention_mask[:, :prefix_len], canvas_mask], dim=-1)
            else:
                prefix_mask = None
        elif auto_canvas:
            raise ValueError("input_ids 和 decoder_input_ids 不能同时为 None")
        else:
            # 显式传入 canvas: input_ids 即 prefix (可能为 None)
            prefix_ids = input_ids
            prefix_mask = attention_mask

        # 模型前向
        outputs = self.model(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            past_key_values=past_key_values,
            decoder_input_ids=decoder_input_ids,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
        )

        # 计算 logits (canvas 上的预测)
        logits = self.lm_head(outputs.hidden_states)

        # final logits softcapping (可选, 参考 DiffusionGemma)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        # 计算损失
        loss = None
        if labels is not None:
            # 1) diffusion loss: canvas 上双向预测交叉熵
            canvas_labels = labels[:, -canvas_len:] if auto_canvas else labels
            loss = F.cross_entropy(
                rearrange(logits, 'b s v -> (b s) v'),
                rearrange(canvas_labels, 'b s -> (b s)'),
                ignore_index=-100,
            )
            # 2) prefix 自回归损失 (可选): 类似 DiffusionGemma 的 encoder AR loss
            if (self.config.use_prefix_ar_loss and prefix_ids is not None
                    and outputs.encoder_last_hidden_state is not None):
                enc_logits = self.lm_head(outputs.encoder_last_hidden_state)
                prefix_labels = labels[:, :prefix_len]
                x, y = enc_logits[..., :-1, :].contiguous(), prefix_labels[..., 1:].contiguous()
                ar_loss = F.cross_entropy(
                    rearrange(x, 'b s v -> (b s) v'),
                    rearrange(y, 'b s -> (b s)'),
                    ignore_index=-100,
                )
                loss = loss + ar_loss

        return MiniMindDiffusionOutputWithPast(
            loss=loss,
            aux_loss=outputs.aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        )

    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=None,
                 num_diffusion_steps=None, temperature=0.85, top_p=0.85, top_k=50,
                 self_conditioning_prob=None, seed=None, **kwargs):
        """
        Block diffusion 去噪采样

        从随机初始化的 canvas 出发, 进行 `num_diffusion_steps` 轮去噪:
        每轮用模型预测 logits 并重采样全部 canvas token (self-conditioning 可选),
        最后一轮取 argmax 得到最终 canvas。

        参数:
            inputs: input_ids (prefix tokens), 形状 (batch, prefix_len)
            attention_mask: prefix 的注意力掩码
            max_new_tokens: canvas 长度 (默认 config.canvas_length)
            num_diffusion_steps: 去噪步数 (默认 config.num_diffusion_steps)
            temperature: 采样温度
            top_p: 核采样概率阈值
            top_k: Top-K 采样
            self_conditioning_prob: 每步使用 self-conditioning 的概率
            seed: 随机种子

        返回:
            完整序列 (prefix + canvas), 形状 (batch, prefix_len + canvas_len)
        """
        if seed is not None:
            torch.manual_seed(seed)

        input_ids = kwargs.pop("input_ids", inputs)
        device = input_ids.device
        batch_size, prefix_len = input_ids.shape
        canvas_len = max_new_tokens or self.config.canvas_length
        num_diffusion_steps = num_diffusion_steps or self.config.num_diffusion_steps
        sc_prob = self.config.self_conditioning_prob if self_conditioning_prob is None else self_conditioning_prob

        # 初始 canvas: 从词表均匀随机采样
        canvas = torch.randint(0, self.config.vocab_size, (batch_size, canvas_len), device=device)
        prev_logits = None

        for step in range(num_diffusion_steps):
            # self-conditioning: 除第一步和最后一步外, 以 sc_prob 概率使用上一轮 logits
            if step > 0 and step < num_diffusion_steps - 1 and torch.rand(()) > sc_prob:
                prev_logits = None

            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=canvas,
                self_conditioning_logits=prev_logits,
            )
            logits = outputs.logits
            prev_logits = logits.detach()

            # 最后一轮: 贪婪取 argmax
            if step == num_diffusion_steps - 1:
                canvas = logits.argmax(dim=-1)
                continue

            # 采样
            logits = logits / temperature

            # Top-K 采样
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

            # Top-P (核) 采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cum_probs > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(-1, sorted_indices, mask)] = -float('inf')

            probs = torch.softmax(logits, dim=-1)
            # multinomial 只支持 1D/2D, 展平 (batch, seq, vocab) -> (batch*seq, vocab) 后采样再恢复
            canvas = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1).squeeze(-1)
            canvas = canvas.reshape(batch_size, canvas_len)

        # 返回 prefix + canvas 的完整序列
        return torch.cat([input_ids, canvas], dim=-1)

    @torch.inference_mode()
    def chat(model, tok, ques, history=[], **kw):
        """对话接口: 以历史对话为 prefix, diffusion 去噪生成回复。"""
        iids = tok.apply_chat_template(
            history + [{'role': 'user', 'content': ques}],
            add_generation_prompt=1,
        )
        iids = torch.tensor([iids]).to(model.device)
        oids = model.generate(
            inputs=iids,
            **(model.generation_config.to_dict() | kw),
        )
        oids = oids[0][len(iids):].tolist()
        if oids[-1] == tok.eos_token_id:
            oids = oids[:-1]
        ans = tok.decode(oids)

        return ans
