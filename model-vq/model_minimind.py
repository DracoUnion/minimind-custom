"""
MiniMind 语言模型实现

基于Transformer架构的轻量级语言模型，支持:
- RMSNorm 层归一化
- RoPE (旋转位置编码)
- GQA (分组查询注意力)
- SwiGLU 激活函数
- MoE (混合专家模型)
- YaRN 长度外推
"""
import math, torch, torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.generation.utils import LogitsProcessorList
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

    优点:
    1. 计算更简单高效 (少一次均值计算)
    2. 在语言模型任务上表现与 LayerNorm 相当甚至更好
    参考论文: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 防止除零的小常数
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
        return (self.weight * self.norm(x.float())).type_as(x)


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

    架构:
    1. 线性投影: Q, K, V 分别投影
    2. RMSNorm: 对 Q 和 K 进行归一化 (Query/Key Norm)
    3. RoPE: 应用旋转位置编码
    4. KV Cache: 支持推理时的缓存
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

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        前向传播

        参数:
            x: 输入，形状 (batch_size, seq_len, hidden_size)
            position_embeddings: (cos, sin) 位置编码
            past_key_value: 缓存的 K, V (用于自回归生成)
            use_cache: 是否缓存 K, V
            attention_mask: 注意力掩码

        返回:
            output: 注意力输出
            past_kv: 更新后的 K, V 缓存
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
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 4. 拼接 KV Cache (如果存在)
        if past_key_value is not None:
            # 将新的 K, V 拼接到缓存的 K, V 后面
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # 保存当前的 K, V 用于返回
        past_kv = (xk, xv) if use_cache else None

        # 5. 调整维度顺序以便计算注意力
        # (batch, seq, head, dim) -> (batch, head, seq, dim)
        xq = rearrange(xq, 'b s h d -> b h s d')
        xk = rearrange(repeat_kv(xk, self.n_rep), 'b s h d -> b h s d')
        xv = rearrange(repeat_kv(xv, self.n_rep), 'b s h d -> b h s d')

        # 6. 注意力计算
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # 使用 Flash Attention (内存高效、计算更快)
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 手动实现注意力
            # Q @ K^T / sqrt(d_k)
            scores = (xq @ rearrange(xk, 'b h s d -> b h d s')) / math.sqrt(self.head_dim)
            # 应用因果掩码 (causal mask): 每个位置只能看到自己和之前的位置
            # triu(1) 创建上三角矩阵，对角线以上全为 -inf
            scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # 应用额外的 attention_mask (如果有)
            if attention_mask is not None:
                # (b, s) => (b, 1, 1, s)
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

    Pre-Norm vs Post-Norm:
    - Pre-Norm: LayerNorm 在残差连接之前，训练更稳定 (现代 LLM 常用)
    - Post-Norm: LayerNorm 在残差连接之后
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

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 1. 自注意力子层 (带残差连接)
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # Pre-Norm
            position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual  # 残差连接

        # 2. MLP 子层 (带残差连接)
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))  # Pre-Norm
        hidden_states += residual  # 残差连接

        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind 基础模型 (不含语言模型头)

    架构:
    1. Token Embedding
    2. 多个 Transformer Block 堆叠
    3. 最终 RMSNorm
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        # Token 嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer 块堆叠
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])

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

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """
        前向传播

        参数:
            input_ids: 输入 token IDs，形状 (batch, seq_len)
            attention_mask: 注意力掩码
            past_key_values: 缓存的 K, V 列表 (用于自回归生成)
            use_cache: 是否使用 KV Cache

        返回:
            hidden_states: 最后的隐藏状态
            presents: 更新后的 K, V 缓存列表
            aux_loss: MoE 辅助损失 (如果有)
        """
        batch_size, seq_length = input_ids.shape

        # 处理 HuggingFace 特定格式的 past_key_values
        if hasattr(past_key_values, 'layers'): past_key_values = None

        # 初始化 KV Cache 列表 (如果未提供)
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算起始位置 (用于 KV Cache)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Token 嵌入
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 获取当前序列的位置编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 逐层前向传播
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # 计算所有层的 MoE 辅助损失之和
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniMind 因果语言模型 (用于文本生成)

    继承:
    - PreTrainedModel: HuggingFace Transformers 基类，支持 save_pretrained/load_pretrained
    - GenerationMixin: 支持 generate() 方法进行文本生成
    """
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)

        # 基础模型
        self.model = MiniMindModel(self.config)

        # 语言模型头: 将隐藏状态映射到词表
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 权重绑定: 输入嵌入和输出投影共享权重
        # 减少参数量，提升性能
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        """
        前向传播

        参数:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            past_key_values: KV Cache
            use_cache: 是否使用缓存
            logits_to_keep: 只保留最后 N 个 logits (用于节省内存)
            labels: 目标标签 (用于计算 loss)

        返回:
            MoeCausalLMOutputWithPast 包含 loss, logits, past_key_values 等
        """
        # 获取隐藏状态
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)

        # 计算 logits (如果只保留部分，切片)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 计算语言模型损失 (如果提供了 labels)
        loss = None
        if labels is not None:
            # 移位: 预测下一个 token
            # logits 去掉最后一个，labels 去掉第一个
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # 展平后计算交叉熵
            loss = F.cross_entropy(rearrange(x, 'b s v -> (b s) v'), rearrange(y, 'b s -> (b s)'), ignore_index=-100)

        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    # 生成方法参考: https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def _generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """
        自回归文本生成

        参数:
            inputs: 输入文本或 input_ids
            attention_mask: 注意力掩码
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度 (越高越随机)
            top_p: 核采样概率阈值
            top_k: Top-K 采样
            eos_token_id: 结束符 ID
            streamer: 流式输出回调
            use_cache: 使用 KV Cache (显著提升生成速度)
            num_return_sequences: 生成序列数
            do_sample: 是否采样 (False 则用贪婪解码)
            repetition_penalty: 重复惩罚 (>1 降低重复)

        返回:
            生成的 token IDs，形状 (batch*num_return_sequences, input_len+new_tokens)
        """
        if max_new_tokens is None:
            max_new_tokens = 8192
        # 处理输入
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)

        # 标记已完成的序列 (遇到 EOS)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        # 流式输出初始输入
        if streamer: streamer.put(input_ids.cpu())

        # 自回归生成循环
        for _ in range(max_new_tokens):
            # 计算已有缓存长度
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0

            # 前向传播 (只传入新 token)
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)

            # 更新 attention_mask
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None

            # 取最后一个位置的 logits 并应用温度
            logits = outputs.logits[:, -1, :] / temperature

            # 重复惩罚
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    # 对已经出现过的 token 降低其概率
                    logits[i, torch.unique(input_ids[i])] /= repetition_penalty

            # Top-K 采样
            if top_k > 0:
                # 将不在 top-k 中的 logits 设为 -inf
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

            # Top-P (核) 采样
            if top_p < 1.0:
                # 按概率降序排序
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # 计算累积概率
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                # 标记超过 top_p 的位置
                mask = cum_probs > top_p
                # 右移一位，确保至少保留第一个 token
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                # 将 mask 中标记的位置设为 -inf
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            # 采样或贪婪选择下一个 token
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # 已完成的序列强制输出 EOS
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)

            # 追加到序列
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # 更新 KV Cache
            past_key_values = outputs.past_key_values if use_cache else None

            # 流式输出
            if streamer: streamer.put(next_token.cpu())

            # 检查是否所有序列都已完成
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        if streamer: streamer.end()

        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}
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
        ans = tok.decode(oids)
        
        return ans
