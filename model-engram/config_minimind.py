import math
from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        ### Engram (n-gram hash 外部记忆) specific configs
        # 是否启用 Engram 模块
        self.use_engram = kwargs.get("use_engram", False)
        # 在哪些层索引上插入 Engram
        self.engram_layer_ids = kwargs.get("engram_layer_ids", [1, num_hidden_layers - 2] if num_hidden_layers >= 3 else [1])
        # 最大 n-gram 阶数
        self.engram_max_ngram_size = kwargs.get("engram_max_ngram_size", 3)
        # 每个 n-gram 的 embedding 维度 / 头数
        self.engram_n_embed_per_ngram = kwargs.get("engram_n_embed_per_ngram", 512)
        self.engram_n_head_per_ngram = kwargs.get("engram_n_head_per_ngram", 8)
        # 每个 n-gram 的头对应的哈希槽数（决定 codebook/嵌入表大小，按 n-gram 阶数排列）。
        # 槽数取「向下的质数」，因此实际值可能略小于此。默认 512/1024，
        # 对 MiniMind 这种小模型是合理量级；可按需增大以接近 Engram 论文的容量。
        self.engram_vocab_size = kwargs.get("engram_vocab_size", [512, 1024][: (self.engram_max_ngram_size - 1)])
        # 短卷积核 / 种子 / pad
        self.engram_kernel_size = kwargs.get("engram_kernel_size", 4)
        self.engram_seed = kwargs.get("engram_seed", 0)
        # 哈希 pad 用的 token（默认 eos）
        self.engram_pad_id = kwargs.get("engram_pad_id", self.eos_token_id)
