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

        # ---- 线性注意力 / 全注意力混合 (Mamba2 + Attention) ----
        # layer_types[i] 决定第 i 层用哪种 mixer:
        #   "linear_attention" -> Mamba2 选择性状态空间 (线性)
        #   "full_attention"   -> 标准 GQA 注意力
        # 若不指定, 按 "3 个 linear + 1 个 full" 的模式生成, 即每 4 层插入一个全注意力层,
        # 与 Nemotron H 的混合范式一致。
        layer_types = kwargs.get("layer_types", None)
        if layer_types is None:
            pattern = ["linear_attention"] * 3 + ["full_attention"]
            self.layer_types = (pattern * ((num_hidden_layers + len(pattern) - 1) // len(pattern)))[:num_hidden_layers]
        else:
            assert len(layer_types) == num_hidden_layers, "layer_types 长度必须等于 num_hidden_layers"
            self.layer_types = layer_types

        # ---- Mamba2 相关超参数 (仅 linear_attention 层使用) ----
        self.mamba_num_heads = kwargs.get("mamba_num_heads", 8)                    # SSM 头数
        self.mamba_head_dim = kwargs.get("mamba_head_dim", hidden_size // self.mamba_num_heads)
        self.mamba_intermediate_size = kwargs.get("mamba_intermediate_size", self.mamba_num_heads * self.mamba_head_dim)
        self.ssm_state_size = kwargs.get("ssm_state_size", 16)                     # 每个 SSM 状态大小 n
        self.conv_kernel = kwargs.get("conv_kernel", 4)                            # causal conv1d 核大小
        self.chunk_size = kwargs.get("chunk_size", 64)                             # 分块并行扫描块大小 (需 >= ssm_state_size)
        self.n_groups = kwargs.get("n_groups", 1)                                  # B/C 的组数
        self.time_step_min = kwargs.get("time_step_min", 1e-3)                    # 时间步 dt 下限
        self.time_step_max = kwargs.get("time_step_max", 0.1)                     # 时间步 dt 上限
        self.layer_norm_eps = kwargs.get("layer_norm_eps", self.rms_norm_eps)
        self.use_conv_bias = kwargs.get("use_conv_bias", False)
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
