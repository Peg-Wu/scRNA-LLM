from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    hidden_size: int = 512,
    num_attention_heads: int = 8,
    num_hidden_layers: int = 6,
    intermediate_size: int = 512,
    moe_intermediate_size: int = 512,
    first_k_dense_replace: int = 0,
    moe_layer_freq: int = 1,
    hidden_act: str = "silu",
    seq_aux: bool = True,
    aux_loss_alpha: float = 0.001,
    num_experts_per_tok: int = 4,
    n_routed_experts: int = 16,
    n_shared_experts: int = 1,
    scoring_func: str = "softmax",
    norm_topk_prob: bool = False,
    attention_dropout: float = 0.0,
    initializer_range: float = 0.02,
    rms_norm_eps: float = 1e-6,
    pretraining_tp: int = 1,

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention heads ({self.num_attention_heads})!")

@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={
            "help": "Huggingface .arrow datasets location!"
                    "Use `TranscriptomeTokenizer` to save your datasets!"
        }
    )

    mlm_probability: float = field(
        default=0.15,
        metadata={
            "help": "Mask gene expression probability"
        }
    )
