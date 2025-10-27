from ..vocab import PAD_TOKEN_ID
from transformers import PretrainedConfig


class STELLAConfig(PretrainedConfig):
    r"""
    Args:
        gene_symbol_vocab_size (`int`, *optional*, defaults to 42571):
            Vocabulary size of gene symbols, including pad token.
        bin_vocab_size (`int`, *optional*, defaults to 102):
            Vocabulary size of gene expression, including pad and mask token.
        pad_token_id (`int`, *optional*):
            Padding token id.
        hidden_size (`int`, *optional*, defaults to 512):
            Dimension of the hidden representations.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 512):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 512):
            Dimension of the MoE representations.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                            \--k dense layers--/
        moe_layer_freq (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder.
        seq_aux = (`bool`, *optional*, defaults to True):
            Whether to compute the auxiliary loss for each individual sample.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Auxiliary loss weight coefficient.
        num_experts_per_tok (`int`, *optional*, defaults to None):
            Number of selected experts, None means dense model.
        n_routed_experts (`int`, *optional*, defaults to None):
            Number of routed experts, None means dense model.
        n_shared_experts (`int`, *optional*, defaults to None):
            Number of shared experts, None means dense model.
        scoring_func (`str`, *optional*, defaults to 'softmax'):
            Method of computing expert weights.
        norm_topk_prob (`bool`, *optional*, defaults to False):
            Whether to normalize the weights of the routed experts.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        input_gene_expr_type (`str`, *optional*, defaults to "bin"):
            Type of input gene expression, can be "bin" or "continuous".
        input_length (`int`, defaults to None):
            Used for `STELLAForPheno2Gene`, the input length of the model (= number of input genes).
    """

    model_type = "stella"
    # If you change number of bins, please modify bin_vocab_size, and MASK_TOKEN_ID in vocab.py!
    def __init__(
        self,
        gene_symbol_vocab_size=42571,
        bin_vocab_size=102,  # w/o mask: 51; w mask: 52
        pad_token_id=PAD_TOKEN_ID,
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=6,
        intermediate_size=1024,
        moe_intermediate_size=1024,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        hidden_act="silu",
        seq_aux=True,
        aux_loss_alpha=0.001,
        num_experts_per_tok=2,
        n_routed_experts=8,
        n_shared_experts=1,
        scoring_func="softmax",
        norm_topk_prob=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        pretraining_tp=1,
        input_gene_expr_type="bin",
        tie_word_embeddings=False,
        input_length=None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.gene_symbol_vocab_size = gene_symbol_vocab_size
        self.bin_vocab_size = bin_vocab_size
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_layers = num_hidden_layers
        self.moe_intermediate_size = moe_intermediate_size
        self.hidden_act = hidden_act
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.seq_aux = seq_aux
        self.aux_loss_alpha = aux_loss_alpha
        self.pretraining_tp = pretraining_tp
        self.intermediate_size = intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.input_gene_expr_type = input_gene_expr_type
        self.tie_word_embeddings = tie_word_embeddings
        self.input_length = input_length
        