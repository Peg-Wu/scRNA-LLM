import math
import torch
import torch.nn.functional as F

from .loss import *
from torch import nn
from .modeling_output import *
from typing import Optional, Tuple
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from .configuration_stella import STELLAConfig


class ContinuousValueEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(1, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        x = x.to(self.linear1.weight.dtype)
        # expand last dimension
        x = x.unsqueeze(-1)
        x = self.act_fn(self.linear1(x))
        x = self.linear2(x)
        return x


class STELLAEmbeddings(nn.Module):
    """Construct the embeddings from gene_symbol and gene_expression embeddings."""

    def __init__(self, config):
        super().__init__()
        self.gene_symbol_embedding = nn.Embedding(
            config.gene_symbol_vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        if config.input_gene_expr_type == "bin":
            self.gene_expression_embedding = nn.Embedding(
                config.bin_vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id,
            )
        elif config.input_gene_expr_type == "continuous":
            self.gene_expression_embedding = ContinuousValueEncoder(config)
        else:
            raise ValueError("input_gene_expr_type should be bin or continuous.")

    def forward(
        self,
        input_ids_gene_symbol: Optional[torch.LongTensor] = None,
        input_ids_gene_expression: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        embeddings = self.gene_symbol_embedding(
            input_ids_gene_symbol
        ) + self.gene_expression_embedding(input_ids_gene_expression)
        return embeddings


class DeepseekRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepseekMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.contiguous().view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList(
            [
                DeepseekMLP(config, intermediate_size=config.moe_intermediate_size)
                for i in range(config.n_routed_experts)
            ]
        )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(
                config=config, intermediate_size=intermediate_size
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.contiguous().view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                expert_output = expert(hidden_states[flat_topk_idx == i])
                y = y.to(expert_output.dtype)
                y[flat_topk_idx == i] = expert_output
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(
                hidden_states, flat_topk_idx, topk_weight.view(-1, 1)
            ).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        if self.config.pretraining_tp > 1:
            qkv_slicing = self.all_head_size // self.config.pretraining_tp
            query_slices = self.query.weight.split(qkv_slicing, dim=0)
            key_slices = self.key.weight.split(qkv_slicing, dim=0)
            value_slices = self.value.weight.split(qkv_slicing, dim=0)

            query_layer = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_layer = self.transpose_for_scores(torch.cat(query_layer, dim=-1))

            key_layer = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_layer = self.transpose_for_scores(torch.cat(key_layer, dim=-1))

            value_layer = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_layer = self.transpose_for_scores(torch.cat(value_layer, dim=-1))
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.config.pretraining_tp > 1:
            context_layer = context_layer.split(
                self.all_head_size // self.config.pretraining_tp, dim=2
            )
            Wo_slices = self.Wo.weight.split(
                self.config.hidden_size // self.config.pretraining_tp, dim=1
            )
            context_layer = self.dropout(
                sum(
                    [
                        F.linear(context_layer[i], Wo_slices[i])
                        for i in range(self.config.pretraining_tp)
                    ]
                )
            )
        else:
            context_layer = self.dropout(self.Wo(context_layer))

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class STELLALayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.mlp = (
            DeepseekMoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekMLP(config)
        )
        self.input_layernorm = DeepseekRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> STELLALayerModelOutput:
        residual = hidden_states

        # RMSNorm
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if output_attentions:
            hidden_states, attention_scores = self.self_attn(
                hidden_states, attention_mask, output_attentions
            )
        else:
            (hidden_states,) = self.self_attn(
                hidden_states, attention_mask, output_attentions
            )

        # Residual Connection
        hidden_states = residual.to(hidden_states.dtype) + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # Residual Connection
        hidden_states = residual + hidden_states

        return STELLALayerModelOutput(
            hidden_states=hidden_states,
            attention_score=attention_scores if output_attentions else None,
        )


class STELLAEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                STELLALayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeepseekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> STELLAEncoderModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer_module,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions += (layer_outputs[1],)

        hidden_states = self.norm(
            hidden_states
        )  # conduct rmsnorm on last layer hidden_states

        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # last layer hidden_states

        return STELLAEncoderModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class STELLAPreTrainedModel(PreTrainedModel):
    config_class = STELLAConfig
    base_model_prefix = "stella"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class STELLAModel(STELLAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.config = config
        self.embeddings = STELLAEmbeddings(config)
        self.encoder = STELLAEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()
    
    def set_input_embeddings(self, value):
        self.embeddings.gene_symbol_embedding = value

    def forward(
        self,
        input_ids_gene_symbol: Optional[torch.Tensor] = None,
        input_ids_gene_expression: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> STELLAModelOutput:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        input_shape = input_ids_gene_symbol.size()

        embedding_output = self.embeddings(
            input_ids_gene_symbol=input_ids_gene_symbol,
            input_ids_gene_expression=input_ids_gene_expression,
        )

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0]  # last_hidden_state

        return STELLAModelOutput(
            last_hidden_state=sequence_output,  # last_hidden_state
            hidden_states=encoder_outputs.hidden_states,  # all_hidden_states
            attentions=encoder_outputs.attentions,  # all_attention_scores
        )


class MLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.decoder = nn.Linear(config.hidden_size, config.bin_vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform_act_fn(self.dense(hidden_states))
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class STELLAForMaskedLM(STELLAPreTrainedModel):
    _tied_weights_keys = ["mlmhead.decoder.weight"]

    # Mask Gene Expression Bin
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.stella = STELLAModel(config)
        self.mlmhead = MLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        assert isinstance(
            self.stella.embeddings.gene_expression_embedding, nn.Embedding
        ), (
            "You must use `nn.Embedding` for the gene expression embedding if you want to tie weights."
        )
        return self.mlmhead.decoder

    def forward(
        self,
        input_ids_gene_symbol: Optional[torch.Tensor] = None,
        input_ids_gene_expression: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> STELLAForMaskedLMOutput:
        if labels is None:
            raise ValueError("Please mask gene expression and generate labels.")

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        outputs = self.stella(
            input_ids_gene_symbol=input_ids_gene_symbol,
            input_ids_gene_expression=input_ids_gene_expression,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]  # last_hidden_states
        logits = self.mlmhead(hidden_states)  # (bsz, seq_len, gene_symbol_vocab_size)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        mlm_loss = loss_fct(
            logits.view(-1, self.config.bin_vocab_size), labels.view(-1)
        )

        return STELLAForMaskedLMOutput(
            loss=mlm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class STELLAForSequenceClassification(STELLAPreTrainedModel):
    r"""
    **Examples:**
        >>> model = STELLAForSequenceClassification.from_pretrained("path_to_pretrained_models", num_labels=...)
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.stella = STELLAModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids_gene_symbol: Optional[torch.Tensor] = None,
        input_ids_gene_expression: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutput:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        outputs = self.stella(
            input_ids_gene_symbol=input_ids_gene_symbol,
            input_ids_gene_expression=input_ids_gene_expression,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]  # last_hidden_states
        # hidden_states = hidden_states.mean(dim=1)  # mean pooling
        hidden_states = (hidden_states * attention_mask.unsqueeze(-1)).sum(
            dim=1
        ) / attention_mask.sum(dim=1, keepdim=True)  # mean pooling
        logits = self.classifier(hidden_states)  # (bsz, num_labels)

        loss_fct = nn.CrossEntropyLoss()
        cls_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=cls_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PerturbationDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """hidden_states is the output of the STELLAModel, (batch, seq_len, d_model)"""
        return self.fc(hidden_states).squeeze(-1)  # (batch, seq_len)


class STELLAForPerturbation(STELLAPreTrainedModel):
    r"""
    **Examples:**
        >>> model = STELLAForPerturbation.from_pretrained("path_to_pretrained_models", input_gene_expr_type="continuous")
    """

    def __init__(self, config):
        super().__init__(config)
        self.stella = STELLAModel(config)
        # 0: not perturbed; 1: perturbed
        self.perturbation_id_embedding = nn.Embedding(2, config.hidden_size)
        self.pertubation_decoder = PerturbationDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.stella.embeddings.gene_symbol_embedding
    
    def set_input_embeddings(self, value):
        self.stella.embeddings.gene_symbol_embedding = value

    def forward(
        self,
        input_ids_gene_symbol: Optional[torch.Tensor] = None,
        input_ids_gene_expression: Optional[torch.Tensor] = None,
        input_pert_flags: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> PertubationOutput:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        total_embeds = self.stella.embeddings(
            input_ids_gene_symbol, input_ids_gene_expression
        ) + self.perturbation_id_embedding(input_pert_flags)

        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(
                attention_mask, input_ids_gene_symbol.size()
            )

        encoder_output = self.stella.encoder(
            total_embeds, attention_mask, output_attentions, output_hidden_states
        )

        logits = F.relu(self.pertubation_decoder(encoder_output[0]))  # (batch, seq_len)

        pert_loss = None
        if labels is not None:
            # loss_fct = nn.MSELoss()
            loss_fct = MMDLoss()
            pert_loss = loss_fct(logits, labels)

        return PertubationOutput(
            loss=pert_loss,
            logits=logits,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )