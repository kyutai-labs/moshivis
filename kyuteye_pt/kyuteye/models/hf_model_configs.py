# pylint: disable=protected-access
"""Configuration for HF-compliant models"""

from typing import Any, Literal, Optional, Sequence

from transformers import PretrainedConfig


class HeliumConfig(PretrainedConfig):
    """Config class for Helium language models (LLM part)"""

    model_type = "Helium_v2"

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 12,
        num_layers: int = 24,
        hidden_scale: int = 4,
        context: int = 2048,
        cross_attention: bool = False,
        card: int = 32000,
        output_card: Optional[int] = None,
        norm: Literal[
            "rms_norm",
            "rms_norm_f32",
            "real_rms_norm",
            "real_rms_norm_f32",
        ] = "real_rms_norm_f32",
        positional_embedding: Literal["none", "sin", "rope", "sin_rope"] = "rope",
        freeze_padding_embedding: bool = False,
        max_period: float = 10000,
        causal: bool = True,
        gating: bool = True,
        activation: Literal[
            "none",
            "identity",
            "sigmoid",
            "tanh",
            "relu",
            "leaky_relu",
            "elu",
            "gelu",
            "silu",
            "mish",
            "softsign",
        ] = "silu",
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 3,
        **kwargs: Any,
    ):
        super().__init__(
            vocab_size=32000,
            hidden_size=dim,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_scale = hidden_scale
        self.context = context
        self.cross_attention = cross_attention
        self.positional_embedding = positional_embedding
        self.card = card
        self.output_card = output_card
        self.norm = norm
        self.max_period = max_period
        self.causal = causal
        self.gating = gating
        self.activation = activation
        self.freeze_padding_embedding = freeze_padding_embedding


class MoshiVisConfig(HeliumConfig):
    """Config for Moshi-Vis"""

    model_type = "Moshi_v1"

    def __init__(
        self,
        text_card: int = 32000,
        text_context: int = 3000,
        n_q: int = 8,
        n_q_per_source: Optional[int] = None,
        audio_card: int = 1024,
        depformer: bool = False,
        depformer_multi_linear: bool = False,
        depformer_pos_emb: Optional[Literal["none", "sin", "rope", "sin_rope"]] = None,
        depformer_dim: Optional[int] = None,
        depformer_dim_feedforward: Optional[int] = None,
        depformer_num_layers: Optional[int] = None,
        depformer_num_heads: Optional[int] = None,
        depformer_weights_per_step: bool = False,
        depformer_input_cumsum: bool = False,
        depformer_skip_self_attn: bool = False,
        delays: Optional[Sequence[int]] = None,
        same_initial: bool = False,
        text_loss_weight: float = 1.0,
        audio_loss_weight: float = 1.0,
        audio_other_channel_loss_weight: float = 1.0,
        audio_semantic_loss_weight: float = 100.0,
        audio_acoustic_loss_weight: float = 1.0,
        padding_loss_weight: float = 1.0,
        textonly_padding_loss_weight: float = 1.0,
        audio_padding_loss_weight: float = 1.0,
        sparsity_loss_weight: float = 0,
        mask_audio_codebooks_from: int = -1,
        add_pad_embed_text_only_other_tokens: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **kwargs,
        )
        # rename some args from HeliumConfig to explicitly
        # distinguish audio from text
        del self.card
        self.text_card = text_card
        del self.context
        self.text_context = text_context
        # add Audio config
        self.n_q = n_q
        self.n_q_per_source = n_q_per_source or n_q
        self.audio_card = audio_card
        self.depformer = depformer
        self.depformer_multi_linear = depformer_multi_linear
        self.depformer_pos_emb = depformer_pos_emb
        self.depformer_dim = depformer_dim
        self.depformer_dim_feedforward = depformer_dim_feedforward
        self.depformer_num_layers = depformer_num_layers
        self.depformer_num_heads = depformer_num_heads
        self.depformer_weights_per_step = depformer_weights_per_step
        self.depformer_input_cumsum = depformer_input_cumsum
        self.depformer_skip_self_attn = depformer_skip_self_attn
        self.delays = list(delays) if delays is not None else None
        self.same_initial = same_initial
        # loss weights for the different codebbooks
        self.text_loss_weight = text_loss_weight
        self.audio_loss_weight = audio_loss_weight
        self._audio_other_channel_loss_weight = audio_other_channel_loss_weight
        self._audio_semantic_loss_weight = audio_semantic_loss_weight
        self._audio_acoustic_loss_weight = audio_acoustic_loss_weight
        self.padding_loss_weight = padding_loss_weight
        self.textonly_padding_loss_weight = textonly_padding_loss_weight
        self.audio_padding_loss_weight = audio_padding_loss_weight
        self._sparsity_loss_weight = sparsity_loss_weight
        self.mask_audio_codebooks_from = mask_audio_codebooks_from
        self.add_pad_embed_text_only_other_tokens = add_pad_embed_text_only_other_tokens

        if self.mask_audio_codebooks_from >= 0:
            self.num_audio_tokens_in_loss_main = self.mask_audio_codebooks_from
            self.num_audio_tokens_in_loss_other = self.mask_audio_codebooks_from
        else:
            self.num_audio_tokens_in_loss_main = self.n_q_per_source - 1
            self.num_audio_tokens_in_loss_other = (self.n_q - self.n_q_per_source) - 1

    @property
    def total_audio_loss_weight(self) -> float:
        """Total weight used to normalize the losses on audio tokens"""
        return (
            # weight for Moshi
            self._audio_semantic_loss_weight
            + self._audio_acoustic_loss_weight * self.num_audio_tokens_in_loss_main
            # weight for Other
            + self._audio_other_channel_loss_weight
            * (
                self._audio_semantic_loss_weight
                + self._audio_acoustic_loss_weight * self.num_audio_tokens_in_loss_other
            )
        )

    @property
    def audio_semantic_loss_weight(self) -> float:
        """Loss weight set on the semantic token"""
        return (
            self.audio_loss_weight
            * self._audio_semantic_loss_weight
            / (self.total_audio_loss_weight + 1e-6)
        )

    @property
    def audio_acoustic_loss_weight(self) -> float:
        """Loss weight set on the semantic token"""
        return (
            self.audio_loss_weight
            * self._audio_acoustic_loss_weight
            / (self.total_audio_loss_weight + 1e-6)
        )

    @property
    def audio_other_semantic_loss_weight(self) -> float:
        """Loss weight set on audio codebooks for OTHER channel"""
        return self._audio_other_channel_loss_weight * self.audio_semantic_loss_weight

    @property
    def audio_other_acoustic_loss_weight(self) -> float:
        """Loss weight set on audio codebooks for OTHER channel"""
        return self._audio_other_channel_loss_weight * self.audio_acoustic_loss_weight

    @property
    def sparsity_loss_weight(self) -> float:
        """Loss weight set on the sparsity loss for the extended transformer."""
        return self._sparsity_loss_weight
