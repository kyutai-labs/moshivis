# pylint: disable=redefined-outer-name, pointless-string-statement
"""Port of Helium from Jax to Pytorch and then HF. 
The architecture is also made to be easily converted to the mimi/audiocraft codebase"""

from typing import Any, Literal, Optional

import torch

from kyuteye.modules.transformer import Transformer
from kyuteye.modules.utils import ClampedEmbedding, NormalizationLayer


class Helium(torch.nn.Module):
    """Jax -> Pytorch port of Helium (text LLM)

    :param dim: Inner dimension of the tokens
    :param num_head: Number of heads
    :param num_layers: Number of layers
    :param hidden_scale: Scale for the inner dimension of the FFN layes (wrt. `dim`)
    :param context: Context size for the model. This is only used to determine the automatic
        causal mask and also set the KV cache accordingly at inference
    :param cross_attention: Whether to add cross-attention layers
    :param card: Cardinality of the vocabulary
    :param output_card: (Optional) can be used to specify a different number of output tokens
        than card (which is used for the embedding layers). This is useful for audio
        models that add an extra *initial token* which is never predicted
    :param norm: Type of normalization layers to use
    :param positional_embedding: Type of positional embedding to use
    :param max_period: Maximum period / theta for RoPE embeddings
    :param causal: Whether to automatically force a causal mask in MHSA
    :param gating: If True, will use gated FFN (Swi-GLU like)
    :param activation: Activation to use for the FFN gating, if `gating` is True
    :param padding_token_id: Padding token ID of the tokenizer
    :param freeze_padding_embedding: If True, the embedding of the padding token ID
        will not receive gradients/updates during training
    :param device: Device to load the model on
    :param dtype: Dtype to define the model parameters as
    """

    def __init__(
        self,
        # Architecture
        dim: int,
        num_heads: int,
        num_layers: int,
        hidden_scale: float = 4.125,
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
        max_period: float = 10000,
        causal: bool = True,
        gating: bool = True,
        activation: str = "silu",
        # Padding token
        padding_token_id: int = 3,
        freeze_padding_embedding: bool = False,
        zero_token_id: int = -1,
        # Other kwargs
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ):
        super().__init__()
        # Initial tokens projection
        self.dim = dim
        self.text_emb = ClampedEmbedding(
            card,
            dim,
            padding_idx=padding_token_id if freeze_padding_embedding else None,
            zero_idx=zero_token_id,
            dtype=dtype,
            device=device,
        )
        # Output RMS Norm and linear layer
        self.out_norm = getattr(NormalizationLayer, norm.upper()).create_norm_fn(
            dim, device=device, dtype=dtype
        )
        self.text_linear = torch.nn.Linear(
            dim, card if output_card is None else output_card, bias=False
        )
        self.cross_attention = cross_attention
        # Main transformer
        self.transformer = Transformer(
            # Architecture
            d_model=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=int(hidden_scale * dim),
            causal=causal,
            context=context,
            cross_attention=cross_attention,
            positional_embedding=positional_embedding,
            max_period=max_period,
            # Transformer Layer kwargs
            gating=gating,
            activation=activation,
            norm=norm,
            # Others
            device=device,
            dtype=dtype,
            **kwargs,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_src: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> torch.Tensor:
        """Forward function

        :param input_ids: Input tokens IDs with size `(batch_size, seq_length)`
        :param inputs_embeds: If given, this will skip input_ids and the initial embedding phase.
            Instead, `inputs_embeds` are directly fed to the transformer
        :param attention_mask: 0/1 Attention mask with shape `(batch_size, seq_length)`.
            Indicates which tokens to mask in the attention, e.g. padding tokens
        :param cross_attention_src: Cross-attention source with shape
            `(batch_size, seq_length, dim)`
        :param cross_attention_mask: Cross-attention mask with shape `(batch_size, seq_length)`
        :param return_features: If True, will skip the last classification layer and only
            output features with dimensions `(batch_size, seq_length, dim)`
        """
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.text_emb(input_ids)

        x = self.transformer(
            inputs_embeds,
            attention_mask=attention_mask,
            cross_attention_src=cross_attention_src,
            cross_attention_mask=cross_attention_mask,
        )
        x = self.out_norm(x)
        if return_features:
            return x
        return self.text_linear(x)
