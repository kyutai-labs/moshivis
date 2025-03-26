# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Transformer base"""

from typing import Any, List, Literal, Optional, Tuple

import torch
from kyuteye.modules.attention import MultiheadAttention
from kyuteye.modules.cross_attention import GatedCrossAttention
from kyuteye.modules.streaming_utils import StreamingModule
from kyuteye.modules.utils import (
    NormalizationLayer,
    RotaryEmbedding,
    create_sin_embedding,
    get_activation,
    make_ffn,
)


class TransformerLayer(StreamingModule):
    """Base TransformerLayer with causal support.
    This also integrates cross_attention, when passing `cross_attention=True`,
    rather than having two separate classes like in PyTorch.

    :param d_model: Dimension of the data.
    :param num_heads: Number of heads.
    :param dim_feedforward: Intermediate dimension of FF module.
    param causal: Causal mask applied automatically.
    :param context: Receptive field for the causal mask, infinite if None.
    :param custom: Use custom MHA implementation, for testing / benchmarking.
    :param cross_attention: If True, expect to get secondary input for cross-attention.
        Cross attention will use the default MHA, as it typically won't require
        special treatment.
    :param rope: Optional Rope embedding to use.
    :param norm: Normalization to use. Currently, only 'layer_norm' is supported.
    :param layer_scale: If not None, LayerScale will be used with the given value as initial scale.
    :param weights_per_step: use different weights per depformer step. If non zero,
        should correspond to the number of possible time steps.
    :param gating: if True, uses SwiGLU like gating in the FFN
    :param activation: Activation function to use in the FFN layer
    :param device: Device on which to initialize the module.
    :param dtype: Dtype to use.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int | List[int] = 2048,
        causal: bool = True,
        context: Optional[int] = None,
        cross_attention: bool = False,
        rope: Optional[RotaryEmbedding] = None,
        norm: Literal[
            "layer_norm",
            "layer_norm_f32",
            "rms_norm",
            "rms_norm_f32",
            "real_rms_norm",
            "real_rms_norm_f32",
        ] = "layer_norm",
        weights_per_step: int = 0,
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
        # Cross attention to image tokens parameters
        xa_gating: Literal["sigmoid", "tanh", "none"] = "tanh",
        xa_conditional_gating: bool = False,
        xa_shared: bool = True,
        xa_gate_shared: bool = False,
        xa_delay: int = 0,
        xa_start: Optional[Literal["start", "boi", "eoi"]] = None,
        xa_end: Optional[Literal["end", "eoi"] | int] = None,
        xa_step: int = 1,
        xa_dim: Optional[int] = None,
        # Factory kwargs
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        assert norm.upper() in [x.name for x in NormalizationLayer]
        factory_kwargs = {"device": device, "dtype": dtype}
        self.causal = causal

        self.self_attn: MultiheadAttention = MultiheadAttention(
            causal=causal,
            context=context,
            rope=rope,
            weights_per_step=weights_per_step,
            embed_dim=d_model,
            num_heads=num_heads,
            **factory_kwargs,  # type: ignore
        )  # type: ignore
        self.norm1 = getattr(NormalizationLayer, norm.upper()).create_norm_fn(
            d_model, **factory_kwargs
        )

        # Cross attention (optional)
        self.cross_attention: Optional[torch.nn.Module] = None
        if cross_attention:
            assert xa_start is not None and xa_end is not None
            self.cross_attention = GatedCrossAttention(
                xa_gating=xa_gating,
                xa_conditional_gating=xa_conditional_gating,
                xa_shared=xa_shared,
                xa_gate_shared=xa_gate_shared,
                xa_delay=xa_delay,
                xa_start=xa_start,
                xa_end=xa_end,
                xa_step=xa_step,
                embed_dim=d_model,
                xa_dim=xa_dim,
                num_heads=num_heads,
                **factory_kwargs,  # type: ignore
            )
            self.norm_cross = getattr(NormalizationLayer, norm.upper()).create_norm_fn(
                d_model, **factory_kwargs
            )

        # gating = FFN/MLP
        self.activation = get_activation(activation)
        self.gating = make_ffn(
            d_model,
            dim_feedforward,
            self.activation,
            gating=gating,
            weights_per_step=weights_per_step,
            **factory_kwargs,
        )
        self.norm2 = getattr(NormalizationLayer, norm.upper()).create_norm_fn(
            d_model, **factory_kwargs
        )

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward block"""
        x_orig = x
        x = self.norm2(x)

        if isinstance(self.gating, torch.nn.ModuleList):
            # Inference
            ys: List[torch.Tensor] = []
            for t in range(x.shape[1]):
                y = self.gating[self.streaming_offset + t](x[:, len(ys) : len(ys) + 1])
                ys.append(y)
            update = torch.cat(ys, dim=1)
        else:
            # Training: Apply all levels in parallel
            update = self.gating(x)

        return x_orig + update

    def _maybe_cross_attend(
        self,
        x: torch.Tensor,
        cross_attention_src: Optional[Tuple[torch.Tensor, torch.Tensor] | torch.Tensor],
        cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Cross attention"""
        if self.cross_attention is not None and cross_attention_src is not None:
            x_orig = x
            update, gate_weight = self.cross_attention(
                self.norm_cross(x),
                cross_attention_src,
                None,
                cross_attention_mask,
            )
            return x_orig + update, gate_weight

        return x, None

    def _self_attend(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-attention"""
        x_orig = x
        update = self.self_attn(
            self.norm1(x),
            attention_mask=attention_mask,
        )
        return x_orig + update

    def forward(
        self,
        x: torch.Tensor,
        cross_attention_src: Optional[
            Tuple[torch.Tensor, torch.Tensor] | torch.Tensor
        ] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """(Optional) MHSA => (Optional) cross-attention => FFN

        :param x: Input query tensor
        :param cross_attention_src: Optional tensor used as source for the
            keys and values in the cross-attention
        :param cross_attention_mask: Mask for the cross-attention
        :param attention_mask: Attention mask for the self-attention. Can be
            used to mask out tokens that shouldn't be involved in the
            self-attention computation
        :param image_tokens_mask: Mask indicating where the image tokens
            are in the stream with shape (B, seq, 1). This is used in two places:
              * (i) in the cross-attention, to determine which token should
                cross-attend to the image
              * (ii) in the self-attention, to allow the attention mask to
                be non causal inside the image tokens
        """
        if self.is_streaming and not self.has_streaming_attribute("offset"):
            self.streaming_offset = 0

        x = self._self_attend(x, attention_mask=attention_mask)

        x, gate_weight = self._maybe_cross_attend(
            x,
            cross_attention_src=cross_attention_src,
            cross_attention_mask=cross_attention_mask,
        )

        x = self._ff_block(x)

        # Update streaming offset for the multi linear FFNs in the depformer
        if self.is_streaming:
            self.streaming_offset += x.shape[1]
        return x, gate_weight


class Transformer(StreamingModule):
    """Transformer with Streaming / Causal support.

    :param d_model: Dimension of the data.
    :param num_heads: Number of heads.
    :param num_layers: Number of transformer layers
    :param dim_feedforward: Intermediate dimension of FF module.
    :param causal: If True, automatically applies a causal mask.
    :param context: Size of the receptive field for the causal mask.
        If None, assumes infinite context.
    :param cross_attention: If True, `forward` will expect to get
        secondary input for cross-attention.
    :param xa_layers: If a non-empty tuple, specified which layers
        to add cross-attention layers to. If None or an empty tuple,
        and cross_attention is True, will apply cross attention
        in every layer
    :param positional_embedding: Positional embedding strategy
        (sin, rope, sin_rope, or none).
    :param max_period: Maximum period for the sin/cos in RoPE embedding.
    :param positional_scale: Scale of positional embedding, set to 0 to deactivate.
    :parma device: Device on which to initialize the model.
    :param dtype: Device type to use.
    **kwargs: Extra arguments fed to the `TransformerLayer` constructor (e.g. layer_scale)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: Optional[int] = None,
        cross_attention: bool = False,
        xa_layers: Optional[Tuple[int, ...]] = None,
        positional_embedding: Literal["none", "sin", "rope", "sin_rope"] = "sin",
        max_period: float = 10000,
        positional_scale: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale

        assert positional_embedding in {"sin", "rope", "sin_rope", "none"}
        self.rope: Optional[RotaryEmbedding] = None
        if self.positional_embedding in {"rope", "sin_rope"}:
            self.rope = RotaryEmbedding(max_period=max_period)
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            cross_attend_layer = (
                xa_layers is None or len(xa_layers) == 0 or layer_idx in xa_layers
            )
            self.layers.append(
                TransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    cross_attention=cross_attention and cross_attend_layer,
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def set_context(self, context: Optional[int] = None) -> None:
        """Update context size in all MHSA layers"""
        for module in self.modules():
            if isinstance(module, MultiheadAttention):
                module.context = context

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, float]:
        """Forward pass"""
        _, seq_len, channels = x.shape

        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = torch.arange(seq_len, device=x.device).view(1, -1, 1)
            pos_emb = create_sin_embedding(
                positions, channels, max_period=self.max_period, dtype=x.dtype
            )
            x = x + self.positional_scale * pos_emb

        alpha = 0.0
        for layer_idx, layer in enumerate(self.layers):
            x, gate_weight = layer(x, *args, **kwargs)
            if gate_weight is not None and layer_idx >= len(self.layers) - 10:
                alpha += torch.mean(gate_weight).cpu().item()

        return x, alpha / min(10, len(self.layers))
