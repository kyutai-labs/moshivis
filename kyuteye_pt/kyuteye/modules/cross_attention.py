# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Gated cross-attention module"""

from typing import Any, Callable, Literal, Optional, Tuple, Union

import torch
from kyuteye.modules.attention import MultiheadAttention
from kyuteye.modules.streaming_utils import StreamingModule


class SharedModuleType(type):
    """Wrapper to build shared Pytorch modules"""

    _instances = {}  # type: ignore

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(SharedModuleType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class XAGate(torch.nn.Module):
    """Learned multiplicative gating per layer"""

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        activation: Literal["tanh", "sigmoid"] = "tanh",
        conditional_gating: bool = False,
        dims: Optional[int] = None,
        hidden_dims_factor: float = 0.125,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.alpha: torch.nn.Parameter | torch.nn.Module
        self.conditional_gating = conditional_gating
        if self.conditional_gating:
            assert dims is not None
            hidden_dims = int(hidden_dims_factor * dims)
            self.alpha = torch.nn.Sequential(
                torch.nn.Linear(dims, hidden_dims, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dims, dims, bias=False),
            )
        else:
            self.alpha = torch.nn.Parameter(
                torch.full((1, 1, 1), 0.0, device=device, dtype=dtype)
            )
        self.act: Callable
        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "sigmoid":
            # shift left to mimic initialization ~ close to 0
            self.act = lambda x: torch.sigmoid(x - 4)
        else:
            raise NotImplementedError("Unknown activation function", activation)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gating (constant scaling or input-dependent)"""
        if isinstance(self.alpha, torch.nn.Parameter):
            gate_weight = self.act(self.alpha)
        else:
            gate_weight = self.act(self.alpha(x))
        return x * gate_weight, gate_weight


class SharedXaGate(XAGate, metaclass=SharedModuleType):
    """Shared XaGate"""

    pass  # pylint: disable=unnecessary-pass


class CrossAttention(MultiheadAttention):
    """Cross attention module"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["cross_attention"] = True
        super().__init__(*args, **kwargs)


class SharedCrossAttention(CrossAttention, metaclass=SharedModuleType):
    """Shared Cross Attention projection across all layers"""

    pass  # pylint: disable=unnecessary-pass


class GatedCrossAttention(StreamingModule):
    """Gated Cross Attention with or without sharing parameters across layers"""

    def __init__(
        self,
        embed_dim: int,
        xa_gating: Literal["tanh", "sigmoid", "none"] = "tanh",
        xa_conditional_gating: bool = False,
        xa_shared: bool = True,
        xa_gate_shared: bool = False,
        xa_delay: int = 0,
        xa_start: Literal["start", "boi", "eoi"] = "start",
        xa_end: Literal["end", "eoi"] | int = "end",
        xa_step: int = 1,
        xa_dim: Optional[int] = None,
        **attn_kwargs: Any,
    ) -> None:
        """
        Initializes a Gated CrossAttention module.

        :param xa_gating: Whether (and which type of) to add multiplicative gating at
            the output of the cross-attention layer
        :param xa_shared: Whether to share the projection parameters of this
            corss-attention module across all layers

            tanh_gate (bool, optional): Whether to apply tanh activation to the gate.
                 Defaults to True.
            share_tanh_gate (bool, optional): Whether to share the tanh gate parameters
                across different attention heads. Defaults to True.
            shared_parameters (bool, optional): Whether to share the attention parameters
                across different attention heads. Defaults to True.
            shift (int, optional): The shift value for the cross attention, i.e., whether
                the update is to be based on past queries.
                 Defaults to 0, i.e., no shifting.
            xa_scope (Literal["images", "all_after_first_boi"], optional): The scope of the
                attention. Can be "images" or "all_after_first_boi". Defaults to "images".
            **attn_kwargs: Additional keyword arguments to be passed to the CrossAttention
                or SharedCrossAttention class.

        """
        super().__init__()
        device = attn_kwargs.get("device", None)
        dtype = attn_kwargs.get("dtype", None)

        # Attention module
        self.mha = (SharedCrossAttention if xa_shared else CrossAttention)(
            embed_dim=embed_dim, xa_dim=xa_dim, **attn_kwargs
        )

        # Output Gating
        self.gate: Optional[torch.nn.Module] = None
        if xa_gating != "none":
            self.gate = (SharedXaGate if xa_gate_shared else XAGate)(
                activation=xa_gating,
                device=device,
                dtype=dtype,
                dims=embed_dim,
                conditional_gating=xa_conditional_gating,
            )

        # If the XA module AND gates are shared, we add a per-layer
        # coefficient to still have some modularity
        self.per_layer_alpha: Optional[torch.nn.Parameter] = None
        if xa_shared and xa_gate_shared:
            self.per_layer_alpha = torch.nn.Parameter(
                torch.full((1, 1, 1), 1.0, device=device, dtype=dtype)
            )

        # Determine the xa scope
        self.xa_start = xa_start
        self.xa_end = xa_end
        self.xa_step = xa_step
        self.xa_delay = xa_delay

        self._active = True

    def get_xa_scope(
        self, x: torch.Tensor, image_tokens_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Build the mask of which tokens should receive contribution from
        the cross-attention image tokens"""
        if self.is_streaming and isinstance(self.xa_start, int):
            return int(self.streaming_offset >= self.xa_start)

        if not (
            self.xa_start in {"start", "boi", "eoi"} or isinstance(self.xa_start, int)
        ) or not (self.xa_end in {"end", "eoi"} or isinstance(self.xa_end, int)):
            raise NotImplementedError(
                f"Unsupported XA scope : {self.xa_start}, {self.xa_end}"
            )

        if self.xa_start == "start":
            # scope = 'all'
            if self.xa_end == "end":
                mask = torch.ones_like(x[:, :, :1])
            # scope = 'start + relative'abs
            elif isinstance(self.xa_end, int):
                mask = torch.zeros_like(x[:, :, :1])
                mask[:, : self.xa_end, :] = 1
            # everything else is kinda weird (0, BoI) ? (0, EoI) ?
            else:
                raise NotImplementedError(
                    f"Unsupported XA scope : {self.xa_start}, {self.xa_end}"
                )
        elif isinstance(self.xa_start, int):
            if isinstance(self.xa_end, int):
                mask = torch.zeros_like(x[:, :, :1])
                # self.xa_end is relative to self.xa_start
                mask[:, self.xa_start : self.xa_start + self.xa_end, :] = 1
            elif self.xa_end == "end":
                # If for some reason the start is further than the end, we do not attend
                mask = torch.zeros_like(x[:, :, :1])
                if not self.xa_start > mask.shape[1]:
                    mask[:, self.xa_start :, :] = 1
            else:
                raise NotImplementedError(
                    f"Unsupported XA scope : {self.xa_start}, {self.xa_end}"
                )
        else:
            assert image_tokens_mask is not None
            # another easy case is attention only to the image tokens
            if self.xa_start == "boi" and self.xa_end == "eoi":
                mask = image_tokens_mask

            # Otherwise, we build the mask  manually
            # first, determine the start, either EoI or BoI
            elif self.xa_start == "boi":
                # everything that comes after boi
                # e.g. (0 0 0 1 1 1 1 0 0 0 ) becomes
                # (0 0 0 1 2 3 4 4 4 4 )
                mask = torch.cumsum(image_tokens_mask, dim=1)
            elif self.xa_start == "eoi":
                # everything that comes after eoi
                # e.g. (0 0 0 1 1 1 1 0 0 0 ) becomes
                # (0 0 0 0 0 0 0 4 4 4 )
                mask = (1 - image_tokens_mask) * torch.cumsum(image_tokens_mask, dim=1)
            else:
                raise NotImplementedError(
                    f"Unsupported XA scope starting at {self.xa_start}"
                )

            # then determine the end of the mask
            mask = torch.greater(mask, 0).to(x.dtype)

            if isinstance(self.xa_end, int):
                mask = torch.cumsum(mask, dim=1)
                mask = torch.lt(mask, self.xa_end + 1).to(x.dtype)

        # then apply xa_step
        if self.xa_step > 1:
            if self.xa_start not in {"start", 0}:
                raise NotImplementedError("xa_step")
            step_mask = torch.eq(
                torch.remainder(torch.arange(x.shape[1]), self.xa_step), 0
            ).to(mask)
            mask *= step_mask[None, :, None].float()
        return mask

    def is_active(self, image_tokens_mask: Optional[torch.Tensor] = None) -> bool:
        """Whether this model is active during the forward pass"""
        if self.is_streaming:
            # case 1: never stop
            if self.xa_end == "end":
                return self.xa_start == "start" or (
                    isinstance(self.xa_start, int)
                    and self.streaming_offset >= self.xa_start
                )
            # case 2: XA applies to the image; we only apply cross-attention
            # in the step where the image is inserted
            if self.xa_end == "eoi" and image_tokens_mask is None:
                return False
            # Case 3: XA end is relative to XA start
            if isinstance(self.xa_end, int):
                if self.xa_start == "start":
                    offset = 0
                elif isinstance(self.xa_start, int):
                    offset = self.xa_start
                elif self.xa_start == "boi":
                    if self.has_streaming_attribute("image_insert_start"):
                        offset = self.get_streaming_info_as_int("image_insert_start")
                    else:
                        return False
                elif self.xa_start == "eoi":
                    if self.has_streaming_attribute("image_insert_end"):
                        offset = self.get_streaming_info_as_int("image_insert_end")
                    else:
                        return False
                else:
                    raise ValueError("Unsupported xa_start option", self.xa_start)
                # if xa_step is active
                if self.xa_step > 1:
                    return (
                        offset <= self.streaming_offset < offset + self.xa_end
                    ) and ((self.streaming_offset - offset) % self.xa_step == 0)
                # base case
                return offset <= self.streaming_offset < offset + self.xa_end
        # In training, we are always active and just build the xa scope mask
        return True

    def forward(
        self,
        x: torch.Tensor,
        key: Optional[Tuple[torch.Tensor, torch.Tensor] | torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        image_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Gated Cross attention

        :param x: Input query tensor
        :param key: Source tokens for the keys
        :param value: Source tokens for the values (typically, equal to keys)
        :param cross_attention_mask: Mask to apply to the cross-attention
        :param image_tokens_mask: Mask indicating where the image tokens
            are in the stream. This is used to determine which token should
            cross-attend to the image
        """
        gate_weight = None
        if self.is_streaming:
            if not self.has_streaming_attribute("offset"):
                self.streaming_offset = 0
            # Mark the last inserted image's position in streaming mode
            if image_tokens_mask is not None:
                self.add_streaming_attribute(
                    "image_insert_start", self.streaming_offset
                )
                image_lengths = torch.sum(image_tokens_mask[:, :, 0], dim=1)
                assert torch.all(
                    torch.eq(image_lengths, int(image_lengths[0].item()))
                ), "All inserted images must have the same number of tokens"
                self.add_streaming_attribute(
                    "image_insert_end",
                    self.streaming_offset + int(image_lengths[0].item()),
                )
        if not self.is_active(image_tokens_mask=image_tokens_mask):
            x = torch.zeros_like(x)
        else:
            x = self.mha(
                query=x, key=key, value=value, attention_mask=cross_attention_mask
            )

            if self.gate is not None:
                x, gate_weight = self.gate(x)

            if self.per_layer_alpha is not None:
                x *= self.per_layer_alpha

            # Mask out tokens that should not receive signal from the image
            x = x * self.get_xa_scope(x, image_tokens_mask=image_tokens_mask)

        # Update streaming offset
        if self.is_streaming:
            self.streaming_offset += x.shape[1]

        return x, gate_weight
