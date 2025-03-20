# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Smaller building blocks:
  * FFN layers with Swi-GLU like gating
  * Normalization layers
  * Input embedding layer
  * Positional embeddings
"""

from enum import Enum, unique
from typing import Any, Callable, List, Literal, Optional, Tuple

import torch


def multi_linear(
    num_linear: int, weight: torch.Tensor, x: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    """Utility to apply a multi linear layer to the given input. A multi linear layer
    applies a different set of weight for each time step.

    Args:
        num_linear (int): Number of possible time steps and so number of linears.
        weight (torch.Tensor): Weight tensor, with shape `[num_linear * chout, chin]`.
        x (torch.Tensor): Input tensor, with shape `[B, T, C]`.
        offset (int): offset for the current time step, in particular for decoding, with
            time steps provided one by one.
    """
    ys = []
    # when calling the depformer, x.shape[1] is always 1, and the offset contains the
    # codebook index we care about
    for t in range(x.shape[1]):
        y = torch.nn.functional.linear(  # pylint: disable=not-callable
            x[:, t], weight.chunk(num_linear)[offset + t]
        )
        ys.append(y)
    out = torch.stack(ys, 1)
    return out


def get_activation(
    name: Literal[
        "sigmoid",
        "tanh",
        "relu",
        "leaky_relu",
        "elu",
        "gelu",
        "silu",
        "mish",
        "softsign",
        "identity",
        "none",
    ]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return correct activation function from the given name"""
    if name in {"sigmoid", "tanh", "relu"}:
        return getattr(torch, name)
    if name in {"leaky_relu", "elu", "gelu", "silu", "mish", "softsign"}:
        return getattr(torch.nn.functional, name)
    if name in {"identity", "none"}:
        return torch.nn.Identity()
    raise NotImplementedError(f"Unknown activation {name}")


def gating_forward_kernel(
    weight_in: torch.Tensor,
    weight_out: torch.Tensor,
    activation: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """Simple multiplicative gating strategy (SwiGLU like)"""
    x = torch.nn.functional.linear(x, weight_in)  # pylint: disable=not-callable
    batch_size, seq_len, _ = x.shape
    x = x.view(batch_size, seq_len, 2, -1)
    x = activation(x[..., 0, :]) * x[..., 1, :]
    x = torch.nn.functional.linear(x, weight_out)  # pylint: disable=not-callable
    return x


class ActivationGating(torch.nn.Module):
    """
    FFN layer with multiplicative gating using the given activation.

    :param dim: Dimensions of the tokens.
    :param activation: Activation function to use.
    :param factory_kwargs: Other kwargs passed to the linear layer, in particular device and dtype.
    """

    def __init__(
        self,
        dim: int,
        dim_feedforward: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        **factory_kwargs: Any,
    ):
        super().__init__()
        # We should have 8 d^2 param, instead we will have
        # 2 * h * d + h * d = 3 h * d = 8 d^2
        # so h = 8 d / 3 but following Hervé's advice we use 21 / 8 as an approx.
        if dim_feedforward == 4 * dim:
            hidden = (21 * dim) // 8
        else:
            hidden = (2 * dim_feedforward) // 3
        self.linear_in = torch.nn.Linear(dim, 2 * hidden, bias=False, **factory_kwargs)
        self.linear_out = torch.nn.Linear(hidden, dim, bias=False, **factory_kwargs)
        self.activation = activation

        max_params = 2 * dim * dim_feedforward
        params = sum(p.numel() for p in self.parameters())
        assert params <= max_params, f"Gating has {params} params, max is {max_params}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN with Swi-GLU like gating but customizable activation"""
        return gating_forward_kernel(
            self.linear_in.weight, self.linear_out.weight, self.activation, x
        )


class NoGating(torch.nn.Module):
    """
    Simple 2 layer MLP FFN layer
    """

    def __init__(
        self,
        dim: int,
        dim_feedforward: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        **factory_kwargs: Any,
    ):
        super().__init__()

        self.linear1 = torch.nn.Linear(
            dim, dim_feedforward, bias=False, **factory_kwargs
        )
        self.linear2 = torch.nn.Linear(
            dim_feedforward, dim, bias=False, **factory_kwargs
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple two layers MLP FFN"""
        return self.linear2(self.activation(self.linear1(x)))


def make_ffn(
    dim: int,
    dim_feedforward: int | List[int],
    activation_fn: Callable[[torch.Tensor], torch.Tensor],
    gating: bool = False,
    weights_per_step: int = 0,
    **factory_kwargs: Any,
) -> torch.nn.Module:
    """Create a FNN module

    :param dim: Number of input dimensions
    :param dim_feedforward: Nubmer of inner dimensions
    :param activation_fn: Activation function
    :param gating: If True, uses FFN with multiplicative gating
    :param factory_kwargs: Any extra argument fed to the Linear layer
        constructors (e.g. device, dtype)
    """
    ffn: torch.nn.Module
    if gating:
        if weights_per_step > 0:
            if isinstance(dim_feedforward, int):
                dim_feedforward = [dim_feedforward] * weights_per_step
            assert isinstance(dim_feedforward, list), dim_feedforward
            ffn = torch.nn.ModuleList(
                [
                    ActivationGating(dim, dim_out, activation_fn, **factory_kwargs)
                    for dim_out in dim_feedforward
                ]
            )
        else:
            assert isinstance(dim_feedforward, int)
            ffn = ActivationGating(
                dim, dim_feedforward, activation_fn, **factory_kwargs
            )
    else:
        assert isinstance(dim_feedforward, int)
        assert (
            weights_per_step == 0
        ), f"weights per step {weights_per_step} > 0 is not supported without gated FFN"
        ffn = NoGating(dim, dim_feedforward, activation_fn, **factory_kwargs)
    return ffn


class LayerNormF32(torch.nn.LayerNorm):
    """Layer norm executed in Float32 for maximal precision"""

    def forward(
        self, input: torch.Tensor  # pylint: disable=redefined-builtin
    ) -> torch.Tensor:
        """Applies the layer norm"""
        x_f32 = input.float()
        out_f32 = super().forward(x_f32)
        return out_f32.to(input.dtype)


def _rms_norm(
    x: torch.Tensor,
    alpha: torch.Tensor,
    dtype: Optional[torch.dtype],
    eps: float,
    use_var: bool,
    return_factor: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3, f"RMSNorm expects 3D inputs but got {x.shape}"
    in_dtype = x.dtype
    if dtype is not None:
        x = x.to(dtype)
    if use_var:
        var = eps + x.var(dim=2, keepdim=True)
    else:
        var = eps + torch.mean(x**2, dim=2, keepdim=True)
    if return_factor:
        factor = alpha.to(var) * torch.rsqrt(var)
        return (x * factor).to(in_dtype), factor.to(in_dtype)
    return (x * (alpha.to(var) * torch.rsqrt(var))).to(in_dtype)


class RMSNorm(torch.nn.Module):
    """RMSNorm layer

    :param dim: Input channels dimension
    :param eps: Epsilon
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        use_var: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.use_var = use_var
        self.alpha = torch.nn.Parameter(
            torch.full((1, 1, dim), 1.0, requires_grad=True, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RMS Norm call"""
        out = _rms_norm(x, self.alpha, self.dtype, self.eps, self.use_var)
        return out  # type: ignore[return-value]


@unique
class NormalizationLayer(Enum):
    """Select one of several normalization layers"""

    LAYER_NORM = 0
    LAYER_NORM_F32 = 1
    RMS_NORM = 2
    RMS_NORM_F32 = 3
    REAL_RMS_NORM = 4
    REAL_RMS_NORM_F32 = 5

    def create_norm_fn(self, dim: int, **kwargs: Any) -> torch.nn.Module:
        """Return the proper normalization layer initializer"""
        # Layer Norm
        if self == NormalizationLayer.LAYER_NORM:
            return torch.nn.LayerNorm(dim, eps=1e-5, **kwargs)
        if self == NormalizationLayer.LAYER_NORM_F32:
            return LayerNormF32(
                dim, eps=1e-8, **{k: v for k, v in kwargs.items() if k != "dtype"}
            )
        # Real RMS Norm using |x**2| normalization
        if self == NormalizationLayer.REAL_RMS_NORM:
            return RMSNorm(dim, eps=1e-5, use_var=False, **kwargs)
        if self == NormalizationLayer.REAL_RMS_NORM_F32:
            return RMSNorm(
                dim,
                eps=1e-8,
                dtype=torch.float32,
                use_var=False,
                **{k: v for k, v in kwargs.items() if k != "dtype"},
            )
        # RMS Norm using variance of the data
        if self == NormalizationLayer.RMS_NORM:
            return RMSNorm(dim, eps=1e-5, **kwargs)
        if self == NormalizationLayer.RMS_NORM_F32:
            return RMSNorm(
                dim,
                eps=1e-8,
                dtype=torch.float32,
                **{k: v for k, v in kwargs.items() if k != "dtype"},
            )
        raise NotImplementedError(f"Unknown norm type: {self.name}")


class ClampedEmbedding(torch.nn.Embedding):
    """An embedding layer such that all input IDs of the ID `zero_idx < 0`
    are mapped to zero at the output of the module

    Args:
        lr (float or None): Learning rate for the embedding layer if provided.
        norm (bool): if True, uses a layer norm after the embedding.
        zero_idx (int): special value indicating that the output should be exactly 0.
    """

    def __init__(
        self, *args: Any, norm: bool = False, zero_idx: int = -1, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm = None
        if norm:
            self.norm = NormalizationLayer.LAYER_NORM.create_norm_fn(self.embedding_dim)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx

    def forward(  # pylint: disable=arguments-renamed
        self, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Embed the input IDs"""
        is_zero = inputs == self.zero_idx
        # TODO(amelie)
        # assert torch.equal(is_zero, (inputs < 0)), (
        #    f"Input IDs contain negative IDs which are not the zero token ({self.zero_idx}). "
        #    "This is likely not the behavior you want"
        # )
        zero = torch.zeros(1, dtype=inputs.dtype, device=inputs.device)
        y = super().forward(inputs.clamp(min=0))
        if self.norm is not None:
            y = self.norm(y)
        y = torch.where(is_zero[..., None], zero, y)
        return y


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create fixed sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # Assumes BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full(
        [], max_period, device=positions.device, dtype=dtype
    )  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    max_period: float = 10_000,
    offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE embedding to he input queries/keys

    :param q: queries, shape `[B, T, H, D]`.
    :param k: keys, shape `[B, T, H, D]`.
    :param max_period: maximum period for the cos and sin (aka. `theta_rope`).
    """

    batch, seq_length, num_heads, dim = q.shape
    assert k.shape == q.shape

    ds = torch.arange(dim // 2, device=q.device, dtype=torch.float32)
    max_period_t = torch.full([1], max_period, device=q.device, dtype=torch.float32)
    freqs = 1.0 / (max_period_t ** (2 * ds / dim))
    ts = torch.arange(
        offset, seq_length + offset, device=q.device, dtype=torch.float32
    ).view(-1, 1, 1)

    q = q.view(batch, seq_length, num_heads, dim // 2, 2)
    k = k.view(batch, seq_length, num_heads, dim // 2, 2)

    # convention is `r` suffix is real part, `i` is imaginary.
    qr = q[..., 0].float()
    qi = q[..., 1].float()

    kr = k[..., 0].float()
    ki = k[..., 1].float()

    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr

    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)

    return qo.view(batch, seq_length, num_heads, dim), ko.view(
        batch, seq_length, num_heads, dim
    )


class RotaryEmbedding(torch.nn.Module):
    """Rotary positional embedding (RoPE) from [Su et al 2022](https://arxiv.org/abs/2104.09864).

    :param max_period: Maximum period of the rotation frequencies (aka `theta_rope`).
    """

    def __init__(self, max_period: float = 10000.0) -> None:
        super().__init__()
        self.max_period = max_period

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rope rotation to query or key tensor with an `offset` on temporal positions."""
        return apply_rope(q, k, max_period=self.max_period, offset=offset)
