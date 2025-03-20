# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Gated cross-attention module"""

from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
from jaxtyping import BFloat16

from .config import TransformerConfig
from .kv_cache import XACache


class SharedModuleType(type):
    """Wrapper to build shared Pytorch modules"""

    _instances = {}  # type: ignore

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(SharedModuleType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class CrossAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias_attn)
        self.scale = cfg.head_dim ** (-0.5)

    def __call__(
        self,
        xs: BFloat16[mx.array, "batch time {self.cfg.d_model}"],
        xa_cache: XACache,
        kv: BFloat16[mx.array, "batch kv_size {self.cfg.d_model}"] | None = None,
    ) -> BFloat16[mx.array, "batch time {self.cfg.d_model}"]:
        k, v = xa_cache.state if xa_cache is not None else (None, None)
        assert kv is not None or (k is not None and v is not None), (
            "Need to provide embeddings or pre-computed keys and values"
        )

        b, t, hd = xs.shape
        q = self.q_proj(xs).reshape(b, self.cfg.num_heads, t, self.cfg.head_dim)

        if k is None and v is None:
            assert kv is not None, "No image embeds given but also no cache found"
            kv = self.kv_proj(kv).reshape(b, -1, 2, self.cfg.num_heads, self.cfg.head_dim)
            k = kv[:, :, 0].transpose(0, 2, 1, 3)
            v = kv[:, :, 1].transpose(0, 2, 1, 3)
            xa_cache.set(k, v)

        xs = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        xs = xs.reshape(b, t, hd)
        xs = self.out_proj(xs)
        return xs


class SharedCrossAttention(CrossAttention, metaclass=SharedModuleType):
    """Shared Cross Attention projection across all layers"""

    pass  # pylint: disable=unnecessary-pass


class XAGate(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        hidden_dims_factor: float = 0.125,
        activation: Literal["tanh", "sigmoid"] = "sigmoid",
        conditional_gating: bool = True,
    ):
        super().__init__()

        assert conditional_gating, "TODO: Support non-conditional gating."
        self.dims = cfg.d_model
        hidden_dims = int(hidden_dims_factor * self.dims)

        self.alpha = nn.Sequential(
            nn.Linear(self.dims, hidden_dims, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dims, self.dims, bias=False),
        )
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "sigmoid":
            # shift left to mimic initialization ~ close to 0
            self.act = lambda x: mx.sigmoid(x - 4)
        else:
            raise NotImplementedError("Unknown activation function", activation)

    def __call__(
        self, xs: BFloat16[mx.array, "batch time {self.dims}"]
    ) -> BFloat16[mx.array, "batch time {self.dims}"]:
        return xs * self.act(self.alpha(xs))


class GatedCrossAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.mha = (SharedCrossAttention if cfg.xa_shared else CrossAttention)(cfg)
        # Output Gating
        self.gate: nn.Module | None = None
        if cfg.xa_gating != "none":
            self.gate = XAGate(cfg)

    def __call__(
        self,
        xs: BFloat16[mx.array, "batch time features"],
        xa_cache: XACache,
        kv: BFloat16[mx.array, "batch kv_size features"] | None = None,
    ) -> BFloat16[mx.array, "batch time features"]:
        if kv is None and not xa_cache.is_set:
            return xs
        xs = self.mha(xs=xs, xa_cache=xa_cache, kv=kv)

        if self.gate is not None:
            xs = self.gate(xs)
        return xs
