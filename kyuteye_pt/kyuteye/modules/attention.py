# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MultiHead Self-Attention module with optional KV Caching"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from einops import rearrange
from kyuteye.modules.streaming_utils import StreamingModule
from kyuteye.modules.utils import RotaryEmbedding, multi_linear


@dataclass
class KVCache:
    """Efficient streaming KVCache to avoid allocating new memory too many times.

    :param batch_size: Batch size.
    :param num_heads: Number of heads in the attention.
    :param dim_per_head: Dimension per head.
    :param context: Context size for the attention, if None, will grow exponentially,
        otherwise will use a fixed allocation with a bit of overhead.
    :param growth: Growth factor for the exponential growth, fraction of overhead
        when context is not None.
    :param initial_size: Initial size of the cache, used only when context is None.
    :param device: Device on which to initialize the cache.
    :param dtype: dtype to use for the cache.
    :param cache: Initial cache, if provided.
    :param current_end: Current end of the cache, used only when cache is provided.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        context: Optional[int] = None,
        growth: float = 1.2,
        initial_size: int = 100,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        cache: Optional[torch.Tensor] = None,
        current_end: int = 0,
    ) -> None:
        if cache is None:
            assert current_end == 0

        assert growth > 1
        self.growth = growth

        if context is not None:
            initial_size = 1 + int(growth * context)

        self.capacity = initial_size
        self.context = context
        self.current_end = current_end

        if cache is None:
            self._cache = torch.full(
                (2, batch_size, initial_size, num_heads, dim_per_head),
                float("NaN"),
                device=device,
                dtype=dtype,
            )
        else:
            self._cache = cache

    def clone(self) -> "KVCache":
        """Return a separate memory copy of the KV cache"""
        return KVCache(
            self._cache.shape[1],
            self._cache.shape[3],
            self._cache.shape[4],
            self.context,
            self.growth,
            self.capacity,
            self._cache.device,
            self._cache.dtype,
            self._cache.clone(),
            self.current_end,
        )

    @property
    def current_start(self) -> int:
        """Current start of the KV cache (0 if no context size)"""
        return 0 if self.context is None else max(self.current_end - self.context, 0)

    def __maybe_increase_capacity__(self, required_capacity: int) -> None:
        """If needed, increase capacity to the `required_capacity`
        using exponential growth strategy"""
        if required_capacity > self.capacity:
            if self.context is None:
                # We take an exponential growth approach.
                new_capacity = self.capacity
                while required_capacity > new_capacity:
                    new_capacity = int(math.ceil(new_capacity * self.growth))
                new_shape = list(self._cache.shape)
                new_shape[2] = new_capacity
                new_cache = torch.full(
                    tuple(new_shape),
                    float("NaN"),
                    device=self._cache.device,
                    dtype=self._cache.dtype,
                )
                new_cache[:, :, : self.current_end] = self._cache[
                    :, :, : self.current_end
                ]
                self._cache = new_cache
                self.capacity = new_capacity
            else:
                # With context, we just have to roll the predict to the left and
                # use the new space on the right.
                assert self.current_start > 0
                self._cache[:] = self._cache.roll(-self.current_start, dims=2)
                self.current_end -= self.current_start

    def complete(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add keys `k` and values `v` to the current cache and returns
        cache up to the context size"""
        assert k.shape[1] == v.shape[1]
        self.__maybe_increase_capacity__(self.current_end + k.shape[1])

        assert self.current_end + k.shape[1] <= self.capacity, (
            self.current_end,
            k.shape[1],
            self.capacity,
        )
        self._cache[0, :, self.current_end : self.current_end + k.shape[1]] = k
        self._cache[1, :, self.current_end : self.current_end + v.shape[1]] = v
        self.current_end += k.shape[1]
        valid = self._cache[:, :, self.current_start : self.current_end]
        return valid[0], valid[1]


class MultiheadAttention(StreamingModule):
    """Similar to `nn.MultiheadAttention` but with support for causal evaluation.

    Args:
        :param embed_dim: Dimension to project to.
        :param num_heads: Number of heads.
        :param causal: If true, applies causal mask automatically.
        :param context: Number of time steps the attention can access to.
            When causal, can access `context` time steps into the past, and when non causal,
            can access `context // 2` steps in the past, and the same in the future.
        :param rope: Rope embedding to use. If None, no rope embedding is applied
        :param cross_attention: Should be true when used as a cross attention.
            Cannot be used with `causal` or `rope` (as it wouldn't make sens to
            interpret the time steps in the keys relative to those in the queries).
        :param use_kv_cache: If True, enables a KV cache with context size `context`.
        :param weights_per_step: use different weights per depformer step. If non zero,
            should correspond to the number of possible time steps.
        :param device: Device on which to initialize the module.
        :param dtype: dtype to use.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: Optional[int] = None,
        rope: Optional[RotaryEmbedding] = None,
        cross_attention: bool = False,
        use_kv_cache: bool = False,
        weights_per_step: int = 0,
        xa_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.causal = causal
        self.context = context
        self.rope = rope
        self.cross_attention = cross_attention
        self.num_heads = num_heads
        self.use_kv_cache = use_kv_cache
        self.weights_per_step = weights_per_step
        mult = max(1, weights_per_step)

        if cross_attention:
            assert not causal, "Cannot set causal mask when `cross attention` is True."
            assert (
                not context
            ), "Cannot set context size when `cross attention` is True."

        # if cross-attention source have != num_dims than the speech tokens,
        # we need to separate the KV and Q embeddings
        if cross_attention and xa_dim is not None and xa_dim != embed_dim:
            in_proj_q = torch.nn.Linear(
                embed_dim, mult * embed_dim, bias=False, **factory_kwargs
            )
            in_proj_kv = torch.nn.Linear(
                xa_dim, mult * 2 * embed_dim, bias=False, **factory_kwargs
            )
            self.in_proj_weight_q = in_proj_q.weight
            self.in_proj_bias_q = in_proj_q.bias
            self.in_proj_weight_kv = in_proj_kv.weight
            self.in_proj_bias_kv = in_proj_kv.bias
            self.in_proj_weight = None
            self.in_proj_bias = None
        else:
            in_proj = torch.nn.Linear(
                embed_dim, mult * 3 * embed_dim, bias=False, **factory_kwargs
            )
            self.in_proj_weight = in_proj.weight
            self.in_proj_bias = in_proj.bias
        self.out_proj = torch.nn.Linear(
            embed_dim, mult * embed_dim, bias=False, **factory_kwargs
        )

    def _complete_kv(
        self, k: torch.Tensor, v: torch.Tensor, initial_kv_cache_size: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add key/values to the KV cache"""
        # With cross attention we assume all keys and values
        # are already available, and streaming is with respect
        # to the queries only.
        if self._is_streaming and not self.cross_attention:
            if "kv_cache" not in self._streaming_state:
                self._streaming_state["kv_cache"] = KVCache(  # type: ignore
                    k.shape[0],
                    k.shape[2],
                    k.shape[3],
                    self.context,
                    initial_size=self.weights_per_step or initial_kv_cache_size,
                    device=k.device,
                    dtype=k.dtype,
                )
                self.streaming_offset = torch.zeros(1)  # type: ignore
            kv_cache: KVCache = self._streaming_state["kv_cache"]  # type: ignore
            self.streaming_offset += k.shape[1]
            return kv_cache.complete(k, v)

        return k, v

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[Tuple[torch.Tensor, torch.Tensor] | torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """If self.cross attention is False, we only expects the first input. Otherwise,
        when using cross attention, we need to explicitly give the source for the
        respective query/key/value embeddings"""
        # Get current streaming offset before it gets potentially modified by the KV cache update
        current_streaming_offset = self.streaming_offset

        if self.cross_attention:
            assert key is not None, "Missing inputs in cross attention"
            if isinstance(key, torch.Tensor):
                value = value or key
                assert value is not None, "Missing inputs in cross attention"
            # Case 1: Inputs x and ca_src have the same number of dimension
            # We have a single big weight for the QKV projections
            if self.in_proj_weight is not None:
                q = torch.nn.functional.linear(  # pylint: disable=not-callable
                    query, self.in_proj_weight[: self.embed_dim]
                )
                if isinstance(key, torch.Tensor):
                    k = torch.nn.functional.linear(  # pylint: disable=not-callable
                        key, self.in_proj_weight[self.embed_dim : 2 * self.embed_dim]
                    )
                    v = torch.nn.functional.linear(  # pylint: disable=not-callable
                        value, self.in_proj_weight[2 * self.embed_dim :]  # type: ignore
                    )
                else:
                    k, v = key
            # Case 2: Inputs x and ca_src have different number of dimension
            # We have to separate the Q and KV proj
            else:
                q = torch.nn.functional.linear(  # pylint: disable=not-callable
                    query, self.in_proj_weight_q[: self.embed_dim]
                )
                if isinstance(key, torch.Tensor):
                    k = torch.nn.functional.linear(  # pylint: disable=not-callable
                        key, self.in_proj_weight_kv[: self.embed_dim]
                    )
                    v = torch.nn.functional.linear(  # pylint: disable=not-callable
                        value, self.in_proj_weight_kv[self.embed_dim :]  # type: ignore
                    )
                else:
                    k, v = key
            q, k, v = [
                rearrange(x, "b t (h d) -> b t h d", h=self.num_heads)
                for x in [q, k, v]
            ]
        else:
            assert self.in_proj_weight is not None
            if self.weights_per_step > 0:
                projected = multi_linear(
                    self.weights_per_step,
                    self.in_proj_weight,
                    query,
                    offset=current_streaming_offset,
                )
            else:
                projected = torch.nn.functional.linear(  # pylint: disable=not-callable
                    query, self.in_proj_weight
                )
            packed = rearrange(
                projected, "b t (p h d) -> b t p h d", p=3, h=self.num_heads
            )
            q, k, v = torch.unbind(packed, dim=2)

        if self.rope:
            q, k = self.rope(q, k, offset=current_streaming_offset)

        k, v = self._complete_kv(k, v)

        # Attention
        q, k, v = [x.transpose(1, 2) for x in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, is_causal=False, attn_mask=attention_mask
        )
        x = x.transpose(1, 2)

        # output projection
        x = rearrange(x, "b t h d -> b t (h d)")
        if self.weights_per_step > 0:
            x = multi_linear(
                self.weights_per_step,
                self.out_proj.weight,
                x,
                offset=current_streaming_offset,
            )
        else:
            x = self.out_proj(x)
        return x
