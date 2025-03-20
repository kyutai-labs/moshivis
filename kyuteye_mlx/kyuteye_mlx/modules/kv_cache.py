# Most of the code below comes from:
# https://github.com/ml-explore/mlx-examples/blob/6c2369e4b97f49fb5906ec46033497b39931b25d/llms/mlx_lm/models/base.py#L1
# Copyright © 2023-2024 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Any, Self

import mlx.core as mx
import numpy as np


class XACache:
    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.is_set: bool = False

    def set(self, k: mx.array, v: mx.array) -> None:
        # See https://github.com/ml-explore/mlx/issues/1918 for an
        # explanation of this hack.
        self.keys = mx.array(np.array(k.astype(mx.float32))).astype(mx.bfloat16)
        self.values = mx.array(np.array(v.astype(mx.float32))).astype(mx.bfloat16)
        self.is_set = True

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self.is_set = False

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        return self.keys, self.values


class KVCache:
    def __init__(self, head_dim: int | tuple[int, int], n_kv_heads: int) -> None:
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        assert self.values is not None
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def reset(self) -> None:
        self.offset = 0

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        return self.keys, self.values


class RotatingKVCache:
    def __init__(
        self,
        head_dim: int | tuple[int, int],
        n_kv_heads: int,
        max_size: int,
        keep: int = 0,
        step: int = 256,
    ) -> None:
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keep = keep
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        self.max_size = max_size
        self.step = step
        self._idx = 0

    def _trim(self, trim_size: int, v: mx.array, append: mx.array | None = None) -> mx.array:
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        prev = self.offset
        B, _, S = keys.shape[:3]

        # Prefill mode
        if S > 1:
            if self.keys is None:
                self.keys = keys
                self.values = values
            else:
                # The largest size is self.max_size + S - 1 to ensure
                # every token gets at least self.max_size context
                trim_size = self.keys.shape[2] - self.max_size + 1
                self.keys = self._trim(trim_size, self.keys, keys)
                self.values = self._trim(trim_size, self.values, values)
            self.offset += S
            self._idx = self.keys.shape[2]
            return self.keys, self.values

        # Generation mode
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        if self.keys is None or (prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size):
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, self.n_kv_heads, new_size, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, new_size, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign
        self.keys[..., self._idx : self._idx + 1, :] = keys
        assert self.values is not None
        self.values[..., self._idx : self._idx + 1, :] = values
        self.offset += 1
        self._idx += 1

        # If the buffer is not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def reset(self) -> None:
        self.offset = 0
        self._idx = 0

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        return self.keys, self.values


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params: dict[str, Any]):
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})
