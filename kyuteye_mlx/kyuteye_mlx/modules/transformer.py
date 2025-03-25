# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import mlx.core as mx
import mlx.nn as nn
from jaxtyping import BFloat16

from .config import TransformerConfig
from .cross_attention import GatedCrossAttention
from .kv_cache import KVCache, RotatingKVCache, XACache


class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()

        num_kv = cfg.num_heads // cfg.kv_repeat
        out_dim = cfg.d_model + 2 * num_kv * cfg.d_model // cfg.num_heads
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, out_dim, bias=cfg.bias_attn)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias_attn)
        self.scale = cfg.head_dim ** (-0.5)
        self.rope = None
        if cfg.positional_embedding == "rope":
            self.rope = nn.RoPE(cfg.head_dim, traditional=True, base=cfg.max_period)

    def __call__(
        self,
        xs: BFloat16[mx.array, "batch time features"],
        cache: KVCache | RotatingKVCache,
        mask: BFloat16[mx.array, "batch kv_size features"] | None = None,
    ) -> BFloat16[mx.array, "batch time features"]:
        assert self.cfg.kv_repeat == 1, "only kv_repeat==1 is supported"

        b, t, hd = xs.shape
        qkv = self.in_proj(xs).reshape(b, t, 3, self.cfg.num_heads, self.cfg.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        if self.rope is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)

        k, v = cache.update_and_fetch(k, v)
        k_len = k.shape[2]
        k_target_len = t + min(self.cfg.context, k_len - t)
        if k_target_len < k_len:
            k = k[:, :, k_len - k_target_len :]
            v = v[:, :, k_len - k_target_len :]

        xs = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        xs = xs.transpose(0, 2, 1, 3).reshape(b, t, hd)
        xs = self.out_proj(xs)
        return xs


class MlpGating(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()

        hidden = 2 * cfg.dim_feedforward // 3
        if cfg.dim_feedforward == 4 * cfg.d_model:
            hidden = 11 * cfg.d_model // 4

        self.linear_in = nn.Linear(cfg.d_model, 2 * hidden, bias=cfg.bias_ff)
        self.linear_out = nn.Linear(hidden, cfg.d_model, bias=cfg.bias_ff)

    def __call__(
        self, xs: BFloat16[mx.array, "batch time features"]
    ) -> BFloat16[mx.array, "batch time features"]:
        xs = self.linear_in(xs)
        b, t, _ = xs.shape
        xs = xs.reshape(b, t, 2, -1)
        return self.linear_out(nn.silu(xs[:, :, 0]) * xs[:, :, 1])


class MlpNoGating(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()

        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward, bias=cfg.bias_ff)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model, bias=cfg.bias_ff)

    def __call__(self, xs: mx.array) -> mx.array:
        return self.linear2(nn.gelu_approx(self.linear1(xs)))


class TransformerLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        assert not cfg.use_conv_block, "conv-block is not supported"
        if cfg.gating:
            self.gating = MlpGating(cfg)
        else:
            self.gating = MlpNoGating(cfg)

        if cfg.norm == "layer_norm":
            self.norm1 = nn.LayerNorm(cfg.d_model, 1e-5)
            self.norm2 = nn.LayerNorm(cfg.d_model, 1e-5)
        elif cfg.norm == "rms_norm":
            self.norm1 = nn.RMSNorm(cfg.d_model, 1e-8)
            self.norm2 = nn.RMSNorm(cfg.d_model, 1e-8)
        else:
            raise ValueError(f"unsupported norm type {cfg.norm}")

        self.self_attn = Attention(cfg)
        if cfg.cross_attention:
            self.cross_attention = GatedCrossAttention(cfg)
            if cfg.norm == "rms_norm":
                self.norm_cross = nn.RMSNorm(cfg.d_model, 1e-8)
            elif cfg.norm == "layer_norm":
                self.norm_cross = nn.LayerNorm(cfg.d_model, 1e-5)
            else:
                raise ValueError(f"unsupported norm type {cfg.norm}")
        else:
            self.cross_attention = None

    def __call__(
        self,
        xs: BFloat16[mx.array, "batch time {self.cfg.d_model}"],
        cache: KVCache | RotatingKVCache,
        img_embeds: (BFloat16[mx.array, "batch kv_size {self.cfg.d_model}"] | None) = None,
        xa_cache: XACache | None = None,
    ) -> BFloat16[mx.array, "batch time {self.cfg.d_model}"]:
        xs = xs + self.self_attn(self.norm1(xs), cache=cache)
        if self.cross_attention is not None:
            if xa_cache is None:
                raise ValueError("xa_cache should never be None when using cross attention.")
            xs = xs + self.cross_attention(self.norm_cross(xs), xa_cache=xa_cache, kv=img_embeds)
            pass
        xs = xs + self.gating(self.norm2(xs))
        return xs


class ImagePrefix(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm_xa = nn.RMSNorm(cfg.d_model, 1e-8)
        self.proj_xa = nn.Linear(cfg.img_emb_dim, cfg.d_model, bias=True)

    def __call__(
        self, xa: BFloat16[mx.array, "batch kv_size {self.cfg.img_emb_dim}"]
    ) -> BFloat16[mx.array, "batch kv_size {self.cfg.d_model}"]:
        xa = self.proj_xa(xa)
        xa = self.norm_xa(xa)
        return xa


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig, with_img_prefix: bool = False) -> None:
        super().__init__()

        self.cfg = cfg
        self.layers = [TransformerLayer(cfg=cfg) for _ in range(cfg.num_layers)]
        if with_img_prefix:
            self.image_prefix = ImagePrefix(cfg)

    def __call__(
        self,
        xs: BFloat16[mx.array, "batch time {self.cfg.d_model}"],
        cache: list[KVCache] | list[RotatingKVCache],
        img_embeds: (BFloat16[mx.array, "batch kv_size {self.cfg.img_emb_dim}"] | None) = None,
        xa_cache: XACache | None = None,
    ) -> BFloat16[mx.array, "batch time {self.cfg.d_model}"]:
        if img_embeds is not None and xa_cache is not None and not xa_cache.is_set:
            img_embeds = self.image_prefix(img_embeds)
        else:
            img_embeds = None
        for layer, c in zip(self.layers, cache):
            xs = layer(xs, cache=c, xa_cache=xa_cache, img_embeds=img_embeds)
        return xs

    def make_cache(self) -> list[KVCache]:
        num_kv_heads = self.cfg.num_heads // self.cfg.kv_repeat
        return [KVCache(head_dim=self.cfg.head_dim, n_kv_heads=num_kv_heads) for _ in self.layers]

    def make_rot_cache(self) -> list[RotatingKVCache]:
        num_kv_heads = self.cfg.num_heads // self.cfg.kv_repeat
        return [
            RotatingKVCache(
                head_dim=self.cfg.head_dim,
                n_kv_heads=num_kv_heads,
                max_size=self.cfg.max_seq_len,
            )
            for _ in self.layers
        ]
