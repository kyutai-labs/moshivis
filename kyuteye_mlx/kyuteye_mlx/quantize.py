# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "mlx==0.18.1",
#     "safetensors >= 0.4.0, < 0.5",
# ]
# ///
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, Optional

import fire
import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from kyuteye_mlx import models
from kyuteye_mlx.utils.loading import (
    remove_shared_weights,
    repeat_shared_weights,
    split_embedder_weights,
)


def quantize(
    model_file: str,
    out_file: Optional[str] = None,
    img_embed: Literal["siglip", "pixtral"] = "siglip",
    bits: int = 8,
    group_size: int = 64,
    quantize_embedder: bool = False,
) -> None:
    if out_file is None:
        out_file = model_file.replace(".safetensors", f".q{bits}.safetensors")
    weights = mx.load(model_file)
    if img_embed == "siglip":
        lm_config = models.config_siglip()
        if quantize_embedder:
            from kyuteye_mlx.kyuteye_mlx.models.siglip import SiglipWrapper

            embedder = SiglipWrapper()
        else:
            embedder = None
    else:
        lm_config = models.config_pixtral()
        if quantize_embedder:
            from kyuteye_mlx.kyuteye_mlx.models.pixtral import PixtralWrapper

            embedder = PixtralWrapper()
        else:
            embedder = None

    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    weights = repeat_shared_weights(weights, lm_config.transformer.num_layers)
    weights, embed_weights = split_embedder_weights(weights)

    model.load_weights(list(weights.items()), strict=True)

    print("weights loaded")

    nn.quantize(model, bits=bits, group_size=group_size)
    if quantize_embedder:
        assert embedder is not None
        nn.quantize(embedder, bits=bits, group_size=group_size)
        embed_weights = dict(tree_flatten(embedder.parameters()))

    print(f"saving the quantized q{bits} weights in {out_file}")
    new_weights = dict(tree_flatten(model.parameters()))
    new_weights = remove_shared_weights(new_weights, lm_config.transformer.num_layers)
    # Re-adding prefix for consistency with other scripts.
    new_weights.update({"img_embedder." + k: v for k, v in embed_weights.items()})
    mx.save_safetensors(out_file, new_weights)


def main():
    fire.Fire(quantize)


if __name__ == "__name__":
    main()
