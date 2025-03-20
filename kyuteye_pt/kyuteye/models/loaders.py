# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Load moshi-vis neccessary components."""

from typing import Any, Dict, Optional, Tuple

import torch

from kyuteye.config.kyuteye_config import KyuteyeConfig
from kyuteye.models.image_projection import ImageProjection
from kyuteye.models.moshivis import MoshiVisGen


def get_moshi_vis(
    kyuteye_config: KyuteyeConfig,
    moshi_weight: Optional[str] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[MoshiVisGen, ImageProjection]:
    """Return main Moshi model"""
    image_proj_state: Dict[str, torch.Tensor] = {}
    model_state: Dict[str, torch.Tensor] = {}

    if moshi_weight is not None:
        from safetensors.torch import load_file

        for key, v in load_file(moshi_weight, device=device).items():  # type: ignore
            if key.startswith("image_prefix."):
                image_proj_state[key[13:]] = v
            else:
                model_state[key] = v

    moshi_vis = MoshiVisGen.from_config(
        kyuteye_config, model_state, device, dtype, **(gen_kwargs or {})
    )
    image_embedder = ImageProjection.from_config(
        kyuteye_config, moshi_vis.model_dim, image_proj_state, device
    )

    return moshi_vis.to(dtype), image_embedder.to(dtype)
