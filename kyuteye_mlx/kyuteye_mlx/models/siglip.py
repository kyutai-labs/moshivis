import json
import os

import mlx
import mlx.core as mx
import mlx.nn

from ..mlx_vlm.models.siglip.vision import VisionConfig, VisionModel


class SiglipWrapper(mlx.nn.Module):
    """Siglip encoder returning penultimate features"""

    def __init__(self) -> None:
        super().__init__()
        with open("siglip448.config", "r") as f:
            vision_config = VisionConfig(**json.load(f))
            print("loaded!")
        self.model = VisionModel(vision_config).vision_model

    def __call__(self, x: mx.array) -> mx.array:
        """Forward to the last hidden states

        We expect the input to be an RGB uint8 image (H, W, C)
        """
        means = mx.array([0.5] * 3)
        std = mx.array([0.5] * 3)
        x = ((x / 255.0) - means[None, None, :]) / std[None, None, :]

        x = x[None, :, :, :].astype(mx.float32)

        out = self.model(x, output_hidden_states=False)[0]
        return out[None]

    def warmup(self) -> None:
        eval(self(mx.zeros((224, 224, 3), dtype=mx.uint8)))
