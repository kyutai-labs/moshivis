import json

import mlx
import mlx.core as mx
import mlx.nn

from ..mlx_vlm.models.pixtral.vision import PixtralVisionModel, VisionConfig


class PixtralWrapper(mlx.nn.Module):
    """Pixtral encoder returning penultimate features"""

    def __init__(self) -> None:
        super().__init__()
        # if not os.path.exists("pixtral-12b-8bit.safetensors"):
        #     self.load_pixtral_weights()
        with open("pixtral-12b-8bit.config", "r") as f:
            vision_config = VisionConfig(**json.load(f))
        self.model = PixtralVisionModel(vision_config)
        weights = mx.load("pixtral-12b-8bit.safetensors")
        mlx.nn.quantize(self.model, bits=8, group_size=64)
        self.model.load_weights(list(weights.items()))

    # def load_pixtral_weights(self):
    #     from mlx_vlm.models.pixtral import Model
    #     from dataclasses import asdict
    #     pixtral = Model.from_pretrained("mlx-community/pixtral-12b-8bit")
    #     self.model = pixtral.vision_tower.vision_model
    #     self.model.save_weights("pixtral-12b-8bit.safetensors")
    #     with open("pixtral-12b-8bit.config", "w") as f:
    #         json.dump(asdict(pixtral.config.vision_config), f)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward to the last hidden states

        We expect the input to be an RGB uint8 image (H, W, C)
        """
        means = mx.array([0.48145466, 0.4578275, 0.40821073])
        std = mx.array([0.26862954, 0.26130258, 0.27577711])
        x = ((x / 255.0) - means[None, None, :]) / std[None, None, :]

        x = [x[None, :, :, :].astype(mx.float32)]

        assert isinstance(x, list), "Pixtral expects a list of tensors."

        return self.model(x, output_hidden_states=False)[0]

    def warmup(self) -> None:
        eval(self(mx.zeros((224, 224, 3), dtype=mx.uint8)))
