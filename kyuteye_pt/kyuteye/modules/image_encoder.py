# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pretrained image encoders from timm and/or transformers"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
from kyuteye.config.enums import ImageEncoder
from kyuteye.modules.image_transforms import (
    Normalize,
    PixtralNormalize,
    SigLIPNormalize,
)
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    LlavaForConditionalGeneration,
    SiglipVisionConfig,
    SiglipVisionModel,
)


class TrimmedFlexiViTWrapper(torch.nn.Module):
    """ViT module without the classification tower"""

    def __init__(
        self, model: torch.nn.Module, interpolate_pos_encoding: bool = False
    ) -> None:
        super().__init__()
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get last hidden states"""
        return self.model(
            x, interpolate_pos_encoding=self.interpolate_pos_encoding
        ).last_hidden_state


def load_paligemma_vision_encoder(
    name: str, device: torch.device | str = "cpu", pretrained: bool = False
) -> torch.nn.Module:
    """Load Paligemma encoder from the shared HuggingFace cache"""
    if pretrained:
        model = AutoModelForImageTextToText.from_pretrained(
            name
        ).vision_tower.vision_model
    else:
        image_size = int(name.rsplit("-", 1)[-1])
        model = SiglipVisionModel(
            SiglipVisionConfig(
                **{
                    "hidden_size": 1152,
                    "image_size": image_size,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": (image_size // 14) ** 2,
                    "num_positions": 256,
                    "patch_size": 14,
                    "projection_dim": 2304,
                    "torch_dtype": "bfloat16",
                    "vision_use_head": False,
                }
            )
        ).vision_model
    return TrimmedFlexiViTWrapper(model.to(device), interpolate_pos_encoding=True)


@dataclass
class PixtralOutput:
    """Pixtral Output"""

    out: torch.Tensor
    mask: torch.Tensor


class PixtralWrapper(torch.nn.Module):
    """Pixtral encoder returning penultimate features"""

    def __init__(
        self, device: Optional[torch.device | str] = None, pretrained: bool = False
    ) -> None:
        super().__init__()
        if pretrained:
            self.model = AutoModelForImageTextToText.from_pretrained(
                "mistral-community/pixtral-12b"
            ).vision_tower.to(device)
        else:
            config = AutoConfig.from_pretrained("mistral-community/pixtral-12b")
            self.model = LlavaForConditionalGeneration(config).vision_tower.to(device)
        self.patch_size = torch.prod(
            torch.tensor(self.model.patch_conv.weight.shape[-2:])  # type: ignore
        )

    def __get_num_output_tokens__(self, x: List[torch.Tensor]) -> List[int]:
        """Get number of tokens for each image in the list"""

        return [
            (torch.prod(torch.tensor(img[1].shape[-2:])) // self.patch_size).item()
            for img in x
        ]

    @staticmethod
    def split_and_pad_output(
        x: torch.Tensor, split_points: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split the output of the model into a list of tensors corresponding
        to the input images and create a mask tensor"""
        splits = list(torch.split(x, split_points, dim=1))
        assert sum(split.shape[1] for split in splits) == x.shape[1]

        # Pad the splits to have the same size along the sequence dimension
        max_len = max(split.shape[1] for split in splits)
        padded_splits = []
        mask = torch.zeros((len(splits), max_len), dtype=torch.bool, device=x.device)

        for i, split in enumerate(splits):
            pad_len = max_len - split.shape[1]
            # Right padding of the second to last (i.e., the sequence) dimension
            padded_split = torch.nn.functional.pad(split, (0, 0, 0, pad_len))
            padded_splits.append(padded_split)
            mask[i, : split.shape[1]] = 1

        return torch.cat(padded_splits, dim=0), mask

    def forward(self, x: List[torch.Tensor] | torch.Tensor) -> PixtralOutput:
        """Forward to the last hidden states"""
        if isinstance(x, torch.Tensor):
            x = list(x)
        assert isinstance(x, list), "Pixtral expects a list of tensors."
        split_points = self.__get_num_output_tokens__(x)

        # Pixtral expects list of images
        model_out = self.model(x).last_hidden_state
        split_output, mask = self.split_and_pad_output(model_out, split_points)

        return PixtralOutput(out=split_output, mask=mask)


def get_img_normalize(
    img_encoder: ImageEncoder,
) -> Callable[..., Normalize | PixtralNormalize]:
    """Return input normalization function"""
    if img_encoder == ImageEncoder.PIXTRAL:
        return PixtralNormalize
    if img_encoder in {
        ImageEncoder.SIGLIP_GEMMA2_224,
        ImageEncoder.SIGLIP_GEMMA2_448,
        ImageEncoder.SIGLIP_GEMMA2_896,
    }:
        return SigLIPNormalize
    raise NotImplementedError("Unknown image encoder", img_encoder.name)


def load_image_encoder(
    img_encoder: ImageEncoder,
    device: torch.device | str = "cpu",
    pretrained: bool = False,
) -> torch.nn.Module:
    """Load Image encoder as a torch module"""

    if img_encoder == ImageEncoder.PIXTRAL:
        return PixtralWrapper(device=device, pretrained=pretrained)

    if img_encoder in {
        ImageEncoder.SIGLIP_GEMMA2_224,
        ImageEncoder.SIGLIP_GEMMA2_448,
        ImageEncoder.SIGLIP_GEMMA2_896,
    }:
        size = int(img_encoder.name.rsplit("_", 1)[-1])
        return load_paligemma_vision_encoder(
            name=f"google/paligemma2-3b-pt-{size}",
            device=device,
            pretrained=pretrained,
        )

    raise NotImplementedError(f"image encoder {img_encoder.name} not recognized")
