# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Image transforms"""

from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms.v2 as T
from PIL.Image import Image

try:
    from transformers.models.pixtral.image_processing_pixtral import (
        PixtralImageProcessor,
    )
except ImportError:
    print("Cannot find Pixtral encoder, you need to upgrade to transformers >= 0.46")


def get_minimal_transforms(
    img_size: Union[Tuple[int], int] = 224,
    interpolation: Literal[
        "bicubic", "bilinear", "nearest", "nearest_exact"
    ] = "bicubic",
    to_tensor: bool = False,
    keep_aspect_ratio: bool = False,
    max_img_size: Optional[int] = None,
) -> T.Transform:
    """Minimal transform from converting a PIL image to a Tensor.
    This is used as default in most of our datasets when the img_transforms
    is not provided. If keep_aspect_ratio is False, it resizes the image to (img_size, img_size),
    without respecting aspect ratio. Otherwise, it only resizes the smaller side to img_size,

    :param img_size: Target image (square) size
    :param interpolation: Resizing interpolation
    :param to_tensor: Whether to also convert to a Tensor type (or leave it to a later transform)
    :param keep_aspect_ratio: Whether to keep the aspect ratio
    :param max_img_size: Maximum size an image can be along the longer side
    """
    if not isinstance(img_size, tuple):
        img_size = img_size if keep_aspect_ratio else (img_size, img_size)  # type: ignore
    return T.Compose(
        [
            T.Resize(
                img_size,
                interpolation=getattr(T.InterpolationMode, interpolation.upper()),
                max_size=max_img_size if keep_aspect_ratio else None,
            ),
            T.PILToTensor() if to_tensor else T.Identity(),
        ]
    )


class Normalize:
    """Normalization types for the different image encoders. These will be
    set in image_projection.py"""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.std = std
        self.mean = mean
        self.transform = T.Compose(
            [
                T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
                T.Normalize(
                    mean,
                    std,
                ),
            ]
        )

    def __call__(
        self, image: Union[Image, List[Image]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(image, list):
            return [self.transform(img) for img in image]
        return self.transform(image)

    def to_pil_transform(self, mode: str = "RGB") -> T.Transform:
        """Returns the function that inverts this normalization"""
        return T.Compose(
            [
                T.Normalize([0 for _ in self.mean], [1 / (x + 1e-6) for x in self.std]),
                T.Normalize([-x for x in self.mean], [1 for _ in self.std]),
                T.ToPILImage(mode=mode),
            ]
        )


class UnitNormalize(Normalize):
    """Normalization for SigLIP encoder"""

    def __init__(self) -> None:
        super().__init__(
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )


class CLIPNormalize(Normalize):
    """Normalization for SigLIP encoder"""

    def __init__(self) -> None:
        super().__init__(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )


class SigLIPNormalize(Normalize):
    """Normalization for SigLIP encoder"""

    def __init__(self) -> None:
        super().__init__(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


class PixtralNormalize:
    """Image preprocessing for Pixtral
    https://github.com/huggingface/transformers/blob/fc1ae7f30f1d16c7652c28dd8d91c5d8a8ed2f15/src/transformers/models/pixtral/image_processing_pixtral.py#L369
    """

    def __init__(self) -> None:
        self.preprocess = PixtralImageProcessor.from_pretrained(
            "mistral-community/pixtral-12b"
        )

    def __call__(
        self, image: Union[Image, List[Image], torch.Tensor]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Image input to pixtral can be an individual image or a list of
        # images or a list of lists of images (multiple images in a single text sequence)

        # Case 1: Single image
        if isinstance(image, Image) or isinstance(image, torch.Tensor):
            # Pixtral converts a single image into [[image]] — a list of lists of images
            return self.preprocess(image, return_tensors="pt")["pixel_values"][0][0]
        # Case2: List of images
        if isinstance(image, list) and isinstance(image[0], Image):
            return [
                self.preprocess(subimg, return_tensors="pt")["pixel_values"][0][0]
                for subimg in image
            ]

        # Case 3: List of lists of images
        # This is not currently supported by us
        raise ValueError(
            "PixtralNormalize does not support list of lists of images currently."
        )
