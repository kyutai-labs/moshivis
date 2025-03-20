"""Knowledge base of useful fixed values used across the codebase"""

from enum import Enum, unique
from typing import List, Optional, Tuple


@unique
class ImageEncoder(Enum):
    """Encapsulate every image encoder"""

    CLIP_VIT = "clip_vit"
    CLIP_VIT_LARGE = "clip_vit_large"
    MOBILECLIP_S1 = "mobileclip_s1"
    MOBILECLIP_S2 = "mobileclip_s2"
    # original siglip
    SIGLIP = "siglip"
    # Siglip Pretrained: Phase 2 of PaliGemma 1 training
    SIGLIP_GEMMA1_224 = "siglip_gemma1_224"
    # Same but for PaliGemma 2
    SIGLIP_GEMMA2_224 = "siglip_gemma2_224"
    SIGLIP_GEMMA2_448 = "siglip_gemma2_448"
    SIGLIP_GEMMA2_896 = "siglip_gemma2_896"
    PIXTRAL = "pixtral"

    @property
    def out_dims(self) -> int:
        """Return the number of dimensions output by the given image encoder"""
        if self in {
            ImageEncoder.PIXTRAL,
            ImageEncoder.CLIP_VIT_LARGE,
            ImageEncoder.MOBILECLIP_S1,
        }:
            return 1024
        if self == ImageEncoder.CLIP_VIT:
            return 512
        if self == ImageEncoder.MOBILECLIP_S2:
            return 1280
        if self in {
            ImageEncoder.SIGLIP,
            ImageEncoder.SIGLIP_GEMMA1_224,
            ImageEncoder.SIGLIP_GEMMA2_224,
            ImageEncoder.SIGLIP_GEMMA2_448,
            ImageEncoder.SIGLIP_GEMMA2_896,
        }:
            return 1152
        raise NotImplementedError("Unknown image encoder", self.name)

    def to_rust(self) -> str:
        """Return the corresponding `ImageEncoder` enum name in the rust codebase"""
        if self == ImageEncoder.PIXTRAL:
            return "Pixtral"
        if self in {
            ImageEncoder.SIGLIP,
            ImageEncoder.SIGLIP_GEMMA1_224,
            ImageEncoder.SIGLIP_GEMMA2_224,
        }:
            return "Siglip224"
        if self == ImageEncoder.SIGLIP_GEMMA2_448:
            return "Siglip448"
        if self == ImageEncoder.SIGLIP_GEMMA2_896:
            return "Siglip896"
        if self == ImageEncoder.MOBILECLIP_S1:
            return "MobileclipS1"
        if self == ImageEncoder.MOBILECLIP_S2:
            return "MobileclipS2"
        raise ValueError(
            f"Image encoder {self.name} is not implemented in the Rust codebase"
        )
