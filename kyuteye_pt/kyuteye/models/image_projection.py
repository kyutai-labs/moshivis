"""Image encoders  (CLIP, SigLIP)"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch
from einops import rearrange
from kyuteye.config.enums import ImageEncoder
from kyuteye.modules.image_encoder import (
    PixtralOutput,
    get_img_normalize,
    load_image_encoder,
)
from kyuteye.modules.utils import NormalizationLayer

if TYPE_CHECKING:
    from kyuteye.config.kyuteye_config import KyuteyeConfig


class ImageProjection(torch.nn.Module):
    """
    Takes in a batch of images and returns a batch of embeddings of the
    same dimensions, which are then fed to the LM, either by inserting
    the tokens in the stream, or by using them as source for cross-attention
    moduels (or both).

    :param config: KyuteyeConfig object
    :param lm_model_dim: Output dimension (number of channels) for this module
    """

    def __init__(
        self,
        kyuteye_config: "KyuteyeConfig",
        lm_model_dim: Optional[int],
        load_pretrained_encoder: bool = True,
    ):
        super().__init__()
        self.kyuteye_config = kyuteye_config

        try:
            self.encoder_type = getattr(
                ImageEncoder, self.kyuteye_config.image.encoder_name.upper()
            )
        except AttributeError as e:
            raise NotImplementedError(
                f"Unknown image encoder {self.encoder_type}"
            ) from e

        # Number of output dimensions of the entire module (i.e. including
        # potential projection after the encoder)
        if self.kyuteye_config.xa_dim is not None:
            self.out_dim = self.kyuteye_config.xa_dim
        else:
            assert lm_model_dim is not None
            self.out_dim = lm_model_dim

        # Load the image encoder
        self.enc = load_image_encoder(
            self.encoder_type, pretrained=load_pretrained_encoder
        )
        # Projection layer; there are two possible projection targets
        # A. for the extra tokens
        self.proj_extra = self.init_proj_module(
            self.kyuteye_config.fuse.num_extra_tokens
        )
        # B. for the cross attention
        self.proj_xa = self.init_proj_module(
            self.kyuteye_config.fuse.num_crossattended_tokens
        )

        # Output normalizations
        self.norm_extra = self.init_norm_module(self.kyuteye_config.image.norm_extra)
        self.norm_xa = self.init_norm_module(self.kyuteye_config.image.norm_xa)

    @classmethod
    def from_config(
        cls,
        kyuteye_config: "KyuteyeConfig",
        lm_model_dim: Optional[int] = None,
        moshi_weight: Optional[Dict[str, Any]] = None,
        device: str | torch.device = "cpu",
    ) -> "ImageProjection":
        """Init image projection from config"""
        load_pretrained_encoder = moshi_weight is None or not any(
            x.startswith("enc.") for x in moshi_weight
        )
        image_projection = cls(
            kyuteye_config,
            lm_model_dim,
            load_pretrained_encoder=load_pretrained_encoder,
        )
        if moshi_weight is not None:
            missing_keys, _ = image_projection.load_state_dict(
                moshi_weight, strict=False
            )
            encoder_keys: List[str] = []
            proj_keys: List[str] = []
            for key in missing_keys:
                (encoder_keys if key.startswith("enc.") else proj_keys).append(key)
            print(proj_keys)
            assert len(proj_keys) == 0, "Failed to load image to speech projections"
            print(encoder_keys)
            assert len(encoder_keys) == 0, "Failed to load frozen image encoder"

        return image_projection.to(device)

    def init_proj_module(self, num_tokens: int) -> Optional[torch.nn.Module]:
        """Init the project module for the inserted and/or cross-attended iamge tokens"""
        if num_tokens == 0:
            return None
        if num_tokens == -1:
            if self.encoder_out_dim != self.out_dim:
                return torch.nn.Linear(self.encoder_out_dim, self.out_dim)
            return torch.nn.Identity()
        raise ValueError(f"Found negative number of tokens for projection {num_tokens}")

    @property
    def encoder_out_dim(self) -> int:
        """Number of dimension output by the encoder"""
        return self.encoder_type.out_dims

    @property
    def to_tensor_and_normalize(self) -> Callable:
        """Image normalization function"""
        return get_img_normalize(self.encoder_type)()

    def init_norm_module(self, norm_type: Optional[str]) -> Optional[torch.nn.Module]:
        """Init normalization module"""
        if norm_type is None:
            return None
        return getattr(NormalizationLayer, norm_type.upper()).create_norm_fn(
            self.out_dim
        )

    def forward(self, x: torch.Tensor | List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Image embedding mapping"""
        # Apply image encoder
        encoded, mask = self.encode(x)

        # Apply different projection for extra vs cross attended tokens
        out = {}
        # The mask will be handled by the QP mapper and this module will output the same
        # number of tokens for every sample in the batch, i.e., no padding is needed anymore.
        # => We will not forward the mask in this case.
        if mask is not None:
            out["cross_attention_mask"] = mask

        if self.proj_extra is not None:
            assert mask is None, "proj_extra is not implemented yet for pixtral."
            out["image_embeds"] = self.project_extra(encoded)

        if self.proj_xa is not None:
            out["cross_attention_src"] = self.project_xa(encoded)

        return out

    def encode(
        self, x: torch.Tensor | List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pass through the image encoder backbone and reshape to (batch, seq, dim) Tensors"""
        logits = self.enc(x)
        if self.encoder_type == ImageEncoder.PIXTRAL:
            assert isinstance(logits, PixtralOutput)
            return logits.out, logits.mask

        if logits.ndim == 4:
            logits = rearrange(logits, "b d h w -> b (h w) d")

        if logits.ndim != 3:
            raise ValueError(
                "The image encoder should output a tensor of shape"
                " (B, Seq, D) (ViT) or (B, D, H, W) (CNN)"
            )

        return logits, None

    def project_extra(self, logits: torch.Tensor) -> torch.Tensor:
        """Projection 1: Used for inserted extra tokens"""
        assert self.proj_extra is not None
        logits = self.proj_extra(logits)
        if self.norm_extra is not None:
            return self.norm_extra(logits)
        return logits

    def project_xa(self, logits: torch.Tensor) -> torch.Tensor:
        """Projection 2: Used for cross-attended tokens"""
        assert self.proj_xa is not None
        logits = self.proj_xa(logits)
        if self.norm_xa is not None:
            return self.norm_xa(logits)
        return logits
