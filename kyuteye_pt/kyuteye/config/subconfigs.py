"""Modular configs for configuring a Kyuteye model training run"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from kyuteye.config.enums import ImageEncoder
from kyuteye.utils.dist_utils import print_main


def __is_nonstring_iterable__(arg: Any) -> bool:
    return isinstance(arg, Iterable) and not isinstance(arg, str)


@dataclass(frozen=True)
class LMConfig:
    """Configure the model weights"""

    # if repo is None, then model, mimi_codec and tokenizer are
    # expected to point to local files
    hf_repo: Optional[str] = "kyutai/moshika-vis-pytorch-bf16"
    # MoshiVis model; Note that if the model doesn't contain weights
    # for the frozen image encoder, those will be loaded from audiocraft
    # directly
    model: str = "model.safetensors"
    # Mimi codec
    mimi_codec: str = "tokenizer-e351c8d8-checkpoint125.safetensors"
    # Tokenizer for Helium
    text_tokenizer: str = "tokenizer_spm_32k_3.model"

    @staticmethod
    def help(field_name: str) -> str:
        """Optional; returns the argparse's help message for each field name
        in this subconfig"""
        if field_name == "model_path":
            return (
                "Path to .safetensors containing model weights (vision encoder + LLM)"
            )
        if field_name == "mimi_codec":
            return "Path to .safetensors containing Mimi weights"
        if field_name == "text_tokenizer":
            return "Path to .safetensors containing text tokenizer"
        return ""


@dataclass(frozen=False)
class ImageEncoderConfig:
    """Configure the image encoder for MoshiVis"""

    # Main image backbone to load
    encoder_name: str = "pixtral"
    interpolation: Literal["bicubic", "bilinear", "nearest", "nearest_exact"] = (
        "bicubic"
    )
    # Whether to add dropout after the learned image linear projection
    image_size: int = 256
    # normalization used after projecting the extra tokens
    norm_extra: Optional[Literal["layer_norm", "rms_norm"]] = "rms_norm"
    # normalization used after projecting the xa tokens
    norm_xa: Optional[Literal["layer_norm", "rms_norm"]] = "rms_norm"

    def __post_init__(self) -> None:
        assert self.image_size > 0
        self.encoder_name = self.encoder_name.lower()
        try:
            ImageEncoder(self.encoder_name)
        except ValueError as e:
            raise ValueError(f"Unknown image encoder {self.encoder_name}") from e

        self.aug_strategy = "Pixtral" if self.encoder_name == "pixtral" else "None"

    @staticmethod
    def help(field_name: str) -> str:
        """Optional; returns the argparse's help message for each field name
        in this subconfig"""
        if field_name == "encoder_name":
            return "Name of the pretrained image encoder to use"
        if field_name == "interpolation":
            return "Interpolation algorithm for image resizing"
        if field_name == "image_size":
            return "Input image size to the encoder"
        return ""


@dataclass(frozen=True)
class MoshiConfig:
    """Configure the backbone Moshi"""

    dim: int = 4096
    text_card: int = 32000
    padding_token_id: int = 3
    n_q: int = 16
    dep_q: int = 8
    audio_card: int = 2048
    num_heads: int = 32
    num_layers: int = 32
    hidden_scale: float = 4.125
    causal: bool = True
    context: int = 3000
    max_period: int = 10000
    gating: bool = True
    activation: str = "silu"
    norm: str = "rms_norm_f32"
    positional_embedding: str = "rope"
    depformer: bool = True
    depformer_dim: int = 1024
    depformer_dim_feedforward: int = int(4.125 * 1024)
    depformer_num_heads: int = 16
    depformer_num_layers: int = 6
    depformer_multi_linear: bool = True
    depformer_context: int = 8
    depformer_gating: bool = True
    depformer_activation: str = "silu"
    depformer_pos_emb: str = "none"
    depformer_weights_per_step: bool = True
    delays: Tuple[int, ...] = (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)

    @staticmethod
    def help(field_name: str) -> str:
        """Optional; returns the argparse's help message for each field name
        in this subconfig"""
        if field_name == "model_path":
            return (
                "Path to .safetensors containing model weights (vision encoder + LLM)"
            )
        if field_name == "mimi_codec":
            return "Path to .safetensors containing Mimi weights"
        if field_name == "text_tokenizer":
            return "Path to .safetensors containing text tokenizer"
        return ""


@dataclass(frozen=False)
class FusionConfig:
    """Configures how we integrate the image information in MoshiVis"""

    num_extra_tokens: int = -1
    add_boi_eoi: bool = False
    token_insertion: Optional[Literal["prefix"]] = "prefix"
    num_crossattended_tokens: int = 0
    align_img_and_speech_tokens_dim: bool = True
    xa_dim: Optional[int] = None
    xa_start: Optional[Literal["start", "boi", "eoi"]] = None
    xa_end: Optional[Literal["end", "eoi"] | int] = None
    xa_step: int = 1
    xa_delay: int = 0
    xa_shared: bool = True
    xa_gate_shared: bool = False
    xa_gating: Literal["tanh", "sigmoid", "none"] = "tanh"
    xa_conditional_gating: bool = False
    xa_layers: Tuple[int, ...] = ()
    xa_extended_layer_dims: Optional[int] = None
    xa_extended_layer_embed_dims: int = 1024
    xa_extended_layer_p_norm: int = 2

    @staticmethod
    def help(field_name: str) -> str:
        """Optional; returns the argparse's help message for each field name
        in this subconfig"""
        if field_name == "num_extra_tokens":
            return (
                "Number of extra tokens (containing image information) to insert into"
                "the text+audio tokens. If -1, uses all the image tokens."
            )
        if field_name == "token_insertion":
            return (
                "An option used later in `fuse_utils` to determine where"
                " to insert the extra tokens"
            )
        if field_name == "num_crossattended_tokens":
            return (
                "Number of extra tokens (containing image information) to use as keys/values"
                " source in the cross-attention mechanism. If -1, uses all the image tokens"
            )
        if field_name == "xa_shared":
            return (
                "If True, the linear layers of the cross-attention mechanism"
                "are shared across layers"
            )
        if field_name == "xa_gate_shared":
            return (
                "If True, the gating modules of the cross-attention mechanism"
                "are shared across layers"
            )
        if field_name == "xa_gating":
            return "Type of multiplicative gating at the output of each cross-attention"
        if field_name == "xa_conditional_gating":
            return "Whether to make the gating input dependent"
        if field_name == "xa_layers":
            return (
                "If given, specifies the subset of layers to apply cross-attention to"
            )
        if field_name == "xa_start":
            return "At which token to start applying the cross-attention"
        if field_name == "xa_end":
            return "Until which token to apply the cross-attention"
        if field_name == "xa_step":
            return "Applies cross-attention every `xa_step` token between xa_start and xa_end"
        if field_name == "xa_delay":
            return "Applies cross-attention to token t with the embedding from token t - xa_delay"
        if field_name == "xa_extended_layer_dims":
            return "By how many dimensions to extend the layers in xa_layers."
        return ""

    def __post_init__(self) -> None:
        """Check that the subconfig is valid"""
        if self.num_extra_tokens == 0 and self.num_crossattended_tokens == 0:
            print_main("[bold yellow]WARN:[/bold yellow]: No fusion mechanisms active")

        if not self.align_img_and_speech_tokens_dim and self.xa_dim is not None:
            print_main(
                "[bold yellow]WARN:[/bold yellow]: xa_dim is set hence it "
                "takes precedence over the value fed to align_img_and_speech_tokens"
            )

        # Convert
        if isinstance(self.xa_layers, list):
            self.xa_layers = tuple(self.xa_layers)
        if self.xa_end is not None and self.xa_end not in {"end", "eoi"}:
            self.xa_end = int(self.xa_end)

        if self.num_extra_tokens == 0:
            self.add_boi_eoi = False

        if self.num_crossattended_tokens == 0:
            self.xa_start = None
            self.xa_end = None
        else:
            assert (
                self.xa_start is not None
            ), "xa_start should be given to use cross-attention"
            assert (
                self.xa_end is not None
            ), "xa_end should be given to use cross-attention"

            if self.num_extra_tokens == 0:
                assert (
                    self.xa_start
                    not in {
                        "boi",
                        "eoi",
                    }
                    and self.xa_end != "eoi"
                ), "BoI and EoI tokens are not inserted if `num_extra_tokens` is 0"
            elif self.xa_start != "start":
                print_main(
                    "[yellow]WARN:[/yellow]Making cross_attention start before BoI is weird",
                    rich=True,
                )

        if self.xa_step > 1:
            raise NotImplementedError("xa_step > 1 is not implemented yet")

        if self.xa_extended_layer_dims is not None:
            assert (
                self.xa_extended_layer_dims > 0
            ), "xa_extended_layer_dims should be positive"

        return

    @property
    def crossattention_kwargs(self) -> Dict[str, Any]:
        """Return crossattention related kwargs used for model initialization.
        These are passed to modules/cross_attention.py:GatedCrossAttention
        down the line"""
        return {
            # passed to Moshi/Helium constructor
            "cross_attention": self.num_crossattended_tokens != 0,
            "xa_layers": self.xa_layers,
            # Futher passed to GatedCrossAttention construction
            "xa_gating": self.xa_gating,
            "xa_conditional_gating": self.xa_conditional_gating,
            "xa_shared": self.xa_shared,
            "xa_gate_shared": self.xa_gate_shared,
            "xa_start": self.xa_start,
            "xa_end": self.xa_end,
            "xa_step": self.xa_step,
            "xa_delay": self.xa_delay,
        }
