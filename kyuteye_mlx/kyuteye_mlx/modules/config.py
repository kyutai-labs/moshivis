from dataclasses import dataclass
from typing import Literal


@dataclass
class TransformerConfig:
    d_model: int
    num_heads: int
    num_layers: int
    causal: bool
    norm_first: bool
    bias_ff: bool
    bias_attn: bool
    layer_scale: float | None
    positional_embedding: str
    use_conv_block: bool
    cross_attention: bool
    xa_shared: bool
    xa_gating: Literal["none", "sigmoid", "tanh"]
    conv_kernel_size: int
    use_conv_bias: bool
    gating: bool
    norm: str
    context: int
    max_period: int
    max_seq_len: int
    kv_repeat: int
    dim_feedforward: int
    conv_layout: bool
    img_emb_dim: int | None = None

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads
