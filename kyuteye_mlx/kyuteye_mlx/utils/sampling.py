# Taken from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/sample_utils.py
# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from functools import partial

import mlx.core as mx
from jaxtyping import BFloat16, UInt32


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_p_sampling(
    logits: BFloat16[mx.array, "batch vocab"], top_p: float, temperature: float
) -> UInt32[mx.array, "batch"]:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output.
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.
    Returns:
        token selected based on the top-p criterion.
    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460  # noqa
    probs = mx.softmax(logits * (1 / temperature), axis=-1)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = probs[..., sorted_indices.squeeze(0)]

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # select tokens with cumulative probs below threshold
    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        0,
    )

    sorted_token = mx.random.categorical(mx.log(top_probs))
    token = sorted_indices.squeeze(0)[sorted_token]

    return token


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def categorical_sampling(logits: BFloat16[mx.array, "batch vocab"], temp: float) -> UInt32[mx.array, "batch"]:
    return mx.random.categorical(logits * (1 / temp))


@dataclass
class Sampler:
    temp: float
    top_p: float

    def __call__(
        self, logits: BFloat16[mx.array, "batch vocab"]
    ) -> tuple[UInt32[mx.array, "batch"], BFloat16[mx.array, "batch vocab"]]:
        logit_bias: dict[int, float] | None = None

        if logit_bias:
            indices = mx.array(list(logit_bias.keys()))
            values = mx.array(list(logit_bias.values()))
            logits[:, indices] += values

        if self.temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            if self.top_p > 0 and self.top_p < 1.0:
                token = top_p_sampling(logits, self.top_p, self.temp)
            else:
                token = categorical_sampling(logits, self.temp)

        logprobs = logits - mx.logsumexp(logits)
        return token, logprobs
