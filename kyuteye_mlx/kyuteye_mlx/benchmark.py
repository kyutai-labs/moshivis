import time

import mlx.core as mx
import numpy as np

from .local_web import get_args_for_main, get_model, predict_text_and_audio


def main():
    args = get_args_for_main()
    args.quantized = 4
    gen = get_model(args, load_weights=False)

    sum_times = 0
    for i in range(100):
        data = mx.arange(8, dtype=mx.uint32).reshape(8, 1)
        uploaded_image_embeddings = mx.arange(1024 * 1152, dtype=mx.bfloat16).reshape(1, 1024, 1152)
        mx.eval((data, uploaded_image_embeddings))
        t1 = time.time()
        predict_text_and_audio(gen, data, uploaded_image_embeddings)
        t2 = time.time()
        if i >= 5:
            sum_times += t2 - t1

    print(f"average time per step: {(sum_times / 95) * 1000:1f} ms")
