import mlx.core as mx
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from kyuteye_mlx.models.siglip import SiglipWrapper


def convert_weights_for_mlx(weights: dict[str, torch.Tensor]) -> dict[str, mx.array]:
    new_weights = {}
    for k, v in weights.items():
        new_key = k.removeprefix("vision_model.")
        new_key = "model." + new_key
        new_weights[new_key] = mx.array(v)
    new_weights["model.embeddings.patch_embedding.weight"] = new_weights[
        "model.embeddings.patch_embedding.weight"
    ].transpose(0, 2, 3, 1)
    return new_weights


@torch.no_grad()
def test_siglip_weights_conversion() -> None:
    model_id = "google/paligemma2-3b-pt-448"

    processor = AutoProcessor.from_pretrained(model_id)
    image_processor = processor.image_processor

    model = AutoModelForImageTextToText.from_pretrained(model_id)
    np.random.seed(99)
    img = np.random.randint(0, 255, size=(448, 448, 3), dtype=np.uint8)
    inp = image_processor(images=img, return_tensors="pt")

    out_pytorch = model.vision_tower(pixel_values=inp["pixel_values"])

    out_pytorch_as_np = out_pytorch.last_hidden_state.detach().numpy()

    as_mlx_weights = convert_weights_for_mlx(model.vision_tower.state_dict())

    siglip_wrapper = SiglipWrapper()

    siglip_wrapper.load_weights(list(as_mlx_weights.items()), strict=True)

    output = siglip_wrapper(mx.array(img))
    out_mlx_as_np = np.array(output, copy=False)

    diff_average = np.mean(np.abs(out_pytorch_as_np - out_mlx_as_np))
    assert diff_average < 2e-5, f"Average difference between the two embeddings is {diff_average}"
