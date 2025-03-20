import mlx.core as mx


def repeat_shared_weights(weights: dict[str, mx.array], num_layers: int) -> dict[str, mx.array]:
    for layer_idx in range(1, num_layers):
        for srckey in [
            "transformer.layers.0.cross_attention.mha.kv_proj",
            "transformer.layers.0.cross_attention.mha.q_proj",
            "transformer.layers.0.cross_attention.mha.out_proj",
        ]:
            for subkey in ["weight", "scales", "biases"]:
                k = srckey + "." + subkey
                if k in weights:
                    weights[k.replace(".0.", f".{layer_idx}.")] = weights[k]
    return weights


def remove_shared_weights(weights: dict[str, mx.array], num_layers: int) -> dict[str, mx.array]:
    for layer_idx in range(1, num_layers):
        k1 = "transformer.layers.0.cross_attention.mha.kv_proj.weight"
        weights.pop(k1.replace(".0.", f".{layer_idx}."))
    return weights


def split_embedder_weights(
    weights: dict[str, mx.array],
) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    embedder_weights = {}
    model_weights = {}
    for k, v in weights.items():
        if k.startswith("img_embedder."):
            embedder_weights[k[len("img_embedder.") :]] = v
        else:
            model_weights[k] = v
    return model_weights, embedder_weights
