# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "numpy",
#     "rich",
#     "safetensors",
#     "torch",
# ]
# ///
"""Utils for converting checkpoint formats across the different backends

uv run scripts/convert_ckpt_utils.py rust_to_pt safetensors_file_path.safetensors
-> will output in safetensors_file_path_pt.safetensors


uv run scripts/convert_ckpt_utils.py pt_to_mlx safetensors_file_path.safetenros
-> will output in safetensors_file_path_mlx.safetensors
"""

import os
import re
from typing import Dict, Optional

import fire
import rich
import torch
from safetensors.torch import load_file, save_file


def remove_other_output_codebooks(
    tensors: Dict[str, torch.Tensor], dep_q: int = 8
) -> None:
    """Remove output codebooks corresponding to OTHER"""
    for key in list(tensors.keys()):
        if (m := re.match(r"^depformer_in.([0-9]+).", key)) or (
            m := re.match(r"^audio_linears.([0-9]+).", key)
        ):
            layer = int(m.groups()[0])
            if layer >= dep_q:
                del tensors[key]
        if m := re.match(r"^depformer_emb.([0-9]+).", key):
            layer = int(m.groups()[0])
            if layer >= dep_q - 1:
                del tensors[key]
        if m := re.match(r"^depformer.layers.[0-9+].gating.([0-9]+).", key):
            layer = int(m.groups()[0])
            if layer >= dep_q:
                del tensors[key]


class Launcher:

    def rust_to_pt(self, safetensors_file: str, out_file: Optional[str] = None) -> None:
        """Rust ckpt to Pytorch ckpt
        Usage: uv run scripts/convert_ckpt_utils.py rust_to_pt path_to_rust_ckpt.safetensors
        """
        safetensors_file = os.path.abspath(safetensors_file)
        assert safetensors_file.endswith(".safetensors")
        if out_file is None:
            out_file = safetensors_file.rsplit(".", 1)[0] + "_pt.safetensors"
        assert out_file is not None
        state_dict = load_file(safetensors_file)
        new_state_dict = {}
        accumulate_attention: Dict[str, Dict] = {}
        for key, value in state_dict.items():
            new_key = key
            if m := re.match(
                r"(.*)cross_attention.(in_proj_weight|out_proj.weight)", key
            ):
                new_key = f"llm.{m.groups()[0]}cross_attention.mha.{m.groups()[1]}"
            elif m := re.match(r"depformer.([0-9]+).emb.(.*)", key):
                idx = int(m.groups()[0])
                new_key = (
                    "depformer_text_emb." if idx == 0 else f"depformer_emb.{idx - 1}."
                ) + m.groups()[1]
            elif m := re.match(r"depformer.([0-9]+).linear_in.(.*)", key):
                new_key = f"depformer_in.{idx}.{m.groups()[1]}"
            elif m := re.match(r"depformer.([0-9]+).linear_out.(.*)", key):
                new_key = f"audio_linears.{idx}.{m.groups()[1]}"
            elif m := re.match(
                r"depformer.([0-9]+).transformer.layers.([0-9]+).gating.(.*)", key
            ):
                layer_idx = int(m.groups()[1])
                codebook_idx = int(m.groups()[0])
                new_key = f"depformer.layers.{layer_idx}.gating.{codebook_idx}.{m.groups()[2]}"
            elif m := re.match(
                r"depformer.([0-9]+).transformer.layers.([0-9]+).self_attn.(.*)", key
            ):
                layer_idx = int(m.groups()[1])
                codebook_idx = int(m.groups()[0])
                new_key = f"depformer.layers.{layer_idx}.self_attn.{m.groups()[2]}"
                if new_key not in accumulate_attention:
                    accumulate_attention[new_key] = {}
                accumulate_attention[new_key][codebook_idx] = value
                continue
            elif m := re.match(
                r"depformer.([0-9]+).transformer.layers.([0-9]+).norm(1|2).(.*)", key
            ):
                layer_idx = int(m.groups()[1])
                codebook_idx = int(m.groups()[0])
                if codebook_idx > 0:
                    continue
                new_key = (
                    f"depformer.layers.{layer_idx}.norm{m.groups()[2]}.{m.groups()[3]}"
                )
            elif m := re.match(r"emb.(.*)", key):
                new_key = "audio_emb." + m.groups()[0]
            elif (
                key.startswith("transformer")
                or key.startswith("audio")
                or key in {"text_emb.weight", "out_norm.alpha", "text_linear.weight"}
            ):
                new_key = f"llm.{key}"
            # be careful not to override anything
            assert new_key not in new_state_dict, new_key
            new_state_dict[new_key] = value

        for key, sd in accumulate_attention.items():
            tensor = torch.concatenate([sd[codebook] for codebook in sorted(sd)])
            new_state_dict[key] = tensor

        save_file(new_state_dict, out_file)
        rich.print(f"Saved converted state dict in [yellow]{out_file}[/yellow]")

    def pt_to_mlx(self, safetensors_file: str, out_file: Optional[str] = None) -> None:
        """Pytorch to MLX ckpt conversion"""
        safetensors_file = os.path.abspath(safetensors_file)
        assert safetensors_file.endswith(".safetensors")
        if out_file is None:
            if safetensors_file.endswith("_pt.safetensors"):
                out_file = safetensors_file.rsplit("_", 1)[0] + "_mlx.safetensors"
            else:
                out_file = safetensors_file.rsplit(".", 1)[0] + "_mlx.safetensors"
        assert out_file is not None
        state_dict = load_file(safetensors_file)
        model = {}
        in_n_q: int | None = None
        for idx in range(999):
            name = f"audio_emb.{idx}.weight"
            if name not in state_dict:
                in_n_q = idx
                break
        assert in_n_q is not None, "audio_emb weights not found in src checkpoint"
        out_n_q: int | None = None
        for idx in range(999):
            name = f"audio_linears.{idx}.weight"
            if name not in state_dict:
                out_n_q = idx
                break
        assert out_n_q is not None, "audio_emb weights not found in src checkpoint"
        for name in ["text_emb.weight", "text_linear.weight"]:
            model[name] = state_dict["llm." + name]
        model["out_norm.weight"] = state_dict["llm." + "out_norm.alpha"][0, 0]
        for idx in range(in_n_q):
            src_name = f"audio_emb.{idx}.weight"
            dst_name = f"audio_embs.{idx}.weight"
            model[dst_name] = state_dict[src_name]

        exported_out_n_q = out_n_q
        for idx in range(exported_out_n_q):
            base = f"depformer.slices.{idx}."
            model[base + "linear_in.weight"] = state_dict[f"depformer_in.{idx}.weight"]
            model[base + "linear_out.weight"] = state_dict[
                f"audio_linears.{idx}.weight"
            ]
            if idx == 0:
                model[base + "emb.weight"] = state_dict["depformer_text_emb.weight"]
            else:
                model[base + "emb.weight"] = state_dict[
                    f"depformer_emb.{idx - 1}.weight"
                ]

            for layer_idx in range(6):
                layer = base + f"transformer.layers.{layer_idx}."
                # WARNING: note that this uses in_proj_weight vs out_proj.weight
                model[layer + "self_attn.in_proj.weight"] = (
                    state_dict[f"depformer.layers.{layer_idx}.self_attn.in_proj_weight"]
                    .chunk(out_n_q)[idx]
                    .clone()
                )
                model[layer + "self_attn.out_proj.weight"] = (
                    state_dict[
                        f"depformer.layers.{layer_idx}.self_attn.out_proj.weight"
                    ]
                    .chunk(out_n_q)[idx]
                    .clone()
                )
                model[layer + "norm1.weight"] = state_dict[
                    f"depformer.layers.{layer_idx}.norm1.alpha"
                ][0, 0].clone()
                model[layer + "norm2.weight"] = state_dict[
                    f"depformer.layers.{layer_idx}.norm2.alpha"
                ][0, 0].clone()
                model[layer + "gating.linear_in.weight"] = state_dict[
                    f"depformer.layers.{layer_idx}.gating.{idx}.linear_in.weight"
                ]
                model[layer + "gating.linear_out.weight"] = state_dict[
                    f"depformer.layers.{layer_idx}.gating.{idx}.linear_out.weight"
                ]

        for key in [
            "image_prefix.norm_xa.alpha",
            "image_prefix.proj_xa.bias",
            "image_prefix.proj_xa.weight",
        ]:
            model["transformer." + key.replace("alpha", "weight")] = (
                state_dict[key][0, 0] if "alpha" in key else state_dict[key]
            )

        for k, v in state_dict.items():
            if k.startswith("image_prefix.enc."):
                model["img_embedder." + k[len("image_prefix.enc.") :]] = state_dict[k]
                continue
            elif not k.startswith("llm.") or k in {
                "llm.out_norm.alpha",
                "llm.text_emb.weight",
                "llm.text_linear.weight",
            }:
                continue

            k = k.replace("llm.", "")
            k = k.replace("in_proj_weight", "in_proj.weight")
            k = k.replace(
                "cross_attention.gate.alpha.", "cross_attention.gate.alpha.layers."
            )
            if k == "transformer.layers.0.cross_attention.mha.in_proj.weight":
                query, key, value = v.chunk(3)

                model["transformer.layers.0.cross_attention.mha.kv_proj.weight"] = (
                    torch.cat([key, value], dim=0)
                )
                model["transformer.layers.0.cross_attention.mha.q_proj.weight"] = query
            elif k == "transformer.layers.0.cross_attention.mha.out_proj.weight":
                model["transformer.layers.0.cross_attention.mha.out_proj.weight"] = v

            elif m := re.match(r"transformer.layers.\d+.norm(\d|_cross).alpha", k):
                model[m.group().replace("alpha", "weight")] = v[0, 0]
            else:
                model[k] = v
        print(f"Total Params: {sum([v.numel() for v in model.values()])/1e6}")

        save_file(model, out_file)


if __name__ == "__main__":
    fire.Fire(Launcher)
