from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np
import torch
import torch
from moshi.models import loaders, LMGen
import random
from pathlib import Path
from huggingface_hub import hf_hub_download


def write_weights_for_analysis(model:  torch.nn.Module):
    file_content = ""
  
    if isinstance(model, torch.nn.Module):
        framework_name = "torch"
        params = model.state_dict().items()
    else:
        print("Unsupported model type")

    for i, (key, value) in enumerate(params):
        file_content += f"{i} {key} {value.shape}\n"
    
    dest = Path(f"/tmp/weights_{random.randint(0, 2**16)}_{framework_name}.txt")
    dest.write_text(file_content)
    print("Wrote layers description into " + dest)


@torch.no_grad()
def test_weights_conversion_moshi():
    moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    moshi = loaders.get_moshi_lm(moshi_weight)
    lm_gen_torch = LMGen(moshi, temp=0.8, temp_text=0.7)

    write_weights_for_analysis(lm_gen_torch)
    
if __name__ == "__main__":
    test_weights_conversion_moshi()