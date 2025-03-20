# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "huggingface-hub",
#     "rich",
# ]
# ///
import shutil
import tarfile
from pathlib import Path

import fire
import rich
from huggingface_hub import hf_hub_download


def get() -> None:
    """Download archived sources and unzip"""
    root_dir = Path(__file__).parents[1]
    rich.print("[green][INFO][/green] retrieving the static content")
    dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "vis_dist.tgz")
    dist_tgz = Path(dist_tgz)
    dist = dist_tgz.parent / "dist"
    if not dist.exists():
        with tarfile.open(dist_tgz, "r:gz") as tar:
            tar.extractall(path=dist_tgz.parent)
    tgt_path = str(root_dir / "client" / "dist")
    shutil.move(dist, tgt_path)
    print(f"Static sources downloaded to {tgt_path}")


if __name__ == "__main__":
    fire.Fire(get)
