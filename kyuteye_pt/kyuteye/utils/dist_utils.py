"""Some utils for distributed training"""

import os
from typing import Any

import torch.distributed as dist
from rich import print as rich_print


def is_main() -> bool:
    """Returns True iff the current process is the main one"""
    # torch distributed
    if dist.is_initialized():
        return dist.get_rank() == 0
    # procid
    if "LOCAL_RANK" in os.environ:
        return int(os.environ.get("LOCAL_RANK", 0)) == 0
    return int(os.environ.get("SLURM_PROCID", 0)) == 0


def print_main(*args: Any, rich: bool = False, **kwargs: Any) -> None:
    """Print function that only activate for the main process"""
    if is_main():
        if rich:
            rich_print(*args, **kwargs)
        else:
            print(*args, **kwargs)
