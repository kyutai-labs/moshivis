"""Some utils for experiment tracking and logging"""

import json
import subprocess
from typing import Dict, Tuple

import rich


def flatten_nested_dict(d: Dict) -> Dict:
    """Flatten a nested config dictionary"""
    flattened_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flattened_dict.update(v)
        else:
            flattened_dict[k] = v
    return flattened_dict


def get_git_revision_hash(verbose: bool = True) -> Tuple[str, str]:
    """Return current git branch and commit"""
    git_branch = (
        subprocess.check_output(["git", "branch", "--show-current"])
        .decode("ascii")
        .strip()
    )
    commit_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
    if verbose:
        rich.print(
            f"[magenta]Git:[/] Commit [bold]{commit_hash}[/] on branch {git_branch}"
        )
    return git_branch, commit_hash


def pretty_json(config_dict: dict) -> str:
    """Pretty print the given dict as json"""
    config_dict = {
        k: (
            v.name
            if hasattr(v, "name")
            else v if isinstance(v, (str, int, float)) else str(v)
        )
        for k, v in config_dict.items()
    }
    json_config_dict = json.dumps(config_dict, indent=4)
    return "".join("\t" + line for line in json_config_dict.splitlines(True))
