"""Main configuration object used to configure the model and training pipeline"""

import os
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import yaml

from kyuteye.config.enums import ImageEncoder
from kyuteye.config.subconfigs import (
    FusionConfig,
    ImageEncoderConfig,
    LMConfig,
    MoshiConfig,
)
from kyuteye.utils.dist_utils import print_main
from kyuteye.utils.logging_utils import flatten_nested_dict


class KyuteyeConfig:
    """Base class encapsulating all options for training and evaluating a multimodal model"""

    fuse: FusionConfig
    image: ImageEncoderConfig
    lm: LMConfig

    def __init__(self, **kwargs: Any):
        self._fields_to_sub: Dict[str, str] = {}

        # Define all modular subconfigs defined in subconfigs.py
        backup_kwargs = deepcopy(kwargs)
        self._subnames = []
        for name, constructor in [
            ("fuse", FusionConfig),
            ("image", ImageEncoderConfig),
            ("lm", LMConfig),
            ("moshi", MoshiConfig),
        ]:
            keys = {f.name for f in fields(constructor)}
            setattr(
                self,
                name,
                constructor(
                    **{k: backup_kwargs.pop(k) for k in keys if k in backup_kwargs}
                ),
            )
            self._subnames.append(name)

        # Extra arguments comming from the CLI argparser will be passed but not used
        # we still mark them to remember them
        if len(backup_kwargs):
            print_main(
                "\n[bold yellow]WARN:[/bold yellow] Found superfluous arguments "
                "passed to [cyan]KyuteyeConfig[/cyan]:\n"
                + "\n".join(f"  - {k} = {v}" for k, v in backup_kwargs.items())
                + "\n",
                rich=True,
                flush=True,
            )

        # Map field names to the correct subconfig it belongs to
        for name in self._subnames:
            for f in fields(getattr(self, name)):
                if f.name in self._fields_to_sub:
                    raise AssertionError(
                        f"Subconfig {name} uses field {f.name} which is already in use"
                    )
                if f.name in self._subnames:
                    raise AssertionError(
                        f"Subconfig {name} uses field {f.name} which"
                        " is already defined in KyuteyeConfig"
                    )
                self._fields_to_sub[f.name] = name

        # Postinit
        if self.fuse.num_crossattended_tokens == 0:
            self.image.norm_xa = None
        if self.fuse.num_extra_tokens == 0:
            self.image.norm_extra = None

        if self.fuse.xa_dim is None:
            if not self.fuse.align_img_and_speech_tokens_dim:
                self.fuse.xa_dim = ImageEncoder(self.image.encoder_name).out_dims

    def __getattribute__(self, name: str) -> Any:
        """Getattr with direct shortcut to all subconfigs' subfields"""
        if name not in ["_fields_to_sub", "_subnames"] and name in self._fields_to_sub:
            return getattr(getattr(self, self._fields_to_sub[name]), name)
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Setattr with direct shortcut to all subconfigs' fields"""
        if hasattr(self, "_fields_to_sub") and name in self._fields_to_sub:
            setattr(getattr(self, self._fields_to_sub[name]), name, value)
        else:
            super().__setattr__(name, value)

    @property
    def moshi_constructor_kwargs(self) -> Dict[str, Any]:
        """Return constructor for constructing a MoshiVis model"""
        return dict(
            **asdict(self.moshi),
            xa_dim=self.fuse.xa_dim,
            **self.fuse.crossattention_kwargs,
        )

    @classmethod
    def from_yml(cls, path: Path | str) -> "KyuteyeConfig":
        """Initialize current config from a yaml file"""
        return cls(**__load_yaml__(str(path)))

    def to_yml(self, path: Optional[Path | str] = None) -> None:
        """Save current config to a yaml file"""
        if path is None:
            path = self.output_dir
        path = str(path)
        __save_yaml__(
            {
                k: (
                    v.name
                    if hasattr(v, "name")
                    else (
                        str(v).replace("torch.", "")
                        if isinstance(v, torch.dtype)
                        else (
                            tuple(x.name for x in v)
                            if k in {"train_dataset", "eval_dataset", "blind_dataset"}
                            else v
                        )
                    )
                )
                for k, v in self.to_dict().items()
            },
            path,
        )

    def print(self, flat: bool = False, only: Optional[Sequence[str]] = None) -> None:
        """Pretty print current config"""
        print_main("-" * 100, "[bold green]Config[/bold green]", rich=True)
        if flat:
            print_main(
                "\n".join(
                    f"\t{k} = {v}"
                    for k, v in self.to_dict(flat=True).items()
                    if (only is None or k in only)
                ),
                rich=True,
            )
        else:
            for name, subd in self.to_dict(flat=False).items():
                if only is not None and name not in only:
                    continue
                print_main("\n\t", "-" * 50, f"[cyan]{name}[/cyan]", rich=True)
                print_main(
                    "\n".join(f"\t\t{k} = {v}" for k, v in subd.items()),
                    rich=True,
                )
        print_main("-" * 100)

    def to_dict(self, flat: bool = True) -> Dict[str, Any]:
        """Returns current config as a (flat) dictionary"""
        d = {
            subconfig: asdict(getattr(self, subconfig)) for subconfig in self._subnames
        }
        if flat:
            return flatten_nested_dict(d)
        return d


def __load_yaml__(path: Path | str) -> Dict:
    """Load a dictionary from a YAML file

    :param path: Path to load the config from

    :return: the dictionary of kwargs that will be fed to `KyuteyeConfig`
    """
    path = str(path)
    assert os.path.exists(path), f"Could not load config {path}: File does not exist"
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)

    # KyuteyeConfig works with flattened dict as inputs
    config = flatten_nested_dict(config)

    # Yaml parse sequences sa list -> tuples
    config = {k: tuple(v) if isinstance(v, list) else v for k, v in config.items()}

    return config


def __save_yaml__(config: Dict, path: Path | str) -> None:
    """Save a config dictionary to a YAML

    :param config: Kyuteye config converted to a dict
    :param path: Path to save the config to
    """
    path = str(path)
    path = path + (".yml" if not path.endswith(".yml") else "")
    base_dir = os.path.abspath(os.path.dirname(path))
    os.makedirs(base_dir, exist_ok=True)
    with open(path, "w") as stream:
        config = yaml.safe_dump(config, stream)
