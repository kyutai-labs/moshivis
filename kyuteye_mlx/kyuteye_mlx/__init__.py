# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# flake8: noqa

"""
kyuteye_mlx is the MLX inference codebase for Kyutai audio+vision generation models.
"""

import os
import subprocess

# TODO: remove this when https://github.com/ml-explore/mlx/issues/1963 is fixed
if subprocess.check_output(["sysctl", "hw.model"]).decode().split(":")[1].strip() == "Mac15,12":
    os.environ["MLX_MAX_OPS_PER_BUFFER"] = "8"
    os.environ["MLX_MAX_MB_PER_BUFFER"] = "1000000"


from jaxtyping import install_import_hook

# Set to True to get runtime type-checking and other small checks
# but will slow down the steps by ~2%
DEBUG_MODE = bool(int(os.environ.get("MOSHI_MLX_DEBUG_MODE", "0")))

if DEBUG_MODE:
    print("debug mode enabled.")
    install_import_hook("kyuteye_mlx", "beartype.beartype")
else:
    print("debug mode disabled.")

from . import models, modules, utils

__version__ = "0.1.0"
