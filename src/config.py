"""Load a YAML configuration file and fix all random seeds."""
import os
import random

import numpy as np
import yaml


def load_config(path, task=None, quick=False):
    """Read the YAML config. `task` overrides the task, `quick` shrinks everything for a smoke test."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if task:
        cfg["task"] = task
    if quick:
        cfg["boruta_shap"]["n_trials"] = 5
        cfg["cwgan_gp"]["epochs"] = 2
        for name, params in cfg["models"].items():
            for key in ("iterations", "n_estimators"):
                if key in params:
                    params[key] = 20
        cfg["models"]["mlp_meta"]["epochs"] = 3
    cfg["output_dir"] = os.path.join(cfg["output_dir"], cfg["task"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg


def set_seed(seed):
    """Fix the seed of Python, NumPy and PyTorch (seed = 46 in the paper)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
