"""Step 1 - Clean the dataset and make the stratified 80/20 train/test split (seed 46).

Outputs:
  data/processed/<dataset>/<task>/train.csv.gz, test.csv.gz
  results/<dataset>/<task>/test_indices.csv       row ids of the test set
  results/<dataset>/<task>/dataset_summary.json   sizes and class counts
  results/<dataset>/<task>/environment.txt        Python, library versions and hardware

Usage:
  python scripts/01_prepare_data.py --config configs/unsw_nb15.yaml --task binary
"""
import json
import os
import platform
from importlib import metadata

from _common import parse_args

from src.data import load_and_clean, make_split, save_split

args, cfg = parse_args(__doc__)
df = load_and_clean(cfg)
print(f"Rows after cleaning: {len(df)}")
train_df, test_df = make_split(df, cfg)
summary = save_split(train_df, test_df, cfg)
print(json.dumps(summary, indent=2))

# record the software environment and hardware (reproducibility)
lines = [f"python {platform.python_version()}", f"platform {platform.platform()}", f"processor {platform.processor()}"]
try:
    import torch

    lines.append(f"cuda_available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"gpu {torch.cuda.get_device_name(0)}")
except ImportError:
    pass
for pkg in ["numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "xgboost", "lightgbm",
            "catboost", "shap", "torch", "matplotlib", "PyYAML"]:
    try:
        lines.append(f"{pkg} {metadata.version(pkg)}")
    except metadata.PackageNotFoundError:
        lines.append(f"{pkg} not installed")
with open(os.path.join(cfg["output_dir"], "environment.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
