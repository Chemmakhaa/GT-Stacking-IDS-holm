"""Step 5 - Ablation study with 5-fold cross-validation (Tables 9 and 10).

Variants:
  full_model       scaling + Boruta-SHAP + cWGAN-GP + MLP stacking
  wo_boruta_shap   without feature selection
  wo_cwgan_gp      without synthetic data
  wo_mlp_stacking  base learners combined by soft voting instead of the MLP

Output: results/<dataset>/<task>/ablation.csv  (mean over the 5 folds)

Usage:
  python scripts/05_ablation.py --config configs/cic_ids2018.yaml --task binary
"""
import os

import pandas as pd
from _common import load_encoded, parse_args
from sklearn.model_selection import StratifiedKFold

from src.metrics import compute_metrics
from src.pipeline import fit_pipeline, predict

VARIANTS = {
    "full_model": dict(use_boruta=True, use_gan=True, model="stacking"),
    "wo_boruta_shap": dict(use_boruta=False, use_gan=True, model="stacking"),
    "wo_cwgan_gp": dict(use_boruta=True, use_gan=False, model="stacking"),
    "wo_mlp_stacking": dict(use_boruta=True, use_gan=True, model="voting"),
}

args, cfg = parse_args(__doc__)
X_train, y_train, _, _, _ = load_encoded(cfg)
seed = cfg["split"]["seed"]
binary = cfg["task"] == "binary"
skf = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)

results = {}
for name, opts in VARIANTS.items():
    folds = []
    for tr, va in skf.split(X_train, y_train):
        pipe = fit_pipeline(X_train.iloc[tr], y_train[tr], cfg, seed=seed, **opts)
        folds.append(compute_metrics(y_train[va], predict(pipe, X_train.iloc[va]), binary))
    results[name] = pd.DataFrame(folds).mean(numeric_only=True)
    print(name, results[name].round(4).to_dict())

table = pd.DataFrame(results).T.round(4)
table.to_csv(os.path.join(cfg["output_dir"], "ablation.csv"))
print(table)
