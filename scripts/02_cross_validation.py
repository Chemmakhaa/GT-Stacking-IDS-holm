"""Step 2 - 5-fold cross-validation on the training set (Tables 3, 4 and 13).

In every fold, the whole pipeline (scaling, Boruta-SHAP, cWGAN-GP, stacking)
is fitted on the training part of the fold only and evaluated on its
validation part.

Outputs:
  results/<dataset>/<task>/cv_folds_<model>.csv     metrics of each fold
  results/<dataset>/<task>/cv_summary.csv           mean and std over the folds

Usage:
  python scripts/02_cross_validation.py --config configs/unsw_nb15.yaml --task binary
  python scripts/02_cross_validation.py --config configs/unsw_nb15.yaml --models stacking catboost xgboost
"""
import os

import pandas as pd
from _common import load_encoded, parse_args
from sklearn.model_selection import StratifiedKFold

from src.metrics import compute_metrics
from src.pipeline import fit_pipeline, predict


def extra(p):
    p.add_argument("--models", nargs="+", default=["stacking"],
                   help="stacking, catboost, xgboost, lightgbm, gradient_boosting, adaboost, random_forest")


args, cfg = parse_args(__doc__, extra)
X_train, y_train, _, _, _ = load_encoded(cfg)
seed = cfg["split"]["seed"]
binary = cfg["task"] == "binary"
skf = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)

summary = []
for model in args.models:
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X_train, y_train), start=1):
        pipe = fit_pipeline(X_train.iloc[tr], y_train[tr], cfg, seed=seed, model=model)
        m = compute_metrics(y_train[va], predict(pipe, X_train.iloc[va]), binary)
        m["fold"] = fold
        rows.append(m)
        print(model, f"fold {fold}:", m)
    folds = pd.DataFrame(rows).set_index("fold")
    folds.to_csv(os.path.join(cfg["output_dir"], f"cv_folds_{model}.csv"))
    mean, std = folds.mean(numeric_only=True), folds.std(numeric_only=True)
    summary.append(pd.concat([mean.add_suffix("_mean"), std.add_suffix("_std")]).rename(model))

table = pd.DataFrame(summary).round(4)
table.to_csv(os.path.join(cfg["output_dir"], "cv_summary.csv"))
print(table.T)
