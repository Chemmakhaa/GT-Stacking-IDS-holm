"""Step 4 - Paired significance tests with Holm adjustment (Tables 12 and 17).

Mode A (default): computes the per-fold accuracy of the proposed model and of
the six base learners with 10-fold cross-validation on the training set
(n = 10 paired observations, df = 9), then runs the corrected resampled
t-test of Nadeau and Bengio (n_test / n_train = 1/9) with Holm adjustment.

Mode B (--scores FILE): reads per-fold scores from a CSV file (one column per
model, one row per fold, same folds for every model), e.g. the published
baselines of Table 17, and runs the tests on it.

Outputs (results/<dataset>/<task>/):
  fold_scores_10fold.csv      raw data: accuracy of every model on every fold (mode A)
  stats_<name>.csv            n, df, mean difference, t, p, Cohen's d_z, Holm p, significant

Usage:
  python scripts/04_statistical_tests.py --config configs/cic_ids2018.yaml
  python scripts/04_statistical_tests.py --config configs/unsw_nb15.yaml \
         --scores results/unsw_nb15/binary/fold_scores_baselines.csv --test standard
"""
import os

import pandas as pd
from _common import load_encoded, parse_args
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.pipeline import fit_pipeline, predict
from src.stats_tests import compare_to_baselines

BASE = ["catboost", "xgboost", "lightgbm", "gradient_boosting", "adaboost", "random_forest"]


def extra(p):
    p.add_argument("--scores", help="CSV of per-fold scores (mode B)")
    p.add_argument("--proposed", default="stacking", help="name of the proposed-model column")
    p.add_argument("--test", choices=["corrected", "standard"], default="corrected")


args, cfg = parse_args(__doc__, extra)
out = cfg["output_dir"]
k = cfg["cv"]["stat_n_splits"]

if args.scores:
    scores = pd.read_csv(args.scores)
    name = os.path.splitext(os.path.basename(args.scores))[0]
else:
    X_train, y_train, _, _, _ = load_encoded(cfg)
    seed = cfg["split"]["seed"]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X_train, y_train), start=1):
        row = {"fold": fold}
        for model in ["stacking"] + BASE:
            pipe = fit_pipeline(X_train.iloc[tr], y_train[tr], cfg, seed=seed, model=model)
            row[model] = accuracy_score(y_train[va], predict(pipe, X_train.iloc[va]))
        rows.append(row)
        print(row)
    scores = pd.DataFrame(rows).set_index("fold")
    scores.to_csv(os.path.join(out, f"fold_scores_{k}fold.csv"))
    name = "base_learners"

scores = scores.drop(columns=[c for c in ["fold"] if c in scores.columns])
ratio = 1 / (len(scores) - 1) if args.test == "corrected" else None  # n_test/n_train of K-fold CV
table = compare_to_baselines(scores, args.proposed, test_over_train=ratio)
table.round(4).to_csv(os.path.join(out, f"stats_{name}.csv"), index=False)
print(table.round(4).to_string(index=False))
