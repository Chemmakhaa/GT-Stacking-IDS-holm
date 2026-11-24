"""Step 3 - Train on the full training set, evaluate once on the held-out test set.

Outputs (results/<dataset>/<task>/):
  test_metrics.json                    metrics on the test set
  confusion_matrix.csv / .png          confusion matrix on the test set (Figs. 7-10)
  per_class_report.csv                 precision, recall, F1, one-vs-rest accuracy, support
  synthetic_samples_per_class.csv      real / synthetic / total training samples per class
  selected_features.csv                features kept by Boruta-SHAP

Usage:
  python scripts/03_final_test.py --config configs/unsw_nb15.yaml --task binary
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from _common import load_encoded, parse_args  # noqa: E402

from src.metrics import compute_metrics, per_class_report  # noqa: E402
from src.pipeline import fit_pipeline, predict  # noqa: E402

args, cfg = parse_args(__doc__)
out = cfg["output_dir"]
X_train, y_train, X_test, y_test, names = load_encoded(cfg)
seed = cfg["split"]["seed"]

pipe = fit_pipeline(X_train, y_train, cfg, seed=seed, model="stacking")
y_pred = predict(pipe, X_test)

metrics = compute_metrics(y_test, y_pred, cfg["task"] == "binary")
with open(os.path.join(out, "test_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Test set:", metrics)

report, cm = per_class_report(y_test, y_pred, names)
report.to_csv(os.path.join(out, "per_class_report.csv"), index=False)
pd.DataFrame(cm, index=names, columns=names).to_csv(os.path.join(out, "confusion_matrix.csv"))

fig, ax = plt.subplots(figsize=(4 + 0.5 * len(names), 3.5 + 0.5 * len(names)))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
ax.set_yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(f"Held-out test set (N = {len(y_test)})")
fig.tight_layout()
fig.savefig(os.path.join(out, "confusion_matrix.png"), dpi=300)

if pipe.synthetic_report:
    syn = pd.DataFrame(pipe.synthetic_report)
    syn["class"] = [names[int(c)] for c in syn["class"]]
    syn.to_csv(os.path.join(out, "synthetic_samples_per_class.csv"), index=False)
    print(syn)
pd.DataFrame({"feature": pipe.feature_names}).to_csv(os.path.join(out, "selected_features.csv"), index=False)
