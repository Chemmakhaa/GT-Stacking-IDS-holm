"""Load the trained model and evaluate it on the held-out test set.

The results are printed on the screen (no result file is written).
Add --plot to display the confusion matrix.

Usage:
  python scripts/test.py --config configs/unsw_nb15.yaml --task binary
"""
import os

import joblib
import numpy as np
from _common import load_encoded, parse_args

from src.metrics import compute_metrics, per_class_report
from src.pipeline import predict


def extra(p):
    p.add_argument("--plot", action="store_true", help="show the confusion matrix in a window")


args, cfg = parse_args(__doc__, extra)
path = os.path.join("models", f"{cfg['dataset']['name']}_{cfg['task']}.joblib")
if not os.path.exists(path):
    raise FileNotFoundError(f"{path} not found. Run scripts/train.py first.")
saved = joblib.load(path)
pipe, names = saved["pipeline"], saved["class_names"]

_, _, X_test, y_test, _ = load_encoded(cfg)
y_pred = predict(pipe, X_test)
binary = cfg["task"] == "binary"
m = compute_metrics(y_test, y_pred, binary)

print(f"\n=== Results on the held-out test set ({cfg['dataset']['name']}, {cfg['task']}, N = {len(y_test)}) ===")
print(f"Accuracy      : {m['accuracy'] * 100:.2f} %")
if binary:
    print(f"Precision     : {m['precision'] * 100:.2f} %   (Attack class)")
    print(f"Recall        : {m['recall'] * 100:.2f} %   (Attack class)")
    print(f"F1-score      : {m['f1_attack'] * 100:.2f} %   (Attack class)")
print(f"Macro F1      : {m['macro_f1'] * 100:.2f} %")
print(f"Weighted F1   : {m['weighted_f1'] * 100:.2f} %")
print(f"MCC           : {m['mcc']:.4f}")
print(f"G-Mean        : {m['g_mean']:.4f}")

report, cm = per_class_report(y_test, y_pred, names)
print("\nPer-class results:")
print(report.to_string(index=False))

width = max(len(n) for n in names) + 2
print("\nConfusion matrix (rows = true class, columns = predicted class):")
print(" " * width + "".join(f"{n[:9]:>10}" for n in names))
for i, n in enumerate(names):
    print(f"{n:<{width}}" + "".join(f"{v:>10}" for v in cm[i]))
print(f"\nCorrect predictions: {np.trace(cm)} / {cm.sum()}")

if args.plot:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay(cm, display_labels=names).plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Test set (N = {len(y_test)})")
    plt.tight_layout()
    plt.show()
