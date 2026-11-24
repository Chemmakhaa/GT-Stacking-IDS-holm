"""Evaluation metrics. Binary: positive class = Attack (label 1)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)


def g_mean(y_true, y_pred):
    """Geometric mean of the per-class recalls (binary: sqrt(TPR x TNR))."""
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    return float(np.prod(rec) ** (1 / len(rec)))


def compute_metrics(y_true, y_pred, binary):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "g_mean": g_mean(y_true, y_pred),
        "n_samples": int(len(y_true)),
    }
    if binary:  # precision / recall / F1 of the Attack class (pos_label = 1)
        m["precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        m["recall"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        m["f1_attack"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    else:
        m["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        m["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in m.items()}


def per_class_report(y_true, y_pred, class_names):
    """Per-class precision, recall, F1, one-vs-rest accuracy and support (Tables 5-6)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    total = cm.sum()
    rows = []
    for k, name in enumerate(class_names):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        tn = total - tp - fp - fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"class": name, "precision": prec, "recall": rec, "f1": f1,
                     "one_vs_rest_accuracy": (tp + tn) / total, "support": int(cm[k, :].sum())})
    return pd.DataFrame(rows).round(4), cm
