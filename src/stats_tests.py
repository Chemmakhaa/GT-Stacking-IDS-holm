"""Paired significance tests on per-fold scores and Holm adjustment.

- Observational unit: one held-out fold (both models trained and tested on the same folds).
- corrected test (Nadeau & Bengio, 2003):
      t = mean(d) / sqrt((1/n + n_test/n_train) * s^2),   df = n - 1
- standard paired t-test:
      t = mean(d) / (s / sqrt(n)),                         df = n - 1
- effect size (paired Cohen's d):  d_z = mean(d) / s
- Holm step-down adjustment within one family of comparisons.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def paired_test(a, b, test_over_train=None):
    """a, b: per-fold scores of the proposed model and of one baseline (same folds).
    test_over_train = n_test / n_train for the corrected test (1/9 for 10-fold CV),
    or None for the standard paired t-test."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    if sd == 0:
        return {"n": n, "df": n - 1, "mean_diff": mean, "t": np.inf, "p_value": 0.0, "cohen_dz": np.inf}
    if test_over_train is None:
        t = mean / (sd / np.sqrt(n))
    else:
        t = mean / np.sqrt((1 / n + test_over_train) * sd ** 2)
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return {"n": n, "df": n - 1, "mean_diff": mean, "t": t, "p_value": p, "cohen_dz": mean / sd}


def compare_to_baselines(scores, proposed, test_over_train=None, alpha=0.05):
    """scores: DataFrame, one column per model, one row per fold.
    Compares `proposed` with every other column (one family) and applies Holm."""
    rows = []
    for col in scores.columns:
        if col == proposed:
            continue
        r = paired_test(scores[proposed], scores[col], test_over_train)
        r["comparison"] = f"{proposed} vs {col}"
        rows.append(r)
    res = pd.DataFrame(rows)
    reject, p_holm, _, _ = multipletests(res["p_value"], alpha=alpha, method="holm")
    res["p_holm"] = p_holm
    res["significant"] = reject
    cols = ["comparison", "n", "df", "mean_diff", "t", "p_value", "cohen_dz", "p_holm", "significant"]
    return res[cols]
