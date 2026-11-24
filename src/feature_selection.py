"""Boruta-SHAP feature selection (cooperative game theory: Shapley values).

At each trial, every feature gets a "shadow" copy whose values are randomly
shuffled. A tree model is trained on real + shadow features and the mean
absolute SHAP value of each feature is computed. A real feature scores a
"hit" when its importance is higher than the best shadow feature. After all
trials, a two-sided binomial test (Bonferroni-corrected) accepts or rejects
each feature. Tentative features are kept when their median importance is
higher than the median of the best shadow importance.
"""
import numpy as np
import shap
from scipy.stats import binomtest


def _base_model(name, n_classes, seed):
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=100, num_leaves=31, random_state=seed, verbose=-1)
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)


def _mean_abs_shap(model, X):
    values = shap.TreeExplainer(model)(X, check_additivity=False).values
    values = np.abs(values)
    if values.ndim == 3:  # (samples, features, classes)
        values = values.mean(axis=2)
    return values.mean(axis=0)


class BorutaShap:
    def __init__(self, n_trials=50, alpha=0.05, shap_sample_size=2000, base_model="lightgbm", seed=46):
        self.n_trials = n_trials
        self.alpha = alpha
        self.shap_sample_size = shap_sample_size
        self.base_model = base_model
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        hits = np.zeros(p, dtype=int)
        history, shadow_max = [], []
        n_classes = len(np.unique(y))
        for t in range(self.n_trials):
            shadow = X.copy()
            for j in range(p):
                shadow[:, j] = rng.permutation(shadow[:, j])
            X_all = np.hstack([X, shadow])
            model = _base_model(self.base_model, n_classes, self.seed + t).fit(X_all, y)
            idx = rng.choice(n, size=min(self.shap_sample_size, n), replace=False)
            imp = _mean_abs_shap(model, X_all[idx])
            real, shad = imp[:p], imp[p:]
            hits += real > shad.max()
            history.append(real)
            shadow_max.append(shad.max())

        alpha = self.alpha / p  # Bonferroni correction
        self.status_ = np.empty(p, dtype=object)
        for j in range(p):
            if binomtest(hits[j], self.n_trials, 0.5, alternative="greater").pvalue < alpha:
                self.status_[j] = "accepted"
            elif binomtest(hits[j], self.n_trials, 0.5, alternative="less").pvalue < alpha:
                self.status_[j] = "rejected"
            else:
                self.status_[j] = "tentative"
        history = np.array(history)
        median_imp = np.median(history, axis=0)
        tentative_keep = (self.status_ == "tentative") & (median_imp > np.median(shadow_max))
        self.mask_ = (self.status_ == "accepted") | tentative_keep
        if not self.mask_.any():  # never return an empty feature set
            self.mask_[np.argmax(median_imp)] = True
        self.importance_ = median_imp
        self.hits_ = hits
        return self

    def transform(self, X):
        return X[:, self.mask_]
