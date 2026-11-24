"""Leakage-free pipeline: everything is fitted on the training data only.

    fit_pipeline(X_train, y_train)  ->  preprocessor, Boruta-SHAP mask, cWGAN-GP, classifier
    predict(pipe, X)                ->  applies the fitted steps (no refit) and predicts

Used inside each cross-validation fold and for the final train/test evaluation,
so validation and test data never influence any fitted parameter.
"""
from dataclasses import dataclass, field

import numpy as np

from .cwgan_gp import CWGANGP
from .feature_selection import BorutaShap
from .models import single_model, stacked_ensemble
from .preprocessing import Preprocessor


@dataclass
class FittedPipeline:
    preprocessor: Preprocessor
    mask: np.ndarray
    model: object
    feature_names: list
    synthetic_report: list = field(default_factory=list)
    boruta_status: list = field(default_factory=list)


def fit_pipeline(X_train, y_train, cfg, seed=46, use_boruta=True, use_gan=True, model="stacking"):
    """y_train must be integer-encoded. `model` = "stacking", "voting" or a base-learner name."""
    pre = Preprocessor(cfg["dataset"].get("categorical_columns", [])).fit(X_train)
    X = pre.transform(X_train)
    y = np.asarray(y_train)
    mask = np.ones(X.shape[1], dtype=bool)
    report, status = [], []

    for step in cfg["pipeline"]["order"]:
        if step == "boruta_shap" and use_boruta:
            bs = BorutaShap(**cfg["boruta_shap"], seed=seed).fit(X, y)
            mask = bs.mask_
            status = list(bs.status_)
            X = bs.transform(X)
        elif step == "cwgan_gp" and use_gan:
            gan = CWGANGP(**cfg["cwgan_gp"], seed=seed).fit(X, y)
            X, y, report = gan.balance(X, y)

    n_classes = len(np.unique(y))
    if model in ("stacking", "voting"):
        clf = stacked_ensemble(cfg, n_classes, seed, meta="mlp" if model == "stacking" else "voting")
    else:
        clf = single_model(model, cfg, n_classes, seed)
    clf.fit(X, y)
    names = [n for n, keep in zip(pre.feature_names_, mask) if keep]
    return FittedPipeline(pre, mask, clf, names, report, status)


def predict(pipe, X):
    Z = pipe.preprocessor.transform(X)[:, pipe.mask]
    return pipe.model.predict(Z)


def predict_proba(pipe, X):
    Z = pipe.preprocessor.transform(X)[:, pipe.mask]
    return pipe.model.predict_proba(Z)
