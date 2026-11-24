"""Base learners, MLP meta-learner and stacked ensemble (Table 2 of the paper)."""
import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """MLP meta-learner: one hidden layer (128, ReLU), dropout 0.3, Adam (lr 0.001),
    batch size 128, 50 epochs. Compatible with scikit-learn (fit / predict_proba)."""

    def __init__(self, hidden=128, dropout=0.3, learning_rate=1e-3, batch_size=128, epochs=50, seed=46):
        self.hidden = hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        self.classes_ = np.unique(y)
        yi = torch.tensor(np.searchsorted(self.classes_, y))
        self.net_ = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
                                  nn.Linear(self.hidden, len(self.classes_)))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(X))
            for start in range(0, len(X), self.batch_size):
                idx = perm[start:start + self.batch_size]
                opt.zero_grad()
                loss_fn(self.net_(X[idx]), yi[idx]).backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.net_.eval()
        logits = self.net_(torch.tensor(np.asarray(X, dtype=np.float32)))
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def base_learners(cfg, n_classes, seed=46):
    m = cfg["models"]
    binary = n_classes == 2
    return [
        ("catboost", CatBoostClassifier(**m["catboost"], loss_function="Logloss" if binary else "MultiClass",
                                        random_seed=seed, verbose=0, thread_count=-1,
                                        allow_writing_files=False)),
        ("xgboost", XGBClassifier(**m["xgboost"], random_state=seed, n_jobs=-1, verbosity=0,
                                  eval_metric="logloss" if binary else "mlogloss")),
        ("lightgbm", LGBMClassifier(**m["lightgbm"], random_state=seed, n_jobs=-1, verbose=-1)),
        ("gradient_boosting", GradientBoostingClassifier(**m["gradient_boosting"], random_state=seed)),
        ("adaboost", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                                        **m["adaboost"], random_state=seed)),
        ("random_forest", RandomForestClassifier(**m["random_forest"], random_state=seed, n_jobs=-1)),
    ]


def meta_learner(cfg, seed=46):
    return TorchMLPClassifier(**cfg["models"]["mlp_meta"], seed=seed)


def stacked_ensemble(cfg, n_classes, seed=46, meta="mlp"):
    """Proposed model. StackingClassifier builds the meta-features with an inner
    stratified K-fold (out-of-fold predictions), then refits the base learners
    on the full training data."""
    estimators = base_learners(cfg, n_classes, seed)
    if meta == "voting":  # ablation "w/o MLP stacking": soft voting instead of the MLP
        return VotingClassifier(estimators, voting="soft", n_jobs=1)
    inner_cv = StratifiedKFold(n_splits=cfg["cv"]["stacking_cv"], shuffle=True, random_state=seed)
    return StackingClassifier(estimators=estimators, final_estimator=meta_learner(cfg, seed),
                              cv=inner_cv, stack_method="predict_proba", n_jobs=1)


def single_model(name, cfg, n_classes, seed=46):
    """One base learner alone (used for the comparisons of Table 12)."""
    return dict(base_learners(cfg, n_classes, seed))[name]
