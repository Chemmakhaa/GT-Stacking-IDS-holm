"""Encoding and scaling, fitted on the training part only."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler


class Preprocessor:
    """Ordinal-encodes the categorical columns and standardizes all features.

    fit() must be called on training data only; transform() is then applied,
    unchanged, to validation and test data (no information leaks from them).
    """

    def __init__(self, categorical_columns):
        self.categorical_columns = list(categorical_columns)

    def fit(self, X):
        self.cat_cols_ = [c for c in self.categorical_columns if c in X.columns]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if self.cat_cols_:
            self.encoder_.fit(X[self.cat_cols_].astype(str))
        self.scaler_ = StandardScaler().fit(self._encode(X))
        self.feature_names_ = self.cat_cols_ + self.num_cols_
        return self

    def _encode(self, X):
        parts = []
        if self.cat_cols_:
            parts.append(self.encoder_.transform(X[self.cat_cols_].astype(str)))
        parts.append(X[self.num_cols_].to_numpy(dtype=float))
        return np.hstack(parts)

    def transform(self, X):
        return self.scaler_.transform(self._encode(X)).astype(np.float32)


def encode_labels(y_train, *others):
    """Label-encode the target (fitted on the training labels)."""
    le = LabelEncoder().fit(y_train)
    return (le, le.transform(y_train)) + tuple(le.transform(y) for y in others)
