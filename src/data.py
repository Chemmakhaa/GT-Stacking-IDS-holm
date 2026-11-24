"""Loading, cleaning and splitting of UNSW-NB15 and CSE-CIC-IDS2018.

Only operations that do not learn anything from the data are done here
(removing duplicates, infinite and missing values). Everything that is
learned (encoding, scaling, feature selection, cWGAN-GP) is fitted later,
on the training part only (see src/pipeline.py).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _read_files(patterns):
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise FileNotFoundError(f"No CSV file found for {patterns}. See data/README.md.")
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_and_clean(cfg):
    """Return the cleaned dataframe with two target columns: y_binary and y_multi."""
    d = cfg["dataset"]
    df = _read_files(d["files"])

    # CIC-IDS2018 CSV files repeat the header line inside the data
    if d["multiclass_label"] in df.columns:
        df = df[df[d["multiclass_label"]] != d["multiclass_label"]]

    df = df.drop(columns=[c for c in d.get("drop_columns", []) if c in df.columns])

    # targets
    df["y_multi"] = df[d["multiclass_label"]].astype(str).str.strip()
    df.loc[df["y_multi"].isin(["", "nan", "-"]), "y_multi"] = "Normal"
    if d.get("binary_label"):
        df["y_binary"] = df[d["binary_label"]].astype(int)
    else:
        df["y_binary"] = (df["y_multi"] != d.get("benign_value", "Benign")).astype(int)
    df = df.drop(columns=[c for c in [d["multiclass_label"], d.get("binary_label")] if c and c in df.columns])

    # numeric conversion, infinite and missing values, duplicates
    cat_cols = [c for c in d.get("categorical_columns", []) if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols + ["y_binary", "y_multi"]]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates().reset_index(drop=True)

    # remove constant columns (no information)
    const = [c for c in num_cols if df[c].nunique() <= 1]
    df = df.drop(columns=const)
    return df


def make_split(df, cfg):
    """Optional stratified subset, then stratified 80/20 train/test split (seed 46)."""
    seed = cfg["split"]["seed"]
    target = "y_binary" if cfg["task"] == "binary" else "y_multi"
    subset = cfg["dataset"].get("subset_size")
    if subset and subset < len(df):
        df, _ = train_test_split(df, train_size=subset, stratify=df[target], random_state=seed)
    train_df, test_df = train_test_split(
        df, test_size=cfg["split"]["test_size"], stratify=df[target], random_state=seed
    )
    return train_df, test_df


def save_split(train_df, test_df, cfg):
    out = os.path.join("data", "processed", cfg["dataset"]["name"], cfg["task"])
    os.makedirs(out, exist_ok=True)
    train_df.to_csv(os.path.join(out, "train.csv.gz"), index=True, index_label="row_id")
    test_df.to_csv(os.path.join(out, "test.csv.gz"), index=True, index_label="row_id")
    pd.Series(test_df.index, name="row_id").to_csv(os.path.join(cfg["output_dir"], "test_indices.csv"), index=False)

    target = "y_binary" if cfg["task"] == "binary" else "y_multi"
    summary = {
        "n_rows_total": int(len(train_df) + len(test_df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(train_df.shape[1] - 2),
        "train_class_counts": train_df[target].value_counts().to_dict(),
        "test_class_counts": test_df[target].value_counts().to_dict(),
    }
    summary = json.loads(json.dumps(summary, default=str))
    with open(os.path.join(cfg["output_dir"], "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_split(cfg):
    """Read the saved split. Returns X_train, y_train, X_test, y_test (y as strings or ints)."""
    path = os.path.join("data", "processed", cfg["dataset"]["name"], cfg["task"])
    if not os.path.exists(os.path.join(path, "train.csv.gz")):
        raise FileNotFoundError("Run scripts/01_prepare_data.py first.")
    target = "y_binary" if cfg["task"] == "binary" else "y_multi"
    out = []
    for part in ("train", "test"):
        df = pd.read_csv(os.path.join(path, f"{part}.csv.gz"), index_col="row_id", low_memory=False)
        y = df[target].to_numpy()
        X = df.drop(columns=["y_binary", "y_multi"])
        out += [X, y]
    return out
