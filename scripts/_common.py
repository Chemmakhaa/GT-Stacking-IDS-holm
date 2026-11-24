"""Shared helpers for the scripts (command-line arguments, paths, label encoding)."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.config import load_config, set_seed  # noqa: E402
from src.data import load_split  # noqa: E402
from src.preprocessing import encode_labels  # noqa: E402


def parse_args(description, extra=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="configs/unsw_nb15.yaml or configs/cic_ids2018.yaml")
    p.add_argument("--task", choices=["binary", "multiclass"], help="overrides the task of the config")
    p.add_argument("--quick", action="store_true", help="very small settings, only to test that the code runs")
    if extra:
        extra(p)
    args = p.parse_args()
    cfg = load_config(args.config, args.task, args.quick)
    set_seed(cfg["split"]["seed"])
    return args, cfg


def load_encoded(cfg):
    """Training and test data with integer labels (encoder fitted on training labels)."""
    X_train, y_train, X_test, y_test = load_split(cfg)
    le, y_train, y_test = encode_labels(y_train, y_test)
    class_names = [str(c) for c in le.classes_]
    if cfg["task"] == "binary":
        class_names = ["Normal", "Attack"]
    return X_train, y_train, X_test, y_test, class_names
