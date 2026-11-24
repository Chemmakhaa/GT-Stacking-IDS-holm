"""Train the proposed model on the training set and save it.

Pipeline fitted on the training data only:
  encoding + scaling -> Boruta-SHAP -> cWGAN-GP (minority classes) -> stacking (6 base learners + MLP)

Run scripts/01_prepare_data.py first (it creates the 80/20 split).

Usage:
  python scripts/train.py --config configs/unsw_nb15.yaml --task binary
The trained model is saved to models/<dataset>_<task>.joblib
"""
import os
import time

import joblib
from _common import load_encoded, parse_args

from src.pipeline import fit_pipeline

args, cfg = parse_args(__doc__)
X_train, y_train, _, _, class_names = load_encoded(cfg)
print(f"Training set: {len(X_train)} samples, {X_train.shape[1]} features, classes: {class_names}")

start = time.time()
pipe = fit_pipeline(X_train, y_train, cfg, seed=cfg["split"]["seed"], model="stacking")
print(f"Training finished in {time.time() - start:.0f} s")

print(f"\nFeatures selected by Boruta-SHAP ({len(pipe.feature_names)}): {', '.join(pipe.feature_names)}")
if pipe.synthetic_report:
    print("\nTraining samples per class (real + synthetic from cWGAN-GP):")
    print(f"{'Class':<20}{'Real':>10}{'Synthetic':>12}{'Total':>10}")
    for r in pipe.synthetic_report:
        print(f"{class_names[int(r['class'])]:<20}{r['real']:>10}{r['synthetic']:>12}{r['total']:>10}")

os.makedirs("models", exist_ok=True)
path = os.path.join("models", f"{cfg['dataset']['name']}_{cfg['task']}.joblib")
joblib.dump({"pipeline": pipe, "class_names": class_names}, path)
print(f"\nModel saved to {path}")
