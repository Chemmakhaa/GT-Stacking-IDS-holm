# Game theory and stacking ensemble learning for multi-stage intrusion detection

Code for the paper **"Integrating game theory and stacking ensemble learning for enhanced multi-stage intrusion detection in data centers"** (Discover Artificial Intelligence).

The framework has two stages:

1. **Game-theory stage**
   - *Cooperative game*: Boruta-SHAP feature selection (Shapley values).
   - *Adversarial minimax game*: conditional Wasserstein GAN with gradient penalty (cWGAN-GP) that generates synthetic samples for the minority classes.
2. **Ensemble stage**: six base learners (CatBoost, XGBoost, LightGBM, Gradient Boosting, AdaBoost, Random Forest) combined by an MLP meta-learner (stacking).

## Leakage-free evaluation protocol

Every step that learns from data is fitted **on the training part only** and then applied, unchanged, to validation and test data:

| Step | Fitted on | Applied to validation / test |
|---|---|---|
| Ordinal encoding + StandardScaler | training part of the fold | yes (transform only) |
| Boruta-SHAP feature mask | training part of the fold | yes (same mask) |
| cWGAN-GP | training part of the fold | **no** – test data are never augmented |
| Base learners + MLP meta-learner | training part of the fold | prediction only |

The MLP meta-learner is trained on **out-of-fold predictions**: scikit-learn's `StackingClassifier` with an inner stratified 5-fold split (`cv = 5`) makes each base learner predict samples it was not trained on; these predictions are the input of the MLP. The base learners are then refitted on the whole training data.

```text
for each outer fold (or for the final model):
    preprocessor = fit(train part)              # encoding + scaling
    mask         = BorutaSHAP(train part)       # feature selection
    gan          = cWGAN-GP(train part)         # synthetic minority samples, train part only
    model        = StackingClassifier(base learners, MLP, cv=5).fit(augmented train part)
    score        = metrics(y_val, model.predict(mask(preprocessor(val part))))
```

See `src/pipeline.py` for the implementation.

## Repository structure

```text
configs/
  unsw_nb15.yaml            dataset, split, CV, Boruta-SHAP, cWGAN-GP and model settings
  cic_ids2018.yaml
src/
  config.py                 configuration loading, random seeds
  data.py                   loading, cleaning, stratified 80/20 split
  preprocessing.py          encoding and scaling (fitted on training data only)
  feature_selection.py      Boruta-SHAP
  cwgan_gp.py               cWGAN-GP (PyTorch)
  models.py                 base learners, MLP meta-learner, stacked ensemble
  pipeline.py               leakage-free fit / predict pipeline
  metrics.py                accuracy, precision, recall, F1, macro/weighted F1, MCC, G-mean
  stats_tests.py            paired t-tests, Cohen's d, Holm adjustment
scripts/
  01_prepare_data.py        cleaning + split, test indices, environment report
  train.py                  trains the proposed model and saves it
  test.py                   evaluates the saved model on the test set (prints the results)
  02_cross_validation.py    5-fold CV on the training set
  03_final_test.py          final model, evaluation on the held-out test set
  04_statistical_tests.py   paired tests on per-fold scores + Holm
  05_ablation.py            ablation study
  run_all.sh                runs everything
holm/                       Holm adjustment of Tables 12 and 17 (standalone, see holm/README.md)
data/README.md              where to download the datasets
```

## Installation

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
pip install -r requirements.txt
```

Download the datasets as explained in `data/README.md`.

## Quick start: train and test

```bash
# 1. cleaning and stratified 80/20 split (seed 46)
python scripts/01_prepare_data.py --config configs/unsw_nb15.yaml --task binary

# 2. train the proposed model on the training set (saved to models/)
python scripts/train.py --config configs/unsw_nb15.yaml --task binary

# 3. evaluate it on the held-out test set (results printed on the screen)
python scripts/test.py --config configs/unsw_nb15.yaml --task binary
python scripts/test.py --config configs/unsw_nb15.yaml --task binary --plot   # + confusion matrix
```

## Full experiments (paper tables)

```bash
# 1. cleaning and stratified 80/20 split (seed 46)
python scripts/01_prepare_data.py --config configs/unsw_nb15.yaml --task binary

# 2. 5-fold cross-validation on the training set
python scripts/02_cross_validation.py --config configs/unsw_nb15.yaml --task binary

# 3. final training and evaluation on the held-out test set
python scripts/03_final_test.py --config configs/unsw_nb15.yaml --task binary

# 4. significance tests (10 folds, corrected resampled t-test, Holm)
python scripts/04_statistical_tests.py --config configs/cic_ids2018.yaml --task binary

# 5. ablation study
python scripts/05_ablation.py --config configs/cic_ids2018.yaml --task binary
```

Use `--task multiclass` for multi-class detection and `--quick` to check that the code runs (tiny settings, results not meaningful). `bash scripts/run_all.sh` runs everything.

## Holm adjustment only

The folder `holm/` reproduces the Holm-adjusted p-values of Tables 12 and 17 in a few seconds, without any dataset:

```bash
python holm/holm.py holm/data/table17_unsw_nb15.csv
python holm/holm.py holm/data/table12_cic_ids2018.csv
```

See `holm/README.md` for the procedure and the expected output.

## Outputs

All results are written to `results/<dataset>/<task>/`:

| File | Content |
|---|---|
| `dataset_summary.json` | number of rows, train/test sizes, class counts |
| `test_indices.csv` | row ids of the held-out test set |
| `environment.txt` | Python and library versions, hardware |
| `cv_folds_<model>.csv`, `cv_summary.csv` | metrics of each CV fold, mean and std |
| `test_metrics.json` | metrics on the held-out test set |
| `confusion_matrix.csv/.png` | confusion matrix on the test set |
| `per_class_report.csv` | per-class precision, recall, F1, one-vs-rest accuracy, support |
| `synthetic_samples_per_class.csv` | real, synthetic and total training samples per class |
| `selected_features.csv` | features kept by Boruta-SHAP |
| `fold_scores_10fold.csv` | per-fold accuracy of every model (raw data of the tests) |
| `stats_*.csv` | n, df, mean difference, t, p, Cohen's d_z, Holm-adjusted p |
| `ablation.csv` | ablation results |

## Reproducibility

- Random seed **46** for Python, NumPy, PyTorch and all models.
- Stratified 80/20 train/test split (`train_test_split(stratify=y, random_state=46)`); optional stratified subset with `subset_size`.
- 5-fold stratified CV (shuffle, seed 46) on the training set; 10 folds for the significance tests (n = 10, df = 9).
- Significance tests: corrected resampled paired t-test (Nadeau & Bengio, 2003), `t = mean(d) / sqrt((1/n + n_test/n_train) s²)`; effect size `d_z = mean(d) / s`; Holm step-down adjustment within each family of comparisons.
- All hyperparameters are in `configs/*.yaml` (Tables 1 and 2 of the paper).

Results may vary slightly with the hardware, the library versions and GPU non-determinism.

## Citation

If you use this code, please cite the paper (reference to be completed after publication).
