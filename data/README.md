# Datasets

Both datasets are public benchmarks; they are not included in this repository.

| Dataset | Source | Put the CSV files in |
|---|---|---|
| UNSW-NB15 (Moustafa & Slay, 2015) | https://research.unsw.edu.au/projects/unsw-nb15-dataset — files `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` | `data/raw/unsw_nb15/` |
| CSE-CIC-IDS2018 (Sharafaldin et al., 2018) | https://www.unb.ca/cic/datasets/ids-2018.html — "Processed Traffic Data for ML Algorithms" (CSV) | `data/raw/cic_ids2018/` |

`scripts/01_prepare_data.py` cleans the data, applies the optional stratified
subset (`subset_size` in the config) and writes the 80/20 split to
`data/processed/`. The row ids of the test set are saved in
`results/<dataset>/<task>/test_indices.csv`.
