# Holm adjustment (Tables 12 and 17)

This folder reproduces the Holm-adjusted p-values of the paper. It is independent of
the rest of the code: it only needs `pandas` and `scipy` (`statsmodels` is used for a
cross-check if installed). It runs in a few seconds and needs no dataset.

## Families of comparisons

| File | Table | Family | m |
|---|---|---|---|
| `data/table12_cic_ids2018.csv` | Table 12 | proposed model vs. the 6 base learners, CSE-CIC-IDS2018 | 6 |
| `data/table17_unsw_nb15.csv` | Table 17 | proposed model vs. 9 published baselines, UNSW-NB15 | 9 |

Each file contains, for every comparison, the t statistic, the degrees of freedom
(df = 9, i.e. n = 10 paired fold scores), the raw two-sided p-value and Cohen's d.

## Procedure (Holm, 1979)

For one family of m comparisons and alpha = 0.05:

1. sort the m raw p-values in increasing order: p(1) <= p(2) <= ... <= p(m);
2. multiply the i-th one by (m - i + 1): the smallest by m, the next by m - 1, ..., the largest by 1;
3. enforce monotonicity: each adjusted value is at least equal to the previous one;
4. cap at 1. A comparison is significant when its adjusted p-value is below 0.05.

Raw p-values reported as "<0.0001" are replaced by the exact two-sided p-value computed
from t and df. Option `--from-t` recomputes every p-value from t and df.

## Usage

```bash
python holm/holm.py holm/data/table17_unsw_nb15.csv
python holm/holm.py holm/data/table12_cic_ids2018.csv
python holm/holm.py holm/data/table17_unsw_nb15.csv --from-t
```

The table is printed on the screen and saved in `holm/results/<file>_holm.csv`
(raw p-value used, rank, multiplier, Holm-adjusted p-value, significant).

## Expected output (Table 17)

| Comparison | raw p | × | Holm p | Significant |
|---|---|---|---|---|
| vs. KNN | <0.0001 | 9 | 0.0002 | yes |
| vs. SVM | <0.0001 | 8 | 0.0005 | yes |
| vs. LSTM | 0.0003 | 7 | 0.0021 | yes |
| vs. Autoencoder | 0.0004 | 6 | 0.0024 | yes |
| vs. 1D-CNN | 0.0009 | 5 | 0.0045 | yes |
| vs. Spiking Neural Network | 0.0017 | 4 | 0.0068 | yes |
| vs. CNN-XGBoost | 0.0107 | 3 | 0.0321 | yes |
| vs. CNN-LSTM | 0.0267 | 2 | 0.0534 | no |
| vs. Graph Convolutional Networks | 0.0804 | 1 | 0.0804 | no |

To apply the procedure to another family, create a CSV file with the same columns
(`comparison, t, df, p_value`, optional `cohen_d`).
