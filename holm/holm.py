"""Holm step-down adjustment for one family of comparisons (Tables 12 and 17).

Input: a CSV file with one row per comparison and the columns
    comparison, t, df, p_value      (cohen_d is optional and copied to the output)
A p-value written as "<0.0001" is replaced by the exact two-sided p-value
computed from t and df. With --from-t, every p-value is recomputed from t and df.

Holm procedure (Holm, 1979), for m comparisons and alpha = 0.05:
  1. sort the m raw p-values in increasing order: p(1) <= p(2) <= ... <= p(m)
  2. multiply the i-th one by (m - i + 1):  m * p(1), (m - 1) * p(2), ..., 1 * p(m)
  3. enforce monotonicity: each adjusted value is at least the previous one
  4. cap at 1; a comparison is significant when its adjusted p-value < alpha

Usage:
  python holm/holm.py holm/data/table17_unsw_nb15.csv
  python holm/holm.py holm/data/table12_cic_ids2018.csv --from-t
"""
import argparse
import os

import pandas as pd
from scipy import stats


def holm(pvalues):
    """Return the Holm-adjusted p-values, in the original order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):  # rank = 0 for the smallest p-value
        value = min(1.0, (m - rank) * pvalues[i])
        running_max = max(running_max, value)  # monotonicity
        adjusted[i] = running_max
    return adjusted


def two_sided_p(t, df):
    return 2 * stats.t.sf(abs(t), df)


def fmt(p):
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="input CSV (comparison, t, df, p_value)")
    parser.add_argument("--from-t", action="store_true", help="recompute all p-values from t and df")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"p_value": str})
    raw = []
    for _, row in df.iterrows():
        reported = str(row["p_value"]).strip()
        if args.from_t or reported.startswith("<"):
            raw.append(two_sided_p(float(row["t"]), int(row["df"])))
        else:
            raw.append(float(reported))
    adjusted = holm(raw)

    m = len(raw)
    order = sorted(range(m), key=lambda i: raw[i])
    rank = {i: r + 1 for r, i in enumerate(order)}
    df["p_raw_used"] = raw
    df["rank"] = [rank[i] for i in range(m)]
    df["multiplier"] = [m - rank[i] + 1 for i in range(m)]
    df["p_holm"] = adjusted
    df["significant"] = [p < args.alpha for p in adjusted]

    # check against statsmodels when it is installed
    try:
        from statsmodels.stats.multitest import multipletests

        ref = multipletests(raw, alpha=args.alpha, method="holm")[1]
        assert all(abs(a - b) < 1e-12 for a, b in zip(adjusted, ref)), "mismatch with statsmodels"
        check = "identical to statsmodels multipletests(method='holm')"
    except ImportError:
        check = "statsmodels not installed, check skipped"

    print(f"\nFamily: {os.path.basename(args.csv)}  (m = {m} comparisons, alpha = {args.alpha})\n")
    print(f"{'Comparison':<62}{'raw p':>9}{'rank':>6}{'x':>4}{'Holm p':>10}  Significant")
    for i in order:
        r = df.iloc[i]
        print(f"{r['comparison']:<62}{fmt(raw[i]):>9}{r['rank']:>6}{r['multiplier']:>4}"
              f"{fmt(adjusted[i]):>10}  {'yes' if r['significant'] else 'no'}")
    print(f"\nCheck: {check}")

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
    name = os.path.splitext(os.path.basename(args.csv))[0] + ("_from_t" if args.from_t else "") + "_holm.csv"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", name)
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
