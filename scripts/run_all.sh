#!/usr/bin/env bash
# Runs the full experimental pipeline for both datasets and both tasks.
# Usage: bash scripts/run_all.sh            (add --quick to test that everything runs)
set -e
EXTRA="$@"
for CFG in configs/unsw_nb15.yaml configs/cic_ids2018.yaml; do
  for TASK in binary multiclass; do
    python scripts/01_prepare_data.py     --config $CFG --task $TASK $EXTRA
    python scripts/02_cross_validation.py --config $CFG --task $TASK $EXTRA
    python scripts/03_final_test.py       --config $CFG --task $TASK $EXTRA
  done
  python scripts/04_statistical_tests.py  --config $CFG --task binary $EXTRA
  python scripts/05_ablation.py           --config $CFG --task binary $EXTRA
done
