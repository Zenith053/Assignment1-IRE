#!/usr/bin/env bash
# Q3 training queue: baseline seeds, the improved model, and its ablation arms.
#
# Ordered so the headline claim is secured first if the queue is cut short:
#   1. improved model, seed 13            (vs the seed-13 baselines already trained)
#   2. baseline + improved, seeds 14, 15   (3-seed comparison)
#   3. single-seed ablation arms
#
# Epoch caps come from the seed-13 baseline curves: MIND peaked at epoch 3,
# EB-NeRD at epoch 4. Resumable: a run whose report JSON exists is skipped.
#
#   tools/run_q3.sh                 # every run
#   tools/run_q3.sh ebnerd          # only EB-NeRD runs (e.g. one worker per GPU)
#   PYTHON=python tools/run_q3.sh   # outside the local .venv (Kaggle/Colab)
set -uo pipefail
ONLY="${1:-}"
PY="${PYTHON:-.venv/bin/python}"
cd "$(dirname "$0")/.."
mkdir -p logs

MIND="--config config/mind.yaml --epochs 3"
EB="--config config/ebnerd.yaml --epochs 4"

RUNS=(
  "mind   nrms_pop           13 $MIND --popularity"
  "ebnerd nrms_popfresh      13 $EB --popularity --freshness"
  "mind   nrms               14 $MIND"
  "mind   nrms_pop           14 $MIND --popularity"
  "ebnerd nrms               14 $EB"
  "ebnerd nrms_popfresh      14 $EB --popularity --freshness"
  "mind   nrms               15 $MIND"
  "mind   nrms_pop           15 $MIND --popularity"
  "ebnerd nrms               15 $EB"
  "ebnerd nrms_popfresh      15 $EB --popularity --freshness"
  "mind   nrms_pop_sum       13 $MIND --popularity --gate sum"
  # Added after seed 13 of pop_sum beat the gated model on MIND test: 3 seeds of
  # each so gate vs sum is chosen on *validation* AUC, never on test.
  "mind   nrms_pop_sum       14 $MIND --popularity --gate sum"
  "mind   nrms_pop_sum       15 $MIND --popularity --gate sum"
  "ebnerd nrms_pop           13 $EB --popularity"
  "ebnerd nrms_fresh         13 $EB --freshness"
  "ebnerd nrms_popfresh_sum  13 $EB --popularity --freshness --gate sum"
)

for run in "${RUNS[@]}"; do
  read -r ds tag seed args <<<"$run"
  [[ -n "$ONLY" && "$ds" != "$ONLY" ]] && continue
  report="reports/q3/${ds}_${tag}_seed${seed}.json"
  if [[ -f "$report" ]]; then
    echo "skip  $ds $tag seed $seed (report exists)"
    continue
  fi
  echo "start $ds $tag seed $seed  $(date '+%H:%M')"
  # shellcheck disable=SC2086
  "$PY" -u src/baseline/train_nrms.py $args --seed "$seed" \
    > "logs/q3_${ds}_${tag}_seed${seed}.log" 2>&1 \
    || echo "FAILED $ds $tag seed $seed (see logs/q3_${ds}_${tag}_seed${seed}.log)"
  grep -A4 "^  test (" "logs/q3_${ds}_${tag}_seed${seed}.log" | head -2
done
echo "queue done $(date '+%H:%M')"
