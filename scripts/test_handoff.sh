#!/bin/bash
# Tests whether the training experiments in EXPERIMENTS.md actually start,
# load their real data, and can save a checkpoint -- the same three-point
# check described in cues/docs/how_to_handoff.md Part 2, as an actual script
# instead of prose someone has to interpret.
#
# What this checks, per experiment arm: (1) the documented launch command
# runs with no crash and logs at least one real training step; (2) the
# documented DATA_CSV file exists; (3) the checkpoint-save code path runs
# and produces a real file on disk. It does NOT check training quality --
# a few seconds to minutes of stepping is enough to prove the pipeline
# works, nothing about final loss or scores.
#
# What this does NOT cover: the near-equilibrium rebuild
# (HANDOFF_NEAR_EQUILIBRIUM_REBUILD.md) is eval-only -- generate, relax,
# score -- with no single launcher script like train_arm.sh, so it isn't
# included here. Verify that one by hand per its own doc.
#
# Usage: run from the cues/public repo root, after SETUP.md steps 1-3
# (clone ../steering, pip install, fetch data) are already done:
#   bash scripts/test_handoff.sh
#
# Exits 0 if every arm passes, 1 if any arm fails or times out.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

TIMEOUT_SEC=480          # 8 minutes per arm -- under the 10-minute budget
POLL_SEC=15
TEST_TOTAL_BATCH_SIZE=12288   # == BATCH_SIZE * max_seq_len -> grad_accum=1, so a
                               # step takes seconds instead of ~900s. Test-only;
                               # never use this for a real experiment run.
TEST_SAVE_INTERVAL=2          # force a checkpoint save within the time budget

# arm_name : seed : data_csv : checkpoint_prefix
ARMS=(
  "ed_control:0:./data/corpus/big_d15_autolabel_binrho_k7.csv"
  "ed_lite:0:./data/corpus/big_d15_autolabel_binrho_k7.csv"
  "dec_only:0:./data/corpus/big_d15_autolabel_binrho_k7.csv"
  "ed_lite:10:./data/corpus/big_d15_lowhull.csv"
  "ed_lite:11:./data/corpus/big_d15_wide_matched.csv"
)

RESULTS=()

cleanup_processes() {
  pkill -f "[t]rain_arm.sh" 2>/dev/null
  sleep 1
  pkill -9 -f "[e]d_train.py" 2>/dev/null
  sleep 1
}

run_one_arm() {
  local arm="$1" seed="$2" data_csv="$3"
  local ckpt_dir="./checkpoints/${arm}_s${seed}"
  local log_file="./checkpoints/${arm}_s${seed}.log"

  echo "=================================================================="
  echo "ARM: $arm  SEED: $seed  DATA: $data_csv"
  echo "=================================================================="

  if [ ! -f "$data_csv" ]; then
    echo "FAIL: data file does not exist: $data_csv"
    RESULTS+=("$arm|$seed|FAIL (missing data: $data_csv)")
    return
  fi

  cleanup_processes
  rm -rf "$ckpt_dir" "$log_file"

  DATA_CSV="$data_csv" TOTAL_BATCH_SIZE="$TEST_TOTAL_BATCH_SIZE" \
    SAVE_INTERVAL="$TEST_SAVE_INTERVAL" \
    bash scripts/train_arm.sh "$arm" "$seed" >/dev/null 2>&1 &
  local driver_pid=$!

  local elapsed=0
  local status="TIMEOUT after ${TIMEOUT_SEC}s"
  while [ "$elapsed" -lt "$TIMEOUT_SEC" ]; do
    sleep "$POLL_SEC"
    elapsed=$((elapsed + POLL_SEC))

    if ls "$ckpt_dir"/*.pt >/dev/null 2>&1; then
      status="PASS (checkpoint saved after ~${elapsed}s)"
      break
    fi
    if grep -qi "traceback\|out of memory" "$log_file" 2>/dev/null; then
      status="FAIL (crash in log after ~${elapsed}s -- see $log_file)"
      break
    fi
    if ! kill -0 "$driver_pid" 2>/dev/null; then
      status="FAIL (train_arm.sh exited early after ~${elapsed}s -- see $log_file)"
      break
    fi
  done

  echo "RESULT: $status"
  tail -5 "$log_file" 2>/dev/null
  RESULTS+=("$arm|$seed|$status")

  cleanup_processes
  rm -rf "$ckpt_dir" "$log_file"
}

for entry in "${ARMS[@]}"; do
  IFS=':' read -r arm seed data_csv <<< "$entry"
  run_one_arm "$arm" "$seed" "$data_csv"
done

echo
echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
overall=0
for r in "${RESULTS[@]}"; do
  IFS='|' read -r arm seed status <<< "$r"
  printf "%-12s seed=%-4s %s\n" "$arm" "$seed" "$status"
  case "$status" in
    PASS*) ;;
    *) overall=1 ;;
  esac
done

if [ "$overall" -eq 0 ]; then
  echo "All arms passed."
else
  echo "At least one arm failed -- see above."
fi
exit "$overall"
