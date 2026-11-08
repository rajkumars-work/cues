#!/bin/bash
# Fast end-to-end plumbing check -- NOT a real experiment, and its checkpoint
# has no scientific meaning. It exercises the exact same code path every real
# HANDOFF_*.md experiment uses (tokenizer training, data loading, optimizer
# steps, checkpoint save) on a tiny architecture and a tiny data slice, so it
# finishes in well under a minute instead of hours. Run this FIRST on any new
# machine, before committing real GPU time to an actual experiment -- it's
# designed to catch exactly the kind of environment/plumbing bugs a cold
# contributor is most likely to hit (a 2026-11-08 cold-start test confirmed
# several real ones: a dependency pin that doesn't resolve, a fetch script
# that fills small disks, a missing --max_steps override that silently
# undertrains a "complete" run -- none of those need a real multi-hour run to
# catch, and this test exists so nobody has to burn one just to find out).
#
# Usage:
#   bash scripts/quick_selftest.sh
#
# What it checks, in order: environment, steering clone, corpus fetch/slice,
# a real (tiny) training run to a real saved checkpoint. Scoring/generation
# is NOT covered here -- that path has its own known gaps, tracked
# separately (see EXPERIMENTS.md's Reproducibility section).
set -u

PASS=0
FAIL=0
report() {
  if [ "$1" = "pass" ]; then
    echo "PASS: $2"; PASS=$((PASS+1))
  else
    echo "FAIL: $2"; FAIL=$((FAIL+1))
  fi
}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

echo "=== Step 0: environment ==="
PY=${PY:-python3}
if "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1; then
  report pass "python/torch import"
else
  report fail "python/torch import -- fix this before anything else"
  exit 1
fi

echo "=== Step 1: steering clone ==="
: "${STEERING_PATH:=../steering/crystals/lemat/code/py}"
STEERING_ABS="$(cd "$REPO_ROOT/$STEERING_PATH" 2>/dev/null && pwd)"
if [ -n "$STEERING_ABS" ] && [ -f "$STEERING_ABS/ed/ed_train.py" ]; then
  report pass "steering cloned and ed_train.py found"
else
  report fail "steering not found at $REPO_ROOT/$STEERING_PATH -- see SETUP.md step 1"
  exit 1
fi
export PYTHONPATH="$STEERING_ABS:$STEERING_ABS/dielectric:${PYTHONPATH:-}"

echo "=== Step 2: corpus (tiny slice, not the full file) ==="
FULL_CSV="./data/corpus/big_d15_autolabel_binrho_k7.csv"
if [ ! -f "$FULL_CSV" ]; then
  echo "Full corpus not found locally -- fetching corpus only (~200MB, not the ~14GB full fetch)..."
  if bash scripts/fetch_assets.sh --corpus-only ./data; then
    report pass "corpus fetch (--corpus-only)"
  else
    report fail "corpus fetch failed -- check disk space and network"
    exit 1
  fi
fi
TOY_CSV="./data/corpus/toy_selftest_2000.csv"
"$PY" -c "
import csv
with open('$FULL_CSV', newline='') as f_in, open('$TOY_CSV', 'w', newline='') as f_out:
    r = csv.reader(f_in)
    w = csv.writer(f_out)
    for i, row in enumerate(r):
        w.writerow(row)
        if i >= 2000:
            break
" && report pass "tiny corpus slice (2000 rows)" || { report fail "tiny corpus slice"; exit 1; }

echo "=== Step 3: tiny training run (real code path, toy architecture) ==="
CKPT_DIR="./checkpoints/toy_selftest"
rm -rf "$CKPT_DIR" "${CKPT_DIR}.log"
mkdir -p ./checkpoints

cd "$STEERING_ABS/ed" || exit 1
"$PY" -u ed_train.py \
  --data "$REPO_ROOT/$TOY_CSV" \
  --checkpoint "$REPO_ROOT/$CKPT_DIR" \
  --model_type ed --enc_layers 2 --dec_layers 2 --edim 64 --heads 2 \
  --vocab_size 500 --lr 1e-4 --warmup_frac 0.1 --grad_clip_norm 0.5 \
  --prop_dropout 0.0 --label_dropout 0.0 --max_seq_len 256 --total_batch_size 32 \
  --batch_size 4 --epochs 1 --max_steps 10 --save_interval 1000 --log_interval 2 \
  --max_keep_checkpoints 1 --prefix toy_selftest \
  > "$REPO_ROOT/${CKPT_DIR}.log" 2>&1
TRAIN_RC=$?
cd "$REPO_ROOT" || exit 1

if [ $TRAIN_RC -eq 0 ] && grep -q "Saved final checkpoint" "${CKPT_DIR}.log" 2>/dev/null; then
  report pass "tiny training run (10 steps) completed and saved a checkpoint"
else
  report fail "tiny training run did not complete cleanly -- see ${CKPT_DIR}.log"
  echo "--- last 20 lines of ${CKPT_DIR}.log ---"
  tail -20 "${CKPT_DIR}.log" 2>/dev/null
fi

echo "=== Step 4: checkpoint sanity ==="
if ls "$CKPT_DIR"/*.pt >/dev/null 2>&1 && [ -f "$CKPT_DIR/model_sp.model" ]; then
  report pass "checkpoint directory has a .pt weight file and a trained tokenizer"
else
  report fail "checkpoint directory missing expected files -- see $CKPT_DIR"
fi

echo
echo "=== Summary: $PASS passed, $FAIL failed ==="
echo "(This checkpoint is a toy, scientifically meaningless artifact -- delete"
echo " $CKPT_DIR when done. Scoring/generation is not covered by this script.)"
[ "$FAIL" -eq 0 ]
