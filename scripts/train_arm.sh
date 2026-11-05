#!/bin/bash
# Single-machine training launcher for the CUES architecture ablation (C2/C3):
# ed_control (12+12), ed_lite (6+18), dec_only (17L, edim=1024).
#
# Adapted from an existing internal launcher's resume/watchdog pattern
# (startup-grace + stall detection + checkpoint-based completion check),
# generalized to accept which architecture arm to train and simplified to one
# continuous invocation per arm rather than a segmented per-epoch respawn —
# see ../docs/dec_only_17L_replication_design.md §5 for why that distinction
# matters (the original run hit two real LR-schedule bugs from segmenting).
#
# Hyperparameters are matched across all three arms per the design doc §3 —
# only architecture flags differ between arms. This matching is the point:
# an earlier study's cross-arm LR mismatch is exactly the confound this avoids.
#
# Requires ed_train.py on PYTHONPATH: clone ../steering (see ../SETUP.md) and
# either export STEERING_PATH yourself or rely on the default below.
#
# This script does NOT self-detach. For a real multi-hour/multi-day run that
# must survive the invoking shell exiting (including a shell a Claude Code
# tool call spawned and then returned from -- plain `nohup ... &` is not
# always enough to survive that), launch it fully decoupled:
#   setsid nohup bash scripts/train_arm.sh ed_lite 0 >/dev/null 2>&1 < /dev/null &
# (matches the existing internal convention in cues/scripts/nxn/launch_train_resilient.sh)
set -u

ARM=${1:?"usage: train_arm.sh <ed_control|ed_lite|dec_only> [seed]"}
SEED=${2:-0}

: "${STEERING_PATH:=../steering/crystals/lemat/code/py}"
: "${DATA_CSV:=./data/corpus/big_d15_autolabel_binrho_k7.csv}"
: "${CKPT_ROOT:=./checkpoints}"
: "${BATCH_SIZE:=24}"   # L4 ceiling for the largest arm (dec_only); raise on bigger GPUs
: "${EPOCHS:=150}"
: "${SAVE_INTERVAL:=1000}"  # lower this (e.g. 50) for a quick smoke test that needs to see a checkpoint sooner
: "${LOG_INTERVAL:=50}"
: "${PY:=python3}"

export PYTHONUNBUFFERED=1
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Resolve STEERING_PATH (default "../steering/...") relative to REPO_ROOT by
# actually cd-ing there, rather than string-stripping "../" -- steering is a
# SIBLING of this repo, not nested inside it, and naive prefix-stripping
# silently produced a wrong, nonexistent path here before.
STEERING_ABS="$(cd "$REPO_ROOT/$STEERING_PATH" && pwd)" || {
  echo "STEERING_PATH does not resolve to a real directory: $REPO_ROOT/$STEERING_PATH" >&2
  exit 1
}
# ed_data.py does `from dielectric_data.reader import ...` -- a top-level
# import, so code/py/dielectric itself (one level deeper than code/py) must
# also be on PYTHONPATH, not just code/py.
export PYTHONPATH="$STEERING_ABS:$STEERING_ABS/dielectric:${PYTHONPATH:-}"

ED_DIR="$STEERING_ABS/ed"
if [ ! -f "$ED_DIR/ed_train.py" ]; then
  echo "ed_train.py not found at $ED_DIR/ed_train.py -- clone steering first (see SETUP.md)" >&2
  exit 1
fi

case "$ARM" in
  ed_control) ARCH_FLAGS="--model_type ed --layers 12 --edim 768 --heads 12" ;;
  ed_lite)    ARCH_FLAGS="--model_type ed --enc_layers 6 --dec_layers 18 --edim 768 --heads 12" ;;
  dec_only)   ARCH_FLAGS="--model_type d --layers 17 --edim 1024 --heads 16" ;;
  *) echo "unknown arm: $ARM (expected ed_control | ed_lite | dec_only)" >&2; exit 1 ;;
esac

CKPT_ROOT_ABS="$REPO_ROOT/${CKPT_ROOT#./}"
CKPT="$CKPT_ROOT_ABS/${ARM}_s${SEED}"
LOG="${CKPT}.log"
mkdir -p "$CKPT_ROOT_ABS"

DATA_CSV_ABS="$REPO_ROOT/${DATA_CSV#./}"

# Matched across all three arms -- see ../docs/dec_only_17L_replication_design.md §3.
COMMON_FLAGS="--vocab_size 16000 --lr 5e-5 --warmup_frac 0.15 --grad_clip_norm 0.5 \
--prop_dropout 0.2 --label_dropout 0.0 --max_seq_len 512 --total_batch_size 524288 \
--epochs $EPOCHS --lr_schedule_epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
--cond_probe_interval 0 --num_workers 4 --log_interval $LOG_INTERVAL --save_interval $SAVE_INTERVAL \
--max_keep_checkpoints 3 --prefix $ARM"

STALL_SEC=900       # once stepping, no log progress this long = genuine stall
STARTUP_GRACE=3000  # tolerate slow tokenizer/startup before the first new step
MAX_RESTARTS=50

cd "$ED_DIR" || exit 1

is_done() { grep -q "Saved final checkpoint" "$LOG" 2>/dev/null; }
mtime()   { stat -f %m "$LOG" 2>/dev/null || stat -c %Y "$LOG" 2>/dev/null || echo 0; }

for attempt in $(seq 1 $MAX_RESTARTS); do
  if is_done; then echo "=== [train_arm $ARM s$SEED] DONE ===" | tee -a "$LOG"; break; fi
  echo "=== [train_arm $ARM s$SEED] attempt $attempt $(date -u +%FT%TZ): checkpoint=$CKPT ===" >> "$LOG"

  "$PY" -u ed_train.py --data "$DATA_CSV_ABS" --checkpoint "$CKPT" $ARCH_FLAGS $COMMON_FLAGS \
    >> "$LOG" 2>&1 &
  TPID=$!

  base_lines=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  stepped=0
  while kill -0 "$TPID" 2>/dev/null; do
    sleep 30
    now_lines=$(wc -l < "$LOG" 2>/dev/null || echo 0)
    [ "$now_lines" -gt "$base_lines" ] 2>/dev/null && stepped=1
    age=$(( $(date +%s) - $(mtime) ))
    limit=$STARTUP_GRACE; [ "$stepped" = 1 ] && limit=$STALL_SEC
    if [ "$age" -gt "$limit" ]; then
      echo "=== [train_arm $ARM s$SEED] STALL ${age}s > ${limit}s (stepped=$stepped); killing $TPID ===" >> "$LOG"
      kill -9 "$TPID" 2>/dev/null; sleep 8; break
    fi
  done
  wait "$TPID" 2>/dev/null

  if is_done; then echo "=== [train_arm $ARM s$SEED] DONE ===" >> "$LOG"; break; fi
  echo "=== [train_arm $ARM s$SEED] attempt $attempt ended; restarting in 10s ===" >> "$LOG"
  sleep 10
done
