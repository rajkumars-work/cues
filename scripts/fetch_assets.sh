#!/bin/bash
# Fetch the C2/C3 ablation's training corpus and reference checkpoints from
# Hugging Face (all public). Requires `huggingface_hub` (pip install huggingface_hub,
# or `pip install -r ../requirements.txt`).
#
# Full fetch is ~14GB (corpus + all three reference checkpoints) -- only C2/C3
# needs the checkpoints at all. If you're running A5 (or anything else that
# only needs the training corpus) and have limited disk, pass --corpus-only:
# a 2026-11-08 cold-start test confirmed a VM with ~8GB free cannot complete
# the full fetch, and a second, separate test on 2026-11-07 hit the same wall
# and worked around it with exactly this (then-undocumented) shortcut.
set -euo pipefail

CORPUS_ONLY=0
if [ "${1:-}" = "--corpus-only" ]; then
  CORPUS_ONLY=1
  shift
fi
DEST=${1:-./data}
mkdir -p "$DEST/corpus"
[ "$CORPUS_ONLY" = 1 ] || mkdir -p "$DEST/checkpoints"

python3 - "$DEST" "$CORPUS_ONLY" <<'PY'
import sys
from huggingface_hub import snapshot_download

dest = sys.argv[1]
corpus_only = sys.argv[2] == "1"

snapshot_download("rajkumars47/data_big_d15", repo_type="dataset", local_dir=f"{dest}/corpus")

if corpus_only:
    print(f"Fetched training corpus only to {dest}/corpus (--corpus-only; skipped ~14GB of reference checkpoints)")
else:
    for repo in [
        "rajkumars47/ckpt_backup_ed_control_12_12_d15",
        "rajkumars47/ckpt_ed_enclight_6_18",
        "rajkumars47/ckpt_dec_only_17L_edim1024_d15_repl",
    ]:
        name = repo.split("/")[-1]
        snapshot_download(repo, local_dir=f"{dest}/checkpoints/{name}")
    print(f"Fetched training corpus to {dest}/corpus and checkpoints to {dest}/checkpoints/")
PY
