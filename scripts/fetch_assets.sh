#!/bin/bash
# Fetch the C2/C3 ablation's training corpus and reference checkpoints from
# Hugging Face (all public). Requires `huggingface_hub` (pip install huggingface_hub,
# or `pip install -r ../requirements.txt`).
set -euo pipefail

DEST=${1:-./data}
mkdir -p "$DEST/corpus" "$DEST/checkpoints"

python3 - "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

dest = sys.argv[1]

snapshot_download("rajkumars47/data_big_d15", repo_type="dataset", local_dir=f"{dest}/corpus")

for repo in [
    "rajkumars47/ckpt_backup_ed_control_12_12_d15",
    "rajkumars47/ckpt_ed_enclight_6_18",
    "rajkumars47/ckpt_dec_only_17L_edim1024_d15_repl",
]:
    name = repo.split("/")[-1]
    snapshot_download(repo, local_dir=f"{dest}/checkpoints/{name}")

print(f"Fetched training corpus to {dest}/corpus and checkpoints to {dest}/checkpoints/")
PY
