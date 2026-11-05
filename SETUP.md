# Setup

## 1. Clone the companion `steering` repo alongside this one

```bash
git clone https://github.com/rajkumars-work/steering.git ../steering
```

The training/eval code lives at `../steering/crystals/lemat/code/py/{ed,dielectric}/`
— `ed_train.py`, `ed_model.py`, `d_model.py`, `ed_data.py`, `configs.py` (training),
and `screening.py`, `combined.py` (eval/scoring). This repo's scripts add
`../steering/crystals/lemat/code/py` to `PYTHONPATH` so `import ed` / `import
dielectric` resolve from there.

## 2. Environment

```bash
pip install -r requirements.txt
```

This installs training deps only — torch is deliberately unpinned (see the file's
own header comment: check for a working CUDA torch already on your machine
before installing a new one). Install `requirements-eval-optional.txt` separately,
once training works, only if you need to score checkpoints (it pulls in
mace-torch/e3nn, which have their own strict torch pins).

Neither file is a full lockfile — not guaranteed to byte-match the environment
the original ablation ran in. `steering`'s own `crystals/requirements.txt` is a
fuller, known-working reference for the eval/MLIP-scoring side specifically, if
something here doesn't resolve cleanly.

## 3. Fetch data and reference checkpoints

```bash
bash scripts/fetch_assets.sh ./data
```

Pulls the training corpus and the three original architecture-ablation
checkpoints from Hugging Face (all public — see the main README's table).

## 4. Train one arm

```bash
STEERING_PATH=../steering/crystals/lemat/code/py \
DATA_CSV=./data/corpus/big_d15_autolabel_binrho_k7.csv \
bash scripts/train_arm.sh ed_lite 0
```

First argument is the arm (`ed_control` | `ed_lite` | `dec_only`), second is the
seed. See `scripts/train_arm.sh`'s header for the full set of overridable env
vars (`BATCH_SIZE`, `EPOCHS`, `CKPT_ROOT`, `PY`).

**Before launching, read `docs/dec_only_17L_replication_design.md` §2 and §6.**
The decoder-only arm in particular silently fails to apply qk-norm to
self-attention unless that fix (already landed in `steering`'s vendored
`ed_model.py`, per the design doc) is the version you're actually running —
verify this before trusting a `dec_only` run's numbers.

## 5. Score a checkpoint

```python
import sys
sys.path.insert(0, "../steering/crystals/lemat/code/py")
sys.path.insert(0, "./eval")
import path_override                    # must come before importing screening
from dielectric.eval import screening
screening.ED_CHECKPOINTS = path_override.ED_CHECKPOINTS
```

Then use `screening.load_model()` / `screening.generate_batch()` /
`screening.screen_batch()` as documented in `steering`'s own `crystals/README.md`
and `RESULTS.md`. Set `ED_CHECKPOINTS=/path/to/your/checkpoints` as an
environment variable before running, or it defaults to the original author's
hardcoded path (see `eval/path_override.py`'s docstring).

## 6. Report results back

You almost certainly do **not** have push access to this repo directly — don't
assume it. Two options, in order of preference:

1. **Fork + pull request (preferred).** Fork `rajkumars-work/cues` to your own
   GitHub account, push your results (a report, logs, anything small — not
   checkpoints, see below) to a branch on your fork, and open a pull request
   against this repo. This is the normal open-source contribution flow and
   needs nothing from the maintainer in advance. If you don't already have git
   push credentials configured for your own GitHub account on the machine
   you're running on, set those up first (an SSH key or a personal access
   token) — this is independent of anything in this repo.
2. **Relay directly, if you'd rather not set up git credentials for a one-off
   run.** Write your report (plain markdown is fine) and send it to the
   maintainer however you're already in contact with them — paste it into a
   shared conversation, email it, whatever's already working. Confirmed to
   work in practice: the smoke test in `smoketests/ed_lite_20261105.md` was
   returned exactly this way.

**Checkpoints are large (hundreds of MB to several GB) — don't try to put them
in a git branch or PR either way.** Upload large files to your own Hugging Face
account (`huggingface_hub.upload_folder`/`upload_file`, same pattern as
`scripts/fetch_assets.sh` uses to download) and just link the HF repo in your
report. Don't request or expect write access to the `rajkumars47` HF account —
host large files under your own account instead.

## Known gaps (as of this scaffolding pass)

- **Smoke test completed successfully end-to-end (2026-11-05)**, including all
  the fixes below: cloned both repos, fetched the corpus from HF, trained
  `ed_lite` to step 50 with a real, monotonically-decreasing loss, saved a
  genuine checkpoint (3.8GB — includes optimizer + EMA state, not just model
  weights), and stopped it cleanly. Everything below was found and fixed
  getting to that point.
- The vendored `steering` copy of `ed_train.py` doesn't support `--seed` as of
  this writing (main lineage does) — `train_arm.sh` now probes for this and
  warns instead of crashing; see its own comment for detail.
- **To stop a run started by `train_arm.sh`, killing only the wrapper script
  (`pkill -f train_arm.sh`) is not sufficient** — `ed_train.py` runs as its
  own background process and will keep holding the GPU. Kill it directly, but
  avoid a naive `pkill -f "ed_train.py.*--prefix <arm>"` if you're running the
  kill command itself via a single remote-exec string (e.g.
  `ssh host "pkill -f ... "`, `gcloud compute ssh --command "..."`) — the
  invoking process's own command line then *literally contains* the search
  pattern, so `pkill -f` matches and kills that process too, which can take
  the whole remote session down (confirmed: this killed an active
  `gcloud compute ssh` session outright during this smoke test). Use the
  standard bracket trick to avoid self-matching:
  `pkill -f "[e]d_train.py.*--prefix <arm>"`. This generalizes beyond this one
  command: any check/kill run as a single remote-exec string should avoid
  putting the literal search pattern anywhere else in that same string either
  (including in an `echo`/fallback message), or it'll self-match there too.
- `scripts/train_arm.sh`'s watchdog/stall-detection logic is adapted from an
  existing internal launcher (`cues/scripts/nxn/train_supervisor.sh`) but
  simplified for a single continuous run rather than the original's segmented
  per-epoch respawn — see the design doc §5.2 for why segmented respawns are
  specifically the thing to avoid (they caused two real LR-schedule bugs).
