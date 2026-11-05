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

This is a curated subset (training + eval deps), not a full lockfile — it's not
guaranteed to byte-match the environment the original ablation ran in.
`steering`'s own `crystals/requirements.txt` is the fuller, known-working
reference for the eval/MLIP-scoring side specifically, if something here doesn't
resolve cleanly.

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

## Known gaps (as of this scaffolding pass)

- **Not yet smoke-tested end-to-end** on this exact file layout. Test on a single
  GPU before handing off to anyone else.
- `requirements.txt` here is hand-curated, not a lockfile; `ed/`'s own
  `requirements.txt` (in `steering`'s vendored copy) doesn't pin `torch` at all —
  the original runs used a specific GCE DLVM image
  (`pytorch-2-9-cu129-ubuntu-2204-nvidia-580`, PyTorch 2.9.1+cu129). Match the
  CUDA build to your own hardware.
- `scripts/train_arm.sh`'s watchdog/stall-detection logic is adapted from an
  existing internal launcher (`cues/scripts/nxn/train_supervisor.sh`) but
  simplified for a single continuous run rather than the original's segmented
  per-epoch respawn — see the design doc §5.2 for why segmented respawns are
  specifically the thing to avoid (they caused two real LR-schedule bugs).
