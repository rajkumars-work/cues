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

**Before going further, run the quick self-test:**

```bash
bash scripts/quick_selftest.sh
```

Takes under a minute on a GPU machine. It exercises the exact same code path
(tokenizer training, data loading, optimizer steps, checkpoint saving) a real
experiment uses, but on a tiny 2000-row data slice and a toy architecture, so
it catches environment/plumbing problems — a bad dependency pin, a disk-space
wall, a broken training flag — before you commit real GPU time to anything in
`EXPERIMENTS.md`. A 2026-11-08 cold-start test hit several real bugs this
would have caught in seconds instead of after a multi-hour run; if this
script fails, fix that before attempting any real experiment below, since a
real run will fail the same way after costing much more time to find out.

## 3. Fetch data and reference checkpoints

```bash
bash scripts/fetch_assets.sh ./data
```

Pulls the training corpus and the three original architecture-ablation
checkpoints from Hugging Face (all public — see the main README's table).
**This is ~14GB total.** If you only need the training corpus (e.g. for A5,
which doesn't use the reference checkpoints at all) or have limited disk,
use `bash scripts/fetch_assets.sh --corpus-only ./data` instead — confirmed
necessary on an 8GB-free VM during a 2026-11-08 cold-start test.

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

**No Hugging Face account, and don't want to set one up just for this?** Ask
the maintainer for a token instead. They'll mint a fine-grained HF access
token scoped to write-only on a single dedicated repo
(`rajkumars47/cues-contrib-dropbox`) and send it to you directly — nothing
else on the account is reachable with it. Use it the same way you'd use any
other HF token:

```bash
export HF_TOKEN=<the token they sent you>
```

then upload into that repo with the same
`huggingface_hub.upload_folder`/`upload_file` calls, pointed at
`rajkumars47/cues-contrib-dropbox` instead of a repo of your own —
`repo_type="dataset"` (it's a dataset repo, not a model repo; pass that
explicitly rather than letting the default guess wrong). No account,
no signup, no invite to accept. Treat the token string itself as a secret —
don't paste it into a PR, a committed file, or anywhere else it could leak.
This same token is shared across everyone currently contributing this way
(not a fresh one per person) and goes away once the maintainer deletes it
after this round of experiments wraps up — don't assume it's permanent or
yours alone.

- **Fixed 2026-11-08, found running a real A5 (`ed_lite`) training arm on a
  single L4 GPU:** `train_arm.sh`'s watchdog had a hardcoded `STALL_SEC=900`
  (no override). With `grad_accum=42` micro-batches per logged step, one real
  step there takes ~920s — just over the old threshold — so the watchdog
  killed the run as a false-positive "stall" on *every* attempt, right before
  step 2 could log, looping at step 1 for hours with zero progress (confirmed
  via `py-spy` that the process was genuinely computing, not hung). Default
  raised to 1800s and both `STALL_SEC`/`STARTUP_GRACE` are now overridable env
  vars like the rest of this script's tunables — if a bigger arm or smaller
  GPU makes 1800s too tight, set `STALL_SEC=<higher>` rather than editing the
  script.

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
- **Fixed 2026-11-08, found by a cold-start reproducibility test:**
  `train_arm.sh` was launching every run with `ed_train.py`'s hardcoded
  default `--max_steps 4000` in effect (never overridden), so a "full,
  150-epoch" run actually stopped around 28 epochs (~19%) with the LR
  schedule still near its peak, while still reporting as a clean, complete
  run. Now passes `--max_steps 0`, `ed_train.py`'s own documented way to
  derive the real step budget from `--epochs` instead. Verified on real
  hardware (a single L4): the diagnostic line now reads `max_steps:32250`
  for a 150-epoch `ed_lite` run, matching this project's own historical
  `dec_only_17L_repl` checkpoint's actual training length, instead of the
  previous `max_steps:4000`. **If you've previously treated any checkpoint
  trained via this script's pre-fix version as a complete 150-epoch run, it
  likely wasn't — re-check against its log's own `max_steps:` line.**
- **Also fixed 2026-11-08:** the `--seed`-unsupported warning (above) used to
  go only to this script's own stderr, which the documented launch command
  discards (`>/dev/null 2>&1`) — now also written into the run's own log
  file, so it's visible however you end up reading the run's output.
- **Open question, not yet resolved (found 2026-11-08):** a short training
  run surfaced `ed_train.py`'s own runtime warning — "Target/source ratio
  13x > 10x but --encoder_lr_scale is 1.0 ... Consider --encoder_lr_scale
  0.27" — for `ed_lite`'s actual token-ratio. `--encoder_lr_scale` defaults
  to `1.0` (no scaling) and `train_arm.sh` never sets it. Whether this
  project's own historical `ed_lite` results were trained with this
  unaddressed, and whether applying the suggested scale would change
  anything material, hasn't been checked — flagging rather than changing the
  default, since this could affect comparability with already-published
  numbers if changed without checking first. The warning's own pointer to an
  `ARCHITECTURE.md` for more detail doesn't resolve to a file anywhere in
  this repo or `steering`, for what that's worth.
