# Handoff: the C2/C3 architecture ablation

**Priority: NICE — not blocking, downgraded from MUST on 2026-11-04.**
Neither baseline this experiment compares (decoder-only, naive 12+12) is
the architecture this paper recommends, so resolving the comparison either
way doesn't change the paper's argument. No longer in
`cues/docs/limitations.md` — worth running if compute is free, not
something this paper's conclusions wait on.

**You have no prior context — read this whole file before touching anything.**
This is a real experiment for the CUES paper (Crystals Using Exemplar Steering),
not a test. If anything below doesn't match what you observe, stop and report
the discrepancy rather than guessing past it — see "Reporting results" at the
end for exactly how.

## What this experiment answers

The paper's architecture ablation compares three configurations of the same
~230M-parameter model, trained from scratch on the same corpus with matched
hyperparameters:

- **ed_control** — encoder-decoder, naive 12 encoder + 12 decoder layers
- **ed_lite** — encoder-decoder, asymmetric 6 encoder + 18 decoder layers (the
  configuration the paper recommends)
- **dec_only** — decoder-only, 17 layers, edim=1024

Two independent prior studies disagree on which of **decoder-only** and
**ed_control** scores higher — both non-recommended baselines — while agreeing
that **ed_lite beats both**. Full background, the exact numbers, and why a
clean matched rerun is the thing that would actually settle this:
[`dec_only_17L_replication_design.md`](dec_only_17L_replication_design.md).
Read its §2 (a real bug found in a prior attempt) and §3 (the matched
hyperparameter table) before launching anything — this repo's launcher already
encodes that table, but you should know what it's doing.

**This run has already been smoke-tested successfully** on a single L4 GPU —
see [`../smoketests/ed_lite_20261105.md`](../smoketests/ed_lite_20261105.md)
for exactly what was validated (cloning, data fetch, a real training run to a
saved checkpoint) and four real bugs that were found and fixed getting there.
The scaffolding works; what follows is the full-scale version of that same
proven path.

## What a complete result looks like

Six runs: each of the three arms above, at **two seeds**, trained to
completion (150 epochs), then scored with the same methodology. If you only
have time/compute for fewer runs, say so explicitly in your report — a partial
result (e.g., one seed per arm, or fewer than all three arms) is still useful
as long as it's reported as partial, not silently treated as complete.

## Setup

Follow [`../SETUP.md`](../SETUP.md) steps 1–3 (clone `steering` alongside this
repo, install dependencies, fetch the training corpus — you don't need the
three reference checkpoints for a from-scratch run, only the corpus). Then
come back here for the actual launch, which differs from `SETUP.md`'s own
quickstart example in that this is a full run, not a smoke test.

**Known issues to expect, already fixed in this repo but worth knowing about:**
read `SETUP.md`'s "Known gaps" section once before you start. In particular:
the vendored `ed_train.py` in `steering` doesn't support `--seed` — the
launcher will print a loud warning and proceed without seed control if so;
note this in your report rather than treating it as a failure, and consider
whether it changes how you interpret "two seeds" for your run (effectively
replication without a controlled seed — still useful, just not literally the
same code path as "seed=0 vs seed=1" if the flag silently doesn't apply).

## Launch: one run per arm per seed

`scripts/train_arm.sh <arm> <seed>` launches one full training run. It does
**not** support multi-GPU DDP within a single run — if you have multiple GPUs,
run multiple *separate* invocations in parallel (one per GPU), not one
invocation spanning several GPUs. Pin each to its own GPU with
`CUDA_VISIBLE_DEVICES`:

```bash
# Six runs total, one per GPU if you have six; fewer GPUs means running some
# sequentially instead — just don't run more than one per GPU concurrently.
CUDA_VISIBLE_DEVICES=0 STEERING_PATH=../steering/crystals/lemat/code/py \
  DATA_CSV=./data/corpus/big_d15_autolabel_binrho_k7.csv BATCH_SIZE=64 \
  setsid nohup bash scripts/train_arm.sh ed_control 0 >/dev/null 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 STEERING_PATH=../steering/crystals/lemat/code/py \
  DATA_CSV=./data/corpus/big_d15_autolabel_binrho_k7.csv BATCH_SIZE=64 \
  setsid nohup bash scripts/train_arm.sh ed_control 1 >/dev/null 2>&1 < /dev/null &

# ... and so on for ed_lite (seeds 0, 1) and dec_only (seeds 0, 1)
```

`setsid ... < /dev/null &` (not plain `nohup ... &`) matters — see `SETUP.md`'s
note on this; a run that dies when your shell exits is a real failure mode
that's already been hit once during this project's own testing.

**`BATCH_SIZE=64` above matches the original ablation's per-GPU batch size on
H100-class hardware** — the design doc's own table (§3) used 64 on 2×H100 and
only dropped to 24 as an L4-specific OOM workaround. Use 64 if your GPU has the
headroom (an H100 or similar; check for CUDA OOM early and drop to 24 if it
doesn't fit — the L4 OOM ceiling for the largest arm, dec_only, is well
established at this point). Leave `EPOCHS`/`SAVE_INTERVAL`/`LOG_INTERVAL` at
their defaults (150/1000/50) for a real run — those were only overridden for
the smoke test.

**Disk:** each run keeps up to 3 checkpoints (`--max_keep_checkpoints 3`,
already set), and one checkpoint is roughly 3.8GB (confirmed during the smoke
test — this includes optimizer + EMA state, not just model weights). Budget
at least ~15GB free per concurrently-running arm, more if you're running
several in parallel on a shared disk.

## Monitor

```bash
tail -f checkpoints/<arm>_s<seed>.log
```

Loss should decrease, roughly monotonically, from the start. If it's NaN,
exploding, or flat from the first step, stop and report it — don't let a run
continue to completion if it looks broken from the start; that's wasted
compute and the broken-ness is itself useful information.

**Wall-clock expectations:** on 4×H100-class hardware, each arm should finish
comfortably under a day (per the CUES paper's own `experiments_todo.md`
compute-tier calibration, not re-derived here). On weaker or shared hardware,
budget proportionally more — the design doc's §5 gives a concrete historical
data point (the decoder-only arm took ~10 days of GPU-active time on a single
L4 at batch_size 24).

## Score each finished run

Once a run reaches its final checkpoint (`<prefix>_ckpt_final.pt`, or the EMA
variant), score it the same way for all six runs — consistency across arms
matters more here than any single methodology choice:

```python
import sys
sys.path.insert(0, "../steering/crystals/lemat/code/py")
sys.path.insert(0, "./eval")
import path_override
from dielectric.eval import screening
screening.ED_CHECKPOINTS = path_override.ED_CHECKPOINTS
```

Then generate and score per `steering`'s own `crystals/README.md`/`RESULTS.md`
conventions — matched `n`, matched prompt pool, both the self-consistent and
external-hull (`--genbench`) scoring paths, exactly as the design doc's §3
methodology table specifies. Install `requirements-eval-optional.txt` first if
you haven't (MLIP scoring deps, not needed for training).

## Report results

**Fill in this table** (Combined = novel + unique + (meta)stable; report both
scoring methods):

| Arm | Seed | Final step reached | Self-consistent Combined | External-hull (genbench) Combined |
|---|---|---|---|---|
| ed_control | 0 | | | |
| ed_control | 1 | | | |
| ed_lite | 0 | | | |
| ed_lite | 1 | | | |
| dec_only | 0 | | | |
| dec_only | 1 | | | |

Along with the table, include:
- Hardware used (GPU model, count, how runs were distributed across them).
- Total wall-clock time, per run if they didn't all take the same time.
- Whether the `--seed` warning appeared (see "Known issues" above) and what
  that means for how seed-controlled your two replicates actually were.
- Anything that required a workaround, however minor — the smoke test found
  four real bugs this way, and a fifth (the `pkill` self-match) while *stopping*
  a run, not even training it. Assume more exist.
- Final loss curves or at least the final loss value per run, so a clearly
  undertrained or diverged run is visible even if its Combined score alone
  wouldn't show it.

**How to send this back:** see `../SETUP.md`'s "Report results back" section
(§6) — fork + pull request is preferred; relaying the report directly to the
maintainer is a documented fallback if you'd rather not set up git credentials
for a one-off contribution. Either way, **do not put checkpoints in a git
branch or PR** — upload them to your own Hugging Face account and link the
repo in your report instead.
