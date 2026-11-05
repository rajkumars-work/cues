# CUES — architecture ablation: encoder-decoder vs. decoder-only

Code and data to reproduce the "C2/C3" architecture ablation from the CUES paper
(Crystals Using Exemplar Steering) — the sibling repo to
[rajkumars-work/steering](https://github.com/rajkumars-work/steering), just as
the CUES paper is the sibling of the steering paper.

A from-scratch, matched-methodology retrain of three architecture points on the
same corpus:

- **ed_control** — encoder-decoder, naive 12 encoder + 12 decoder layers
- **ed_lite** — encoder-decoder, asymmetric 6 encoder + 18 decoder layers (the
  configuration the paper recommends)
- **dec_only** — decoder-only, 17 layers, edim=1024

Background and exactly what question this settles:
[`docs/dec_only_17L_replication_design.md`](docs/dec_only_17L_replication_design.md).

**Running this as a real experiment (not just trying it out)?** Start at
[`docs/HANDOFF_C2_C3.md`](docs/HANDOFF_C2_C3.md) instead of the quick start
below — it has the full launch (all three arms × two seeds, multi-GPU
guidance), scoring, and results-reporting template.

## What's here vs. what's in `steering`

Most of the actual model/training code already lives, public, in the companion
repo: [github.com/rajkumars-work/steering](https://github.com/rajkumars-work/steering)
(`crystals/lemat/code/py/{ed,dielectric}/` — training loop, model definitions,
data loader, eval/scoring pipeline). This repo does **not** duplicate that code;
it adds only what's different or missing for this specific experiment:

- `scripts/train_arm.sh` — a single-machine training launcher for the three arms,
  with matched hyperparameters across all three (same LR/warmup/batch/dropout —
  the earlier ablation's own cross-arm LR mismatch is exactly the confound this
  avoids). The original runs used GPU-VM-specific orchestration (preemption
  recovery, segmented relaunches); this keeps only the resume/watchdog pattern
  that's actually load-bearing.
- `scripts/fetch_assets.sh` — pulls the training corpus and the three original
  checkpoints from Hugging Face (all public, see below).
- `eval/path_override.py` — a small override for one hardcoded path in the
  companion repo's `screening.py` (`ED_CHECKPOINTS`), so scoring works outside
  the original author's machine. See its docstring.
- `docs/dec_only_17L_replication_design.md` — the design doc for this experiment,
  including three real bugs (an LR-floor exposure, an EMA-reset-on-resume bug, and
  dead qk-norm code) a naive rerun will silently hit again if not already fixed.
  **Read §2 and §6 before launching anything.**
- `requirements.txt` — a curated environment spec (training + eval deps).

## Quick start

See [`SETUP.md`](SETUP.md) for full setup. Short version:

```bash
git clone https://github.com/rajkumars-work/steering.git ../steering   # training/eval code
bash scripts/fetch_assets.sh                                           # data + checkpoints from HF
pip install -r requirements.txt
bash scripts/train_arm.sh ed_lite 0                                    # or ed_control / dec_only, seed
```

## Data and checkpoints (Hugging Face, public)

| Asset | HF repo |
|---|---|
| Training corpus (`big_d15_autolabel_binrho_k7.csv`, ~242k rows[^corpus-rows]) | [`rajkumars47/data_big_d15`](https://huggingface.co/datasets/rajkumars47/data_big_d15) |
| ed_control checkpoint (original run) | [`rajkumars47/ckpt_backup_ed_control_12_12_d15`](https://huggingface.co/rajkumars47/ckpt_backup_ed_control_12_12_d15) |
| ed_lite checkpoint (original run) | [`rajkumars47/ckpt_ed_enclight_6_18`](https://huggingface.co/rajkumars47/ckpt_ed_enclight_6_18) |
| dec_only checkpoint (original run) | [`rajkumars47/ckpt_dec_only_17L_edim1024_d15_repl`](https://huggingface.co/rajkumars47/ckpt_dec_only_17L_edim1024_d15_repl) |

The checkpoints are the *original* runs this replication is checking — a
reference to compare a new run against, not required as training input.

[^corpus-rows]: Confirmed 241,741 rows via `wc -l` on the published CSV
    (2026-11-05). The original design doc's "~219k" figure referred to an
    earlier copy on a different machine, before the version that was
    ultimately backed up to this HF repo — the HF dataset card's own
    description states 242,170 rows for the final backed-up file. Treat
    ~242k as the authoritative current figure; the small residual difference
    between 242,170 and 241,741 wasn't investigated further.

## Status

Smoke-tested successfully end-to-end on a single external L4 GPU
(2026-11-05) — see [`smoketests/ed_lite_20261105.md`](smoketests/ed_lite_20261105.md)
for what was validated and the bugs found and fixed along the way. Ready for
a real full-scale run; see [`docs/HANDOFF_C2_C3.md`](docs/HANDOFF_C2_C3.md).

## License

MIT, matching the companion `steering` repo. See `LICENSE`.
