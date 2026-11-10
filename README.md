# CUES — experiment handoffs for the CUES paper

Code and data to reproduce experiments that would strengthen the CUES paper
(Crystals Using Exemplar Steering)'s weaker-evidenced claims, or make an
already-accepted claim independently verifiable where it currently rests on
a checkpoint or dataset that no longer exists — the sibling repo to
[rajkumars-work/steering](https://github.com/rajkumars-work/steering), just
as the CUES paper is the sibling of the steering paper. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the standard this
repo is trying to meet.

**Running this as a real experiment (not just trying it out)?** Start at
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — the full, current index of
every experiment hosted here, labeled **Must** (a real, stated paper
limitation), **Nice-to-have** (a genuine but secondary evidentiary gap), or
**Reproducibility** (not an evidentiary gap at all, just artifact
availability or unfinished work). That page is the source of truth for
what's here and why — this README doesn't duplicate its full list, to avoid
the two drifting apart as items get added, resolved, or re-labeled.

The two full train-from-scratch experiments, with their own design docs:

- **C2/C3 — architecture ablation** (Nice-to-have). A from-scratch,
  matched-methodology retrain of three architecture points on the same
  corpus — `ed_control` (12+12), `ed_lite` (6+18, the paper's
  recommendation), `dec_only` (17 layers, decoder-only).

  Background: [`docs/dec_only_17L_replication_design.md`](docs/dec_only_17L_replication_design.md).
  Run it: [`docs/HANDOFF_C2_C3.md`](docs/HANDOFF_C2_C3.md).

- **A5 — corpus ablation** (Must — `limitations.md` item 2). Wide vs.
  stability-filtered training corpus, architecture held fixed at `ed_lite`.

  Background: [`docs/A5_wide_vs_narrow_corpus_design.md`](docs/A5_wide_vs_narrow_corpus_design.md).
  Run it: [`docs/HANDOFF_A5.md`](docs/HANDOFF_A5.md).

Everything else — including the other two Must items, every Nice-to-have
reconstruction, and the Reproducibility-tier artifact-availability gaps —
is listed in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), not here.

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
- `docs/dec_only_17L_replication_design.md` — the design doc for the C2/C3
  architecture experiment, including three real bugs (an LR-floor exposure, an
  EMA-reset-on-resume bug, and dead qk-norm code) a naive rerun will silently
  hit again if not already fixed. **Read §2 and §6 before launching anything.**
- `scripts/build_lowhull_corpus.py` — derives the A5 corpus experiment's two
  training CSVs (stability-filtered and size-matched-wide) from the same wide
  corpus `fetch_assets.sh` already pulls — no second download.
- `docs/A5_wide_vs_narrow_corpus_design.md` — the design doc for the A5 corpus
  experiment.
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

**Before anything else, run `bash scripts/quick_selftest.sh`** (under a
minute on a GPU machine) — it's a fast, repeatable plumbing check (tokenizer,
data, training steps, checkpoint save) meant to catch environment problems
before you commit real GPU time to an experiment. See `SETUP.md` for detail.

The base plumbing (clone, fetch, train, checkpoint) was smoke-tested
successfully end-to-end on a single external L4 GPU (2026-11-05) — see
[`smoketests/ed_lite_20261105.md`](smoketests/ed_lite_20261105.md) for what
was validated and the bugs found and fixed along the way. The A5 corpus path
specifically has its own, narrower smoke test — see the most recent
`smoketests/ed_lite_lowhull_*.md` entry.

**A cold-start test (2026-11-08) — an independent session given no context
beyond this repo, trying to actually run an experiment — found and led to
fixing a real correctness bug:** `train_arm.sh` wasn't overriding
`ed_train.py`'s default step cap, so a "full, 150-epoch" run was silently
stopping around 28 epochs with the LR schedule nowhere near decayed, while
still reporting as complete. Fixed and verified on real hardware; see
`SETUP.md`'s "Known gaps" for detail, and that same test's findings for
several smaller fixes (a disk-space wall in `fetch_assets.sh`, a `--seed`
warning that went nowhere, undocumented single-GPU guidance for A5, and
`HANDOFF_CHECKPOINT_RECONSTRUCTION.md` citing files that weren't actually
public). With those fixed, C2/C3 and A5 are ready for a real full-scale run;
see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## License

MIT, matching the companion `steering` repo. See `LICENSE`.
