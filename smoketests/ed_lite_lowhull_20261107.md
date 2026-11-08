# Smoke test: A5 corpus-filter path, 2026-11-07

**Goal:** confirm the A5 wide-vs-narrow-corpus experiment's *new* piece —
`scripts/build_lowhull_corpus.py` plus an `ed_lite` run pointed at its
output — actually works end-to-end on real GPU hardware, before handing
`HANDOFF_A5.md` to an external collaborator. This is narrower than the
original `ed_lite_20261105.md` smoke test: that test already proved the base
plumbing (clone → fetch → `train_arm.sh` → checkpoint → report). This one
assumes that plumbing works and checks only "does the filtered corpus load
and train" — same host, same arm, same bar (a real checkpoint with a finite,
decreasing loss), narrower scope.

**Final result: PASSED**, first attempt, no fixes needed. Host `gpu-train-2`
(single NVIDIA L4, same VM as the 2026-11-05 test), run directly via
`gcloud compute ssh` from the repo-maintaining Claude Code session (no
separate human relay or VM-side Claude session needed — same pattern the
2026-11-05 test used for its later rounds).

| Step | Result |
|---|---|
| 0 Environment | Pass — L4 visible, Python 3.10.12, torch 2.9.1+cu129 (CUDA available), `steering` and training deps already present from the prior test |
| 1 Clone `cues` (fresh dir, picking up the new A5 files) | Pass, no credentials needed |
| 2 Fetch wide corpus (`fetch_assets.sh`'s dataset step, corpus only) | Pass — 3.7s, 182MB, 241,741 rows, matches `data_big_d15`'s dataset-card count |
| 3 `build_lowhull_corpus.py` | Pass — 163,227 rows in both the `lowhull` and wide-matched outputs, exactly matching the design doc's precomputed table |
| 4 Launch `train_arm.sh ed_lite 0` with `DATA_CSV` pointed at the lowhull CSV | Pass |
| 5 Checkpoint | **Pass** — real checkpoint saved at step 50, 3,809,288,735 bytes (~3.8GB, matches the 2026-11-05 test's checkpoint size exactly) |
| 6 Stop cleanly | Pass — GPU fully released per `nvidia-smi --query-compute-apps` |

## What actually happened

**Corpus fetch (step 2).** Used a direct `huggingface_hub.snapshot_download`
call instead of the full `fetch_assets.sh` (which also pulls all three
reference checkpoints) — intentional, not a bug: this VM only had ~9GB free,
and `fetch_assets.sh`'s default full fetch (corpus + 3 checkpoints, several
GB combined) hit `No space left on device` mid-download the first time it was
tried here. Corpus-only is sufficient for this smoke test and for a real A5
run (A5 doesn't need the reference checkpoints at all — those are for the
C2/C3 architecture comparison). **Worth flagging to a real external
collaborator with limited disk:** run `fetch_assets.sh`'s corpus step alone,
or free up space before running the full script, rather than assuming the
default invocation fits on a small disk.

**Filter script (step 3).** `build_lowhull_corpus.py` ran without
modification and reproduced the exact row counts (163,227 for both outputs)
computed on this Mac beforehand (`A5_wide_vs_narrow_corpus_design.md` §1) —
the `e_above_hull` column read cleanly, no schema surprises.

**Training launch (step 4).** No fixes needed — `train_arm.sh`'s existing
`DATA_CSV` parameterization (already proven generic by the 2026-11-05 test's
own fixes) worked immediately for a corpus the script had never seen before.
Loss decreased monotonically and stayed finite for all 50 steps observed:

```
step 1/4000  loss 9.811
step 10/4000 loss 8.410
step 20/4000 loss 7.733
step 30/4000 loss 7.072
step 40/4000 loss 6.528
step 50/4000 loss 6.052
Saved checkpoint: .../ed_lite_s0/ed_lite_ckpt_000050.pt
```

**Stopping (step 6).** Used the bracket-escaped `pkill` pattern
(`pkill -f "[t]rain_arm.sh..."` / `"[e]d_train.py..."`) documented in
`SETUP.md`'s "Known gaps" from the 2026-11-05 test — no self-match incident
this time, consistent with that fix being the right one.

## Scope note

This test did **not** exercise: the wide-matched arm (only `lowhull` was
run — the wide-matched CSV was generated and row-count-verified in step 3,
but not separately trained, since the goal was "does the filtered corpus
load," which `lowhull` alone answers), scoring (`screening.py`/`combined.py`
— not reached at 50 steps), or a full 150-epoch run. None of these are new
risk relative to the 2026-11-05 test, which already validated the full
training-to-scoring path for a differently-sourced `DATA_CSV` pointed at the
same underlying file.

## Open items, not resolved by this test

- **Wide-matched arm untested directly** — it's a plain CSV subsample read
  through the identical code path `lowhull` just validated, so the risk this
  leaves open is low, but it wasn't separately run.
- **Disk headroom remains tight on this VM** (same observation as
  2026-11-05) — a real A5 run (two arms, each keeping up to 3 checkpoints at
  ~3.8GB) needs active cleanup between runs here, or a bigger disk.
- **Total GPU-VM time for this test was kept under 30 minutes** (fetch
  through stop) per instruction — a real A5 run goes to full training budget
  and will take substantially longer; this test only confirms the corpus
  loads and the first 50 steps are healthy, not full-run behavior.
