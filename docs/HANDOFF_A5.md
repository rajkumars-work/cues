# Handoff: the A5 wide-vs-narrow corpus ablation

**You have no prior context — read this whole file before touching anything.**
This is a real experiment for the CUES paper (Crystals Using Exemplar
Steering), not a test. If anything below doesn't match what you observe, stop
and report the discrepancy rather than guessing past it — see "Report
results" at the end for exactly how.

## What this experiment answers

CUES trains on a corpus spanning stable, metastable, and unstable structures,
reasoning that stability is better controlled at generation time (exemplar
steering) than by curation. The one prior *direct* test of this claim was
withdrawn as a measurement artifact (details, and why it's not reused here:
[`A5_wide_vs_narrow_corpus_design.md`](A5_wide_vs_narrow_corpus_design.md)
§0). This experiment is a clean rerun: same architecture (`ed_lite`), same
corpus size, same training budget — only the training corpus's stability
filtering varies between two arms:

- **narrow / `lowhull`** — the wide corpus filtered to `e_above_hull <= 0.1`
  (163,227 rows).
- **wide, size-matched** — a random 163,227-row subsample of the full wide
  corpus, same row count as the narrow arm, so only stability composition
  differs.

Read the design doc's §1 before launching — it has the exact filter, the
measured row counts, and why size-matching (not comparing the narrow arm
against the *full* wide corpus) is the primary comparison.

**This corpus path has been smoke-tested** on a single L4 GPU — see the
smoketest file linked from [`../README.md`](../README.md)'s Status section
for exactly what was validated. The underlying plumbing (clone, fetch,
`train_arm.sh`) was already proven by the C2/C3 architecture experiment's own
smoke test; this one checks specifically that the filtered corpus loads and
trains.

## What a complete result looks like

Two arms (narrow, wide-matched) trained to completion (150 epochs) and
scored, at a minimum one seed each. A second seed per arm is a nice-to-have,
not required — unlike C2/C3's three-way comparison, this is a two-arm test
with no known prior direction-disagreement to resolve. If you only have time
for one arm, or a partial run, report it as partial — don't present it as
complete.

If a C2/C3 run (or you, concurrently) already trains `ed_lite` on the **full**
wide corpus, that run covers this experiment's secondary full-wide arm for
free (see design doc §2) — don't retrain it. The narrow and wide-*matched*
arms below still need their own runs regardless; a full-wide run doesn't
substitute for the size-matched one.

## Setup

Follow [`../SETUP.md`](../SETUP.md) steps 1–3 (clone `steering` alongside this
repo, install dependencies, fetch the training corpus) if you haven't already
— same corpus as C2/C3, don't fetch it twice if a C2/C3 run already did. **A5
doesn't use the three reference checkpoints C2/C3 needs** — use `bash
scripts/fetch_assets.sh --corpus-only ./data` instead of the plain form in
step 3 to skip ~14GB you don't need here.

## Build the two A5 corpora

```bash
python3 scripts/build_lowhull_corpus.py \
  --wide ./data/corpus/big_d15_autolabel_binrho_k7.csv \
  --out-dir ./data/corpus
```

Confirm the printed row counts match the design doc §1's table (163,227 for
both the `lowhull` and wide-matched files) before launching — a mismatch
means the wide corpus changed since the design doc was written and its
hardcoded numbers are stale; report that rather than proceeding silently.

## Launch

Architecture is held fixed at `ed_lite` (the paper's recommended
configuration) — this is a data ablation, not an architecture one. Only
`DATA_CSV` differs between the two launches:

```bash
# Narrow (lowhull) arm
CUDA_VISIBLE_DEVICES=0 STEERING_PATH=../steering/crystals/lemat/code/py \
  DATA_CSV=./data/corpus/big_d15_lowhull.csv BATCH_SIZE=64 \
  setsid nohup bash scripts/train_arm.sh ed_lite 0 >/dev/null 2>&1 < /dev/null &

# Wide, size-matched arm
CUDA_VISIBLE_DEVICES=1 STEERING_PATH=../steering/crystals/lemat/code/py \
  DATA_CSV=./data/corpus/big_d15_wide_matched.csv BATCH_SIZE=64 \
  setsid nohup bash scripts/train_arm.sh ed_lite 1 >/dev/null 2>&1 < /dev/null &
```

(`train_arm.sh ed_lite <seed>` — the second argument there is the run's seed
label, not a meaningful distinction between the two corpora; it's just there
so the two runs don't collide on the same checkpoint prefix. Use whichever
seed numbering you like, including matching seeds across both arms, as long
as the two runs don't share a `<arm>_s<seed>` checkpoint name.)

**Only one GPU?** The two launches above assume two separate GPUs
(`CUDA_VISIBLE_DEVICES=0` / `=1`) running in parallel. On a single GPU, run
them sequentially instead — drop `CUDA_VISIBLE_DEVICES` entirely (or set both
to `0`) and launch the narrow arm, wait for it to finish, then launch the
wide-matched arm. Don't run both at once on one GPU; use whichever `BATCH_SIZE`
fits (see below) and budget roughly double the wall-clock time of a single run.

`setsid ... < /dev/null &` matters — see `SETUP.md`'s note on this (a run that
dies when your shell exits is a real, previously-hit failure mode).
**`BATCH_SIZE=64`** matches the original ablation's per-GPU batch size on
H100-class hardware (see `dec_only_17L_replication_design.md §3`); drop to 24
on an L4 or similar if you hit CUDA OOM. Leave `EPOCHS`/`SAVE_INTERVAL`/
`LOG_INTERVAL` at their defaults (150/1000/50) for a real run.

**Disk:** same per-checkpoint footprint as C2/C3 (~3.8GB, up to 3 kept per
run) — budget at least ~15GB free per concurrently-running arm.

## Monitor

```bash
tail -f checkpoints/ed_lite_s0.log    # narrow arm
tail -f checkpoints/ed_lite_s1.log    # wide-matched arm
```

Loss should decrease, roughly monotonically, from the start on both arms. If
either is NaN, exploding, or flat from the first step, stop and report it —
don't let a run continue to completion if it looks broken from the start.

**Wall-clock expectations:** per the design doc §5, each arm should finish
comfortably under a day on 4×H100-class hardware. On weaker or shared
hardware, budget proportionally more.

## Score each finished run

Identical to C2/C3's scoring path — see
[`HANDOFF_C2_C3.md`](HANDOFF_C2_C3.md) "Score each finished run" for the full
`path_override.py`/`screening.py`/`combined.py` setup. Score both arms with
matched `n` and the same prompt pool; consistency across arms matters more
here than any single methodology choice.

## Report results

**Fill in this table** (Combined = novel + unique + (meta)stable; report both
scoring methods):

| Arm | Seed | Corpus rows | Final step reached | Self-consistent Combined | External-hull (genbench) Combined |
|---|---|---|---|---|---|
| narrow (lowhull) | | 163,227 | | | |
| wide, size-matched | | 163,227 | | | |
| wide, full (secondary, if run) | | 241,741 | | | |

Along with the table, include:
- Hardware used (GPU model, count, how runs were distributed across them).
- Total wall-clock time, per run if they didn't all take the same time.
- Whether a C2/C3 `ed_lite`-on-full-wide-corpus run already existed and was
  reused for the secondary row above, rather than retrained.
- Anything that required a workaround, however minor — see
  `SETUP.md`'s "Known gaps" and the existing smoke tests for examples of the
  kind of thing worth flagging (lineage-drift flags, path-resolution quirks,
  etc.) even if it seems minor.
- Final loss curves or at least the final loss value per run, so a clearly
  undertrained or diverged run is visible even if its Combined score alone
  wouldn't show it.

**How to send this back:** see `../SETUP.md`'s "Report results back" section
(§6) — fork + pull request is preferred; relaying the report directly to the
maintainer is a documented fallback. Either way, **do not put checkpoints in
a git branch or PR** — upload them to your own Hugging Face account and link
the repo in your report instead.
