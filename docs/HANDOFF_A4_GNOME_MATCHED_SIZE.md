# Handoff: GNoME augmentation, size-matched rerun

**Priority: NICE — a genuine but secondary evidentiary gap, not a stated
paper limitation.** Not in `limitations.md`. Resolving this either way
doesn't change the paper's thesis — see `EXPERIMENTS.md`'s Nice-to-have
table.

**You have no prior context — read this whole file before touching
anything.**

## What this answers

The NxN data-ablation matrix (§7) found that adding GNoME structures to the
production training corpus ("d15+gn") improves generation quality over
production alone — a real, ~3σ effect. But the comparison as run isn't
size-matched: `d15+gn` (489,860 rows) is roughly **2× production's row
count** (242,170 rows). The matrix can't currently tell you how much of the
gain is "GNoME's chemistry is genuinely more useful" versus "the augmented
model just saw more training data, full stop." Both arms already exist as
corpora (reused from the C2/C3 and A5 experiments and the NxN matrix
reconstruction, respectively) — what's missing is a **size-matched** pair.

## The plan

Two arms, same architecture (`ed_lite`), same epoch budget as the NxN
matrix's own recipe (75 epochs — see `HANDOFF_CHECKPOINT_RECONSTRUCTION.md`
Part A §2 for the full matched hyperparameter table, identical here):

- **GNoME-augmented, size-matched** — a random subsample of the `d15+gn`
  corpus, cut down to production's row count (242,170), same way
  `build_lowhull_corpus.py` size-matched A5's wide/narrow arms (a fixed-seed
  random sample, not a stratified one — document the seed used).
- **Production, unmatched** — the existing production corpus
  (`rajkumars47/data_big_d15`), already fetched for C2/C3 and A5; no new
  corpus work needed for this arm.

If you'd rather upsample instead of downsample (train production at
`d15+gn`'s larger size), that's not viable here — production can't be
inflated without duplicating rows, which would introduce a different
confound (repeated-structure memorization) than the one this experiment is
trying to remove. Downsampling the augmented corpus is the cleaner choice,
and the recommended one.

**Building the size-matched corpus:** `d15+gn`'s own row-level construction
isn't yet ported to this public repo (same gap `HANDOFF_CHECKPOINT_RECONSTRUCTION.md`
Part A flags for rebuilding that corpus at all) — build the full `d15+gn`
corpus first per that doc's recipe, then subsample it down to 242,170 rows
with a short deterministic script, same pattern as
`scripts/build_lowhull_corpus.py` (random sample, fixed seed, print the
resulting row count for confirmation before training).

## Launch

Once the size-matched corpus exists:

```bash
DATA_CSV=./data/corpus/d15gn_matched.csv \
EPOCHS=75 \
STEERING_PATH=../steering/crystals/lemat/code/py \
bash scripts/train_arm.sh ed_lite 0
```

No script changes needed — same pattern as every other corpus-varying
experiment here.

## Scoring and comparison

Score both arms (size-matched GNoME-augmented, and production) the same way
as the rest of the NxN matrix — 100 prompts per pool, `natoms < 20`, seed
`1234`, naked format, same `screening.py`/`combined.py` path. Compare:

| Arm | Rows | Combined |
|---|---:|---:|
| Production (unmatched) | 242,170 | (recorded matrix value — see `cues_nxn.md`, not public) |
| GNoME-augmented, full size | 489,860 | (recorded matrix value) |
| **GNoME-augmented, size-matched (new)** | **242,170** | **?** |

If the size-matched arm's Combined still beats production by most of the
original margin, that's real support for "GNoME's chemistry matters, not
just its volume." If the gain mostly disappears once size is matched, the
original finding was more about data volume than data content — report
either outcome plainly.

## Report results

Row count actually used, the subsampling seed, the Combined score, and
whether the gap to production narrowed relative to the full-size comparison.
Same report-back mechanism as every experiment here: `../SETUP.md` §6 — fork
+ PR preferred, relay as fallback.
