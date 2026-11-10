# Handoff: a clean corpus-size ablation

**Priority: NICE — a genuine but secondary evidentiary gap, not a stated
paper limitation.** Not in `limitations.md`. Resolving this either way
doesn't change the paper's thesis — see `EXPERIMENTS.md`'s Nice-to-have
table.

**You have no prior context — read this whole file before touching
anything.**

## What this answers

The NxN matrix's smallest-corpus row (`mp20_ref`, 45,229 rows) scores worst
in the entire matrix by a wide margin — the matrix's own documentation
already calls this "the floor cell," not a controlled size ablation. The
reason it can't support a clean "small corpora hit a hard floor" claim as-is:
`mp20_ref`'s corpus differs from every other row in **three** ways at once —
smaller size, no property tokens (it mirrors MatterGen's published
no-property-token split), and whatever epoch-budget interaction comes with
training a much smaller corpus for the matrix's fixed 75 epochs. Any one of
the three could be driving the result; the matrix can't tell you which.

## The plan

Hold everything except row count fixed, and sweep size alone:

- **Same architecture**: `ed_lite`.
- **Same property-token schema**: start from a corpus that *does* carry
  property tokens (`big_d15` or `alex_nolemat_lowhull` — either works;
  pick one and use it for every size point, don't mix sources across
  points).
- **Same epoch budget**: 75 epochs at every size point, matching the NxN
  matrix's own convention (see `HANDOFF_CHECKPOINT_RECONSTRUCTION.md` Part A
  §2's hyperparameter table) — this is a deliberate choice, not the only
  reasonable one: holding *epoch count* fixed means smaller corpora see
  fewer total gradient steps, which is itself a confound with "epoch budget"
  in a different sense (steps, not epochs). If you'd rather hold *total
  training steps* fixed instead (more epochs for smaller corpora), that's
  an equally defensible choice — just pick one explicitly and say which in
  your report, rather than leaving it ambiguous the way the original floor
  cell did.
- **Only row count varies.**

**Suggested size points** (not prescribed — five points spanning roughly an
order of magnitude is enough to show a trend without needing many parallel
runs): 10%, 25%, 50%, 75%, and 100% of the full corpus's row count. Using
`big_d15` (242,170 rows) as the source, that's roughly 24k / 60k / 121k /
182k / 242k rows.

## Building the size points

A short deterministic subsampling script — same pattern as
`scripts/build_lowhull_corpus.py` (random sample at each target row count,
fixed seed, print the resulting counts for confirmation). Five fixed-size
random subsamples of one already-public corpus; no new data source needed.

## Launch

One `train_arm.sh` invocation per size point, same as every other
corpus-varying experiment here:

```bash
DATA_CSV=./data/corpus/sizeablation_<pct>.csv \
EPOCHS=75 \
STEERING_PATH=../steering/crystals/lemat/code/py \
bash scripts/train_arm.sh ed_lite 0
```

Five size points parallelize cleanly across up to five GPUs; on fewer GPUs,
run sequentially — same guidance as A5's single-GPU fallback.

## Scoring and comparison

Score each size point the same way as every other arm here — 100 prompts,
`natoms < 20`, seed `1234`, naked format, same `screening.py`/`combined.py`
path, scored against the *source* corpus's own held-out pool (not a
different pool per size point — that would reintroduce a confound).

| Corpus size (rows) | Combined |
|---:|---:|
| ~24k (10%) | |
| ~60k (25%) | |
| ~121k (50%) | |
| ~182k (75%) | |
| ~242k (100%) | |

A genuine floor would show Combined flattening out below some row count; a
smooth, roughly monotone curve with no flattening would suggest the matrix's
original floor cell was more about missing property tokens than row count
itself.

## Report results

The five Combined scores, which epoch/step convention you chose and why,
and whether the curve shows a real floor or not. Same report-back
mechanism as every experiment here: `../SETUP.md` §6 — fork + PR preferred,
relay as fallback.
