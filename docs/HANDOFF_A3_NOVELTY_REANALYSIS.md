# Handoff: recompute novelty against the training corpus, not just LeMat-Bulk

**Priority: NICE — a genuine but secondary evidentiary gap, not a stated
paper limitation.** Not in `limitations.md`. Resolving this either way
doesn't change the paper's thesis — see `EXPERIMENTS.md`'s Nice-to-have
table. Possibly the cheapest real improvement in this entire repo, in the
best case (see below).

**You have no prior context — read this whole file before touching
anything.**

## What this answers

The NxN matrix's "curation beats volume" reading — the curated,
single-source `alex_nolemat*` variants outscoring the larger, more diverse
production corpus — is entangled with an artifact: those variants are
pre-filtered, *at build time*, to exclude anything overlapping
`LeMat-Bulk` (that's what "nolemat" means in their name). Scoring their
generations as "novel" against LeMat-Bulk is partly mechanical — the
training data itself was already selected to be LeMat-Bulk-disjoint, so a
model trained on it starts with a structural head start on that specific
novelty metric, independent of whether it's actually generating anything
creative. The question this doesn't yet answer: are these variants also
novel against **their own training corpus** — i.e., are they generating
genuinely new structures, or substantially reproducing what they were
shown?

## Two different costs, depending on what's already in hand

**If the original generation outputs are available** (the `generated.extxyz`
files the NxN matrix's `alex_nolemat*` rows produced — private,
maintainer-side artifacts, not yet published alongside this repo): **this is
a pure re-analysis, no GPU needed at all.** Compute a second novelty score
for each existing generation, using each variant's own training CSV as the
reference set instead of LeMat-Bulk (same structure-matching methodology
already used everywhere else in this project — pymatgen `StructureMatcher`,
`ltol=0.2, stol=0.3, angle=5`, novelty-vs-LeMat-Bulk's own matcher
settings). No regeneration, no training, just a different reference set for
an analysis that already exists in some form.

**If those outputs aren't available to whoever picks this up:** the
cheap path isn't available, and this becomes dependent on
`HANDOFF_CHECKPOINT_RECONSTRUCTION.md` Part A's audit (do the
`alex_nolemat*` checkpoints survive?) — if a checkpoint survives, regenerate
~100 structures per pool the same way the rest of this repo does, then run
the same training-corpus-novelty analysis on the fresh generations. If the
checkpoint doesn't survive either, this folds into that doc's broader
retrain-and-rescore plan instead of standing alone. **Check which situation
you're in before starting** — ask the maintainer whether the original
`generated.extxyz` files can be shared before assuming a GPU is needed.

## What to compute

For each `alex_nolemat*` row's generations, two novelty numbers side by
side:

| Variant | Novel vs. LeMat-Bulk (existing) | Novel vs. own training corpus (new) |
|---|---:|---:|
| `alex_nolemat` | | |
| `alex_nolemat_onhull` | | |
| `alex_nolemat_lowhull` | | |
| `alex_nolemat_stability_sampled` | | |
| production (`big_d15`, for comparison — not pre-filtered against LeMat-Bulk, so this should be a smaller gap for this row) | | |

If the two numbers track closely for the `alex_nolemat*` rows, the artifact
concern is unfounded — their novelty is real, not mechanical. If
novel-vs-own-training-corpus comes in much lower than novel-vs-LeMat-Bulk,
that's direct evidence the original "curation beats volume" reading
overstates how much of the gain is genuine generative novelty rather than
the pre-filtering's own selection effect.

## Report results

The table above, filled in, plus which path you took (pure re-analysis vs.
regeneration) and why. Same report-back mechanism as every experiment here:
`../SETUP.md` §6 — fork + PR preferred, relay as fallback.
