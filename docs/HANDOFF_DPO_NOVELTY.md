# Handoff: completing the DPO-for-novelty experiment

**Priority: REPRODUCIBILITY — not an evidentiary gap, just unfinished work,
and not blocking.** This doesn't touch the paper's central thesis either
way — it's an open side-question about whether a cheap post-hoc fine-tune
can improve one specific metric (novelty) without hurting another
(structural quality). Of this repo's four Reproducibility-tier items, this
is the cheapest and most concretely specified.

**You have no prior context — read this whole file before touching
anything.**

## What this answers

A second, separate DPO (Direct Preference Optimization) experiment — distinct
from the property-compliance DPO thread, which is **done and needs no further
work** (an early eval's low compliance numbers traced to two evaluation bugs,
not a model deficiency; fixed, standard training alone reached 92%/94%
compliance) — was designed to test whether a short preference fine-tune can
raise generation **novelty** without a material cost to structural quality
(tier-0 validity / MSUN). Pair generation was started but the experiment was
never carried to a citable result: the comparison table in its own design
doc is still blank.

## What's already fully specified

Unlike the other three nice-to-have items, this one has a complete, concrete
design already written (internally; being restated here against the current
production checkpoint instead of the original, now-superseded one):

**Pair generation.** For each of ~5,000+ held-out prompts, generate several
structures (`top_k=10`, stochastic), classify each via a novelty check
against the training set (formula + spacegroup + Wyckoff prototype match —
fast, no MLIP needed), and keep pairs where the same prompt produced both a
**novel** (chosen) and a **known/copy** (rejected) output. Target 2,000+
pairs. Expect a low yield (~8-20% of sources produce both in a bounded number
of tries, per the original attempt) — that's expected, not a sign something's
broken.

**DPO fine-tune**, conservative hyperparameters (chosen specifically to
avoid mode collapse, which an earlier, more aggressive attempt hit):

| Param | Value | Why |
|---|---|---|
| Learning rate | `5e-7` | Half of an earlier attempt's `1e-6`, which mode-collapsed |
| Beta (KL penalty) | `0.1` | Keeps the fine-tune close to the base model |
| Epochs | `1` | More than 1 risks overfitting a small preference set |
| Batch size | `16` | Matched to dataset scale |
| Early stop | On entropy dropping below `0.3` | Catches mode collapse before it fully sets in |

**Base checkpoint:** use the current production checkpoint, public on HF at
`rajkumars47/cues-alex-nolemat-lowhull` (not `ckpt_alex_nolemat_lowhull`,
which doesn't resolve — see `experiments_todo.md`'s Reproducibility section, R3), rather than the
original design's now-superseded checkpoint — same fine-tune recipe, current
base model, so the result is about something the paper still stands behind.

## The one real gap: porting the code

The pair-generation and DPO-training scripts (`dpo_train.py`, a novelty-pair
generator) exist in this project's private codebase but **are not yet in the
public `steering` repo** — this is the actual blocker, not the experiment
design. Porting them is a bounded, well-specified task (the scripts already
exist and work; they need cleaning up and adding to the public repo, not
writing from scratch), comparable in spirit to what `build_lowhull_corpus.py`
did for the A5 experiment. If you're picking this up: start by locating and
porting these two scripts before attempting the recipe above from scratch.

## Evaluation

Generate 250+ structures from both the base checkpoint and the DPO fine-tune
on the same held-out sources, score both with the same
`screening.py`/`combined.py` path used everywhere else in this repo, and fill
in:

| Metric | Base model | DPO model | Change |
|---|---:|---:|---:|
| Novelty (GenBench) | | | |
| Tier-0 / validity | | | |
| MSUN | | | |
| Combined | | | |

**Success criteria** (from the original design, still the right bar):
novelty should increase meaningfully; tier-0/validity should not drop more
than ~5 points; MSUN should ideally improve (more novel structures that are
*also* metastable, not just novel).

## Report results

Fill in the table above, report whether entropy stayed stable through
training (a collapse is itself a useful negative result — report it plainly
rather than discarding the run), and the final pair-generation yield.
Same report-back mechanism as every other experiment here: see
`../SETUP.md` §6 — fork + PR preferred, relay as fallback, the fine-tuned
checkpoint to your own HF account.
