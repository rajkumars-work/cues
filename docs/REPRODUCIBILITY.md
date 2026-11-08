# Reproducibility: the standard this repo is trying to meet

## The north star

**Every number this paper claims should be independently recreatable by a
reader** — given the public datasets, the scripts, and either a public
checkpoint or a documented recipe to retrain one that produces an equivalent
result. Not "trust us, we ran this once." Not "the original run's numbers,
cited because nothing better exists." Recreatable, by someone who is not us,
using only what's in these two public repos (`cues`, `steering`) and public
Hugging Face assets.

This repo — its design docs, its `HANDOFF_*.md` files, `EXPERIMENTS.md`'s
index — exists because of that goal, not as a loosely related collection of
"things a colleague with a GPU could help with."

## What follows from that

**1. Every claimed number traces to one of two things, and should say which.**
Either (a) a public checkpoint, trained on a public dataset, scored by a
public script — someone can literally rerun it and get the same answer — or
(b) if the original artifact is lost, a documented reconstruction recipe:
how to rebuild the dataset from public sources, what hyperparameters produced
the checkpoint, and what score the reconstruction should land near. "We ran
this in May and the checkpoint is gone now" is never the end state — it's a
flag that a reconstruction recipe needs to be written down, even if nobody
has run it yet.

**2. Not every gap gets the same urgency, and that's fine — but every gap
gets *something*.** A result this matters to (`EXPERIMENTS.md`'s Must/Should
tiers — currently the near-equilibrium benchmark rebuild and the
corpus-stability ablation; priorities are re-triaged as the paper's actual
argument changes, so check that index rather than assuming this list stays
fixed) gets a fully worked, ready-to-run handoff package, because the paper
shouldn't ship an unresolved or irreproducible number at that tier. A result
that's secondary, historical, or a side-curiosity (the NxN matrix's
underlying checkpoints, the abandoned diffusion lineage, an informal
fine-tuning account, an incomplete DPO side-experiment) still gets a written
reconstruction plan and an honest cost estimate — marked low priority /
nice-to-have, not silently dropped. The difference between MUST/SHOULD and
NICE is urgency, not whether a plan exists at all.

**3. When something's lost, write down exactly what and what it would take
back — don't just cite the old number forever.** The useful unit of work
isn't "rerun the experiment" (sometimes that's genuinely expensive or
requires infrastructure that doesn't exist yet) — it's **writing down**
precisely what's missing, what rebuilding it would require, and an honest
compute-vs-engineering cost estimate, so the gap is legible and actionable
by anyone who later has the time or hardware to close it. A gap with a
written plan and a NICE tag is a very different thing from a gap nobody's
looked at.

**4. Reconstructions reuse existing public infrastructure wherever
possible, rather than accumulating one-off scripts.** `train_arm.sh`,
`fetch_assets.sh`, the `screening.py`/`combined.py` scoring path — every
reconstruction plan in this repo is written to reuse these rather than invent
a parallel pipeline. This repo's own scripts (`build_lowhull_corpus.py`, for
instance) exist because a *clean, scripted, deterministic derivation from an
already-public file* was cheaper and more auditable than fetching or hand-
building a new one. The same discipline should apply to anything added later.

**5. Retraction discipline and reproducibility discipline are the same
discipline, pointed in opposite directions.** `sources_helper.md`'s
DO-NOT-CITE ledger exists to stop a withdrawn or artifact-contaminated number
from quietly surviving in the paper. This doc exists to make sure a number
that *is* trustworthy doesn't quietly become unverifiable once its checkpoint
disappears. Both are about the same thing: a reader should always be able to
tell, for any number in this paper, whether it currently stands on something
real and checkable.

**6. New results should be born reproducible, not added to the backlog.**
The point isn't to eventually clear every historical gap — some (the
abandoned diffusion lineage, especially) are expensive enough that they may
never be worth closing, and this doc says so plainly where that's the honest
assessment (`HANDOFF_DIFFUSION_REPRO.md`). The point is that *new* work
shouldn't create new instances of the same problem: a result computed for
this paper from here forward should ship with its checkpoint and dataset
public, or it's not done yet.

## How this maps to what's actually in this repo

- `EXPERIMENTS.md` is the live index — every entry there is either closing a
  blocking gap (MUST/SHOULD, cross-referenced to a specific `limitations.md`
  item) or a nice-to-have reconstruction (NICE, cross-referenced the same
  way). Both categories exist in the same index on purpose — reproducibility
  gaps aren't sorted into "things we track" and "things we don't," only into
  "urgent" and "not urgent."
- Every `HANDOFF_*.md` states its priority up front, states exactly what's
  missing and why, and gives either a ready-to-run recipe or an honest
  statement of what infrastructure doesn't exist yet and would need to be
  built first. None of them paper over an engineering gap as if it were
  already solved.
- This is a living document, not a one-time audit. If you're adding a new
  experiment, a new checkpoint, or a new claimed number to the paper, use
  this doc's six points as the checklist for whether it's actually being
  shipped reproducibly — not just correctly.
