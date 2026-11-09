# Handoff: publishing the production checkpoint's own training corpus

**RESOLVED, 2026-11-09 — nothing left to do here.** The corpus is now
public: `rajkumars47/data_alex_nolemat_lowhull` (225,103 rows — a close
approximation, not the exact 249,117-row original; that repo's own README
explains why, same explanation as below). Both citations (checkpoint name,
corpus location) are fixed across this repo's docs. Kept below for the full
"what was wrong and why" context, in case anyone lands here from an old
link — **the "What to do" section describes the original plan (a full
`StructureMatcher` rebuild), which is not what actually happened**; see
"What actually happened" right below instead.

## What actually happened

The original plan (below) was to rebuild the exact 249,117-row corpus from
public Alexandria via the real `StructureMatcher` filter. Picking this back
up, a closer look found that an already-existing, already-correct backup of
the production checkpoint's actual training data was sitting in a private
HF dataset repo (`rajkumars47/data_alex_nolemat_lowhull`, pushed
2026-07-13) — at 225,103 rows, not 249,117, because that backup's own build
script substituted a faster `composition_absent` filter for the real
`StructureMatcher` pass (see its docstring). Rather than spend the
nontrivial compute to rebuild the exact original (Alexandria's
`full_pbe_2025.07.02` release is ~5.78M entries), the cheaper and still
honest path was taken: publish the close approximation that already
existed, with the discrepancy documented plainly, rather than silently
presenting it as exact. The repo's README states this; so does the
provenance note in `cues/docs/experiments.md`.

---

**The rest of this file is the original, now-superseded plan — read
"What actually happened" above first.**

**You have no prior context — read this whole file before touching
anything.**

## What's actually wrong

`ckpt_alex_nolemat_lowhull` is this paper's designated production/"CUES"
checkpoint — the one result after result cites as "current and
reproducible," unlike the historical checkpoints `HANDOFF_CHECKPOINT_RECONSTRUCTION.md`
covers. That confidence is almost entirely justified, but a direct check
against the HuggingFace API (account `rajkumars47`, 2026-11-08) turned up two
real problems, not one:

1. **The checkpoint is public, but not at the name this repo's own docs
   cite.** `rajkumars47/ckpt_alex_nolemat_lowhull` — the name quoted in
   `experiments.md`'s provenance note and elsewhere — **does not exist**.
   The actual public checkpoint lives at `rajkumars47/cues-alex-nolemat-lowhull`
   (confirmed public; this is also the name `steering/crystals/fetch_checkpoint.sh`
   already uses correctly). A reader who copies the name as written in this
   repo's own docs gets a 404, not a checkpoint.
2. **The training corpus is not public at all, under either name.**
   `rajkumars47/data_alex_nolemat_lowhull` exists but is **private**. No
   public dataset repo for this corpus exists under any naming variant
   checked (`cues-alex-nolemat-lowhull[-data/-dataset]`, `data-alex-nolemat-lowhull`).
   This matches the `steering` repo's own internal note that the training
   CSV "isn't published" (`steering/No_Model_Experiment_Design.md`) — this
   isn't a new discovery so much as a gap that was never carried over into
   this paper's reproducibility docs.

Put together: the single checkpoint this paper repeatedly holds up as the
reproducible one has a wrong citation for its weights and no public copy at
all of the data that produced them. Nothing about the paper's *results* is
in question — this is the same kind of gap as `experiments_todo.md`'s
Reproducibility items R1/R2 (this is R3): the finding stands, the artifact
trail has a hole. It isn't filed as a paper limitation at all, for exactly
that reason — see that section's own note on why.

## Why this is cheap, not a research problem

The corpus's construction is already fully specified in this repo —
`HANDOFF_CHECKPOINT_RECONSTRUCTION.md`'s Part A already documents exactly
how to build it, because row 1 of the NxN matrix uses this same corpus:

1. Start from Alexandria (public, RWTH's release).
2. Filter out anything structurally matching `LeMaterial/LeMat-Bulk`
   (pymatgen `StructureMatcher`, `ltol=0.2, stol=0.3, angle=5` — the same
   matcher settings this project's novelty scoring already uses elsewhere).
   This produces `alex_nolemat`.
3. Filter `alex_nolemat` to `e_above_hull <= 0.1` (this project's "lowhull"
   convention). This produces `alex_nolemat_lowhull` — 249,117 rows per the
   NxN matrix's own record.

No training, no GPU, no new matcher logic — this is the exact same
mechanical derivation `build_lowhull_corpus.py` already demonstrates for a
different corpus (the A5 wide-vs-narrow comparison), just with the
LeMat-disjoint filter added first. Writing the equivalent script for this
corpus (call it `build_alex_nolemat_lowhull_corpus.py`, or extend the
existing script with a `--lemat-filter` flag) is the only new code this
item needs.

## What to do

1. **Build the corpus.** Starting from public Alexandria, apply the
   `alex_nolemat` structure-match filter, then `e_above_hull <= 0.1`, per
   the recipe above. Confirm the row count lands at 249,117 (±a handful —
   Alexandria and LeMat-Bulk are both versioned; exact matches to a
   snapshot from months ago aren't guaranteed, and a close match is enough).
2. **Publish it.** Either as a new public HF dataset repo (matching the
   existing `rajkumars47/data_big_d15` pattern — public, dataset-type), or
   as a CSV added to this public repo's own assets if a new HF repo isn't
   available to you. Either way, the point is a reader can get the file
   without asking anyone.
3. **Fix the two citations.** This repo's own docs should point to
   `rajkumars47/cues-alex-nolemat-lowhull` for the checkpoint (not
   `ckpt_alex_nolemat_lowhull`), and to wherever step 2 publishes the
   corpus (not the private `rajkumars47/data_alex_nolemat_lowhull`). This
   handoff's companion edit already fixes the citations inside this repo's
   own docs (`experiments.md`'s provenance note, this repo's other
   `HANDOFF_*.md` files) — if you're publishing a new dataset repo under a
   different name than assumed here, update those pointers again to match.
4. **Spot-check (optional, cheap).** If you rebuild the corpus, a quick
   sanity check costs nothing: generate ~100 structures from the production
   checkpoint against a held-out slice of your rebuilt corpus and confirm
   Combined lands near the recorded NxN row-1 cell. Not required — the
   checkpoint already works and isn't in question — but a free confirmation
   that the rebuilt corpus matches the one actually used.

## What this doesn't require

- No retraining. The checkpoint exists, is public, and works.
- No new measurement. Every number currently cited on this checkpoint
  (§7's NxN row 1, §9.6, §10) stays exactly as reported.
- No new matcher logic, filter design, or research judgment call — the
  filter is already fully specified by `HANDOFF_CHECKPOINT_RECONSTRUCTION.md`'s
  Part A, written for a different purpose but directly reusable here.

## Report results

State the final row count, where the corpus was published (HF dataset repo
name, or a note that it was added as a repo asset instead), and confirm the
two citation fixes landed. If the optional spot-check in step 4 was run,
report the Combined number alongside the recorded NxN row-1 cell.

Same report-back mechanism as every other experiment here: see
`../SETUP.md` §6 — fork + PR preferred, relay as fallback.
