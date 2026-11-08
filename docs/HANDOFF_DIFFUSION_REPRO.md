# Handoff: reproducing the diffusion-vs-autoregressive comparison

**Priority: LOW — nice-to-have, not blocking.** The paper's design decision
(autoregressive, not diffusion) is already well-evidenced by a broader lineage
(see "Already-supported claim" below); nothing here is required to defend
that decision. This doc exists only because we want every claimed number to
be independently reproducible where feasible, and this one currently isn't.

**You have no prior context — read this whole file before touching
anything.**

## What's actually missing

Early in this project, three generation schemes were tried and abandoned
before settling on the autoregressive, discrete-token pipeline this paper
uses: a GemNet-backbone diffusion model, an ADiT-style latent-diffusion
model, and a two-stage model (autoregressive composition + diffusion-refined
geometry, whose refinement stage turned net-negative once the autoregressive
stage alone got strong enough). **None of these have surviving checkpoints,
training data snapshots, or raw results** — a sweep found six abandoned code
trees with empty checkpoint directories. This is a real, confirmed gap, not
an oversight: nobody has been able to locate the original numbers despite
looking.

## Already-supported claim — don't reproduce this part

The design decision itself does **not** rest on the missing comparison. The
final autoregressive pipeline has a real, sourced, independently-checkable
head-to-head against four *published* baselines on this project's own
screening pipeline:

| Model | Combined (project's own screening) |
|---|---:|
| **This project's final pipeline** | **38%** |
| MatterGen | 19% |
| ADiT | 10% |
| OMatG | 3% |
| CDVAE | 0% |

Reproducing *this* table needs no new model-building — MatterGen, ADiT,
OMatG, and CDVAE are all public, and this project's own checkpoint is public
(`rajkumars47/cues-alex-nolemat-lowhull` — not `ckpt_alex_nolemat_lowhull`,
which doesn't resolve; see `experiments_todo.md`'s Reproducibility section (R3) for the full citation
fix and the separate, still-open gap that this checkpoint's training corpus
isn't public). If anyone wants to independently
verify a number from this project, **start here, not with the sections
below** — it's a straightforward "run four public baselines + our public
checkpoint through the same scoring pipeline" task, not a research
reconstruction.

## What's genuinely missing, and its real cost

Two different asks, very different costs — don't conflate them:

### Cheap: reproduce the lineage's surviving design (G0–G2 / E1)

The two-stage model (autoregressive composition + diffusion-refined geometry)
and the final autoregressive-only pipeline are both **within this project's
existing training infrastructure** — no new model class needed, since the
two-stage model's autoregressive half is the same architecture family as
`ed_lite`/`ed_control`, just with an added diffusion-refinement head on top
of the geometry output. If someone wants to make the *lineage itself*
independently checkable (not the original missing numbers, but the
qualitative pattern — refinement helps early, then turns net-negative once
the base model is strong enough) — reproducing the two-stage model and
comparing it to the final pipeline on the current production corpus is a
genuinely tractable, moderate-cost project. This is the recommended
starting point if anyone wants to work on this at all.

**What's missing to actually do this:** the diffusion-refinement head's code
does not currently exist in the public `steering` repo (only the final,
refinement-free pipeline is public, since that's what CUES ships). Writing
it requires the original architecture's design (not currently documented in
enough detail in this public repo to reconstruct exactly) — treat this as a
real, if modest, engineering task, not a known quantity.

### Expensive, not recommended: a full diffusion/flow-matching baseline from scratch (G2 / E3)

Getting a true from-scratch number for "our own in-house diffusion attempt"
(as opposed to the lineage's later, more tractable two-stage model) means
building and training a new diffusion or flow-matching model class on the
production corpus — not a rerun, a new implementation. The compute itself is
unremarkable once built (under a day on 4×H100, same scale as everything
else in this project). The real cost is implementation and debugging time:
this project's own history — six abandoned diffusion/DiT code trees — is
evidence this is a multi-week undertaking independent of GPU access. **Not
recommended** as a place to spend volunteer GPU time; more GPUs would not
meaningfully speed this up, since the bottleneck is engineering, not
compute. Only worth attempting if someone specifically wants the experience
of building a diffusion baseline for its own sake.

## If you pick this up

Report which of the two asks above you attempted. For the cheap path
(two-stage lineage), report the diffusion-refinement head's design as you
implemented it (since the original isn't preserved, any faithful attempt is
itself a useful artifact going forward) alongside the Combined
before/after-refinement numbers. For the expensive path, we'd honestly
recommend reconsidering — but if you proceed anyway, same report-back
mechanism as every other experiment here (`../SETUP.md` §6).
