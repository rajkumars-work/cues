# Handoff: rebuilding the near-equilibrium (raw-vs-relaxed) benchmark table

**Priority: MUST — the only MUST-tier item in this entire index.** Not
because the underlying finding is in doubt or because this is central to
the paper's three-contributor thesis — it isn't; §10 is explicitly
supporting context. It's MUST because an unship-able section blocks
calling the draft complete, regardless of how peripheral that section is.
Every other item in `EXPERIMENTS.md` is SHOULD or NICE; this is the one
thing actually gating a submission-ready draft.

**You have no prior context — read this whole file before touching
anything.** This is a real gap in a real paper, not a test.

## What this answers

A conditional generator only earns its keep if its raw, unrelaxed output is
already close to usable — otherwise the benchmark is measuring an MLIP
relaxation pipeline as much as the model. `experiments.md §10` makes this
point by comparing, for CUES and four published baselines (MatterGen,
OMatG, CrystaLLM, ADiT), how much an MLIP relaxation pass changes (a) each
model's raw structures geometrically (median volume change, `|ΔV|`) and (b)
each model's headline benchmark score (Combined, SUN, strict-on-hull rate).

## What's actually broken

The only number currently in `experiments.md §10` traces to a decommissioned
checkpoint:

> 🛑 The numbers below are placeholders pending a rebuild — not final. The
> only number this subsection had (median |ΔV| = 0.18%) does not survive a
> recheck: it traced to a decommissioned checkpoint...

Specifically: the original measurement
(`cues/experiments/relaxation_dependence/findings.md`, internal-only, not in
this public repo) used the `ed_premium` checkpoint — since superseded by
this paper's current, designated production checkpoint,
`rajkumars47/cues-alex-nolemat-lowhull`. A partial recompute (just the
median `|ΔV|` statistic, giving ≈1.6%) exists on the current checkpoint and
is used as a stand-in — but the **full** table (Combined, SUN, strict-on-hull
deltas) has never been rebuilt on it. The baseline models' own numbers
(MatterGen/OMatG/CrystaLLM/ADiT) are *not* the problem — those were measured
in a real, matched, already-documented run; they just haven't had a fresh
eyes-on sourcing check since.

## What's needed — two parts of very different cost

### Part A — rebuild CUES's own row (G1/E1: a few hours of GPU time, no new code)

This reuses the exact same generation → relaxation → scoring pipeline the
original internal run used, just pointed at the current checkpoint:

1. **Generate** n≈1000 raw structures (natoms ≤ 20, the same matched prompt
   pool / protocol used for every other Combined number in this paper — see
   `HANDOFF_C2_C3.md`'s "Score each finished run" for the
   `path_override.py`/`screening.py` setup) from
   `rajkumars47/cues-alex-nolemat-lowhull`.
2. **Relax** that same population with MACE-MP — `steering`'s own
   `crystals/lemat/code/py/lemat-genbench/scripts/relax_structures.py
   --mlip mace` does this (see its own `--help`/docstring for exact flags).
   The original run used FIRE, `ExpCellFilter`, `fmax=0.05`, ≤200 steps —
   match these unless you have a specific reason to deviate, and say so if
   you do.
3. **Score both populations** (raw and relaxed) through the same
   `comprehensive_multi_mlip_hull` (ORB+MACE ensemble) LeMat-GenBench
   configuration used everywhere else in this paper — this is the same
   `screening.py`/`combined.py` path as `HANDOFF_C2_C3.md`, just run twice
   (once per population) instead of once.
4. **Compute `|ΔV|`** directly from the relax step's own before/after
   volumes. Report the median (the headline figure the original run used —
   it's heavy-tailed, so mean alone is misleading) and the mean, same as
   the original.

### Part B — sourcing check on the four baselines (G0/E0: no GPU, no retrain)

The baseline numbers already in `experiments.md §10` and quoted in the
table below were measured in-house in a real, matched run — not lifted from
each model's own paper — documented in
`cues/experiments/relaxation_dependence/findings.md` (internal) and mirrored
publicly in `sources_helper.md §10` (also internal, not in this repo, but
the maintainer can check it directly). "Reconfirm" here means: re-read that
source doc, check the transcription into `experiments.md §10` against it,
and flag anything that doesn't match — not re-running MatterGen/OMatG/
CrystaLLM/ADiT yourselves. If you don't have access to the internal docs,
report Part A's results and leave Part B to the maintainer; it's zero
GPU-time either way.

## Deliverable table

Baseline rows are already known (don't re-derive them, see Part B above) —
fill in CUES's row and leave everything else as a sourcing check:

| Model | Combined raw | Combined pre-relax | Δ Combined | Stable (strict on-hull) raw | Stable pre-relax | median \|ΔV\| (%) |
|---|---:|---:|---:|---:|---:|---:|
| **CUES** (current production ckpt) | | | | | | |
| MatterGen | 15.9% | 22.3% | +6.4 | 2.2% | 4.6% | 0.20 |
| OMatG | 1.9% | 15.5% | +13.6 | 0.2% | 5.4% | 0.35 |
| CrystaLLM | 1.0% | 5.8% | +4.8 | 0.0% | 1.0% | 15.9 |
| ADiT | 17.7% | 19.1% | +1.4 | — | 14.4% | 909 |

(ADiT's raw strict-on-hull rate is blank in the original source too — that
run predated a MACE-in-spawn-worker fix and MACE returned zero valid
structures for the raw pass; don't backfill it, just carry the gap forward
honestly, same as the existing table does.)

## Report results

Fill in the CUES row above, plus:
- Whether Part B's sourcing check turned up any discrepancy in the four
  baseline rows (expected: no, but say so explicitly either way).
- The exact checkpoint/commit used for generation, and the exact MACE-MP
  relaxation settings if you deviated from FIRE/`ExpCellFilter`/`fmax=0.05`/
  ≤200 steps.
- n actually achieved for both the raw and relaxed populations (the
  original run lost a handful of structures per model between raw and
  relaxed scoring — report both n's, don't assume they match).

Same report-back mechanism as every other experiment here: see
`../SETUP.md` §6 — fork + PR preferred, relay as fallback. This doesn't
produce a large checkpoint of its own (it reuses the existing production
checkpoint), so there's nothing to separately upload to HF unless you also
save the generated/relaxed structure files — if you do, link them the same
way as any other large artifact.
