# Handoff: rebuilding the near-equilibrium (raw-vs-relaxed) benchmark table

**Priority: MUST — the only MUST-tier item in this entire index.** Not
because the underlying finding is in doubt or because this is central to
the paper's three-contributor thesis — it isn't; §10 is explicitly
supporting context. It's MUST because an unship-able section blocks
calling the draft complete, regardless of how peripheral that section is.
Every other item in `EXPERIMENTS.md` is Nice-to-have or Reproducibility;
this is the one thing actually gating a submission-ready draft.

**You have no prior context — read this whole file before touching
anything.** This is a real gap in a real paper, not a test.

**Verified end-to-end, 2026-11-09.** A fresh clone of both public repos, run
on a clean GPU VM with no prior state, completed the full generate → relax
→ score-raw → score-relaxed pipeline at a small `n` with no crash. Getting
there found and fixed five real bugs along the way — see "What running this
for real actually found" below. None of them require any special setup
beyond what's already in `SETUP.md`/this doc; `scripts/test_handoff.sh`
exercises this automatically now.

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

**Verified working recipe, 2026-11-09** (`scripts/smoke_near_equilibrium.py`
runs exactly this, at a tiny n, as part of `scripts/test_handoff.sh` — read
it for the actual working code, not just this prose):

1. **Generate** n≈1000 raw structures (natoms ≤ 20, the same matched prompt
   pool / protocol used for every other Combined number in this paper) from
   `rajkumars47/cues-alex-nolemat-lowhull`, via `screening.load_model()` +
   `screening.load_sources(..., split="eval")` + `screening.generate_batch()`.
   **Pass `version_id="d15_binrho_k7"` explicitly** —
   `generate_batch`'s own default (`"d6"`) has the wrong number of target
   segments for this checkpoint (misses the scratchpad segment), which makes
   `parse_target` fall back to a fuzzy parser that then misreads a lattice
   parameter as an element symbol and crashes every generation. Found running
   this for real; not documented anywhere else.
2. **Relax** that same population with MACE — `screening.relax_batch()`
   does this directly (FIRE, `fmax`/max-steps per its own `DEFAULTS`); no
   separate script needed, despite what an earlier draft of this doc said.
3. **Score both populations** (raw and relaxed) via `screening.screen_batch()`
   with `full=True` — called twice: once as
   `screen_batch(gen_results, gen_results, ..., full=True)` for the **raw**
   pass (both `GenResult` and `RelaxResult` objects expose `.atoms`, so
   passing the unrelaxed `gen_results` in the `relax_results` slot scores the
   raw structures), once as `screen_batch(gen_results, relax_results, ...,
   full=True)` for the **relaxed** pass. This `full=True` mode needs
   `dielectric/validity/` and `smact` — both now present/installable (see
   "What running this for real actually found" below), and a real MACE
   model file — see that section for how to get one without `/data/assets/...`
   or sudo.
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

## What running this for real actually found

Five real bugs, each only visible by actually running the pipeline fresh —
none were apparent from reading the code. All are now fixed; the pipeline
was re-verified end-to-end after each one, on a completely fresh clone of
both public repos:

1. **`dielectric/validity/` was missing from the public `steering` repo
   entirely.** `screen_batch(full=True)` imports `from validity.structural
   import load_stability_calc, load_mace` — a fresh clone had no
   `dielectric/validity/` directory at all, confirmed by directly listing
   it. Published, along with the one other file it needs
   (`dielectric/blocked_elements.py`) that wasn't already public either.
   **This also means `HANDOFF_C2_C3.md`'s "Score each finished run" step
   should now work too** (it uses the same `full=True` path) — not
   separately re-verified end-to-end, but it was blocked on the exact same
   missing directory.
2. **`requirements.txt`'s `pip install` failed unconditionally.**
   `pymatgen-core>=2025.10.7` isn't a real PyPI package name — there's no
   separate `pymatgen-core` distribution, the functionality lives in
   `pymatgen` itself. Pip aborted the whole file's install on this, silently
   taking `huggingface_hub` and everything listed after it down with it.
   Removed.
3. **`smact` (used by `validity/structural.py` for chemistry-filter checks)
   was never listed in either requirements file.** Added to
   `requirements-eval-optional.txt`.
4. **Two separate hardcoded MACE model paths, each pointing at
   `/data/assets/checkpoints/mace/...` on the original author's own
   machine**, in `validity/structural.py` (`MACE_MODEL_PATH`) and
   `chem/stability.py` (`DEFAULT_MACE_PATH`) — both bound as default
   *function-argument* values at import time, so the usual
   `path_override.py`-style "reassign the module attribute after import"
   trick silently doesn't work for either (the function's default was
   already fixed by the time any override code runs). Both now read from a
   shared `MACE_MODEL_PATH` environment variable instead, falling back to
   the original hardcoded path unchanged for existing internal use.
   `scripts/smoke_near_equilibrium.py` sets this env var and auto-downloads
   the public MACE-MP-0b3 model to `~/.cache/mace/` if it's not already
   there — no `/data/assets/...`, no sudo needed.
5. **A third, different hardcoded MACE path** (`chem/props/relax.py`'s
   `MACE_MODELS["mpa-0"]`, used by `relax_batch`'s own default) — this one
   *is* read fresh at call time rather than bound as a default, so no code
   patch was needed: `scripts/smoke_near_equilibrium.py` just passes
   `relax_model=` explicitly, reusing the same model already downloaded for
   scoring instead of needing a second download and a third vendored-file
   patch.

Also found and fixed, separately: `generate_batch`'s own default
`version_id` ("d6") has the wrong target-segment count for this checkpoint
and crashes every generation via a fuzzy-parse fallback — see Part A step 1
above for that one.

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
