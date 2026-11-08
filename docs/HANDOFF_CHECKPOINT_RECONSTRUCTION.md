# Handoff: reconstructing evidence that rests on checkpoints we no longer have

**Priority: LOW — nice-to-have, not blocking.** Nothing in this doc is
required for the paper's central thesis. It exists so that results currently
described as "historical record, not independently re-runnable" can become
independently re-runnable, for anyone with spare compute who wants to do it.

**You have no prior context — read this whole file before touching
anything.** Two unrelated reconstruction jobs live here, bundled because both
are "retrain with existing, already-public infra" rather than "build something
new" (that harder case — reproducing the project's abandoned diffusion
lineage — is a separate doc, `HANDOFF_DIFFUSION_REPRO.md`, don't confuse the
two).

## Part A — the NxN matrix's ten underlying checkpoints

### What this is

`cues_design.md §3`'s data-side evidence (§7 of the results, "the cleanest
data-side evidence this paper has") is a 10×10 matrix: ten training corpora ×
ten exemplar pools, one architecture (`ed_lite`) held fixed throughout. The
100-cell matrix itself — the actual numbers — is fully recorded and audited
(see the private `cues_nxn.md` for the complete methodology and tables; not
reproduced here since it's not public). **What is not confirmed is whether
all ten of the underlying trained checkpoints still exist anywhere** — unlike
the architecture-ablation checkpoints (`HANDOFF_C2_C3.md`), no explicit
audit of these ten has been done. Two of the ten are known to survive (see
below); the other eight have not been checked.

### Step 0 — audit before retraining anything (free)

Before training anything, check whether each checkpoint already exists —
local disk, the project's GCS buckets, or Hugging Face. This costs nothing
and might mean most of this doc is unnecessary:

| # | NxN row | Training corpus | Checkpoint status |
|---|---|---|---|
| 1 | `alex_nolemat_lowhull` | Alexandria, LeMat-disjoint, `e_above_hull ≤ 0.1` (249,117 rows) | **Known to survive** — this is the current production checkpoint, public on HF at `rajkumars47/cues-alex-nolemat-lowhull` (not `ckpt_alex_nolemat_lowhull` — that name doesn't resolve). Its training corpus is **not** public under any name checked; see `HANDOFF_PRODUCTION_CORPUS_PUBLICATION.md` (`experiments_todo.md`'s Reproducibility section, R3). Ranked #1 in the matrix. |
| 2 | `production` (`big_d15`) | MP+Alex+WBM+diel+MAX mix (242,170 rows) | **Known to survive** — this row reused the existing `ckpt_ed_enclight_6_18` checkpoint rather than training a new one for the matrix. Public on HF: `rajkumars47/ckpt_ed_enclight_6_18` / `rajkumars47/data_big_d15`. |
| 3 | `alex` | Alexandria, unfiltered (250,010 rows) | Not confirmed either way — audit first |
| 4 | `alex_onhull` | Alexandria, `e_above_hull ≤ 0` (174,986 rows) | Not confirmed — audit first |
| 5 | `alex_nolemat` | Alexandria, LeMat-disjoint, unfiltered (250,000 rows) | Not confirmed — audit first |
| 6 | `alex_nolemat_onhull` | Alexandria, LeMat-disjoint, `e_above_hull ≤ 0` (84,117 rows) | Not confirmed — audit first |
| 7 | `alex_nolemat_stability_sampled` | Alexandria, LeMat-disjoint, preferentially low-`e_above_hull` (not a strict filter; 241,323 rows) | Not confirmed — audit first |
| 8 | `mp20_ref` | MatterGen's published MP-20 split, no property tokens (45,229 rows) | Not confirmed — audit first |
| 9 | `alex_mp20_ref` | MatterGen's published Alex-MP-20 split, no property tokens (675,199 rows) | Not confirmed — audit first |
| 10 | `d15+gn` | `big_d15` + GNoME mixed in (gnome 50.6% / mp 24.4% / alex 21.1% / wbm 2.2% / diel 1.6% / max 0.1%; 489,860 rows) | Not confirmed — audit first |

For whichever of rows 3–10 turn out to still exist (on your own access, or
the maintainer's), skip retraining that row — just confirm it still scores
consistently with its recorded matrix cells (a cheap spot-check: regenerate
~100 structures on its own held-out pool, compare Combined to the recorded
cell).

### Rebuilding missing datasets (rows 3–10, as needed)

All source data is public:

- **Alexandria** (rows 3–7): the full dataset is public (RWTH's Alexandria
  release). Each row's filter is fully specified and mechanical:
  - `alex`: no filter.
  - `alex_onhull`: `e_above_hull <= 0`.
  - `alex_nolemat`: filtered against `LeMaterial/LeMat-Bulk` overlap
    (structure match, not just composition — same matcher settings as this
    project's novelty scoring: pymatgen `StructureMatcher`, `ltol=0.2,
    stol=0.3, angle=5`).
  - `alex_nolemat_onhull`: `alex_nolemat` ∩ `e_above_hull <= 0`.
  - `alex_nolemat_stability_sampled`: `alex_nolemat`, sampled with
    preference toward low `e_above_hull` (not a hard cutoff — a weighted
    sample; the exact sampling weight isn't preserved in the public record,
    so this one row is a close approximation at best, not an exact rebuild,
    if attempted from scratch).
- **`mp20_ref` / `alex_mp20_ref`** (rows 8–9): MatterGen's own published
  MP-20 / Alex-MP-20 splits — pull directly from MatterGen's public release
  rather than rebuilding from Alexandria/MP yourself; this project's copies
  are just those published splits, reformatted to this project's
  element/atom-count prompt schema.
- **`d15+gn`** (row 10): `big_d15` (`rajkumars47/data_big_d15`, already
  fetched for C2/C3) mixed with GNoME (DeepMind's public release) at the
  stated composition. The exact row-level mixing script isn't yet ported to
  this public repo — porting it is a small, well-specified task (filter
  GNoME to the stated proportion, concatenate with a matching subsample of
  `big_d15`), not a research question. Flag this as the one real engineering
  gap in this part, if you pick up row 10.

### Retraining (for whichever rows are confirmed missing)

Same recipe for every row — `ed_lite` (6 encoder + 18 decoder, edim=768),
**75 epochs** (not the usual 150 — this matrix used a shorter, matched
budget across all ten rows):

```bash
DATA_CSV=./data/corpus/<rebuilt_row>.csv \
EPOCHS=75 \
STEERING_PATH=../steering/crystals/lemat/code/py \
bash scripts/train_arm.sh ed_lite 0
```

`train_arm.sh` already supports this — no script changes needed, just
`EPOCHS=75` instead of the default 150 and `DATA_CSV` pointed at the row's
rebuilt corpus.

### Re-scoring

Sample 100 prompts per row (`natoms < 20`, seed `1234`, "naked" format —
`<elements> | <natoms>`, no property tokens) from that row's own held-out
split, generate, and score with the same `screening.py`/`combined.py` path
as every other experiment here. Compare the new Combined score against the
recorded matrix cell for that row × its own pool (the diagonal of the
matrix) as the first sanity check before attempting the full 10×10
re-score.

## Part B — results computed on `ckpt_autolabel_v1_binrho_k7` (lost)

### What this is

A different, older checkpoint — `ckpt_autolabel_v1_binrho_k7` (encoder-decoder,
12+12, trained on the same `big_d15_autolabel_binrho_k7` corpus this project
still has) — has **no weight files on disk anywhere** (confirmed internally;
only its training corpus survives). Several numbers currently cited as
historical record rest on it:

- A steerability baseline (element-set 99.6%, atom-count 95.0%, density
  0.872, band gap 0.774, dielectric 0.722, aggregate $I_{\mathrm{vec}} =
  0.890 \pm 0.030$).
- A composition-fidelity reference (encoder-decoder: 99.6% element-faithful,
  95.0% exact atom count, n=50).
- Four of §8's five exemplar-steering target-kind results (tacit expert
  intuition, Pareto-front probing, disjunctions, joint targets). The fifth
  (population-level Combined range) was already spot-checked on a current
  checkpoint (`ckpt_ed_control_12_12_d15`; public on HF as
  `rajkumars47/ckpt_backup_ed_control_12_12_d15`) on 2026-08-12 and confirmed
  qualitatively — no further work needed there.
- The per-prompt steerability-prediction analysis in `analysis.md §11.4`
  (the AUC ≤ 0.56 finding, and the underlying SUN-driver base rates and
  composition-rarity regression it's drawn from) — added to this list
  2026-11-08; not previously covered here, but the same checkpoint and the
  same "re-run the measurement, don't retrain" fix as everything else in
  this part.

### Why this is cheap, not a retrain

**This does not require recovering or reconstructing the lost checkpoint.**
Every number above is a measurement, not a training result — re-running the
same measurement on a checkpoint that already exists and is public
(`ckpt_ed_control_12_12_d15` / `rajkumars47/ckpt_backup_ed_control_12_12_d15`,
or the current production `ckpt_alex_nolemat_lowhull` /
`rajkumars47/cues-alex-nolemat-lowhull`) answers the same question. If the new numbers
land close to the historical ones, that's independent confirmation; if they
diverge, that's worth knowing too. Either way, no GPU training is needed —
just generation + scoring on a checkpoint already in hand.

### What to run

**Note on sourcing (added 2026-11-08):** steps 2–5 below used to just point at
section numbers in this project's private paper drafts
(`experimental_design.md`, `experiments.md`, `analysis.md`) — not usable by
anyone without access to those files. The methodology each needs is inlined
below instead; a 2026-11-08 cold-start test confirmed this was a real,
blocking gap, not a nitpick.

1. Pick one current checkpoint: **`cues-alex-nolemat-lowhull`** (public on HF
   at `rajkumars47/cues-alex-nolemat-lowhull` — not `ckpt_alex_nolemat_lowhull`,
   which doesn't resolve), the production one, for consistency with the rest
   of the paper going forward.
2. **Steerability baseline.** Generate against five property axes (band gap,
   dielectric, density, element set, atom count). For each axis, steerability
   is *chance-corrected* agreement between what was requested and what was
   generated: $S = \max(0, (\bar c - 1/K) / (1 - 1/K))$, where $\bar c$ is the
   raw fraction of generations landing in the requested bin and $K$ is that
   axis's number of bins. Report $S$ per axis and compare against the
   historical reference (element set 99.6%, atom count 95.0%, density 0.872,
   band gap 0.774, dielectric 0.722 raw compliance; chance-corrected ceiling
   $I_{\mathrm{vec}} = 0.890 \pm 0.030$ for band gap+dielectric+density
   jointly).
3. **Composition-fidelity reference.** n=50 generations: for each, check
   whether the generated structure's element set exactly matches the
   requested elements ("element-faithful") and whether its atom count exactly
   matches the requested count ("exact atom count"). Report both rates
   against the historical reference (99.6% / 95.0%).
4. **§8.1–§8.4 target-kind experiments (tacit intuition, Pareto-front,
   disjunction, joint-target) — currently blocked, not just under-documented.**
   Each of these needs the *original* exemplar pools/seed sets (e.g. the six
   ionic-fluoride prototypes behind the tacit-intuition result) to reproduce
   the same comparison, and those pools aren't yet in this public repo. If you
   want to attempt this anyway, construct a comparable exemplar pool yourself
   using this project's own "prospecting" method: draw a seed set $S$ from a
   structural class or property region you can name (anything a user can
   point at rather than name numerically), estimate $S$'s empirical joint
   distribution over (element set, atom count), then sample from that
   distribution and generate conditioned on each draw — this guarantees every
   request is realizable, since the conditioning values come from real
   materials. Report the result as a fresh analogous test, not a reproduction
   of the original numbers. Porting the original pools is a separate,
   still-open task — flag it if you hit this.
5. **§11.4's per-prompt steerability-prediction check.** Generate a sample
   (the original used n≈754–814), score each for Combined/SUN/MSUN via the
   standard screening path, and check whether a prompt-level predictor built
   from `chem/sun_predictor` (in `steering/crystals/lemat/code/py/dielectric/`
   — already in the public vendored copy, no porting needed) achieves better
   than AUC 0.56 at predicting per-prompt stability. The historical finding is
   that it doesn't reliably exceed chance at the per-prompt level, only at a
   cell-aggregate level — confirming or overturning that is the point of
   re-running it.

## Report results

For either part, report: which rows/checkpoints you confirmed still exist
(Part A step 0) vs. retrained, the new numbers alongside the historical ones
in a side-by-side table, and anything that didn't match the documented
recipe exactly (filter definitions, row counts, etc. — this doc's numbers
were last confirmed 2026-11-07; if a public source has since changed,
report the discrepancy rather than silently using the new numbers).

Same report-back mechanism as every other experiment here: see
`../SETUP.md` §6 — fork + PR preferred, relay as fallback, checkpoints to
your own HF account.
