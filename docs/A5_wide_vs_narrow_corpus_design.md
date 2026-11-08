# Design doc: wide vs. stability-filtered corpus (A5)

Date: 2026-11-07. New design, not an internal-doc adaptation — no sanitization
pass needed.

## 0. Why this experiment

CUES's design (`cues_design.md §3`) trains on a corpus that spans stable,
metastable, and unstable regions of the energy landscape, rather than one
filtered to the convex hull — the stated reasoning is that stability is
better controlled at generation time (exemplar steering, §4) than imposed by
curation. This is a specific, nameable claim, and the one *direct* test of it
on record was withdrawn: a prior wide-vs-narrow comparison reported "wide/
stable corpus, no-hull conditioning beats control by +15pp Combined (→93%)",
and that result was later traced to a measurement artifact — roughly 11pp
from an `ema_final`-checkpoint capture bug (memorization 17%→84-88%) and
roughly 3pp from hull-label removal, not a real effect. **That run's setup
and numbers are not reused anywhere in this document** — see `sources_helper.md`
§7d/§15 for the retraction detail if you want the full account. The claim
currently rests on indirect evidence from a separate experiment (the
data/exemplar NxN matrix, `experiments.md §7`), which is good evidence but
wasn't designed as a dedicated test of this specific question.

This experiment is a clean rerun from scratch: same architecture, same corpus
size, same training budget as the production recipe — only the training
corpus's stability filtering varies. It is the direct counterpart to
`dec_only_17L_replication_design.md` (the C2/C3 architecture ablation): same
spirit of root-causing a prior artifact and designing around it rather than
repeating it, applied to the data axis instead of the architecture axis.

**The question:** does training on a wide corpus (stable + metastable +
unstable) genuinely match or beat training on a stability-filtered (on-hull /
low-`e_above_hull`) corpus, when exemplar steering handles stability at
generation time? A "yes" supports `cues_design.md §3`'s claim directly; a
"no" would mean the wide corpus's apparent sufficiency in the NxN matrix
doesn't hold up to a dedicated, controlled test.

## 1. Corpus construction

The wide corpus is the same file the C2/C3 experiment already uses — **do not
refetch it separately**: `rajkumars47/data_big_d15` →
`big_d15_autolabel_binrho_k7.csv`, fetched via `scripts/fetch_assets.sh` (same
as C2/C3's setup, step 3). Confirmed by downloading and inspecting the file
directly (2026-11-07): 241,741 data rows (3 rows have missing/unparseable
`e_above_hull`, excluded below), columns `source, target, id, origin, label,
e_above_hull, in_lemat`. The column is literally named `e_above_hull` — no
schema translation needed from this project's own `_onhull`/`_lowhull`
convention. (This is a direct training CSV, not the same schema as
`cues/scripts/nxn/build_big_d15_recon.py`'s membership-reconstruction file —
confirmed separately rather than assumed, since that script's own
`e_above_hull` field name belongs to a different file with a different
provenance.)

Threshold, this project's own convention: **`lowhull` = `e_above_hull <=
0.1`** (matching production's own `ckpt_alex_nolemat_lowhull` tier — see
`cues_checkpoint_designation`). Measured on the actual file:

| Filter | Row count | Fraction of wide corpus |
|---|---|---|
| `onhull` (`e_above_hull <= 0`) | 36,900 | 15.3% |
| **`lowhull` (`e_above_hull <= 0.1`)** | **163,227** | **67.5%** |
| Full wide corpus | 241,741 | 100% |

`lowhull` is used, not `onhull`, per the task's instruction to match
production's own tier unless there's a specific reason not to — there isn't
one here, and `onhull` at 15.3% would make the narrow arm's row count a much
larger departure from the wide arm's, confounding "stability filtering" with
"much less data" more than necessary.

**Row-count matching (the design decision):** `experiments_todo.md`'s A5
entry says "same architecture, size, and epoch budget... only
stability-filtering varies" — read literally. The **primary** comparison is
therefore two arms of equal row count:

- **Narrow arm:** the 163,227-row `lowhull` filter above.
- **Wide, size-matched arm:** a random 163,227-row subsample of the full wide
  corpus (same seed-determinism as any other arm — see §4), so total
  corpus size is held fixed and only stability composition varies.

The **full, unsubsampled wide corpus (241,741 rows)** is a secondary arm, not
the clean ablation — it varies both stability filtering and corpus size at
once relative to the narrow arm, so a result there doesn't isolate the
stability-filtering question by itself. Include it only if compute allows;
don't substitute it for the row-count-matched pair above.

**Script:** `scripts/build_lowhull_corpus.py` derives both corpora
deterministically from the already-fetched wide CSV — no new HF dataset repo,
no second download, one scripted filter step so provenance stays obviously
clean:

```bash
python3 scripts/build_lowhull_corpus.py \
  --wide ./data/corpus/big_d15_autolabel_binrho_k7.csv \
  --out-dir ./data/corpus
```

Writes `data/corpus/big_d15_lowhull.csv` (163,227 rows) and
`data/corpus/big_d15_wide_matched.csv` (163,227-row random subsample, seed 0
by default). The untouched wide CSV at the `--wide` path remains available
for the secondary, full-size arm.

## 2. Efficiency note: don't train the wide arm twice

If the colleague (or whoever runs C2/C3) already trains `ed_lite` on the
**full** wide corpus for the architecture experiment, that run is **not** a
substitute for this experiment's wide-matched arm — C2/C3's `ed_lite` run
uses the full 241,741-row file, not the 163,227-row size-matched subsample
this experiment's primary comparison needs. It *is*, however, directly usable
as this experiment's **secondary, full-wide-corpus arm** (§1) for free — say
so explicitly in any report so that arm isn't retrained from scratch if a
C2/C3 `ed_lite` run already exists. The size-matched primary arm still needs
its own run regardless.

## 3. Architecture and hyperparameters

This is a **data ablation, not an architecture one** — architecture is held
fixed at **`ed_lite`** (6 encoder + 18 decoder layers, edim 768), the paper's
recommended configuration, at exactly the matched hyperparameters
`scripts/train_arm.sh` already encodes for that arm (vocab 16,000, LR 5e-5,
warmup 0.15, grad-clip 0.5, prop-dropout 0.2, label-dropout 0.0, max_seq_len
512, total batch 524,288 tokens/step, 150 epochs) — see
`dec_only_17L_replication_design.md §3` for the full table and rationale;
nothing in that table changes here. The only thing that varies between the
two primary arms is `DATA_CSV`.

## 4. Launch

`train_arm.sh` is already parameterized by `DATA_CSV` as an environment
variable — no script changes needed, just point it at a different corpus:

```bash
# Narrow (lowhull) arm
DATA_CSV=./data/corpus/big_d15_lowhull.csv \
STEERING_PATH=../steering/crystals/lemat/code/py \
bash scripts/train_arm.sh ed_lite 0

# Wide, size-matched arm
DATA_CSV=./data/corpus/big_d15_wide_matched.csv \
STEERING_PATH=../steering/crystals/lemat/code/py \
bash scripts/train_arm.sh ed_lite 0
```

Run both arms at (at minimum) one seed each; a second seed per arm is
valuable but not required the way it is for C2/C3's three-way architecture
comparison — this is a two-arm test, not a three-way one with a known
historical direction-disagreement to resolve.

## 5. Wall-clock / hardware estimate

Same architecture (`ed_lite`, 238M params) and same per-step cost as C2/C3's
`ed_lite` arm — corpus size differs (163,227 rows vs. 241,741), which affects
epoch wall-clock roughly linearly but does not change architecture-driven
per-step cost. `experiments_todo.md`'s own calibration for this item: **G2**
compute tier, "each arm comfortably under a day on 4×H100, both arms in
parallel on separate GPUs." On a single L4 (this repo's smoke-test-scale
hardware, not the recommended scale for a real run), expect roughly the same
order of magnitude as C2/C3's `ed_lite` arm at the corresponding corpus
fraction — budget proportionally more on weaker/shared hardware, per
`HANDOFF_C2_C3.md`'s own wall-clock guidance.

## 6. Scoring methodology

Identical to C2/C3's scoring path (`HANDOFF_C2_C3.md` "Score each finished
run") — same `screening.py`/`combined.py` pipeline, both the self-consistent
and external-hull (`--genbench`) scoring paths, same `path_override.py`
setup. Score both arms with matched `n` and the same prompt pool; consistency
across arms matters more here than any single methodology choice, same as
C2/C3.

## 7. Checklist before launching

1. Fetch the wide corpus if not already present (`scripts/fetch_assets.sh`) —
   do **not** refetch it if the C2/C3 experiment already pulled it; same file.
2. Run `scripts/build_lowhull_corpus.py` to derive the two A5-specific CSVs
   (§1). Confirm both land at the expected row count (163,227 each) before
   launching — a mismatch here means the wide CSV changed since this doc was
   written (2026-11-07) and the hardcoded row counts above are stale.
3. Confirm which arm(s), if any, a concurrent C2/C3 run already covers (§2) —
   don't duplicate the full-wide-corpus secondary arm if it already exists.
4. Launch narrow and wide-matched arms per §4. Keep `DATA_CSV` as the only
   difference between their launch commands.
5. Score both arms identically (§6) and report per `HANDOFF_A5.md`.
