# Experiments index

This repo hosts handoff packages for experiments that would strengthen the
CUES paper's weaker-evidenced claims — each one maps to a specific item in
the paper's own `limitations.md`. Start here, then follow the design/handoff
doc pair for the experiment you're running.

| `limitations.md` item | Gap | Handoff available? | Docs |
|---|---|---|---|
| 1 | Narrow side-question: does decoder-only or the naive 12+12 split score higher on aggregate quality, independent of the (settled) allocation finding? | **Yes** | Design: [`dec_only_17L_replication_design.md`](dec_only_17L_replication_design.md) · Run: [`HANDOFF_C2_C3.md`](HANDOFF_C2_C3.md) |
| 2 | Original in-house diffusion-vs-transformer comparison is unrecoverable (no surviving checkpoints/data) | No | Out of scope — `limitations.md` item 2 states this directly: "We do not treat the original, narrower in-house comparison as worth new compute to recover, given the broader evidence already in hand." |
| 3 | Large-model fine-tuning comparison rests on an informal, qualitative account | No | Out of scope — `limitations.md` item 3: "not something we view as a priority: the paper's actual argument only needs the qualitative finding, stated plainly as such, which we already have." |
| 4 | Training on a wide, non-stability-filtered corpus rests on indirect evidence, since the one direct test was withdrawn as a measurement artifact | **Yes** | Design: [`A5_wide_vs_narrow_corpus_design.md`](A5_wide_vs_narrow_corpus_design.md) · Run: [`HANDOFF_A5.md`](HANDOFF_A5.md) |
| 5 | Near-equilibrium benchmark table needs a rebuild | No | Self-serve, no external handoff needed — `limitations.md` item 5: "this needs no new training, only evaluation already within reach on an existing checkpoint." |

Items 2, 3, and 5 have no handoff package here because none of them need an
external collaborator's GPU time — see the quoted reasoning above for each
(copied verbatim from `limitations.md`, not re-derived).

## Running an experiment

Both available experiments share this repo's general setup
([`../SETUP.md`](../SETUP.md)) and training launcher
(`../scripts/train_arm.sh`). Each experiment's own `HANDOFF_*.md` is
self-contained from there — background, exact launch commands, scoring, and
how to report results back. Read the matching design doc first if you want
the full "why," not just the "how."

- **C2/C3 (architecture ablation)** — encoder-decoder vs. decoder-only,
  three architecture arms × two seeds. Already smoke-tested end-to-end
  ([`../smoketests/ed_lite_20261105.md`](../smoketests/ed_lite_20261105.md)).
- **A5 (corpus ablation)** — wide vs. stability-filtered training corpus,
  two arms, architecture held fixed at `ed_lite`. Smoke-tested for the
  corpus-filtering path specifically — see `../smoketests/` for the most
  recent `ed_lite_lowhull_*.md` entry.

If you're running both, note the efficiency overlap documented in the A5
design doc §2: a C2/C3 `ed_lite` run on the full wide corpus already covers
A5's secondary (full-wide) arm — don't train it twice.
