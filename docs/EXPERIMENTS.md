# Experiments index

This repo hosts handoff packages for experiments that would strengthen the
CUES paper's weaker-evidenced claims, or make an already-accepted claim
independently reproducible where it currently isn't. **Read
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) first** — it states the standard
every entry below is trying to meet and why both tiers (blocking and
nice-to-have) exist in the same index. Then come back here and follow the
design/handoff doc pair for the experiment you're running.

## Blocking gaps — the paper's thesis depends on these

| `limitations.md` item | Gap | Handoff available? | Docs |
|---|---|---|---|
| 1 | Narrow side-question: does decoder-only or the naive 12+12 split score higher on aggregate quality, independent of the (settled) allocation finding? | **Yes** | Design: [`dec_only_17L_replication_design.md`](dec_only_17L_replication_design.md) · Run: [`HANDOFF_C2_C3.md`](HANDOFF_C2_C3.md) |
| 4 | Training on a wide, non-stability-filtered corpus rests on indirect evidence, since the one direct test was withdrawn as a measurement artifact | **Yes** | Design: [`A5_wide_vs_narrow_corpus_design.md`](A5_wide_vs_narrow_corpus_design.md) · Run: [`HANDOFF_A5.md`](HANDOFF_A5.md) |
| 5 | Near-equilibrium benchmark table needs a rebuild | No | Self-serve, no external handoff needed — `limitations.md` item 5: "this needs no new training, only evaluation already within reach on an existing checkpoint." |

## Nice-to-have — low priority, not blocking, but worth having a plan

These don't gate anything the paper argues — see each item's own
`limitations.md` text for why — but per `REPRODUCIBILITY.md`, every gap gets
a written plan, not just the urgent ones.

| `limitations.md` item | Gap | Docs |
|---|---|---|
| 2 | Original in-house diffusion-vs-transformer comparison is unrecoverable (no surviving checkpoints/data) | [`HANDOFF_DIFFUSION_REPRO.md`](HANDOFF_DIFFUSION_REPRO.md) |
| 3 | Large-model (~7B) fine-tuning comparison rests on an informal, qualitative account | [`HANDOFF_LLAMA7B_REPRO.md`](HANDOFF_LLAMA7B_REPRO.md) |
| 6 | The NxN data-ablation matrix's underlying checkpoints, and several exemplar-steering/composition-fidelity numbers, rest on checkpoints with no surviving weights | [`HANDOFF_CHECKPOINT_RECONSTRUCTION.md`](HANDOFF_CHECKPOINT_RECONSTRUCTION.md) |
| 7 | A second DPO experiment (targeting generation novelty) was designed and partially run but never completed to a citable result | [`HANDOFF_DPO_NOVELTY.md`](HANDOFF_DPO_NOVELTY.md) |
| 8 | The production checkpoint's own training corpus was never published, and one checkpoint citation in this repo pointed to the wrong HF repo name | [`HANDOFF_PRODUCTION_CORPUS_PUBLICATION.md`](HANDOFF_PRODUCTION_CORPUS_PUBLICATION.md) |
| 9 | The build-order (§9.4) and scratchpad-field (§9.5) ablations rest on dedicated checkpoints never individually audited for survival | [`HANDOFF_TARGET_ABLATION_REPRO.md`](HANDOFF_TARGET_ABLATION_REPRO.md) |

Item 5 has no handoff package because it needs no external collaborator's
GPU time at all (self-serve, per the quoted reasoning above) — it's the one
entry in either tier without a `HANDOFF_*.md`. Item 8 is the cheapest entry
in this entire table — no training or retraining at all, just a filter
script already specified elsewhere in this repo and two corrected
citations.

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

The nice-to-have items are each a self-contained `HANDOFF_*.md` with no
companion design doc (lower priority warrants less scaffolding, not less
honesty) — each states its own priority, exactly what's missing, and either
a ready-to-run recipe or an honest statement of what engineering would need
to exist first. `HANDOFF_PRODUCTION_CORPUS_PUBLICATION.md` (item 8) is the
cheapest thing in this entire repo to pick up — no training, no retraining,
not even a measurement re-run, just a filter script and two citation fixes.
`HANDOFF_CHECKPOINT_RECONSTRUCTION.md`'s "Part B" is the next cheapest: no
training at all, just re-running existing measurements on a checkpoint that
already exists.
