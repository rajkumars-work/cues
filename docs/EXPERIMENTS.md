# Experiments index

This repo hosts handoff packages for experiments that would strengthen the
CUES paper's weaker-evidenced claims, or make an already-accepted claim
independently reproducible where it currently isn't. **Read
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) first** — it states the standard
every entry below is trying to meet and why every tier, urgent or not, gets
a written plan in the same index. Then come back here and follow the
design/handoff doc pair for the experiment you're running.

## At a glance — every experiment, one label and one order

**Three labels, nothing else.** Every experiment here carries exactly one:
**Must** (a real, stated paper limitation — tracked in `limitations.md`
too), **Nice-to-have** (a genuine but secondary evidentiary gap — *not* in
`limitations.md`), or **Reproducibility** (not an evidentiary gap at all,
just artifact availability or unfinished work — also not in
`limitations.md`). This mirrors `experiments_todo.md`'s own "At a glance"
table 1:1 — **that internal tracker is the source of truth; if this table
and that one ever disagree, trust `experiments_todo.md` and treat this page
as stale until fixed.** Ordered within each label by ascending cost
(cheapest/fastest first) — if you only have time for one thing, start at #1.

| # | Experiment | Label | Gap | Docs |
|---|---|---|---|---|
| 1 | §10 near-equilibrium rebuild | **Must** | Only published number traces to a superseded checkpoint, never rebuilt on the current one | Run: [`HANDOFF_NEAR_EQUILIBRIUM_REBUILD.md`](HANDOFF_NEAR_EQUILIBRIUM_REBUILD.md) |
| 2 | A5 — corpus breadth (wide vs. stability-filtered) | **Must** | The one direct wide-vs-narrow test was withdrawn as a measurement artifact; the claim rests on indirect evidence only | Design: [`A5_wide_vs_narrow_corpus_design.md`](A5_wide_vs_narrow_corpus_design.md) · Run: [`HANDOFF_A5.md`](HANDOFF_A5.md) |
| 3 | C4 — Llama-7B fine-tune failure | **Must** | Informal, unquantified account; no shared eval pipeline, no numbers | Run: [`HANDOFF_LLAMA7B_REPRO.md`](HANDOFF_LLAMA7B_REPRO.md) |
| 4 | A3 — curation vs. volume | Nice-to-have | Curation-beats-volume claim entangled with a novelty-reference artifact | — |
| 5 | B10/C8 — residual corpus-content difference | Nice-to-have | A checked but not fully cleared caveat on one architecture comparison | — |
| 6 | C2/C3 — architecture ablation | Nice-to-have | Decoder-only-vs-12+12 direction disagreement, between two baselines this paper doesn't recommend | Design: [`dec_only_17L_replication_design.md`](dec_only_17L_replication_design.md) · Run: [`HANDOFF_C2_C3.md`](HANDOFF_C2_C3.md) |
| 7 | A4 — GNoME cross-source augmentation | Nice-to-have | Augmentation gain confounded with corpus size | — |
| 8 | A2 — corpus-size floor | Nice-to-have | Size floor confounded with other training-recipe differences | — |
| 9 | C1 — diffusion-vs-transformer comparison | Nice-to-have | Original in-house comparison unrecoverable — not recommended to pursue | Run: [`HANDOFF_DIFFUSION_REPRO.md`](HANDOFF_DIFFUSION_REPRO.md) |
| 10 | R3 — production corpus publication | Reproducibility | Training corpus never published — **blocked on a pending maintainer decision**, see the doc | Run: [`HANDOFF_PRODUCTION_CORPUS_PUBLICATION.md`](HANDOFF_PRODUCTION_CORPUS_PUBLICATION.md) |
| 11 | R2 — DPO novelty experiment | Reproducibility | A designed DPO experiment was started but never finished | Run: [`HANDOFF_DPO_NOVELTY.md`](HANDOFF_DPO_NOVELTY.md) |
| 12 | R1 — NxN checkpoint audit | Reproducibility | 8 of 10 NxN-matrix checkpoints never individually audited for survival | Run: [`HANDOFF_CHECKPOINT_RECONSTRUCTION.md`](HANDOFF_CHECKPOINT_RECONSTRUCTION.md) |
| 13 | R4 — ablation checkpoints | Reproducibility | Build-order/scratchpad-field ablation checkpoints never audited | Run: [`HANDOFF_TARGET_ABLATION_REPRO.md`](HANDOFF_TARGET_ABLATION_REPRO.md) |

Items 4, 5, 7, 8 have no handoff doc yet (lower priority, nothing's been
packaged) — see `experiments_todo.md`'s own entries for their next steps if
you want to pick one up anyway.

## Must — real, stated paper limitations

Items here are written up in `limitations.md` itself — not evidentiary
side-notes, actual stated limitations of the paper, ordered cheapest first.
Being Must means "stays a limitation until closed," not "urgent to close" —
#3 below (C4) is explicitly *not* recommended to rush, for instance.

| `limitations.md` item | Gap | Docs |
|---|---|---|
| 3 | Near-equilibrium benchmark table's only number traces to a now-superseded checkpoint and has never been rebuilt on the current production one | Run: [`HANDOFF_NEAR_EQUILIBRIUM_REBUILD.md`](HANDOFF_NEAR_EQUILIBRIUM_REBUILD.md) |
| 2 | Training on a wide, non-stability-filtered corpus rests on indirect evidence, since the one direct test was withdrawn as a measurement artifact | Design: [`A5_wide_vs_narrow_corpus_design.md`](A5_wide_vs_narrow_corpus_design.md) · Run: [`HANDOFF_A5.md`](HANDOFF_A5.md) |
| 1 | Large-model (~7B) fine-tuning comparison rests on an informal, qualitative account | Run: [`HANDOFF_LLAMA7B_REPRO.md`](HANDOFF_LLAMA7B_REPRO.md) |

## Nice-to-have — genuine evidentiary gaps, but not stated limitations

These don't gate anything the paper argues, and per the three-label rule
(2026-11-08) they're deliberately **not** in `limitations.md` — but per
`REPRODUCIBILITY.md`, every gap still gets a written plan, not just the
urgent ones. Unlike the Reproducibility table below, these are genuine (if
secondary) evidentiary gaps — the underlying evidence really is informal or
unrecoverable, not just an artifact that became hard to get hold of after
the fact.

| Gap | Docs |
|---|---|
| Curation-vs-volume claim entangled with a novelty-reference artifact | — (see `experiments_todo.md` A3) |
| A checked-but-not-fully-cleared corpus-content caveat | — (see `experiments_todo.md` B10/C8) |
| Decoder-only-vs-12+12 direction disagreement, between two baselines this paper doesn't recommend. **Downgraded from Must, 2026-11-04** — resolving it either way doesn't change the paper's argument, and it was removed from `limitations.md` on 2026-11-08 for exactly that reason. | Design: [`dec_only_17L_replication_design.md`](dec_only_17L_replication_design.md) · Run: [`HANDOFF_C2_C3.md`](HANDOFF_C2_C3.md) |
| GNoME augmentation gain confounded with corpus size | — (see `experiments_todo.md` A4) |
| Corpus-size floor confounded with other training-recipe differences | — (see `experiments_todo.md` A2) |
| Original in-house diffusion-vs-transformer comparison is unrecoverable (no surviving checkpoints/data). **Removed from `limitations.md`, 2026-11-08** for the same reason as above — the design decision is otherwise well-evidenced. | Run: [`HANDOFF_DIFFUSION_REPRO.md`](HANDOFF_DIFFUSION_REPRO.md) |

## Reproducibility — artifact availability, not an evidentiary gap

**[Moved out of `limitations.md`/`conclusion.md` Appendix A, 2026-11-08.]**
None of these weaken any claim the paper makes: in every case the result
stands, and what's missing is purely a reader's ability to independently
obtain the checkpoint or dataset that produced it (or, for R2, an
experiment that was simply never finished). They're tracked as R1–R4 in
`experiments_todo.md`'s own "Reproducibility" section — this table is the
public mirror of that section.

| Tracked as | Gap | Docs |
|---|---|---|
| R3 | The production checkpoint's own training corpus was never published, and one checkpoint citation in this repo pointed to the wrong HF repo name | [`HANDOFF_PRODUCTION_CORPUS_PUBLICATION.md`](HANDOFF_PRODUCTION_CORPUS_PUBLICATION.md) |
| R2 | A second DPO experiment (targeting generation novelty) was designed and partially run but never completed to a citable result | [`HANDOFF_DPO_NOVELTY.md`](HANDOFF_DPO_NOVELTY.md) |
| R1 | The NxN data-ablation matrix's underlying checkpoints, and several exemplar-steering/composition-fidelity numbers, rest on checkpoints with no surviving weights | [`HANDOFF_CHECKPOINT_RECONSTRUCTION.md`](HANDOFF_CHECKPOINT_RECONSTRUCTION.md) |
| R4 | The build-order (§9.4) and scratchpad-field (§9.5) ablations rest on dedicated checkpoints never individually audited for survival | [`HANDOFF_TARGET_ABLATION_REPRO.md`](HANDOFF_TARGET_ABLATION_REPRO.md) |

R3 is the cheapest entry in this entire repo — no training or retraining at
all, just a filter script already specified elsewhere in this repo and two
corrected citations.

## Running an experiment

**If you only have time for one thing, pick #1 above** —
[`HANDOFF_NEAR_EQUILIBRIUM_REBUILD.md`](HANDOFF_NEAR_EQUILIBRIUM_REBUILD.md).
It's also one of the cheapest: no training at all, eval-only on the
existing production checkpoint, a few hours of GPU time.

C2/C3 and A5 are the two full train-from-scratch experiments — they share
this repo's general setup ([`../SETUP.md`](../SETUP.md)) and training
launcher (`../scripts/train_arm.sh`). Each experiment's own `HANDOFF_*.md`
is self-contained from there — background, exact launch commands, scoring,
and how to report results back. Read the matching design doc first if you
want the full "why," not just the "how."

- **C2/C3 (architecture ablation)** — encoder-decoder vs. decoder-only,
  three architecture arms × two seeds. Already smoke-tested end-to-end
  ([`../smoketests/ed_lite_20261105.md`](../smoketests/ed_lite_20261105.md)),
  plus a fast kickoff/checkpoint-save check for all three arches,
  2026-11-08.
- **A5 (corpus ablation)** — wide vs. stability-filtered training corpus,
  two arms, architecture held fixed at `ed_lite`. Smoke-tested for the
  corpus-filtering path specifically — see `../smoketests/` for the most
  recent `ed_lite_lowhull_*.md` entry, plus the same 2026-11-08 kickoff
  check as C2/C3.

If you're running both, note the efficiency overlap documented in the A5
design doc §2: a C2/C3 `ed_lite` run on the full wide corpus already covers
A5's secondary (full-wide) arm — don't train it twice.

Every item in the Nice-to-have and Reproducibility tables, except C2/C3
(which kept its design doc from its Must days), is a self-contained
`HANDOFF_*.md` with no companion design doc (lower priority warrants less
scaffolding, not less honesty) — each states its own priority, exactly
what's missing, and either a ready-to-run recipe or an honest statement of
what engineering would need to exist first.
`HANDOFF_PRODUCTION_CORPUS_PUBLICATION.md` (R3) is the cheapest thing in
this entire repo to pick up — no training, no retraining, not even a
measurement re-run, just a filter script and two citation fixes.
`HANDOFF_CHECKPOINT_RECONSTRUCTION.md`'s "Part B" (part of R1) is the next
cheapest: no training at all, just re-running existing measurements on a
checkpoint that already exists.
