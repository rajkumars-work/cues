# Handoff: reproducing the large-model fine-tuning comparison

**Priority: MUST — this gap is tracked as a real paper limitation
(`cues/docs/limitations.md` item 1), unlike most other items in this index.**
That said, closing it is not urgent: the paper's design decision (train a
task-sized model from scratch rather than fine-tune a much larger pretrained
model) doesn't depend on a precise, controlled number — the qualitative
finding alone is sufficient and already in hand, and stated as a limitation
rather than overclaimed. This doc exists so the finding can eventually move
from "informal account" to "independently reproducible," for anyone with the
infrastructure to do it — but the E3 engineering cost below is real, and
rushing this isn't recommended.

**You have no prior context — read this whole file before touching
anything.**

## What's missing

Before training CUES from scratch, a colleague fine-tuned a much larger
pretrained model (on the order of 7B parameters) on a crystal-generation
task. The account is real but informal: no shared evaluation pipeline, no
numbers, just two qualitative failure modes reported — (a) the output was
often difficult to parse as a valid structure at all, and (b) once
parseable, it was only weakly responsive to property conditioning. Nobody
has been able to locate logs, checkpoints, or outputs from the original
attempt.

## Step 0 — ask before building anything (free)

If you're in contact with the original colleague, ask directly whether
anything survives (logs, a checkpoint, even partial output) before
attempting a full reproduction — this could resolve the gap at zero cost.
Only proceed past this step if nothing turns up.

## What a controlled replication actually needs

This is **not** a small extension of the other experiments in this repo — it
is a different scale and kind of infrastructure, and that's the real reason
it's never been done, not lack of interest:

- **A comparable open-weight ~7-8B model.** No specific checkpoint is
  prescribed here — pick any permissively-licensed model in that size class
  available to you (e.g., something in the Llama-3-8B class, or a comparably
  sized open alternative). The point is parameter count in the same rough
  range as the original attempt, not a specific model family.
- **A new tokenizer bridge.** This project's own pipeline uses a
  from-scratch 16k-vocab SentencePiece tokenizer trained on its own
  structure/property text format. A 7B-class model comes with its own
  pretrained tokenizer; bridging the two (either by adapting this project's
  structure encoding to the target model's existing vocabulary, or by
  extending that vocabulary) is new engineering, not a config change.
- **A different training stack.** Fine-tuning a model at this scale (even
  with LoRA/parameter-efficient methods) is a different infrastructure
  problem than training this project's ~230M-parameter models from scratch
  — different memory profile, different library stack, likely different
  hardware requirements than a single L4/consumer GPU.
- **A new eval-pipeline bridge.** To score the fine-tuned model's output
  with this project's own metrics (SUN/MSUN/Combined, steerability), its
  output needs to be parsed into the same structure representation
  `screening.py`/`combined.py` expect — this doesn't exist yet for an
  arbitrary foreign model's output format.

## Honest cost estimate

Compute is the least certain part and probably not the bottleneck:
parameter-efficient fine-tuning (LoRA) might fit comfortably under a day on
4×H100-class hardware; a full fine-tune would plausibly need more. Either
way, the gating cost is the engineering above — a new tokenizer bridge, new
training infrastructure, and a new eval bridge, none of which exist yet. This
is comparable in kind (if not scale) to the diffusion-reproduction gap in
`HANDOFF_DIFFUSION_REPRO.md`: real, bounded, but a multi-day-to-multi-week
engineering project in its own right, separate from any GPU-hours question.

## What a complete result looks like

A fine-tuned 7-8B-class model, evaluated with this project's own
screening/scoring pipeline, on the production corpus used elsewhere in this
paper. **Note:** that corpus (`alex_nolemat_lowhull`) is now public —
`rajkumars47/data_alex_nolemat_lowhull`, as of 2026-11-09 — but at 225,103
rows, a close approximation rather than the exact 249,117-row original used
to train the production checkpoint (see that repo's own README, or
`experiments_todo.md`'s Reproducibility section R3, for why). Either use it
as-is and note the approximation, or substitute the public
`rajkumars47/data_big_d15` instead (the corpus behind
`ckpt_autolabel_v1_binrho_k7`/`ckpt_ed_enclight_6_18`) if an exact match to a
production checkpoint's own training file matters more than corpus choice,
noting the substitution in your report.
Reporting at minimum:

- **Parseability rate** — fraction of generations that parse into a valid
  structure at all (the first reported failure mode).
- **Steerability** — property-conditioning agreement, chance-corrected per
  axis: $S = \max(0, (\bar c - 1/K) / (1 - 1/K))$, where $\bar c$ is the raw
  fraction of generations landing in the requested bin and $K$ is that axis's
  number of bins. Same definition this project's task-sized models are
  scored on, so directly comparable to their numbers.

Even a negative or inconclusive result is useful here — the original
account's qualitative claim stands or falls on whatever a controlled attempt
actually finds.

## Report results

Report the two metrics above, which base model and tokenizer-bridging
approach you used, hardware and wall-clock time, and anything that broke
along the way (tokenizer bridge issues especially — that's the part most
likely to surface a genuine surprise). Same report-back mechanism as every
other experiment here: see `../SETUP.md` §6 — fork + PR preferred, relay as
fallback, large model weights to your own HF account, not this repo.
