# Handoff: fine-tuning comparison against a 2026 frontier open-weight model (Kimi)

**Priority: Nice-to-have — a genuine secondary evidentiary gap, but not a
paper limitation.** This is a *new* experiment, not a replacement for
`HANDOFF_LLAMA7B_REPRO.md` (`limitations.md` item 1 / `EXPERIMENTS.md` #3,
C4). That doc stays exactly as it is and still needs to be run — it's
reproducing a specific informal account of a ~7-8B-class fine-tune failing at
this task, and that account is still the paper's actual stated limitation.
This doc exists because a colleague pointed out that a reader in late 2026
is at least as likely to ask "sure, but what about a real frontier-scale
open model?" as they are to ask about a Llama-3-8B-class one — and that's a
fair question the paper doesn't currently address at all, in either
direction.

**You have no prior context — read this whole file, and
`HANDOFF_LLAMA7B_REPRO.md`, before touching anything.**

## Why this is a separate doc, not an edit to the Llama-7B one

The two experiments are answering different questions, at genuinely
different scales — not the same question with a newer model name swapped in:

- **`HANDOFF_LLAMA7B_REPRO.md`** reproduces a specific claim: a model "on
  the order of 7B parameters" was fine-tuned and failed in two specific,
  qualitative ways. The model class there is load-bearing — it's what makes
  the reproduction *of that account*, not a different experiment.
- **This doc** asks a question the original account never covered at all:
  does a current frontier-scale open-weight model do any better? As of this
  writing (2026-11-25), there is no ~7-8B successor to swap in — the
  open-weight ecosystem's "comparable to Llama-3-8B" tier has largely moved
  on, and the frontier open-weight models people actually mean today when
  they say "a strong open model" are two to three orders of magnitude
  larger:
  - **Kimi** (Moonshot AI): mainline releases (K2, K2.5, K2.6, K3) are
    1T–2.8T-parameter MoE models (~32B active for the K2 line). The
    smallest openly released Kimi variant, Kimi Linear, is still 48B total
    / 3B active — already well past "7-8B class."
  - **Muse** (Meta's post-Llama line, first released April 2026): Muse
    Glimmer is a dense 30B model; Muse Spark Open is an 816B-total / 42B-
    active MoE model. Neither is a 7-8B successor either.

  Given a choice between the two, this doc follows the colleague's
  suggestion and uses **Kimi**, but the real point is the same as the
  original doc's: the gap is "evaluate a current frontier open model," not
  "evaluate this one specific checkpoint." **Verify the current Kimi
  lineup before committing to one** — this landscape moves fast enough
  that something newer than K3 may exist by the time you read this.
  `huggingface.co/moonshotai` is the authoritative source, not this
  paragraph.

That jump in scale is the whole reason this is a new doc rather than a
find-and-replace on the old one: it invalidates almost every concrete
estimate in `HANDOFF_LLAMA7B_REPRO.md` (hardware, wall-clock time, even
whether a full/LoRA fine-tune is a sane thing to attempt at all), while the
*shape* of the engineering problem (tokenizer bridge, eval bridge, cost
honesty) carries over.

## Step 0 — ask before building anything (free)

Same as the original doc: if anyone has already tried prompting or
fine-tuning a frontier open model on a crystal-generation-shaped task (even
informally, even on a different corpus), ask first. This is a new enough
question that the answer is plausibly "nobody's tried it yet" — but check.

## A scale-aware decision you'll need to make: fine-tune, or prompt only?

The original doc assumed fine-tuning (LoRA, specifically) was the obvious
approach, because a 7-8B dense model fits that assumption. At 1T-parameter
MoE scale, that assumption doesn't carry over for free, and picking the
wrong one wastes real time:

- **Full or LoRA fine-tuning a 1T-class MoE model** needs multi-node
  infrastructure, not a single box. Even with only ~32B parameters active
  per forward pass, all experts generally still need to be resident (sharded
  across nodes) for training, and the standard open fine-tuning stacks for
  models at this scale assume multi-GPU-node setups (e.g. multiple 8×H100 or
  8×H200 nodes), not a single consumer/workstation GPU or even a single
  4×H100 box. If you have access to that kind of cluster, this is the
  faithful continuation of the original experiment's design (comparable
  fine-tuning methodology, just at the model's real scale) — do it if you
  can.
- **Prompting / in-context conditioning only (no fine-tuning)** is a much
  cheaper first cut and answers a related but different question: not "how
  does a fine-tuned frontier model do," but "how does a frontier model do
  out of the box, given examples and property-conditioning instructions in
  its prompt." This is almost certainly within reach on far more modest
  hardware (inference-only, via the model's own API/endpoint or a hosted
  serving provider — no local weights needed for the 1T-class checkpoints
  at all). It does **not** reproduce the original fine-tuning comparison,
  and your report must say so plainly rather than letting the two get
  conflated.

**If you don't have multi-node fine-tuning infrastructure, the prompting
path is the honest and recommended option** — it's a real result, just a
different and more modest one than a full repro. State which path you took
up front in your report; don't present a prompting-only result as if it
answered the fine-tuning question.

## What a controlled attempt needs (either path)

- **A current Kimi checkpoint**, confirmed available to you (open weights on
  Hugging Face for self-hosted inference/fine-tuning, or the model's own
  hosted API if you're going the prompting-only route and don't want to
  stand up multi-node serving yourself just to run inference).
- **An output-parsing bridge.** Whatever the model emits — however you
  prompt it to represent a crystal structure — needs to be parsed into the
  same structure representation `screening.py`/`combined.py` expect, same
  as the original doc's eval-pipeline-bridge requirement. This exists
  regardless of which path (fine-tune or prompt-only) you take.
- **A tokenizer bridge — fine-tuning path only.** If you fine-tune, you hit
  the same problem the original doc describes: this project's from-scratch
  16k-vocab SentencePiece tokenizer doesn't match Kimi's own pretrained
  vocabulary, and bridging them is new engineering. The prompting-only path
  sidesteps this entirely — you're writing text prompts in the model's
  native tokenization, not training against this project's tokenizer.
- **A new training stack — fine-tuning path only.** Multi-node MoE
  fine-tuning (even parameter-efficient) is a different infrastructure
  problem than either this project's from-scratch training or the original
  doc's single-box 7-8B LoRA plan. Expect to need whatever distributed
  training stack the model's own release recommends (check
  `huggingface.co/moonshotai` for the current guidance) rather than
  improvising one.

## Honest cost estimate

- **Prompting-only path:** low. Inference against a hosted frontier model
  API costs API credits, not GPU procurement, and a few hundred generations
  (enough for the parseability/steerability metrics below to be meaningful)
  is a cheap, fast experiment — plausibly an afternoon, dominated by writing
  and iterating on the output-parsing bridge, not compute.
- **Fine-tuning path:** high, and genuinely uncertain without knowing your
  specific cluster access. This is a materially bigger ask than the
  original doc's "LoRA on 4×H100-class hardware, under a day" estimate —
  think multi-node, multi-day, and don't commit to it without confirming
  you actually have that infrastructure available. If you don't, take the
  prompting-only path instead rather than stalling on infra that doesn't
  exist yet.

Either way, as with the original doc: the engineering (output-parsing
bridge, and for fine-tuning, the tokenizer bridge and distributed training
setup) is the real gating cost, not raw GPU-hours.

## What a complete result looks like

Kimi (fine-tuned, or prompted — state which), evaluated with this project's
own screening/scoring pipeline, on the production corpus used elsewhere in
this paper. **Note:** that corpus (`alex_nolemat_lowhull`) is now public —
`rajkumars47/data_alex_nolemat_lowhull`, as of 2026-11-09 — but at 225,103
rows, a close approximation rather than the exact 249,117-row original used
to train the production checkpoint (see that repo's own README, or
`experiments_todo.md`'s Reproducibility section R3, for why). Either use it
as-is and note the approximation, or substitute the public
`rajkumars47/data_big_d15` instead (the corpus behind
`ckpt_autolabel_v1_binrho_k7`/`ckpt_ed_enclight_6_18`) if an exact match to a
production checkpoint's own training file matters more than corpus choice,
noting the substitution in your report. Reporting at minimum:

- **Parseability rate** — fraction of generations that parse into a valid
  structure at all.
- **Steerability** — property-conditioning agreement, chance-corrected per
  axis: $S = \max(0, (\bar c - 1/K) / (1 - 1/K))$, where $\bar c$ is the raw
  fraction of generations landing in the requested bin and $K$ is that axis's
  number of bins. Same definition this project's task-sized models (and the
  Llama-7B-class reproduction) are scored on, so directly comparable to both.

Even a negative or inconclusive result is useful here — this is new
evidence either way, not a result anyone's already committed to.

## Report results

Report the two metrics above; which Kimi checkpoint you used and when you
checked it was current; whether you took the fine-tuning or prompting-only
path and why; hardware/API costs and wall-clock time; and anything that
broke along the way. Same report-back mechanism as every other experiment
here: see `../SETUP.md` §6 — fork + PR preferred, relay as fallback, large
model weights to your own HF account, not this repo.

## If you add this to the experiment index

This doc isn't yet cross-referenced from `docs/EXPERIMENTS.md`'s table or
from `experiments_todo.md` (the latter lives outside this repo and is the
source of truth per that file's own instructions) — add a row for it there,
labeled Nice-to-have, if you want it tracked alongside the rest rather than
standing alone.
