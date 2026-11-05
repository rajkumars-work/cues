# Design doc: from-scratch replication of `dec-only 17L` (ArchitectureStudy.md ablation)

**Adapted for public release (2026-11-05)** from the original internal design doc.
`§5`/`§6` below have been trimmed to drop internal GCP account/project/bucket
identifiers and day-by-day operational back-and-forth that don't apply to a
different machine — the scientific content (why this replication exists, the
qk-norm bug, the matched hyperparameter table, the expected trajectory) is
otherwise unchanged. Date of original: 2026-09-03.

## 0. Why replicate

The completed `dec_only_17L_edim1024_d15` run (finished 2026-09-01, see
[[project_dec_only_17L_run_status]]) scored 0% SUN / ~10% MSUN at n=100, far below
`ArchitectureStudy.md`'s reported 81.8/82.4 Combined SUN+MSUN for the same architecture. Root-cause
work this week found three distinct defects in that run, only two of which were previously known:

1. **LR-floor exposure (known, root-caused 2026-09-03).** The run spent 59.6% of its budget
   (steps 0–19,215 of 32,250) under a ~10%-of-peak floored LR from a supervisor bug, vs. 15.7% for
   the healthy `ed_control_12_12` run. Fixed in the supervisor; this run's checkpoint is not
   recoverable by further training (see [[project_dec_only_17L_run_status]] for the dose-response
   argument).
2. **EMA-reset-on-resume (known, fixed 2026-09-03).** `ed_train.py` never called
   `EMA.load_state_dict()` on resume, so the EMA shadow reset every stop-and-check epoch instead of
   accumulating across the run. Fixed in `wip/ed/ed_train.py` (commit `bd9579f`). Does not
   discriminate this run from the healthy control (both had the bug), so it's a real defect but not
   the explanation for the gap.
3. **QK-norm never applied to decoder-only self-attention — NEW FINDING, not yet fixed.** This is
   the headline finding of this design doc's code review (§2 below): the actual code that trained
   this run never applied qk-norm to causal self-attention at all, despite `ArchitectureStudy.md`
   §3.1 stating this exact fix is *required* for decoder-only DDP stability, and despite
   `d_model.py`'s own docstring asserting it's applied. This looks like the most architecturally
   fundamental of the three issues and was not part of the LR/EMA writeup already pushed.

Given three real defects (one now confirmed as likely the most fundamental, and unfixed), a
faithful from-scratch replication — not a continuation of the existing checkpoint — is the only way
to find out whether the original ablation numbers are reproducible on this codebase today.

## 1. Dataset

`~/data/assets/datasets/big_d15/big_d15_autolabel_binrho_k7.csv` — confirmed present on `basic`
(219k rows via `.audit.md.json`/`.json` siblings alongside it). This is the exact file named in
`ArchitectureStudy.md` §2 and the exact file the completed `dec_only_17L` run itself used
(`stop_and_check_supervisor_dec_only.py`'s `DATA_CSV`). **No dataset change needed** — same file,
same split logic (`--split eval` for the held-out set used by `run_screen.py`/genbench).

SentencePiece vocab: ablation retrains a vocab=16,000 tokenizer per architecture from the training
corpus (§2). The completed run also used `--vocab_size 16000`. **Decided 2026-09-04 (per user):
reuse the existing run's `model_sp.model`/`model_sp.vocab` as-is**, rather than retraining fresh —
same corpus, same vocab size, low risk. Implemented by seeding the new run's `bucket_dir` with those
two files before first launch; `stop_and_check_supervisor_dec_only.py`'s existing sync-in step
(copies `model_sp.model`/`model_sp.vocab` from `bucket_dir` to `local_dir` if present) picks them up
automatically, and `ed_train.py` skips retraining a tokenizer it finds already in the checkpoint
dir. No code changes needed.

## 2. Code — the qk-norm finding, in detail

**Confirmed by direct code read this session**, comparing `~/packages/wip/ed/ed_model.py` (the
main/live lineage that trained the actual run) against
`~/packages/wip/steering/crystals/lemat/code/py/ed/ed_model.py` (the bundled lineage that mirrors
byte-for-byte to the public `github.com/rajkumars-work/steering` repo, and which has clearly been
kept in sync with the ablation's own conclusions — see its `configs.py`'s `config_ed` preset
comment referencing `ArchitectureStudy.md` directly).

Both lineages' `MHA.__init__` construct `self.q_norm`/`self.k_norm` (`nn.LayerNorm(edim/heads)`)
whenever `qk_norm=True`. Both lineages' `MHA.forward()` apply them correctly in the cross-attention
branch (`if self.is_cross_attn and o is not None:`). They diverge in the **self-attention / causal
branch** (the `else:`, used by decoder-only's `DBlock` since `d_model.py` calls
`MHA(edim, heads, dropout=dropout, causal=True, qk_norm=qk_norm)` with no `o`):

- **Bundled/public `ed_model.py` (lines 220–228)** — applies the norm:
  ```python
  else:
      qkv = self.c_attn(x)
      q, k, v = qkv.split(self.n_embd, dim=2)
      k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
      q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
      v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
      if self.qk_norm:
          q = self.q_norm(q)
          k = self.k_norm(k)
      if self.causal and key_padding_mask is not None:
          ...
  ```
- **Main `wip/ed/ed_model.py`** — the equivalent `else:` branch has no `if self.qk_norm:` block at
  all. `q_norm`/`k_norm` are constructed in `__init__` but never invoked outside cross-attention.

`d_model.py` (identical in both lineages) documents the self-attn qk-norm as **required**:

> "QK-norm on self-attn: encoder-decoder models inherit stabilization from the cross-attention
> pathway; a pure causal stack lacks that and is prone to attention-logit explosion under DDP.
> Required."

...and its `DBlock.__init__` defaults `qk_norm: bool = True`. So the completed `dec_only_17L` run
had `qk_norm=True` configured and threaded all the way down to `MHA.__init__`, constructing live
`LayerNorm` modules — but those modules were **dead code** for every decoder-only self-attention
call in the entire run. This is exactly the bug `ArchitectureStudy.md` §3.1 says was found and
fixed for the ablation's own decoder-only training; the fix evidently never made it from the
bundled/public lineage back into the main lineage that actually trains production runs.

**Two other DDP-stability fixes from §3, checked this session:**

- **Fix #2 (fp32 lm_head + logit clamp for bf16 overflow):** already present in `d_model.py`
  (identical file in both lineages, lines ~107–119: `autocast(enabled=False)` block, fp32 matmul,
  `torch.clamp(logits, min=-30.0, max=30.0)`). **No action needed.**
- **Fix #3 (cond-probe DDP-unsafe, `--cond_probe_interval 0`):** the `cond_probe` feature doesn't
  exist at all in the main lineage's `ed_train.py` (bundled-only code). Moot regardless, since the
  completed run and any near-term replication run train on a single GPU (no DDP), so the specific
  cross-rank desync this fix guards against cannot occur. **No action needed unless a multi-GPU run
  is chosen (§5) — if so, need to re-check the DDP-safety of any probing before enabling it.**

**Fixed 2026-09-03**: ported the 3-line fix into `~/packages/wip/ed/ed_model.py`'s `MHA.forward()`
self-attention branch, matching the bundled version exactly (diff-confirmed identical). Strict
correctness fix (dead code → active); no-op for the encoder-decoder path since `Block`'s
self-attention `MHA` never requested `qk_norm=True` in the first place. Documented in
`~/packages/wip/dielectric/BUGS.md` (all three `ArchitectureStudy.md` §3 fixes, with
per-fix status in this codebase). Does not retroactively fix the completed `dec_only_17L` checkpoint
— relevant for this replication run and any future decoder-only training.

## 3. Hyperparameters — comparison table

| Parameter | Ablation (`ArchitectureStudy.md` §2, §8) | Completed run's actual flags (`stop_and_check_supervisor_dec_only.py`) | Recommended for replication |
|---|---|---|---|
| Data | `big_d15_autolabel_binrho_k7.csv` (219k) | same | same |
| LR (peak, cosine) | 5e-5 | 5e-5 | 5e-5 |
| Warmup frac | 0.15 | 0.15 | 0.15 |
| Batch size / GPU | 64 | 24 (L4 OOM ceiling for this arch; 32/36/40 all OOM'd) | 24, single GPU (unless hardware changes — §5) |
| Max seq len | 512 | 512 | 512 |
| Grad clip norm | 0.5 | 0.5 | 0.5 |
| Prop dropout | 0.2 | 0.2 | 0.2 |
| Label dropout | 0.0 | not in flag list (default — confirm default is 0.0) | 0.0 |
| Model type | decoder-only (`d`) | `--model_type d` | same |
| Layers | 17 | 17 | 17 |
| Heads | 16 | 16 | 16 |
| edim | 1024 | 1024 | 1024 |
| Vocab size | 16,000 | 16,000 | 16,000 |
| qk-norm | on (cross + decoder self-attn) | **configured on, but self-attn path dead code (§2 bug)** | on, **with the §2 fix applied** |
| Total batch (tokens/step) | 524,288 via DDP grad-accum | 524,288 via auto-derived grad-accum, single GPU (~1008 seqs/step vs. ablation's ~1024) | 524,288 (keep matched) |
| Epochs / steps | 150 / ~32,100 | 150 / 32,250 (214 rounds to a slightly different per-epoch step count on this hardware) | 150 / ~32,100–32,250, from scratch, one continuous run |
| cond_probe | disabled (`--cond_probe_interval 0`), required for their 2-4 GPU DDP | N/A (single GPU, feature doesn't exist in this lineage) | N/A |
| Hardware | 2× H100 80GB DDP | 1× L4 24GB, no DDP | see §5 |

**Conclusion: the hyperparameters were already essentially matched.** The completed run was not a
hyperparameter-mismatch problem — it was the LR-floor severity, EMA-reset, and (newly found)
qk-norm dead-code bugs layered on otherwise-correct config. This replication is mainly a "fix the
three bugs and run clean, uninterrupted, from scratch" exercise, not a hyperparameter redesign.

## 4. Expected quick-eval / trajectory — rough estimate, explicitly caveated

**Important caveat:** `ArchitectureStudy.md` §9's per-step trajectory table (loss/bg/eps/rho/all-5
at steps 8k/12k/16k/20k/24k/32.1k) is for **ed-lite** (6+18 enc-dec, edim=768), not dec-only-17L —
confirmed by matching final eval-loss values (0.74 in the §9 table vs. 0.741 for ed-lite in §4;
dec-only-17L's own final eval loss is 0.976, which doesn't match the table's endpoint at all).
`ArchitectureStudy.md` has **no genuine per-step trajectory for dec-only-17L**, only its final
(step-32,100) numbers. Anything below is an extrapolation, not a recorded fact for this
architecture — treat as a rough prior to sanity-check against, not a target to hit exactly.

Known dec-only-17L anchor points (from §4):
- Final eval loss (EMA, step 32,100): **0.976** (vs. ed-lite's 0.741 — a different architecture
  converges to a different absolute loss floor, so don't expect ed-lite-shaped numbers).
- Final Combined SUN+MSUN: **81.8 LeMat-raw / 82.4 MG-prerelax**, Steerability all-5 strict **68.6**.

Qualitative shape borrowed from ed-lite's §9 trajectory (same recipe: same total_batch_size,
same 150-epoch budget, similar param count) as the best available prior:
- Eval loss should drop fastest in the first ~30-40 epochs, reach a minimum somewhere in the
  75–110 epoch range (~16,000–24,000 steps), then flatten or drift slightly upward through 150
  epochs — NOT necessarily at the same epoch as ed-lite's minimum (epoch 75), since dec-only's
  final loss (0.976) is markedly higher and its convergence dynamics differ (single causal stack
  vs. cross-attention-stabilized decoder).
- Easy axes (parse-rate, element-set, natoms) should saturate early (by ~epoch 40), matching
  ed-lite.
- Hard property-bin axes (bg_bin, eps_bin) likely track the eval-loss minimum and may decline
  slightly afterward; rho_bin (structure-direct, no MLIP in the loop) likely keeps improving past
  the loss minimum, similar to ed-lite's pattern — but this is the most speculative extrapolation
  in this section.
- quick_eval's SUNMetric-based combined score is the most important signal to actually watch: the
  bar for "this run isn't structurally broken" is simply **any nonzero SUN rate at all**, since
  every increment of the completed run scored ~0.0–0.14 combined quick_eval throughout, and the
  final real n=100 genbench check confirmed a hard 0% SUN. If the qk-norm fix is the dominant
  cause, we'd expect quick_eval's combined score to clear that noise floor meaningfully earlier
  in the new run (there is no principled way to predict at which step, hence the recommendation
  below).

**Recommendation:** run the same offline trajectory probe methodology `ArchitectureStudy.md` §9
used (n=30 stratified eval prompts, every ~4,000 steps: parse-rate, element-set, natoms, bg_bin,
eps_bin, rho_bin, all-5-strict) on this new run. This produces the first real dec-only-17L
trajectory record, which the ablation itself never had — useful for this replication and for
future dec-only-17L work regardless of outcome. In addition, keep the existing quick_eval-per-epoch
signal from the stop-and-check supervisor as a cheap continuous check, understanding it's noisy at
this scale (n≈30-100 depending on config) per [[project_quick_eval_zero_diagnosis]].

## 5. Hardware / wall-clock

The original ablation trained on 2× H100 80GB via DDP, ~17h wall time for 32,100 steps. On a single
L4, the decoder-only (17L, edim=1024) arm OOMs above `--batch_size 24` and runs at roughly
**~2.15–2.2 steps/min** steady-state — at that rate a full continuous 32,100-step run is
**≈243–249 hours ≈ 10.1–10.4 days** of GPU-active time, an order of magnitude longer than the
original H100 run. The ed-lite/ed-control arms (fewer params) are faster per step, but budget for
the same order of magnitude on a single L4 if running all three arms there. **This is exactly the
kind of gap external GPU access (multiple GPUs, or H100-class hardware) is meant to close** — on
4×H100, each arm should land comfortably under a day (see the main CUES paper's
`experiments_todo.md` G2 compute tier for the calibration this estimate is based on).

**Run one continuous invocation per arm, not a segmented per-epoch respawn.** The original run hit
two separate LR-schedule bugs specifically because its supervisor killed and relaunched
`ed_train.py` once per epoch, and `--max_steps_cap` silently doubled as `--lr_max_steps` (the cosine
schedule's horizon) across relaunches — each relaunch re-floored the LR schedule, then jumped it
back up on the next segment. There is no technical requirement for the per-epoch respawn:
`ed_train.py` already checkpoints mid-run (`--save_interval`) and does not need to be
killed/relaunched to make progress durable. `scripts/train_arm.sh` in this repo launches
`ed_train.py` once, with `--max_steps`/`--lr_max_steps` (via `--epochs`/`--lr_schedule_epochs`) set
to the same constant for the whole run, and only restarts on a genuine stall (no new log output for
15 minutes after startup), not on any fixed schedule. If you build your own launcher instead, keep
this property — it's the single most consequential lesson from the original run's bug history.

**Watch for silent qk-norm regressions.** The dead-code bug in §2 above was specifically a case
where a fix landed in one code lineage (the version vendored into the public `steering` repo) but
not another (the lineage that actually trained the completed run). If you're running code from
somewhere other than this repo's pinned dependency on `steering`, diff `d_model.py`'s/`ed_model.py`'s
`MHA.forward()` self-attention branch against the fix described in §2 before trusting a `dec_only`
result.

## 6. Checklist before launching

1. Confirm the qk-norm self-attention fix (§2) is present in whatever `ed_model.py`/`d_model.py`
   you're running (it already is in `steering`'s vendored copy, which this repo depends on).
2. Confirm the tokenizer: reuse a single `model_sp.model`/`model_sp.vocab` (vocab=16,000) across all
   three arms rather than retraining one per arm — same corpus, same vocab size, lower variance, and
   this is what the hyperparameter table in §3 assumes.
3. Launch continuously per arm (§5) — do not segment into per-epoch relaunches.
4. Score each arm the same way: `run_screen.py`/`screening.py` to generate, then
   `eval_combined.py`/`combined.py` for Combined SUN+MSUN, both self-consistent and against an
   external hull reference, matching this design's methodology table (§3) and n.
5. Record results per arm per seed; see the main repo README for where final numbers should land.
