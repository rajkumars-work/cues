# Cold-start test: does `EXPERIMENTS.md` actually work for someone with zero context?

**Final result: real bugs found and fixed.** Host `gpu-train-2` (single
NVIDIA L4). Unlike the two prior smoke tests (`ed_lite_20261105.md`,
`ed_lite_lowhull_20261107.md`), which validated one specific `HANDOFF_*.md`
end-to-end, this test started from `EXPERIMENTS.md` itself with a genuinely
cold Claude Code session — no context from the repo-maintaining session,
told only to clone the repo, read `EXPERIMENTS.md`, and try to contribute.
The question was whether the *index*, not just one handoff doc, actually
gets a stranger to something runnable.

## What the cold session found

Chose the A5 corpus ablation (narrow/`lowhull` arm, `ed_lite`, one L4).

| # | Finding | Severity |
|---|---|---|
| 1 | `train_arm.sh` never overrode `ed_train.py`'s default `--max_steps 4000` — a "full" 150-epoch run silently stopped at ~28 epochs (19%) with the LR schedule nowhere near decayed, while still reporting as complete | **Critical — a correctness bug, not a docs gap** |
| 2 | Scoring scripts referenced everywhere (`combined.py`, `run_screen.py`, `eval_combined.py`) aren't in the public `steering` repo; `crystals/score.py` doesn't cover the two documented scoring methods, and `n`/prompt-pool are never specified | Blocking — no documented way to complete scoring |
| 3 | The `--seed`-unsupported warning printed to stderr, which the documented launch command discards | Easy fix |
| 4 | Tokenizer-sharing across arms (required by the design doc) isn't explained anywhere | Doc gap |
| 5 | A5's launch commands assume two GPUs, no single-GPU note | Doc gap |
| 6 | `fetch_assets.sh` always pulls ~14GB (corpus + 3 checkpoints) with no corpus-only option; the one VM-disk warning that already existed was buried in a different smoke-test file, never propagated | Known since 2026-11-07, never fixed until now |
| 7 | `HANDOFF_CHECKPOINT_RECONSTRUCTION.md` Part B recommended `ckpt_alex_nolemat_lowhull` (404s) and cited private doc sections (`experimental_design.md §6.4`, `experiments.md §9.2`, §8.1–8.4) a public reader can't open | Blocking for that handoff specifically |
| 8 | Minor naming inconsistency, `ckpt_ed_control_12_12_d15` vs. the actual HF repo `ckpt_backup_ed_control_12_12_d15` | Confusing but not blocking |

**What checked out:** the qk-norm fix is present in `steering`'s `ed_model.py`;
EMA state restores correctly on resume; all documented training flags exist;
`path_override.py` works as described; the model builds correctly (238.0M
params, matching docs) and trained up to the point of a documented,
anticipated OOM at the undocumented-for-this-arm batch size.

The cold session also surfaced that items 7/8 were stale — the clone it was
working from predated a same-day reproducibility-audit session's fixes to
those exact citations, a useful reminder that a contributor's clone being a
few commits behind can resurface already-fixed problems.

## What got fixed as a result (2026-11-08, same day)

1. **`train_arm.sh` now passes `--max_steps 0`** — `ed_train.py`'s own
   documented way to derive the step budget from `--epochs` instead of its
   4000-step default. **Verified on real hardware**: relaunched `ed_lite` on
   this same L4 and confirmed the startup diagnostic now reads
   `max_steps:32250` (= 215 steps/epoch × 150 epochs) instead of `4000` —
   and 32250 independently matches this project's own `dec_only_17L_repl`
   checkpoint's actual historical training length, a reassuring cross-check.
2. **The `--seed` warning now also writes to the run's own log file**, not
   just stderr, so it's visible regardless of how the wrapper's own output
   is redirected.
3. **`fetch_assets.sh` gained a `--corpus-only` flag**, documented in both
   `SETUP.md` and `HANDOFF_A5.md`.
4. **`HANDOFF_A5.md` gained an explicit single-GPU note.**
5. **`HANDOFF_CHECKPOINT_RECONSTRUCTION.md`'s Part B rewritten**: correct
   checkpoint name, and steps 2/3/5 now inline the actual methodology
   (chance-corrected steerability formula, composition-fidelity protocol,
   the SUN-driver check's public code path) instead of citing private files.
   Step 4 (§8.1–§8.4) is honestly flagged as currently blocked — the
   original exemplar pools aren't public yet — rather than papered over.
6. **`HANDOFF_LLAMA7B_REPRO.md` had the same private-citation bug**,
   independently caught during the cleanup pass above — fixed the same way.
7. **New: `scripts/quick_selftest.sh`**, a fast (seconds, not hours)
   end-to-end plumbing check — tiny architecture, 2000-row data slice, 10
   real training steps, a real checkpoint — covering exactly the kind of bug
   this test found, so the next contributor doesn't need a multi-hour run to
   discover it. Verified working on this same L4: 5/5 checks pass in 12.4s.
   Wired into `SETUP.md` as the first thing to run on a new machine.
8. **Still open, flagged rather than fixed**: item 2 (scoring scripts) needs
   either porting missing scripts into the public `steering` repo or
   rewriting the documented scoring path to match what's actually public —
   a real, not-yet-scoped task. Item 4 (tokenizer sharing) still
   undocumented. A new, separate finding from the verification run itself —
   an `ed_train.py` runtime warning about `--encoder_lr_scale` possibly
   mattering for `ed_lite`'s token ratio — is logged in `SETUP.md`'s Known
   gaps as an open question, not changed, since altering a training
   hyperparameter default without checking its effect on existing published
   numbers isn't a call to make silently.

## Lesson for next time

A cold-context test that's allowed to actually run something (not stop at a
disk-space wall or a 50-step smoke ceiling) finds bugs that careful doc
review and short smoke tests both miss — item 1 existed through two prior,
successful-looking smoke tests because neither ran long enough to hit it.
`quick_selftest.sh` exists so that kind of plumbing bug surfaces in seconds
going forward, but it is deliberately narrow (training plumbing only, no
scoring) — it is not a substitute for an occasional real cold-start test like
this one.
