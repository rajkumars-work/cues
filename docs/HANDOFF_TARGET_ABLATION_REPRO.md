# Handoff: reproducing the build-order and scratchpad-field ablations

**Priority: LOW — nice-to-have, not blocking.** Nothing here is required for
the paper's central thesis; §9.4/§9.5's conclusions are stated plainly and
aren't in dispute. This exists because the specific checkpoints behind these
two sections were never individually audited for survival, unlike the
checkpoints `HANDOFF_CHECKPOINT_RECONSTRUCTION.md` already covers.

**You have no prior context — read this whole file before touching
anything.** Two unrelated ablation families live here, bundled the same way
`HANDOFF_CHECKPOINT_RECONSTRUCTION.md` bundles its two parts: both are
"retrain with existing, documented infra," at different levels of
documentation quality.

## What this is

`experiments.md §9.4` (build order) and `§9.5` (which scratchpad fields
matter) each rest on dedicated ablation checkpoints — not the production
checkpoint, not the three C2/C3 architecture arms, and not any checkpoint
`HANDOFF_CHECKPOINT_RECONSTRUCTION.md` discusses. None of the checkpoints
below turned up on the HuggingFace `rajkumars47` account (checked
2026-11-08) under their natural names.

## Part A — the scratchpad-field ablations (§9.5, and part of §9.4)

### What's missing

Three checkpoints back the `bg_bin_match`/`eps_bin_match`/`rho_bin_match`/
`all_5_match` table in §9.5 and the reversed-order check in §9.4:

| Checkpoint | Role | Family | Checked on HF? |
|---|---|---|---|
| `ckpt_autolabel_v1_binrho_k7_no_scratchpad` | scratchpad removed, formula kept | fork of `ckpt_autolabel_v1_binrho_k7` | Not found |
| `ckpt_autolabel_v1_binrho_k7_comp_only` | scratchpad + space group + anon removed, formula kept | fork of `ckpt_autolabel_v1_binrho_k7` | Not found |
| `lattice_atoms_only_ep75` (ed-lite, 6+18) | formula also removed — the ablation that actually collapses steerability | fork of `ckpt_ed_enclight_6_18` | Not found |

The first two are forks of `ckpt_autolabel_v1_binrho_k7`, which
`HANDOFF_CHECKPOINT_RECONSTRUCTION.md` already confirms has **no surviving
weights** — these two were trained in the same directory tree
(`ed/checkpoints/ckpt_autolabel_v1_binrho_k7_no_scratchpad/`, etc.), at the
same time, as 50-epoch pilots branching off the same baseline run. Nothing
suggests they survived when their parent didn't; this hasn't been
individually confirmed, but treat it as likely lost rather than merely
unaudited.

`lattice_atoms_only_ep75` is a separate case: an `ed-lite` (6 encoder + 18
decoder) fork trained specifically for this one ablation ("Stage D" in
`TargetAblation.md`), at 75 epochs, forked from the production ed-lite
recipe but never itself backed up — it isn't the production
`ckpt_ed_enclight_6_18` checkpoint, and per `dielectric/MODELS.md` §0 the HF
backup process keeps only the single latest checkpoint per model, not
per-epoch snapshots, so a dedicated ablation-only fork like this one was
never in scope for that automation to begin with.

Separately, the *matched-step baselines* used alongside these ablations —
`ed_ckpt_022000.pt` (step 22,000, ≈ep 51.4, from the `ckpt_autolabel_v1_binrho_k7`
run) and `ckpt_ed_enclight_6_18` at step 16,000 (≈ep 75) — are intermediate
training snapshots of checkpoints whose *final* weights are lost
(`ckpt_autolabel_v1_binrho_k7`) or public (`ckpt_ed_enclight_6_18`). The
ed-lite intermediate snapshot is the same backup-policy gap as
`lattice_atoms_only_ep75`'s own checkpoint — "only the latest" doesn't cover
a mid-training comparison point either, even though the model's final
checkpoint is fine.

### Why this is tractable

All three ablations were already run once via fully-documented,
mechanical infra — `TargetAblation.md` gives the exact target-schema
variants, hyperparameters, and script:

```
data/big_d15_autolabel_binrho_k7_no_scratchpad.csv   # built by build_target_ablations.py
data/big_d15_autolabel_binrho_k7_comp_only.csv       # built by build_target_ablations.py
```

- **`no_scratchpad`** / **`comp_only`**: same hyperparameters as the
  `ckpt_autolabel_v1_binrho_k7` baseline (`--epochs 50`, `--batch_size 64
  --lr 1e-4 --warmup_frac 0.1 --prop_dropout 0.2 --label_dropout 0.2`),
  trained on the ablation CSVs above (built from the production corpus by
  `build_target_ablations.py`, already in this project's scripts). 50
  epochs, not 150 — these were always pilots, not full production runs, so
  retraining them to their original scope is G2 at most.
- **`lattice_atoms_only_ep75`**: `ed-lite` recipe (`--enc_layers 6
  --dec_layers 18 --edim 768 --heads 12 --lr 5e-5 --max_seq_len 512
  --warmup_frac 0.15 --label_dropout 0.0`), 75 epochs, on a target schema
  with segments 0–3 (anon/formula/sg/scratchpad) all replaced with
  `<none>`. Same corpus, same tokenizer (seeded from the production ed-lite
  checkpoint's vocab, sha256 `eb24318d…`), just the target-string content
  changes.

None of this needs new research judgment — the ablation CSVs' construction
rule and every hyperparameter are already pinned down in `TargetAblation.md`.
`build_target_ablations.py` isn't yet ported to this public repo; that's the
one real engineering gap (**E1**: port one existing, working script, not
design a new one).

### What to run

1. Port `build_target_ablations.py` to this public repo (or request it from
   the maintainer — it's a small, mechanical CSV transform, not a design
   decision).
2. Retrain `no_scratchpad` and `comp_only` at 50 epochs on
   `ckpt_autolabel_v1_binrho_k7`'s corpus (public: `rajkumars47/data_big_d15`),
   using `train_arm.sh`'s existing parameterization with the hyperparameters
   above. **G2 / E1** each.
3. Retrain `lattice_atoms_only` at 75 epochs on the production ed-lite
   recipe, using the all-`<none>` target schema. **G2 / E1**.
4. Score all three at n=1000 (matched eval split, same steerability/Combined
   pipeline as every other experiment here) and compare against the
   recorded §9.5 table and §9.4's 48.2%/58.1% pair.

## Part B — the build-order progression (§9.4's 36.7% → 50.9% → 58.1%)

### What's missing

§9.4's other evidence — tier-0 validity climbing 36.7% → 50.9% → 58.1% as
space-group/Wyckoff conditioning, then the scratchpad, were added to the
target sequence one at a time — comes from `PROGRESSION_SUMMARY.md`'s
"Plain / Symmetry / Scratchpad" comparison, and the related "V2/V3 → V11"
lineage. These checkpoints predate the project's current naming and HF
backup conventions entirely (they're from the same early exploratory era as
the `ckpt_d*`/`ckpt_v*` family in `data/checkpoint_registry.json`'s
naming-convention table) — none turned up on HF under any name checked
(`ckpt_d12_np_think_250ep`, `ckpt_d11_np_s30k`, and siblings).

### Why this one is less tractable than Part A

Unlike Part A, this progression's exact training recipe per stage isn't
pinned down with the same precision in the available docs — `TargetAblation.md`
gives hyperparameters to the batch size and warmup fraction;
`PROGRESSION_SUMMARY.md` gives the schema progression (plain →
+symmetry → +scratchpad) and the resulting numbers, but not a
run-reproducible recipe at the same level of detail. Reconstructing this
cleanly would mean re-deriving a training config from a qualitative
description, which is a real (if modest) engineering task, not a known
quantity — closer to **E2** than Part A's **E1**.

### If you pick this up

Treat it as a from-scratch three-point ablation (plain target schema →
+space-group/Wyckoff → +scratchpad, each on the production-era corpus),
not a literal reconstruction of the original runs. State clearly that any
numbers produced are a fresh replication of the *pattern*, not a recovery of
the original 36.7%/50.9%/58.1% figures — the same distinction
`HANDOFF_DIFFUSION_REPRO.md` draws between reproducing a lineage's pattern
and recovering its original numbers.

## Report results

For Part A: report the three retrained checkpoints' n=1000 numbers
side-by-side with the recorded §9.4/§9.5 tables, and flag anything that
didn't match the documented recipe exactly. For Part B, if attempted: report
it explicitly as a fresh three-point replication, not a recovery, per the
caveat above.

Same report-back mechanism as every other experiment here: see
`../SETUP.md` §6 — fork + PR preferred, relay as fallback, checkpoints to
your own Hugging Face account.
