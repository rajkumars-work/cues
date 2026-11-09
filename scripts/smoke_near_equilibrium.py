#!/usr/bin/env python3
"""Smoke-test HANDOFF_NEAR_EQUILIBRIUM_REBUILD.md's generate -> relax -> score
pipeline at a tiny n (NEAR_EQ_N, default 3) -- confirms the pipeline runs
end to end with no crash. Does NOT produce scientifically meaningful numbers
at this n; this is a kickoff/plumbing check, not a real measurement, matching
how scripts/test_handoff.sh checks the five training arms.

Env vars (all required except NEAR_EQ_N):
  NEAR_EQ_CKPT_DIR  -- directory holding the production checkpoint
                       (ed_ckpt_final.pt, model_sp.model, model_sp.vocab)
  NEAR_EQ_DATA_CSV  -- a corpus CSV with a "label" column including "eval" rows
  NEAR_EQ_N         -- how many eval sources to pull (default 3)
  STEERING_PATH     -- path to steering's code/py, relative to this repo's
                       root (default ../steering/crystals/lemat/code/py)

Exits 0 on success, 1 on any failure, with a clear FAIL line explaining why.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # cues/public
STEERING_PATH = os.environ.get("STEERING_PATH", "../steering/crystals/lemat/code/py")
STEERING_ABS = os.path.abspath(os.path.join(REPO_ROOT, STEERING_PATH))

CKPT_DIR = os.environ.get("NEAR_EQ_CKPT_DIR", "")
DATA_CSV = os.environ.get("NEAR_EQ_DATA_CSV", "")
N = int(os.environ.get("NEAR_EQ_N", "3"))
DEVICE = os.environ.get("NEAR_EQ_DEVICE", "cuda")
# screening.generate_batch()'s own default version_id ("d6") has only 5
# target segments and silently mis-parses this checkpoint's real 6-segment
# target (it includes a scratchpad segment "d6" doesn't expect), falling
# back to a fuzzy parser that then mistakes a lattice parameter ("12.69")
# for an element symbol -- found running this script for real, 2026-11-09.
# "d15_binrho_k7" matches alex_nolemat_lowhull's actual target schema;
# override via NEAR_EQ_VERSION_ID if pointed at a different checkpoint.
VERSION_ID = os.environ.get("NEAR_EQ_VERSION_ID", "d15_binrho_k7")

# validity/structural.py's load_mace() hardcodes MACE_MODEL_PATH as a
# default *argument* value (bound at import time), so it must be set via
# this env var BEFORE that module is first imported -- a module-attribute
# monkeypatch after the fact (the path_override.py pattern used for
# ED_CHECKPOINTS) does not work here. Point it at a local cache and
# auto-download the public MACE-MP-0b3 model if it's not already there,
# rather than requiring the hardcoded /data/assets/... path (and the sudo
# it would need) to exist. Found running this for real, 2026-11-09.
if "MACE_MODEL_PATH" not in os.environ:
    _cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mace")
    _model_path = os.path.join(_cache_dir, "mace-mp-0b3-medium.model")
    if not os.path.exists(_model_path):
        os.makedirs(_cache_dir, exist_ok=True)
        import urllib.request
        _url = ("https://github.com/ACEsuit/mace-mp/releases/download/"
                "mace_mp_0b3/mace-mp-0b3-medium.model")
        print(f"Downloading MACE model from {_url} ...", flush=True)
        urllib.request.urlretrieve(_url, _model_path)
    os.environ["MACE_MODEL_PATH"] = _model_path


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


if not os.path.isdir(STEERING_ABS):
    fail(f"STEERING_PATH does not resolve to a real directory: {STEERING_ABS}")
if not CKPT_DIR or not os.path.isdir(CKPT_DIR):
    fail(f"NEAR_EQ_CKPT_DIR missing or not a directory: {CKPT_DIR!r}")
if not DATA_CSV or not os.path.exists(DATA_CSV):
    fail(f"NEAR_EQ_DATA_CSV missing or not found: {DATA_CSV!r}")

sys.path.insert(0, STEERING_ABS)
# dielectric/ itself must also be on sys.path -- validity.structural and
# chem.props.relax (imported lazily inside screening.py's screen_one/
# relax_one, used by screen_batch(full=True) and relax_batch()) are
# top-level packages under dielectric/, not under code/py directly. Same
# requirement as train_arm.sh's PYTHONPATH export; missing this surfaced
# as "No module named 'validity'" only once screen_batch actually ran,
# since load_model/generate_batch alone don't need it -- found 2026-11-09.
sys.path.insert(0, os.path.join(STEERING_ABS, "dielectric"))
sys.path.insert(0, os.path.join(REPO_ROOT, "eval"))

try:
    import path_override
    from dielectric.eval import screening
except Exception as e:
    fail(f"could not import screening.py -- {e}")

screening.ED_CHECKPOINTS = path_override.ED_CHECKPOINTS

print(f"Loading model from {CKPT_DIR} ...", flush=True)
try:
    model, sp = screening.load_model(CKPT_DIR, device=DEVICE)
except Exception as e:
    fail(f"load_model crashed -- {e}")
print("Model loaded OK", flush=True)

sources = screening.load_sources(DATA_CSV, n=N, split="eval")
print(f"Loaded {len(sources)} eval sources from {DATA_CSV}", flush=True)
if not sources:
    fail("load_sources returned zero rows -- check the CSV has a 'label' "
         "column with 'eval'/'test_1k'/'test_100' rows")

gen_results = screening.generate_batch(model, sp, sources, DEVICE, version_id=VERSION_ID)
n_ok = sum(1 for r in gen_results if r.status == "SUCCESS")
print(f"Generated {n_ok}/{len(gen_results)} successfully", flush=True)
if n_ok == 0:
    fail("no successful generations -- every source failed to parse or errored")

print("Scoring RAW (unrelaxed) ...", flush=True)
try:
    raw_scores = screening.screen_batch(gen_results, gen_results, DEVICE, full=True)
except Exception as e:
    fail(f"screen_batch (raw) crashed -- {e}")
n_raw = sum(1 for s in raw_scores if s is not None)
print(f"RAW screen produced {n_raw} result(s)", flush=True)

print("Relaxing ...", flush=True)
try:
    # Pass the already-downloaded model explicitly -- relax_batch()'s own
    # default (relax_model=None) falls through to chem/props/relax.py's
    # MACE_MODELS["mpa-0"], a DIFFERENT hardcoded /data/assets/... path (a
    # third one, distinct from MACE_MODEL_PATH/DEFAULT_MACE_PATH above) for
    # a different MACE variant. Reusing the one model already in hand here
    # avoids a second download and a third vendored-file patch. Found
    # running this for real, 2026-11-09.
    relax_results = screening.relax_batch(gen_results, DEVICE,
                                           relax_model=os.environ["MACE_MODEL_PATH"])
except Exception as e:
    fail(f"relax_batch crashed -- {e}")
n_conv = sum(1 for r in relax_results if r is not None and r.converged)
print(f"Relaxed {n_conv}/{len(relax_results)} converged", flush=True)

print("Scoring RELAXED ...", flush=True)
try:
    relaxed_scores = screening.screen_batch(gen_results, relax_results, DEVICE, full=True)
except Exception as e:
    fail(f"screen_batch (relaxed) crashed -- {e}")
n_relaxed = sum(1 for s in relaxed_scores if s is not None)
print(f"RELAXED screen produced {n_relaxed} result(s)", flush=True)

print("PASS: near-equilibrium pipeline ran end-to-end with no crash "
      f"(n={N}, gen_ok={n_ok}, raw_scored={n_raw}, relaxed={n_conv}, "
      f"relaxed_scored={n_relaxed})")
