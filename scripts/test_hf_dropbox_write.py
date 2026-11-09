#!/usr/bin/env python3
"""Tests that the shared HF dropbox token (SETUP.md Sec.6's no-account
fallback for sending results back) actually works: can write into
rajkumars47/cues-contrib-dropbox, and genuinely cannot write anywhere else.

No GPU needed -- this is pure Hugging Face Hub API calls, runs anywhere,
including straight from your own machine. Run this whenever you (re-)mint
the shared token, before sending it to a colleague, or any time you want to
confirm it still works.

Usage:
    export HF_TOKEN=<the shared dropbox token>
    python3 scripts/test_hf_dropbox_write.py

Exits 0 on PASS, 1 on any failure, with a clear explanation either way.
"""
import os
import sys
import tempfile
import time

DROPBOX_REPO = "rajkumars47/cues-contrib-dropbox"
# A real, existing repo the token should have no access to at all -- proves
# the token's scope is actually enforced, not just documented as narrow.
OUT_OF_SCOPE_REPO = "rajkumars47/data_big_d15"

TOKEN = os.environ.get("HF_TOKEN", "")


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


if not TOKEN:
    fail("HF_TOKEN not set -- export HF_TOKEN=<the shared dropbox token> first")

try:
    from huggingface_hub import HfApi
except ImportError:
    fail("huggingface_hub not installed -- pip install huggingface_hub")

api = HfApi(token=TOKEN)

test_filename = f"token_test_{int(time.time())}.txt"
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write(f"test upload from test_hf_dropbox_write.py at {time.time()}\n")
    local_path = f.name

print(f"Uploading {test_filename} to {DROPBOX_REPO} ...", flush=True)
try:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=test_filename,
        repo_id=DROPBOX_REPO,
        repo_type="dataset",
    )
except Exception as e:
    fail(f"could not write to {DROPBOX_REPO} -- {e}")
print("Upload OK", flush=True)

print(f"Confirming {test_filename} is actually there ...", flush=True)
files = api.list_repo_files(DROPBOX_REPO, repo_type="dataset")
if test_filename not in files:
    fail(f"{test_filename} not found in repo listing after upload -- "
         "upload call succeeded but the file isn't actually there")
print("Confirmed present", flush=True)

print(f"Negative test: confirming this token CANNOT write to "
      f"{OUT_OF_SCOPE_REPO} ...", flush=True)
try:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=test_filename,
        repo_id=OUT_OF_SCOPE_REPO,
        repo_type="dataset",
    )
    # If this line is reached, the write succeeded -- the token is NOT
    # actually scoped the way it's documented to be. Clean up immediately
    # and fail loudly; this is a real security problem, not a test bug.
    api.delete_file(path_in_repo=test_filename, repo_id=OUT_OF_SCOPE_REPO,
                     repo_type="dataset")
    fail(f"token was able to write to {OUT_OF_SCOPE_REPO} (cleaned up the "
         f"stray file) -- its scoping is broken, revoke this token now and "
         f"mint a correctly-scoped replacement before sending it to anyone")
except Exception as e:
    msg = str(e)
    if "403" in msg or "Forbidden" in msg:
        print(f"Confirmed: write to {OUT_OF_SCOPE_REPO} correctly rejected "
              f"(403)", flush=True)
    else:
        fail(f"write to {OUT_OF_SCOPE_REPO} was rejected, but not with the "
             f"expected 403 -- got: {e}")

print(f"Cleaning up: deleting {test_filename} from {DROPBOX_REPO} ...",
      flush=True)
api.delete_file(path_in_repo=test_filename, repo_id=DROPBOX_REPO,
                 repo_type="dataset")
os.unlink(local_path)

print("PASS: dropbox token can write to its own repo and nowhere else")
