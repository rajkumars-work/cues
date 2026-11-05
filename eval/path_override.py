"""
Override for one hardcoded path in the companion steering repo's
`dielectric/eval/screening.py` (crystals/lemat/code/py/dielectric/eval/screening.py):

    ED_CHECKPOINTS = Path("/data/rkumar/code/py/ed/checkpoints")

baked in for the original author's own machine, with no CLI flag or env var to
override it as shipped. This module provides that override without forking the
989-line file it lives in.

Usage — import this *after* the steering repo's code directory is on
sys.path/PYTHONPATH, and *before* calling load_model()/load_sources() etc.:

    import sys
    sys.path.insert(0, "../steering/crystals/lemat/code/py")
    sys.path.insert(0, "./eval")
    import path_override
    from dielectric.eval import screening
    screening.ED_CHECKPOINTS = path_override.ED_CHECKPOINTS

Set the ED_CHECKPOINTS environment variable to point at your own checkpoints
directory (e.g. the output of scripts/train_arm.sh, or scripts/fetch_assets.sh's
download location) before running anything that imports this module.
"""
import os
from pathlib import Path

ED_CHECKPOINTS = Path(os.environ.get("ED_CHECKPOINTS", "/data/rkumar/code/py/ed/checkpoints"))
