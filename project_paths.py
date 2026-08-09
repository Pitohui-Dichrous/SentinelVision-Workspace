"""SentinelVision project paths.

All paths are resolved from this file's location instead of the current working
directory or a Windows drive letter.  This keeps the project usable after the
folder is moved to another computer or the external drive receives a new
letter.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / "RUNTIME"
RUNTIME_PYTHON = RUNTIME_DIR / "python" / "python.exe"
STATE_DIR = PROJECT_ROOT / ".runtime"
EVENTS_DIR = STATE_DIR / "events"
CONFIG_DIR = PROJECT_ROOT / "config"
SAFETY_PIPELINE_CONFIG = CONFIG_DIR / "safety_pipeline.yaml"
RESULTS_DIR = PROJECT_ROOT / "RESULTS"
TRAINING_OUTPUTS_DIR = PROJECT_ROOT / "TRAINING_OUTPUTS"
PRETRAINED_WEIGHTS_DIR = PROJECT_ROOT / "PRETRAINED_WEIGHTS"
DATA_DIR = PROJECT_ROOT / "data"


def _configured_path(environment_name, default_relative_path):
    """Return an optional environment override or a project-relative default."""
    configured = os.environ.get(environment_name)
    path = Path(configured).expanduser() if configured else PROJECT_ROOT / default_relative_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


YOLO_REPO_DIR = PROJECT_ROOT

# Historical aliases retained only for TEST.py and old notebooks.  The current
# Sentinel application discovers every deployment dynamically from RESULTS.
MODEL_PATH_COMBINED = (
    RESULTS_DIR / "Amphoreus_long_run_combined" / "weights" / "best.pt"
).resolve()
MODEL_PATH_FIRE = (
    RESULTS_DIR / "Amphoreus_Fire_only" / "weights" / "best.pt"
).resolve()
MODEL_PATH_PPE = (
    RESULTS_DIR / "Amphoreus_long_run_helmet_only" / "weights" / "best.pt"
).resolve()

# Model used by the older DETECT_GPT.py demo.
LEGACY_MODEL_PATH = _configured_path(
    "SENTINEL_MODEL_LEGACY",
    Path("runs") / "train" / "fire_exp_800_300" / "weights" / "best.pt",
)
