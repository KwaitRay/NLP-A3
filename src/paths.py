"""Centralised path constants for both local Windows dev and Colab runs."""
from __future__ import annotations
import os
from pathlib import Path


def _detect_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root)
    if os.environ.get("IS_COLAB") == "1":
        return Path("/content/drive/MyDrive/NLP_Assignment3/Assignment3")
    here = Path(__file__).resolve().parent
    return here.parent


PROJECT_ROOT = _detect_root()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TRAIN_CLAIMS = DATA_DIR / "train-claims.json"
DEV_CLAIMS = DATA_DIR / "dev-claims.json"
TEST_CLAIMS_UNLABELLED = DATA_DIR / "test-claims-unlabelled.json"
DEV_BASELINE = DATA_DIR / "dev-claims-baseline.json"
EVIDENCE_JSON = DATA_DIR / "evidence.json"

EDA_DIR = OUTPUTS_DIR / "eda"
SPLITS_DIR = OUTPUTS_DIR / "splits"
SFT_DIR = OUTPUTS_DIR / "sft_data"

LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED")
