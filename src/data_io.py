"""I/O helpers for claim files, evidence corpus, and prediction outputs.

Designed to run identically on local Windows and on Colab. Heavy artifacts
(evidence.json, embeddings, indices) are loaded lazily so that Stage 0 work
can proceed before evidence.json is downloaded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .paths import (
    DEV_BASELINE,
    DEV_CLAIMS,
    EVIDENCE_JSON,
    LABELS,
    TEST_CLAIMS_UNLABELLED,
    TRAIN_CLAIMS,
)


def load_claims(path: str | Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_train() -> dict[str, dict]:
    return load_claims(TRAIN_CLAIMS)


def load_dev() -> dict[str, dict]:
    return load_claims(DEV_CLAIMS)


def load_test_unlabelled() -> dict[str, dict]:
    return load_claims(TEST_CLAIMS_UNLABELLED)


def load_dev_baseline() -> dict[str, dict]:
    return load_claims(DEV_BASELINE)


def load_evidence(path: str | Path = EVIDENCE_JSON) -> dict[str, str]:
    """Load the full evidence corpus (~120k+ passages, ~174 MB).

    Raises FileNotFoundError with a clear next-step hint if not yet downloaded.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download from one of:\n"
            "  - https://drive.google.com/file/d/1JlUzRufknsHzKzvrEjgw8D3n_IRpjzo6/view\n"
            "  - https://canvas.lms.unimelb.edu.au/courses/234957/pages/evidence-dot-json-download\n"
            f"and place at {p}."
        )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: Iterable[dict], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_predictions(preds: dict[str, dict], path: str | Path) -> None:
    """Write predictions in the format expected by eval.py.

    Each value must contain `claim_label` (one of LABELS) and `evidences`
    (non-empty list of evidence IDs). `claim_text` is optional (eval ignores).
    """
    for cid, rec in preds.items():
        if rec.get("claim_label") not in LABELS:
            raise ValueError(f"{cid}: invalid claim_label {rec.get('claim_label')}")
        evs = rec.get("evidences")
        if not isinstance(evs, list) or len(evs) == 0:
            raise ValueError(f"{cid}: evidences must be a non-empty list")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
