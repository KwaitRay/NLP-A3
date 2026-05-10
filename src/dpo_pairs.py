"""Stage 4 — DPO preference-pair construction.

Mines wrong predictions on ``dev_holdout`` (NEVER on official dev) and emits
ms-swift DPO format records, where ``response`` is the gold output (chosen)
and ``rejected_response`` is the model's mispredicted output.

Optionally augments with hand-crafted DISPUTED-vs-SUPPORTS contrast pairs:
the most common confusion in 4-class climate-claim verification.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from .prompt import SYSTEM_PROMPT, build_target_response, build_user_query


def _normalise_response(label: str, evidences: Iterable[str], shown_ids: list[str]) -> str:
    """Reuse the SFT response builder so chosen/rejected match training fmt."""
    return build_target_response(label, list(evidences), shown_ids)


def build_dpo_pair(
    *,
    sft_record: dict,
    pred_label: str,
    pred_evidences: list[str],
    gold_label: str,
    gold_evidences: list[str],
) -> dict | None:
    """Build a single DPO record from one mispredicted dev_holdout example.

    Returns None when prediction matches gold (no preference signal).
    The ``sft_record`` is the row we passed to the model — its ``query`` and
    ``_meta.shown`` carry the exact evidences the model saw, so the reference
    response can be reconstructed identically.
    """
    if pred_label == gold_label and set(pred_evidences) == set(gold_evidences):
        return None
    shown_ids: list[str] = sft_record.get("_meta", {}).get("shown") or []
    chosen = _normalise_response(gold_label, gold_evidences, shown_ids)
    rejected = _normalise_response(pred_label, pred_evidences or shown_ids, shown_ids)
    if chosen == rejected:  # Both happen to round-trip identically — skip.
        return None
    return {
        "id": sft_record["id"],
        "system": sft_record.get("system", SYSTEM_PROMPT),
        "query": sft_record["query"],
        "response": chosen,
        "rejected_response": rejected,
        "_meta": {
            **sft_record.get("_meta", {}),
            "pred_label": pred_label,
            "gold_label": gold_label,
        },
    }


def build_dpo_dataset(
    sft_records: list[dict],
    predictions: dict[str, dict],
    gold: dict[str, dict],
    *,
    extra_pairs: Iterable[dict] = (),
) -> list[dict]:
    """Walk SFT records, look up the model's predictions for the same ids.

    ``predictions[claim_id]`` should match the format used by ``eval.py``:
    ``{"claim_label": ..., "evidences": [...]}``.
    ``gold[claim_id]`` is the labelled claim row from train-claims.json.
    """
    out: list[dict] = []
    for rec in sft_records:
        cid = rec["id"]
        if cid not in predictions or cid not in gold:
            continue
        pair = build_dpo_pair(
            sft_record=rec,
            pred_label=predictions[cid].get("claim_label", ""),
            pred_evidences=predictions[cid].get("evidences", []) or [],
            gold_label=gold[cid]["claim_label"],
            gold_evidences=gold[cid]["evidences"],
        )
        if pair is not None:
            out.append(pair)
    out.extend(extra_pairs)
    return out


# -- DISPUTED contrast augmentation ------------------------------------------

_DISPUTED_TEMPLATES = [
    # Hand-crafted templates that flip a SUPPORTS claim into a DISPUTED-style
    # response by adding a dissenting evidence framing. Used sparingly — these
    # are the hardest pairs to learn.
    "A subset of researchers contests this conclusion citing alternative data.",
    "Recent peer-reviewed work has questioned the strength of this finding.",
    "Independent reanalyses have produced inconsistent results.",
]


def synthesise_disputed_contrast(
    sft_records: list[dict],
    *,
    n: int = 30,
    seed: int = 42,
) -> list[dict]:
    """Synthesise ``n`` DPO pairs where SUPPORTS rows are paired against a
    rejected response that picks DISPUTED. Trains the model to be sceptical
    when only a tiny minority of evidence dissents.

    Operates only on records whose meta scenario is ``supports_clear``. Skip
    if the dataset has fewer such records than requested."""
    rng = random.Random(seed)
    candidates = [r for r in sft_records if r.get("_meta", {}).get("scenario") == "supports_clear"]
    rng.shuffle(candidates)
    out: list[dict] = []
    for rec in candidates[:n]:
        shown_ids = rec.get("_meta", {}).get("shown") or []
        if not shown_ids:
            continue
        chosen = rec["response"]  # Already SUPPORTS ##[..]##
        rejected = _normalise_response("DISPUTED", shown_ids[:1], shown_ids)
        if chosen == rejected:
            continue
        out.append({
            "id": f"{rec['id']}__synth_disputed",
            "system": rec.get("system", SYSTEM_PROMPT),
            "query": rec["query"],
            "response": chosen,
            "rejected_response": rejected,
            "_meta": {**rec.get("_meta", {}), "augmented": "supports_vs_disputed"},
        })
    return out


# -- I/O ---------------------------------------------------------------------

def write_dpo_jsonl(records: list[dict], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            # Strip ``_meta`` for the actual training file — ms-swift only
            # consumes system/query/response/rejected_response.
            payload = {k: v for k, v in r.items() if not k.startswith("_") and k != "id"}
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n
