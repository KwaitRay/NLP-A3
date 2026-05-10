"""Smoke tests for dpo_pairs.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dpo_pairs import (  # noqa: E402
    build_dpo_dataset,
    build_dpo_pair,
    synthesise_disputed_contrast,
)


SFT_REC_TPL = {
    "id": "claim-X",
    "messages": [
        {"role": "system",    "content": "sys"},
        {"role": "user",      "content": "Claim: ...\nEvidence:\n[1] e1\n[2] e2\nAnswer:"},
        {"role": "assistant", "content": "SUPPORTS ##[1,2]##"},
    ],
    "_meta": {"shown": ["ev-1", "ev-2"], "scenario": "supports_clear"},
}


def test_build_pair_skips_correct_predictions() -> None:
    pair = build_dpo_pair(
        sft_record=SFT_REC_TPL,
        pred_label="SUPPORTS", pred_evidences=["ev-1", "ev-2"],
        gold_label="SUPPORTS", gold_evidences=["ev-1", "ev-2"],
    )
    assert pair is None
    print("  [pass] no pair when prediction is correct")


def test_build_pair_emits_when_label_wrong() -> None:
    pair = build_dpo_pair(
        sft_record=SFT_REC_TPL,
        pred_label="REFUTES", pred_evidences=["ev-1"],
        gold_label="SUPPORTS", gold_evidences=["ev-1", "ev-2"],
    )
    assert pair is not None
    # Chosen lives at messages[2] (assistant turn); rejected_response is top-level.
    assert pair["messages"][2]["content"].startswith("SUPPORTS")
    assert pair["rejected_response"].startswith("REFUTES")
    assert pair["_meta"]["pred_label"] == "REFUTES"
    print("  [pass] pair emitted on label mismatch")


def test_build_dataset_walks_sft_records() -> None:
    sft = [
        dict(SFT_REC_TPL, id="c1"),
        dict(SFT_REC_TPL, id="c2"),
        dict(SFT_REC_TPL, id="c3"),
    ]
    preds = {
        "c1": {"claim_label": "REFUTES", "evidences": ["ev-1"]},
        "c2": {"claim_label": "SUPPORTS", "evidences": ["ev-1", "ev-2"]},  # correct
        # c3 missing → skipped
    }
    gold = {
        "c1": {"claim_label": "SUPPORTS", "evidences": ["ev-1", "ev-2"]},
        "c2": {"claim_label": "SUPPORTS", "evidences": ["ev-1", "ev-2"]},
        "c3": {"claim_label": "DISPUTED", "evidences": ["ev-1"]},
    }
    out = build_dpo_dataset(sft, preds, gold)
    assert len(out) == 1, f"expected 1 emitted pair, got {len(out)}"
    assert out[0]["id"] == "c1"
    print("  [pass] dataset walk produced 1 pair from 3 records")


def test_synthesise_disputed_contrast_only_targets_supports_clear() -> None:
    sft = [
        dict(SFT_REC_TPL, id=f"sup{i}", _meta={"shown": ["ev-1", "ev-2"], "scenario": "supports_clear"})
        for i in range(5)
    ] + [
        dict(SFT_REC_TPL, id="ref1", _meta={"shown": ["ev-1", "ev-2"], "scenario": "refutes_clear"}),
    ]
    syn = synthesise_disputed_contrast(sft, n=10)
    assert all(r["_meta"]["augmented"] == "supports_vs_disputed" for r in syn)
    assert all(r["rejected_response"].startswith("DISPUTED") for r in syn)
    assert len(syn) <= 5  # only supports_clear records eligible
    print(f"  [pass] synthesised {len(syn)} disputed contrast pairs")


if __name__ == "__main__":
    print("test_dpo_pairs")
    test_build_pair_skips_correct_predictions()
    test_build_pair_emits_when_label_wrong()
    test_build_dataset_walks_sft_records()
    test_synthesise_disputed_contrast_only_targets_supports_clear()
    print("all green")
