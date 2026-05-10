"""Smoke tests for inference.py — exercises the retrieval-only path end-to-end
without needing a real model or evidence corpus.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference import (  # noqa: E402
    RetrievalOnlyInferer,
    load_predictions,
    predict_all,
)


class FakeRetriever:
    """Returns a fixed list of (id, text) for any claim — enough to test the
    plumbing without bringing up bm25s / sentence-transformers."""

    def __init__(self, hits: list[tuple[str, str]]):
        self._hits = hits

    def retrieve(self, claim_text: str):
        return list(self._hits)


def test_retrieval_only_predict_shape() -> None:
    inferer = RetrievalOnlyInferer(FakeRetriever([("ev-1", "x"), ("ev-2", "y")]))
    out = inferer.predict("any claim")
    assert out["claim_label"] == "SUPPORTS"
    assert out["evidences"] == ["ev-1", "ev-2"]
    print("  [pass] retrieval-only majority predict shape")


def test_retrieval_only_random_label_is_valid() -> None:
    inferer = RetrievalOnlyInferer(FakeRetriever([("ev-1", "x")]), label_strategy="random")
    out = inferer.predict("claim 7")
    assert out["claim_label"] in {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"}
    print(f"  [pass] random label ok: {out['claim_label']}")


def test_predict_all_writes_valid_json() -> None:
    inferer = RetrievalOnlyInferer(FakeRetriever([("ev-1", "x"), ("ev-2", "y")]))
    claims = {
        "claim-1": {"claim_text": "x"},
        "claim-2": {"claim_text": "y"},
    }
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "preds.json"
        preds = predict_all(claims, inferer, out_path, progress=False)
        assert set(preds.keys()) == {"claim-1", "claim-2"}
        loaded = load_predictions(out_path)
        for cid, rec in loaded.items():
            assert rec["claim_label"] in {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"}
            assert isinstance(rec["evidences"], list) and rec["evidences"]
    print("  [pass] predict_all wrote valid eval.py-format json")


def test_predict_all_handles_failure() -> None:
    """If a claim raises, fall back to NEI + dummy evidence so the output JSON
    still validates against eval.py's schema."""

    class BoomRetriever:
        def retrieve(self, claim_text):
            raise RuntimeError("synthetic failure")

    inferer = RetrievalOnlyInferer(BoomRetriever())
    preds = predict_all({"claim-1": {"claim_text": "x"}}, inferer, progress=False)
    assert preds["claim-1"]["claim_label"] == "NOT_ENOUGH_INFO"
    assert preds["claim-1"]["evidences"], "evidences must be non-empty"
    print("  [pass] predict_all degrades gracefully on retriever failure")


if __name__ == "__main__":
    print("test_inference")
    test_retrieval_only_predict_shape()
    test_retrieval_only_random_label_is_valid()
    test_predict_all_writes_valid_json()
    test_predict_all_handles_failure()
    print("all green")
