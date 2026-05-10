"""Quick sanity tests for prompt building and parsing.

Run with: ``python -m tests.test_prompt`` (no pytest required).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompt import build_target_response, build_user_query, parse_response  # noqa: E402


def _check(name: str, got, expected) -> None:
    if got == expected:
        print(f"  [pass] {name}")
        return
    print(f"  [FAIL] {name}: got={got!r}\n           expected={expected!r}")
    raise SystemExit(1)


def test_build_user_query() -> None:
    q = build_user_query(
        "South Australia has the most expensive electricity in the world.",
        [("evidence-67732", "Australia has high electricity prices."),
         ("evidence-572512", "South Australia leads in renewable energy.")],
    )
    assert "Claim: South Australia" in q
    assert "[1] Australia has high" in q
    assert "[2] South Australia leads" in q
    assert q.endswith("Answer:")
    print("  [pass] build_user_query")


def test_build_target_response_keeps_gold_indices() -> None:
    out = build_target_response(
        "SUPPORTS",
        gold_evidence_ids=["evidence-2", "evidence-9"],
        shown_evidence_ids=["evidence-1", "evidence-2", "evidence-9", "evidence-3"],
    )
    _check("response keeps gold indices", out, "SUPPORTS ##[2,3]##")


def test_build_target_response_no_gold_in_shown() -> None:
    """If retriever missed all gold evidence, fall back to all shown indices.

    Otherwise SFT teaches the model to emit empty citation, which eval.py
    rejects (evidences list must be non-empty)."""
    out = build_target_response(
        "REFUTES",
        gold_evidence_ids=["evidence-99"],
        shown_evidence_ids=["evidence-1", "evidence-2"],
    )
    _check("fallback to all when no gold present", out, "REFUTES ##[1,2]##")


def test_parse_basic() -> None:
    label, evs = parse_response(
        "SUPPORTS ##[1,3]##",
        shown_evidence_ids=["e-a", "e-b", "e-c"],
    )
    _check("parse basic", (label, evs), ("SUPPORTS", ["e-a", "e-c"]))


def test_parse_loose_label() -> None:
    label, evs = parse_response(
        "not enough info ##[2]##",
        shown_evidence_ids=["e-a", "e-b"],
    )
    _check("parse loose NEI", (label, evs), ("NOT_ENOUGH_INFO", ["e-b"]))


def test_parse_no_citation_falls_back_to_all() -> None:
    label, evs = parse_response("DISPUTED.", shown_evidence_ids=["e-a", "e-b"])
    _check("no citation fallback", (label, evs), ("DISPUTED", ["e-a", "e-b"]))


def test_parse_garbage_label_uses_default() -> None:
    label, evs = parse_response("foo bar", shown_evidence_ids=["e-a"])
    _check("garbage label default", (label, evs), ("NOT_ENOUGH_INFO", ["e-a"]))


def test_parse_drops_out_of_range_indices() -> None:
    label, evs = parse_response(
        "REFUTES ##[1,7,2,99]##", shown_evidence_ids=["e-a", "e-b"]
    )
    _check("oob dropped", (label, evs), ("REFUTES", ["e-a", "e-b"]))


if __name__ == "__main__":
    print("test_prompt")
    test_build_user_query()
    test_build_target_response_keeps_gold_indices()
    test_build_target_response_no_gold_in_shown()
    test_parse_basic()
    test_parse_loose_label()
    test_parse_no_citation_falls_back_to_all()
    test_parse_garbage_label_uses_default()
    test_parse_drops_out_of_range_indices()
    print("all green")
