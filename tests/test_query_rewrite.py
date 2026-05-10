"""Smoke tests for query_rewrite.py.

Synonym expansion needs WordNet on first run; we tolerate the download but
make sure the function returns *something* sensible.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.query_rewrite import (  # noqa: E402
    blend_query_text,
    decompose_subclaims_prompt,
    expand_query,
    hyde_prompt,
    parse_subclaims,
    synonym_expand,
)


def test_synonym_expand_includes_original_first() -> None:
    out = synonym_expand("Sea level is rising fast.")
    assert out[0] == "Sea level is rising fast."
    assert all(isinstance(s, str) and s for s in out)
    print(f"  [pass] synonym_expand returned {len(out)} variants (incl. original)")


def test_subclaim_prompt_has_claim() -> None:
    p = decompose_subclaims_prompt("CO2 causes warming and warming causes melting.")
    assert "CO2 causes warming" in p
    assert "atomic sub-claims" in p
    print("  [pass] subclaim prompt contains claim")


def test_parse_subclaims_handles_numbered() -> None:
    text = "1. CO2 causes warming.\n2. Warming causes melting.\n3. Melting raises sea level.\n"
    subs = parse_subclaims(text)
    assert len(subs) == 3
    assert subs[0] == "CO2 causes warming."
    print(f"  [pass] parsed {len(subs)} sub-claims")


def test_parse_subclaims_fallback_when_empty() -> None:
    subs = parse_subclaims("???", fallback="original claim")
    assert subs == ["original claim"]
    print("  [pass] fallback works")


def test_hyde_prompt_shape() -> None:
    p = hyde_prompt("Greenland is losing ice mass.")
    assert "Greenland is losing ice mass" in p
    assert "1-2 sentences" in p
    print("  [pass] hyde prompt shape")


def test_blend_query_text_repeats_claim() -> None:
    out = blend_query_text("A", "B", alpha=0.75)
    # alpha=0.75 → claim×3, hyp×1
    assert out.count("A") >= out.count("B")
    print(f"  [pass] blend repeats: {out!r}")


def test_expand_query_dedupes() -> None:
    qs = expand_query(
        "Sea level rises.",
        use_synonym=True,
        use_subclaim=["Sea level rises."],   # collides with original
        use_hyde="Glaciers melt.",
    )
    # First entry must be the original.
    assert qs[0] == "Sea level rises."
    assert len(qs) == len(set(qs))
    print(f"  [pass] expand_query produced {len(qs)} deduped queries")


if __name__ == "__main__":
    print("test_query_rewrite")
    test_synonym_expand_includes_original_first()
    test_subclaim_prompt_has_claim()
    test_parse_subclaims_handles_numbered()
    test_parse_subclaims_fallback_when_empty()
    test_hyde_prompt_shape()
    test_blend_query_text_repeats_claim()
    test_expand_query_dedupes()
    print("all green")
