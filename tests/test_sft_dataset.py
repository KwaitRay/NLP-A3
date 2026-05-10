"""Smoke test: build SFT records from a fake mini-corpus.

Doesn't need evidence.json; uses a hand-crafted dict so we can exercise both
gold-only and retrieval-driven paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sft_dataset import build_dataset, curriculum_sort_key  # noqa: E402


FAKE_EV = {
    "ev-1": "Sea level has risen 8 inches since 1900.",
    "ev-2": "Antarctic ice sheets are losing mass annually.",
    "ev-3": "South Australia has high renewable share.",
    "ev-4": "Climate models predict warming of 1.5-4 °C this century.",
    "ev-5": "Carbon dioxide concentration is now 420 ppm.",
    "ev-6": "Ice cores from Greenland record 800k years of climate.",
}

TAGGED = [
    {
        "id": "c-1", "claim_text": "Sea level is rising fast.", "claim_label": "SUPPORTS",
        "evidences": ["ev-1", "ev-2"],
        "domain": "sea_level", "scenario": "supports_clear",
        "difficulty": {"level": "easy", "score": 0.3, "source": "heuristic"},
    },
    {
        "id": "c-2", "claim_text": "South Australia has cheap electricity.", "claim_label": "REFUTES",
        "evidences": ["ev-3"],
        "domain": "policy_economics", "scenario": "refutes_clear",
        "difficulty": {"level": "medium", "score": 0.55, "source": "heuristic"},
    },
    {
        "id": "c-3", "claim_text": "Climate sensitivity is exactly 3 °C.", "claim_label": "DISPUTED",
        "evidences": ["ev-4", "ev-6"],
        "domain": "models_attribution", "scenario": "disputed_conflict",
        "difficulty": {"level": "hard", "score": 0.9, "source": "heuristic"},
    },
]


def main() -> None:
    out = build_dataset(TAGGED, FAKE_EV, k=3, pad_with_random=True, n_hard_neg=1, seed=7)

    # Each claim → 1 normal + 1 hard-neg = 6 records total.
    assert len(out) == 6, f"expected 6, got {len(out)}"

    # Curriculum sort: easy first, hard last among the *normal* records.
    normals = [r for r in out if "augmented" not in r["_meta"]]
    keys = [curriculum_sort_key(r) for r in normals]
    assert keys == sorted(keys), f"curriculum out of order: {keys}"
    print(f"  [pass] {len(out)} records, curriculum sorted")

    # Inspect one normal SFT record.
    one = next(r for r in normals if r["id"] == "c-1")
    assert "Claim: Sea level is rising fast." in one["query"]
    assert one["response"].startswith("SUPPORTS"), one["response"]
    cited = one["response"]
    assert "##[" in cited and "]##" in cited
    print(f"  [pass] c-1 response = {cited!r}")

    # Hard-neg should always be NOT_ENOUGH_INFO.
    hns = [r for r in out if "augmented" in r["_meta"]]
    for hn in hns:
        assert hn["response"].startswith("NOT_ENOUGH_INFO"), hn["response"]
        assert hn["_meta"]["scenario"] == "nei_topic_off"
    print(f"  [pass] {len(hns)} hard negatives all NEI/topic_off")

    # Print one full record so a human can eyeball the prompt.
    print("\nSample record (c-1, gold path):")
    print("  system:", one["system"][:80] + "...")
    print("  query[:200]:", one["query"][:200].replace("\n", " | "))
    print("  response:", one["response"])
    print("  _meta:", one["_meta"])


if __name__ == "__main__":
    main()
    print("all green")
