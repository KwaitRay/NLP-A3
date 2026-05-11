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
    # New schema: messages = [system, user, assistant]
    assert isinstance(one["messages"], list) and len(one["messages"]) == 3
    assert [m["role"] for m in one["messages"]] == ["system", "user", "assistant"]
    user_content = one["messages"][1]["content"]
    assistant_content = one["messages"][2]["content"]
    assert "Claim: Sea level is rising fast." in user_content
    assert assistant_content.startswith("SUPPORTS"), assistant_content
    assert "##[" in assistant_content and "]##" in assistant_content
    print(f"  [pass] c-1 assistant content = {assistant_content!r}")

    # Hard-neg should always be NOT_ENOUGH_INFO.
    hns = [r for r in out if "augmented" in r["_meta"]]
    for hn in hns:
        assert hn["messages"][2]["content"].startswith("NOT_ENOUGH_INFO"), hn["messages"][2]["content"]
        assert hn["_meta"]["scenario"] == "nei_topic_off"
    print(f"  [pass] {len(hns)} hard negatives all NEI/topic_off")

    # Print one full record so a human can eyeball the prompt.
    print("\nSample record (c-1, gold path):")
    print("  system:", one["messages"][0]["content"][:80] + "...")
    print("  user[:200]:", user_content[:200].replace("\n", " | "))
    print("  assistant:", assistant_content)
    print("  _meta:", one["_meta"])

    # --- Phase 4 weak_buckets oversampling ---
    # c-2 scenario=refutes_clear, c-3 scenario=disputed_conflict, c-3 also
    # difficulty=hard. With factor 3 on refutes_clear and 2 on hard:
    #   - c-1 (supports_clear, easy): factor 1 → 1 real + 1 hn = 2
    #   - c-2 (refutes_clear, medium): factor 3 → 3 real + 3 hn = 6
    #   - c-3 (disputed_conflict, hard): max(1, 2) = 2 → 2 real + 2 hn = 4
    # Total: 12. Confirms (a) factor multiplies both real + hard-neg, and
    # (b) overlapping matches resolve to max not product.
    out2 = build_dataset(
        TAGGED, FAKE_EV, k=3, pad_with_random=True, n_hard_neg=1, seed=7,
        weak_buckets={
            ("scenario", "refutes_clear"): 3,
            ("difficulty", "hard"): 2,
        },
    )
    assert len(out2) == 12, f"expected 12 with weak_buckets, got {len(out2)}"
    c2_normals = [r for r in out2 if r["id"] == "c-2"]
    c3_normals = [r for r in out2 if r["id"] == "c-3"]
    assert len(c2_normals) == 3, f"c-2 should appear 3× (factor 3), got {len(c2_normals)}"
    assert len(c3_normals) == 2, f"c-3 should appear 2× (max factor 2), got {len(c3_normals)}"
    print(f"  [pass] weak_buckets: 12 records (c-2×3, c-3×2 via max)")

    # No-op when weak_buckets={} or None → identical count to baseline.
    out3 = build_dataset(
        TAGGED, FAKE_EV, k=3, pad_with_random=True, n_hard_neg=1, seed=7,
        weak_buckets={},
    )
    assert len(out3) == 6, f"empty weak_buckets should be no-op, got {len(out3)}"
    print(f"  [pass] empty weak_buckets is no-op (6 records)")


if __name__ == "__main__":
    main()
    print("all green")
