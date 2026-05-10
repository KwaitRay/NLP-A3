"""Stage 0.3 driver: apply scenario × domain × difficulty tagging.

Reads train + dev claim files, writes:
  - outputs/sft_data/claims_tagged.jsonl    (all train+dev rows with tags)
  - outputs/sft_data/tag_distribution.md    (cross-tab tables, sanity check)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .data_io import load_dev, load_train, write_jsonl
from .paths import LABELS, SFT_DIR
from .tagging import tag_claim


_DOMAINS = (
    "temperature",
    "co2_atmospheric",
    "sea_level",
    "extreme_weather",
    "paleoclimate",
    "models_attribution",
    "policy_economics",
    "general_other",
)
_SCENARIOS = (
    "supports_clear",
    "supports_aggregated",
    "refutes_clear",
    "refutes_aggregated",
    "nei_topic_off",
    "nei_underspec",
    "disputed_conflict",
)
_DIFFICULTIES = ("easy", "medium", "hard")


def _build_records() -> list[dict]:
    rows: list[dict] = []
    for split, claims in [("train", load_train()), ("dev", load_dev())]:
        for cid, c in claims.items():
            r = tag_claim(cid, c)
            r["split"] = split
            rows.append(r)
    return rows


def _md_table(title: str, header: list[str], rows: list[list[str]]) -> str:
    out = [f"### {title}", ""]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    out.append("")
    return "\n".join(out)


def _build_distribution_report(records: list[dict]) -> str:
    train_rows = [r for r in records if r["split"] == "train"]

    label_x_domain = Counter()
    for r in train_rows:
        label_x_domain[(r["claim_label"], r["domain"])] += 1

    scen_count = Counter(r["scenario"] for r in train_rows)
    diff_count = Counter(r["difficulty"]["level"] for r in train_rows)

    label_x_diff = Counter()
    for r in train_rows:
        label_x_diff[(r["claim_label"], r["difficulty"]["level"])] += 1

    pieces: list[str] = ["# Stage 0.3 — tag distribution (train split)", ""]

    pieces.append(
        _md_table(
            "Domain distribution by label",
            ["domain", *LABELS, "total"],
            [
                [
                    d,
                    *[str(label_x_domain.get((lbl, d), 0)) for lbl in LABELS],
                    str(sum(label_x_domain.get((lbl, d), 0) for lbl in LABELS)),
                ]
                for d in _DOMAINS
            ]
            + [
                [
                    "**total**",
                    *[
                        str(sum(label_x_domain.get((lbl, d), 0) for d in _DOMAINS))
                        for lbl in LABELS
                    ],
                    str(sum(label_x_domain.values())),
                ]
            ],
        )
    )

    pieces.append(
        _md_table(
            "Scenario distribution",
            ["scenario", "count"],
            [[s, str(scen_count.get(s, 0))] for s in _SCENARIOS],
        )
    )

    pieces.append(
        _md_table(
            "Difficulty by label",
            ["label", *_DIFFICULTIES, "total"],
            [
                [
                    lbl,
                    *[str(label_x_diff.get((lbl, d), 0)) for d in _DIFFICULTIES],
                    str(sum(label_x_diff.get((lbl, d), 0) for d in _DIFFICULTIES)),
                ]
                for lbl in LABELS
            ],
        )
    )

    return "\n".join(pieces)


def run() -> tuple[Path, Path]:
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    records = _build_records()
    out_jsonl = SFT_DIR / "claims_tagged.jsonl"
    write_jsonl(records, out_jsonl)

    out_md = SFT_DIR / "tag_distribution.md"
    out_md.write_text(_build_distribution_report(records), encoding="utf-8")
    return out_jsonl, out_md


if __name__ == "__main__":
    j, m = run()
    print(f"Wrote {j} ({sum(1 for _ in open(j, encoding='utf-8'))} rows)")
    print(f"Wrote {m}")
