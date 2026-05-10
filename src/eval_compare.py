"""4-track evaluation harness — Base / Base+RAG / SFT / SFT+DPO.

Each track produces an ``eval.py``-compatible prediction JSON, then a
single rendering pass writes the comparison Markdown:

    outputs/predictions/track1_base.json
    outputs/predictions/track2_base_rag.json
    outputs/predictions/track3_sft.json
    outputs/predictions/track4_dpo.json
    outputs/eval_compare.md

Reading the table:
    Track 1  isolates label accuracy without retrieval (F=0 by design).
    Track 2  adds RAG  → marginal gain of retrieval.
    Track 3  adds SFT  → marginal gain of supervised fine-tuning.
    Track 4  adds DPO + self-consistency → marginal gain of preference align + sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .eval_helpers import score_predictions
from .inference import predict_all


@dataclass
class TrackResult:
    name: str
    label_acc: float
    retrieval_f: float
    harmonic: float
    n_claims: int
    pred_path: Path


def evaluate_track(
    track_name: str,
    inferer: Any,
    claims: dict,
    gold: dict,
    out_dir: Path,
) -> TrackResult:
    """Run an inferer over claims, persist preds, return metrics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{track_name}.json"
    preds = predict_all(claims, inferer, out_path=pred_path)
    metrics = score_predictions(preds, gold)
    return TrackResult(
        name=track_name,
        label_acc=metrics["accuracy"],
        retrieval_f=metrics["f_score"],
        harmonic=metrics["harmonic_mean"],
        n_claims=metrics["n"],
        pred_path=pred_path,
    )


def render_compare_table(rows: list[TrackResult], out_md: Path) -> Path:
    """Markdown table with delta-vs-prev column."""
    out_md = Path(out_md)
    lines = [
        "# 4-Track Evaluation Comparison",
        "",
        "| # | Track | Label Acc | Retrieval F | Harmonic | Δ Harmonic vs prev |",
        "|---|-------|-----------|-------------|----------|--------------------|",
    ]
    prev = None
    for i, r in enumerate(rows, start=1):
        delta = "—" if prev is None else f"{r.harmonic - prev:+.4f}"
        lines.append(
            f"| {i} | {r.name} | {r.label_acc:.4f} | {r.retrieval_f:.4f} | "
            f"{r.harmonic:.4f} | {delta} |"
        )
        prev = r.harmonic

    n = rows[0].n_claims if rows else 0
    lines += [
        "",
        f"_Evaluated on {n} claims._",
        "",
        "**Reading guide:**",
        "- Track 1 expected harmonic ≈ 0 (no retrieval → F=0). "
        "Read its **Label Acc** column in isolation as the parametric-knowledge baseline.",
        "- Track 2 → 1: marginal value of RAG.",
        "- Track 3 → 2: marginal value of SFT.",
        "- Track 4 → 3: marginal value of DPO + self-consistency.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def run_4tracks(
    claims: dict,
    gold: dict,
    inferers: dict[str, Any],
    out_dir: Path,
) -> tuple[list[TrackResult], Path]:
    """Run all four inferers, persist preds, write the compare table.

    ``inferers`` maps the canonical track key to the prepared inferer:
        {
            "track1_base":     NoRagInferer(...),
            "track2_base_rag": ZeroShotInferer(retriever, base_model, ...),
            "track3_sft":      ZeroShotInferer(retriever, sft_model, ...),
            "track4_dpo":      ModelInferer(retriever, dpo_model, ...),
        }
    Missing keys are skipped (so you can run a partial comparison while
    still iterating on SFT/DPO).
    """
    canonical = ("track1_base", "track2_base_rag", "track3_sft", "track4_dpo")
    out_dir = Path(out_dir)
    rows: list[TrackResult] = []
    for key in canonical:
        if key not in inferers:
            print(f"  skip {key} (no inferer provided)")
            continue
        print(f"== {key} ==")
        rows.append(evaluate_track(key, inferers[key], claims, gold, out_dir / "predictions"))
    table_path = render_compare_table(rows, out_dir / "eval_compare.md")
    return rows, table_path
