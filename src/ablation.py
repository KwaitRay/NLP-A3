"""Stage 6 — ablation harness + diagnostic slicer.

Two layers:

1. ``AblationConfig`` — declarative toggles for each pipeline stage. Same
   config object drives the inferer factory in the notebook.
2. ``AblationHarness`` — accepts a list of (config_id, predictions_path or
   predictions_dict) and renders:
     - main ablation table on **official dev** (one row per config)
     - per-domain / per-scenario / per-difficulty diagnostic slices on
       **diag_test** (only for the configs the user marks as ``flagship``)

The harness is intentionally **model-agnostic**: you hand it predictions
already produced by ``inference.predict_all``. This way, the slow part
(model inference on Colab) can be persisted to JSON, and the report
rendering / table generation runs locally in seconds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .data_io import read_jsonl
from .eval_helpers import score_per_bucket, score_predictions
from .paths import LABELS, OUTPUTS_DIR, SPLITS_DIR


# -- Config ------------------------------------------------------------------

@dataclass
class AblationConfig:
    """Declarative description of one pipeline configuration.

    Two purposes: (a) drives the inferer factory in the notebook, (b)
    documents what each row in the table actually ran. The dataclass is
    serialised next to the predictions JSON for traceability.
    """
    config_id: str
    description: str
    # Retrieval stage toggles
    use_bm25: bool = True
    use_dense: bool = True
    fuse_strategy: str = "weighted"  # weighted | rrf
    use_rerank: bool = True
    use_rule_reorder: bool = True
    use_hyde: bool = False
    # Model stage
    model_kind: str = "sft"  # zero_shot | sft | dpo | sft_9b_int4 | retrieval_only
    n_samples: int = 5
    # Curriculum (only meaningful for SFT-trained variants)
    curriculum: bool = True
    # Marks rows that should also appear on the diagnostic slice tables.
    flagship: bool = False


# -- Predictions loader ------------------------------------------------------

def _load_preds(p: str | Path | dict) -> dict[str, dict]:
    if isinstance(p, dict):
        return p
    pp = Path(p)
    return json.loads(pp.read_text(encoding="utf-8"))


# -- Diag-test tag lookup ----------------------------------------------------

def load_diag_tag_lookup() -> tuple[dict[str, dict], dict[str, dict]]:
    """Load `diag_test.jsonl` once and return both the gold dict (claim_id →
    {claim_label, evidences}) and the tag dict (claim_id → tagged row)."""
    rows = list(read_jsonl(SPLITS_DIR / "diag_test.jsonl"))
    gold = {r["id"]: {"claim_label": r["claim_label"], "evidences": r["evidences"]} for r in rows}
    tags = {r["id"]: r for r in rows}
    return gold, tags


# -- Markdown rendering ------------------------------------------------------

_DOMAINS = (
    "temperature", "co2_atmospheric", "sea_level", "extreme_weather",
    "paleoclimate", "models_attribution", "policy_economics", "general_other",
)
_SCENARIOS = (
    "supports_clear", "supports_aggregated", "refutes_clear", "refutes_aggregated",
    "nei_topic_off", "nei_underspec", "disputed_conflict",
)
_DIFFICULTIES = ("easy", "medium", "hard")


def _md_row(cells: Sequence) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def render_main_table(
    rows: list[tuple[AblationConfig, dict[str, float]]],
) -> str:
    out: list[str] = ["## Ablation table (official dev)", ""]
    out.append(_md_row(["config", "description", "F", "A", "HM", "n"]))
    out.append("|---|---|---|---|---|---|")
    for cfg, m in rows:
        out.append(_md_row([
            cfg.config_id, cfg.description,
            f"{m['f_score']:.4f}", f"{m['accuracy']:.4f}",
            f"{m['harmonic_mean']:.4f}", m["n"],
        ]))
    return "\n".join(out)


def _format_metric_pair(m: dict) -> str:
    return f"F={m['f_score']:.3f}/A={m['accuracy']:.3f}"


def render_slice_table(
    title: str,
    bucket_keys: Sequence[str],
    rows_per_config: dict[str, dict[str, dict[str, float]]],
) -> str:
    """Render one diagnostic slice table.

    ``rows_per_config[config_id][bucket_key] -> {f_score, accuracy, n}``."""
    out: list[str] = [f"### {title}", ""]
    header = ["bucket"] + list(rows_per_config.keys()) + ["n"]
    out.append(_md_row(header))
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for bk in bucket_keys:
        # n = sample count for this bucket (taken from any config — they all
        # operate on the same ground truth).
        any_cfg = next(iter(rows_per_config))
        n = rows_per_config[any_cfg].get(bk, {}).get("n", 0)
        cells = [bk]
        for cfg_id, slices in rows_per_config.items():
            m = slices.get(bk)
            cells.append(_format_metric_pair(m) if m else "—")
        cells.append(n)
        out.append(_md_row(cells))
    return "\n".join(out)


# -- Top-level harness -------------------------------------------------------

@dataclass
class AblationHarness:
    """Holds gold (official dev + diag_test) and renders everything in one shot."""
    dev_gold: dict[str, dict]
    diag_gold: dict[str, dict]
    diag_tags: dict[str, dict]
    rows: list[tuple[AblationConfig, dict[str, dict]]] = field(default_factory=list)

    def add(self, cfg: AblationConfig, predictions: str | Path | dict) -> None:
        """Register one (config, predictions) pair. ``predictions`` may be a
        dict already in memory, or a path to a JSON file produced by
        ``inference.predict_all``."""
        self.rows.append((cfg, _load_preds(predictions)))

    def main_metrics(self) -> list[tuple[AblationConfig, dict[str, float]]]:
        return [(cfg, score_predictions(p, self.dev_gold)) for cfg, p in self.rows]

    def diag_slice_metrics(
        self, axis: str
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Slice each flagship config's predictions on ``diag_test`` by the
        given axis (``domain``, ``scenario``, ``difficulty``)."""
        if axis not in {"domain", "scenario", "difficulty"}:
            raise ValueError(f"axis must be domain|scenario|difficulty, got {axis!r}")

        def lookup(cid: str) -> str | None:
            row = self.diag_tags.get(cid)
            if row is None:
                return None
            if axis == "difficulty":
                return row.get("difficulty", {}).get("level")
            return row.get(axis)

        out: dict[str, dict[str, dict[str, float]]] = {}
        for cfg, preds in self.rows:
            if not cfg.flagship:
                continue
            # Restrict preds to diag_test claim_ids before slicing.
            restricted = {c: preds[c] for c in self.diag_gold if c in preds}
            out[cfg.config_id] = score_per_bucket(restricted, self.diag_gold, lookup)
        return out

    def render(self, out_dir: str | Path | None = None) -> str:
        """Build the full markdown report. Optionally write to disk."""
        out: list[str] = ["# Ablation report", ""]

        out.append(render_main_table(self.main_metrics()))
        out.append("")

        out.append("## Diagnostic slices on `diag_test`")
        out.append("")
        out.append("Format: F-score / Accuracy. Buckets follow Stage 0.3 tagging.")
        out.append("")

        for axis, keys, title in [
            ("domain", _DOMAINS, "By climate-science domain"),
            ("scenario", _SCENARIOS, "By scenario"),
            ("difficulty", _DIFFICULTIES, "By difficulty"),
        ]:
            slices = self.diag_slice_metrics(axis)
            if slices:
                out.append(render_slice_table(title, keys, slices))
                out.append("")

        # Append per-label slice on official dev for the worst-fit row.
        if self.rows:
            cfg_label_slices = {
                cfg.config_id: score_per_bucket(p, self.dev_gold, lambda c: self.dev_gold[c]["claim_label"])
                for cfg, p in self.rows
                if cfg.flagship
            }
            if cfg_label_slices:
                out.append(render_slice_table("By gold label (official dev)", LABELS, cfg_label_slices))
                out.append("")

        report = "\n".join(out)
        if out_dir is not None:
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / "ablation_report.md").write_text(report, encoding="utf-8")
        return report


def quick_harness() -> AblationHarness:
    """Convenience constructor wired against the standard splits."""
    from .data_io import load_dev
    dev = load_dev()
    diag_gold, diag_tags = load_diag_tag_lookup()
    return AblationHarness(dev_gold=dev, diag_gold=diag_gold, diag_tags=diag_tags)


# -- Default ablation configs (Plan §6.1) -----------------------------------

DEFAULT_CONFIGS: list[AblationConfig] = [
    AblationConfig("A1", "BM25 + zero-shot Qwen3.5",
                   use_dense=False, use_rerank=False, use_rule_reorder=False,
                   model_kind="zero_shot", n_samples=1),
    AblationConfig("A2", "+ dense (bge-m3)",
                   use_rerank=False, use_rule_reorder=False,
                   model_kind="zero_shot", n_samples=1),
    AblationConfig("A3", "+ cross-encoder rerank",
                   use_rule_reorder=False,
                   model_kind="zero_shot", n_samples=1),
    AblationConfig("A4", "+ rule reorder + HyDE",
                   use_hyde=True,
                   model_kind="zero_shot", n_samples=1),
    AblationConfig("B1", "A4 + Qwen3.5 SFT", use_hyde=True,
                   model_kind="sft", n_samples=1),
    AblationConfig("B2", "+ DPO", use_hyde=True,
                   model_kind="dpo", n_samples=1),
    AblationConfig("B3", "+ self-consistency", use_hyde=True,
                   model_kind="dpo", n_samples=5, flagship=True),
    AblationConfig("C1", "B3 with Qwen3.5-9B int4 (inference only)",
                   use_hyde=True, model_kind="sft_9b_int4", n_samples=5),
    AblationConfig("C2", "B3 without curriculum", use_hyde=True,
                   model_kind="dpo", n_samples=5, curriculum=False),
]
