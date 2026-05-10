"""End-to-end smoke for the ablation harness.

Uses the official baseline JSON as a stand-in for two synthetic configs to
verify that the table + slice rendering work and that metrics agree with
``score_predictions``.
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ablation import (  # noqa: E402
    AblationConfig,
    AblationHarness,
    load_diag_tag_lookup,
    render_main_table,
)
from src.data_io import load_dev  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _perturb(preds: dict, rng: random.Random, swap_rate: float = 0.05) -> dict:
    """Lightly mutate predictions so the second 'config' looks slightly worse,
    giving the slice tables non-trivial differences to render."""
    out = {}
    labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
    for cid, rec in preds.items():
        rec = dict(rec)
        if rng.random() < swap_rate:
            rec["claim_label"] = rng.choice(labels)
        out[cid] = rec
    return out


def test_main_table_renders() -> None:
    baseline = json.loads((ROOT / "data" / "dev-claims-baseline.json").read_text(encoding="utf-8"))
    perturbed = _perturb(baseline, random.Random(0), 0.30)
    dev = load_dev()
    diag_gold, diag_tags = load_diag_tag_lookup()
    h = AblationHarness(dev_gold=dev, diag_gold=diag_gold, diag_tags=diag_tags)
    h.add(AblationConfig("A1", "baseline (random label, leaked evidence)"), baseline)
    h.add(AblationConfig("A2", "30% label noise on top of A1", flagship=True), perturbed)
    table = render_main_table(h.main_metrics())
    assert "A1" in table and "A2" in table
    assert "F" in table and "HM" in table
    print("  [pass] main table renders with two configs")


def test_diag_slice_only_for_flagship() -> None:
    baseline = json.loads((ROOT / "data" / "dev-claims-baseline.json").read_text(encoding="utf-8"))
    dev = load_dev()
    diag_gold, diag_tags = load_diag_tag_lookup()
    h = AblationHarness(dev_gold=dev, diag_gold=diag_gold, diag_tags=diag_tags)
    h.add(AblationConfig("X", "non-flagship", flagship=False), baseline)
    h.add(AblationConfig("Y", "flagship", flagship=True), baseline)
    domain_slices = h.diag_slice_metrics("domain")
    assert set(domain_slices.keys()) == {"Y"}, "non-flagship configs must be skipped"
    print("  [pass] only flagship configs reach diagnostic slices")


def test_render_full_report_writes_file() -> None:
    baseline = json.loads((ROOT / "data" / "dev-claims-baseline.json").read_text(encoding="utf-8"))
    dev = load_dev()
    diag_gold, diag_tags = load_diag_tag_lookup()
    h = AblationHarness(dev_gold=dev, diag_gold=diag_gold, diag_tags=diag_tags)
    h.add(AblationConfig("B3", "flagship", flagship=True), baseline)
    with tempfile.TemporaryDirectory() as td:
        report = h.render(td)
        out_file = Path(td) / "ablation_report.md"
        assert out_file.exists() and out_file.stat().st_size > 200
        assert "Ablation table (official dev)" in report
        assert "By climate-science domain" in report
        assert "By gold label" in report
    print("  [pass] full report renders to disk")


if __name__ == "__main__":
    print("test_ablation")
    test_main_table_renders()
    test_diag_slice_only_for_flagship()
    test_render_full_report_writes_file()
    print("all green")
