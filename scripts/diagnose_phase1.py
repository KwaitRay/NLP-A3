"""Phase 1 diagnostic — analyse saved predictions without re-running inference.

Why this exists:
    Phase 1 baseline (Track 1 v1 on diag_test) returned Acc=0.3223, suspiciously
    close to the NEI fraction in diag_test (40/121 = 0.3306). The headline
    metric alone cannot tell us whether the model is:
      (a) outputting unparseable text → ``predict_all`` falls back to
          ``NOT_ENOUGH_INFO`` for ~all claims, accidentally hitting the NEI
          gold majority, or
      (b) genuinely predicting a mix of labels but biased toward NEI for the
          non-NEI claims.
    The fix path differs sharply: (a) → tighten parser / prompt; (b) → SFT
    data tilt in Phase 4.

What this script does:
    Pure analysis on the saved ``outputs/eval_phase1/track*_*_<dataset>.json``
    prediction dumps + ``outputs/splits/<dataset>.jsonl`` gold. No model
    loading, no GPU.

    For each (track, prompt) combo found on disk it reports:
      - Predicted label distribution
      - 4x4 confusion matrix (rows=gold, cols=pred)
      - Per-gold-label correctness
      - Evidence recall (Track 2+ only — Track 1's stub is skipped)
      - "Defaulting to NEI" heuristic flag
      - Sample mispredictions for human eyeballing

    Output:
      stdout summary + ``outputs/eval_phase1/diagnose_<dataset>.md``.

Usage::

    # all (track, prompt) combos that exist on disk for diag_test
    python -m scripts.diagnose_phase1 --dataset diag_test

    # restrict to a single combo
    python -m scripts.diagnose_phase1 --dataset diag_test --tracks 1 --prompts v1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import load_dev, read_jsonl  # noqa: E402
from src.paths import LABELS, OUTPUTS_DIR, SPLITS_DIR  # noqa: E402

EVAL_DIR = OUTPUTS_DIR / "eval_phase1"

# Cells of (track, prompt, dataset) decoded from filenames like
# ``track2_v1_diag_test.json``.
_FN_RE = re.compile(r"^track(?P<track>\d+)_(?P<prompt>[^_]+)_(?P<dataset>.+)\.json$")


_K_SUFFIX_RE = re.compile(r"_k\d+$")


def _strip_k_suffix(dataset_label: str) -> str:
    """``diag_test_k20`` -> ``diag_test``.

    phase1_eval.py adds the ``_k{N}`` suffix to output filenames when
    ``--final-k`` is non-default. The underlying gold split is the same,
    so we strip the suffix before looking it up.
    """
    return _K_SUFFIX_RE.sub("", dataset_label)


def _load_gold(dataset_label: str) -> dict[str, dict]:
    """Return ``{claim_id: {claim_label, claim_text, evidences}}``.

    ``dataset_label`` may carry a ``_k{N}`` suffix from phase1_eval's
    ``--final-k`` mode; we strip it before reading the splits file.
    """
    dataset = _strip_k_suffix(dataset_label)
    if dataset == "official_dev":
        return load_dev()
    path = SPLITS_DIR / f"{dataset}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return {
        r["id"]: {
            "claim_label": r["claim_label"],
            "claim_text": r["claim_text"],
            "evidences": r.get("evidences", []),
        }
        for r in read_jsonl(path)
    }


def _discover_runs(dataset: str, tracks: list[int] | None,
                   prompts: list[str] | None) -> list[tuple[int, str, Path]]:
    """Walk eval_phase1/ and return matching (track, prompt, json_path)."""
    out: list[tuple[int, str, Path]] = []
    if not EVAL_DIR.exists():
        return out
    for p in sorted(EVAL_DIR.glob("track*.json")):
        m = _FN_RE.match(p.name)
        if not m or m.group("dataset") != dataset:
            continue
        t = int(m.group("track"))
        v = m.group("prompt")
        if tracks and t not in tracks:
            continue
        if prompts and v not in prompts:
            continue
        out.append((t, v, p))
    return out


def _confusion(preds: dict, gold: dict) -> dict[str, dict[str, int]]:
    """Rows = gold label, cols = predicted label. Counts only."""
    mat = {g: {p: 0 for p in LABELS} for g in LABELS}
    for cid, g in gold.items():
        gl = g["claim_label"]
        pl = preds.get(cid, {}).get("claim_label", "NOT_ENOUGH_INFO")
        if gl in mat and pl in mat[gl]:
            mat[gl][pl] += 1
    return mat


def _evidence_recall(preds: dict, gold: dict) -> tuple[float, float, int]:
    """Macro / micro evidence recall on claims that have gold evidence.

    Returns (macro_recall, micro_recall, n_claims_evaluated). Skips claims
    whose gold evidence list is empty.
    """
    macro_sum = 0.0
    macro_n = 0
    tp = 0
    total_gold = 0
    for cid, g in gold.items():
        gold_ev = set(g.get("evidences") or [])
        if not gold_ev:
            continue
        pred_ev = set(preds.get(cid, {}).get("evidences") or [])
        hits = len(gold_ev & pred_ev)
        macro_sum += hits / len(gold_ev)
        macro_n += 1
        tp += hits
        total_gold += len(gold_ev)
    macro = (macro_sum / macro_n) if macro_n else 0.0
    micro = (tp / total_gold) if total_gold else 0.0
    return macro, micro, macro_n


def _is_nei_default_pattern(matrix: dict[str, dict[str, int]],
                            pred_dist: Counter, n: int) -> tuple[bool, str]:
    """Heuristic: model is *probably* defaulting NEI for non-parseable outputs.

    Trigger when:
      - predicted NEI > 50% of all claims, AND
      - of the non-NEI gold claims, NEI is the dominant prediction
        (i.e. the model isn't seeing those as non-NEI even when they are).

    Returns (is_pattern, human_explanation).
    """
    pred_nei = pred_dist.get("NOT_ENOUGH_INFO", 0)
    if pred_nei < 0.5 * n:
        return False, ""

    non_nei_total = sum(
        sum(matrix[g].values()) for g in LABELS if g != "NOT_ENOUGH_INFO"
    )
    non_nei_predicted_nei = sum(
        matrix[g]["NOT_ENOUGH_INFO"] for g in LABELS if g != "NOT_ENOUGH_INFO"
    )
    if non_nei_total == 0:
        return False, ""
    leakage = non_nei_predicted_nei / non_nei_total
    if leakage < 0.5:
        return False, ""
    return True, (
        f"Predicted NEI on {pred_nei}/{n} claims ({pred_nei/n:.1%}); "
        f"{non_nei_predicted_nei}/{non_nei_total} ({leakage:.1%}) of the "
        f"non-NEI gold claims were *also* labelled NEI — consistent with "
        f"the parser falling back to default_label on un-parseable model "
        f"output."
    )


def _format_confusion(matrix: dict[str, dict[str, int]]) -> str:
    """Markdown 4x4 confusion matrix with row/col totals."""
    cols = LABELS
    header = "| gold\\pred | " + " | ".join(cols) + " | **total** |"
    sep = "|---" * (len(cols) + 2) + "|"
    lines = [header, sep]
    col_tot = {c: 0 for c in cols}
    grand = 0
    for r in cols:
        row = matrix.get(r, {})
        row_tot = sum(row.get(c, 0) for c in cols)
        grand += row_tot
        cells = []
        for c in cols:
            v = row.get(c, 0)
            col_tot[c] += v
            marker = "**" if c == r and v else ""
            cells.append(f"{marker}{v}{marker}")
        lines.append(f"| {r} | " + " | ".join(cells) + f" | {row_tot} |")
    lines.append(
        "| **total** | " + " | ".join(str(col_tot[c]) for c in cols)
        + f" | {grand} |"
    )
    return "\n".join(lines)


def _samples(preds: dict, gold: dict, *, want: str, n: int = 3) -> list[str]:
    """Return ``n`` markdown bullets for samples matching ``want``.

    ``want`` is one of:
      - ``"miss_to_nei"``: gold != NEI, pred == NEI
      - ``"miss_other"``: gold != pred, neither is NEI
      - ``"correct_non_nei"``: gold == pred, gold != NEI
    """
    out: list[str] = []
    for cid, g in gold.items():
        gl = g["claim_label"]
        pl = preds.get(cid, {}).get("claim_label")
        cond = False
        if want == "miss_to_nei":
            cond = gl != "NOT_ENOUGH_INFO" and pl == "NOT_ENOUGH_INFO"
        elif want == "miss_other":
            cond = gl != pl and gl != "NOT_ENOUGH_INFO" and pl != "NOT_ENOUGH_INFO"
        elif want == "correct_non_nei":
            cond = gl == pl and gl != "NOT_ENOUGH_INFO"
        if not cond:
            continue
        text = g["claim_text"].replace("\n", " ").strip()
        if len(text) > 140:
            text = text[:137] + "..."
        out.append(f"  - `{cid}` gold={gl} pred={pl} — \"{text}\"")
        if len(out) >= n:
            break
    return out


# -- Main per-run report ----------------------------------------------------

def _diagnose_one(track: int, prompt: str, dataset: str,
                  preds: dict, gold: dict) -> tuple[str, dict]:
    """Build the markdown section + a small machine-readable summary."""
    n = len(gold)
    pred_dist = Counter(p.get("claim_label", "NOT_ENOUGH_INFO") for p in preds.values())
    gold_dist = Counter(g["claim_label"] for g in gold.values())
    matrix = _confusion(preds, gold)

    # Per-gold-label correctness.
    diag_acc = {}
    for g_label in LABELS:
        ok = matrix[g_label].get(g_label, 0)
        tot = sum(matrix[g_label].values())
        diag_acc[g_label] = (ok, tot, (ok / tot) if tot else 0.0)

    overall_correct = sum(diag_acc[g][0] for g in LABELS)
    acc = overall_correct / n if n else 0.0

    # Evidence recall — only meaningful when predicted evidences aren't a stub.
    has_real_ev = any(
        p.get("evidences") and p["evidences"] != ["evidence-0"]
        for p in preds.values()
    )
    if has_real_ev:
        macro_r, micro_r, ev_n = _evidence_recall(preds, gold)
    else:
        macro_r = micro_r = 0.0
        ev_n = 0

    is_nei, nei_msg = _is_nei_default_pattern(matrix, pred_dist, n)

    # Build the markdown section.
    parts = [
        f"## Track {track} — prompt `{prompt}`  (n={n}, Acc={acc:.4f})",
        "",
        "### Predicted vs gold label distribution",
        "",
        "| label | gold | predicted | Δ |",
        "|---|---|---|---|",
    ]
    for lab in LABELS:
        g = gold_dist.get(lab, 0)
        p = pred_dist.get(lab, 0)
        parts.append(f"| {lab} | {g} ({g/n:.1%}) | {p} ({p/n:.1%}) | {p - g:+d} |")
    parts.append("")
    parts.append("### Confusion matrix (rows = gold, columns = pred)")
    parts.append("")
    parts.append(_format_confusion(matrix))
    parts.append("")
    parts.append("### Per-gold-label correctness")
    parts.append("")
    parts.append("| gold label | correct / n | accuracy |")
    parts.append("|---|---|---|")
    for lab in LABELS:
        ok, tot, a = diag_acc[lab]
        parts.append(f"| {lab} | {ok} / {tot} | {a:.3f} |")
    parts.append("")

    if has_real_ev:
        parts.extend([
            "### Evidence recall (predicted ∩ gold) / gold",
            "",
            f"- macro: {macro_r:.4f}  (mean over {ev_n} claims with gold ev)",
            f"- micro: {micro_r:.4f}",
            "",
        ])

    parts.append("### Diagnostic flag")
    parts.append("")
    if is_nei:
        parts.append(f"- ⚠️ **Defaulting-to-NEI suspected**. {nei_msg}")
        parts.append(
            "- Likely root cause: model output not matching the `LABEL ##[..]##` "
            "format → `parse_response` returns ``default_label='NOT_ENOUGH_INFO'``."
        )
        parts.append(
            "- Next step: re-run a small sample with raw-output logging "
            "(`scripts.test_qwen35_inference` smoke 4b shape) to inspect generated "
            "text; consider prompt v2 (NEI explicit) to see if a different system "
            "prompt elicits parseable answers on non-NEI claims."
        )
    else:
        parts.append("- No defaulting-to-NEI pattern detected.")
    parts.append("")

    # Sample mispredictions.
    parts.append("### Sample mispredictions")
    parts.append("")
    sec = _samples(preds, gold, want="miss_to_nei", n=3)
    if sec:
        parts.append("- non-NEI gold predicted as NEI:")
        parts.extend(sec)
    sec = _samples(preds, gold, want="miss_other", n=3)
    if sec:
        parts.append("- non-NEI gold predicted as a *different* non-NEI label:")
        parts.extend(sec)
    sec = _samples(preds, gold, want="correct_non_nei", n=3)
    if sec:
        parts.append("- correct non-NEI predictions (sanity):")
        parts.extend(sec)
    parts.append("")

    summary = {
        "track": track,
        "prompt": prompt,
        "n": n,
        "acc": acc,
        "pred_dist": dict(pred_dist),
        "gold_dist": dict(gold_dist),
        "per_label_acc": {k: v[2] for k, v in diag_acc.items()},
        "evidence_macro_recall": macro_r if has_real_ev else None,
        "evidence_micro_recall": micro_r if has_real_ev else None,
        "nei_default_pattern": is_nei,
    }
    return "\n".join(parts), summary


# -- Cross-run summary ------------------------------------------------------

def _render_summary_table(rows: list[dict]) -> str:
    lines = [
        "| Track | Prompt | n | Acc | non-NEI acc | predicted NEI share | NEI-default? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        non_nei_total = sum(
            v for k, v in r["gold_dist"].items() if k != "NOT_ENOUGH_INFO"
        )
        # accuracy on non-NEI gold = sum(per-label correct for non-NEI) / non_nei_total
        per = r["per_label_acc"]
        non_nei_acc = 0.0
        if non_nei_total > 0:
            for lab, a in per.items():
                if lab == "NOT_ENOUGH_INFO":
                    continue
                non_nei_acc += a * (r["gold_dist"].get(lab, 0) / non_nei_total)
        pred_nei_share = r["pred_dist"].get("NOT_ENOUGH_INFO", 0) / r["n"] if r["n"] else 0.0
        flag = "⚠️ yes" if r["nei_default_pattern"] else "no"
        lines.append(
            f"| {r['track']} | {r['prompt']} | {r['n']} | {r['acc']:.4f} | "
            f"{non_nei_acc:.4f} | {pred_nei_share:.1%} | {flag} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1 prediction diagnostic")
    p.add_argument("--dataset", default="diag_test",
                   help="Base name (diag_test / dev_holdout / official_dev) or "
                        "a suffixed label such as 'diag_test_k20' to inspect "
                        "phase1_eval runs from a non-default --final-k.")
    p.add_argument("--tracks", default=None,
                   help="Comma-separated track ids (default: all found on disk).")
    p.add_argument("--prompts", default=None,
                   help="Comma-separated prompt versions (default: all found).")
    p.add_argument("--out", default=None,
                   help="Output markdown path (default: outputs/eval_phase1/"
                        "diagnose_<dataset>.md).")
    args = p.parse_args()

    tracks = [int(x) for x in args.tracks.split(",")] if args.tracks else None
    prompts = args.prompts.split(",") if args.prompts else None

    runs = _discover_runs(args.dataset, tracks, prompts)
    if not runs:
        raise SystemExit(
            f"No track*_<prompt>_{args.dataset}.json found under {EVAL_DIR}. "
            f"Run scripts.phase1_eval first."
        )

    print(f"Found {len(runs)} prediction file(s) for dataset={args.dataset}:")
    for t, v, path in runs:
        print(f"  track{t}_{v}  ←  {path.name}")
    print()

    gold = _load_gold(args.dataset)
    sections: list[str] = []
    summaries: list[dict] = []
    for track, prompt, path in runs:
        preds = json.loads(path.read_text(encoding="utf-8"))
        md, summ = _diagnose_one(track, prompt, args.dataset, preds, gold)
        sections.append(md)
        summaries.append(summ)

    # Header + cross-run summary first.
    head = [
        f"# Phase 1 diagnostic — {args.dataset}",
        "",
        f"Source: `outputs/eval_phase1/track*_*_{args.dataset}.json`  "
        f"({len(runs)} run(s))",
        "",
        "## Cross-run summary",
        "",
        _render_summary_table(summaries),
        "",
        "Legend:",
        "- *non-NEI acc*: accuracy on the non-NEI gold claims only. If this "
        "is near 0 while overall acc ≈ gold NEI share, the model is probably "
        "defaulting to NEI for everything (parse-fallback pattern).",
        "- *predicted NEI share*: fraction of claims the parser labelled NEI.",
        "- *NEI-default?*: see per-run heuristic in each section below.",
        "",
    ]
    doc = "\n".join(head + sections)
    out_path = Path(args.out) if args.out else EVAL_DIR / f"diagnose_{args.dataset}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")

    # Stdout: print the cross-run summary so the user sees the headline.
    print(_render_summary_table(summaries))
    print()
    print(f"→ full report written to {out_path}")


if __name__ == "__main__":
    main()
