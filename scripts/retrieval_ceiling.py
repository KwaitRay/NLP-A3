"""Phase 3.5 — retrieval ceiling audit.

Why this exists:
    Phase 2 prompt sweep showed evidence recall locked at ~0.11 across
    all four prompt variants (debug_log 复用经验 22). That makes the
    F-score ceiling ~0.12 / HM ceiling ~0.21, even with perfect labels.
    SFT can lift label classification but not retrieval. Must audit
    retrieval first.

What this script does:
    Sweeps retrieval configurations and measures evidence recall@k on
    `outputs/splits/<dataset>.jsonl` gold. No LLM (synonym expansion
    uses WordNet only). Reuses BM25 / dense / reranker / RetrievalConfig
    from the existing pipeline.

Modes (selectable via --mode, comma-separated or `all`):

  final_k       Run the full pipeline to top-100 ONCE, then slice into
                final_k ∈ {5, 10, 20, 50, 100}. Cheap (one rerank pass).
  retriever     Ablate components: BM25-only, dense-only, fused
                (no rerank), full pipeline.
  fusion_w      Sweep w_bm25 ∈ {0.1, 0.3, 0.5, 0.7, 0.9} with the full
                pipeline. Slowest mode (5 rerank passes).
  synonym_expand
                Multi-query: original claim alone vs claim + WordNet
                synonym variants (`src.query_rewrite.synonym_expand`),
                fused via RRF across all variants. Two configs.
  llm_rewrite   Multi-query with LLM rewrites: baseline / HyDE only /
                sub-claim only / HyDE + sub-claim. Requires
                `outputs/query_rewrite/claim_rewrites.jsonl` from
                `scripts.rewrite_queries`. Phase 3.5b escape hatch when
                non-LLM modes can't push recall@20 past 0.50.

  all           Run all five modes (NOTE: llm_rewrite needs cache built
                first; otherwise it errors out).

Output:
    outputs/eval_phase1/retrieval_ceiling_<dataset>.md  — per-mode tables
    + best overall config callout.

Runtime: ~30-60 min on AutoDL 4080 SUPER for `--mode all` (reranker
dominates). Single mode is ~5-20 min. The `final_k` mode is the
quickest sanity check.

Usage::

    # Full audit
    python -m scripts.retrieval_ceiling --dataset diag_test --mode all

    # Quick sanity (just final_k)
    python -m scripts.retrieval_ceiling --dataset diag_test --mode final_k

    # Skip rerank to halve runtime (recall numbers slightly lower)
    python -m scripts.retrieval_ceiling --dataset diag_test --mode all --no-rerank
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import load_evidence, read_jsonl  # noqa: E402
from src.paths import LABELS, OUTPUTS_DIR, SPLITS_DIR  # noqa: E402
from src.retrieval.fuse import rrf_fuse  # noqa: E402
from src.retrieval.pipeline import RetrievalConfig, RetrievalPipeline  # noqa: E402

EVAL_DIR = OUTPUTS_DIR / "eval_phase1"
OUT_PATH_TEMPLATE = "retrieval_ceiling_{dataset}.md"

# Final-k slice points used by every mode for reporting. We always slice the
# top-100 result list at these k values; the per-mode table picks one.
K_GRID = [5, 10, 20, 50, 100]


# -- Data ------------------------------------------------------------------

def _load_gold(dataset: str) -> dict[str, dict]:
    """{claim_id: {claim_label, claim_text, evidences}} from splits/*.jsonl."""
    if dataset not in {"diag_test", "dev_holdout"}:
        raise ValueError(f"unsupported dataset for retrieval audit: {dataset!r}")
    path = SPLITS_DIR / f"{dataset}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")
    return {
        r["id"]: {
            "claim_label": r["claim_label"],
            "claim_text": r["claim_text"],
            "evidences": r.get("evidences", []),
        }
        for r in read_jsonl(path)
    }


def _build_components(use_rerank: bool):
    """Load BM25, dense, reranker once. Returns (bm25, dense, reranker)."""
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import DenseRetriever

    bm25_dir = OUTPUTS_DIR / "bm25_index"
    dense_dir = OUTPUTS_DIR / "dense_index"
    if not bm25_dir.exists():
        raise FileNotFoundError(
            f"BM25 index missing at {bm25_dir}. "
            f"Run: python -m scripts.build_indexes --skip-dense"
        )
    bm25 = BM25Retriever.load(bm25_dir)

    dense = None
    if (dense_dir / "faiss.index").exists():
        dense = DenseRetriever.load(dense_dir, max_seq_length=256, fp16=True)
    else:
        print(f"  WARN: dense index missing at {dense_dir}; BM25-only audit")

    reranker = None
    if use_rerank:
        try:
            from src.retrieval.rerank import CrossEncoderReranker
            reranker = CrossEncoderReranker()
        except Exception as e:
            print(f"  WARN: reranker load failed ({type(e).__name__}: {e}); audit without rerank")
    return bm25, dense, reranker


# -- Recall computation ----------------------------------------------------

def _recall_at_k_list(retrieved: list[str], gold: set[str], k_grid: list[int]) -> dict[int, float]:
    """Per-claim recall at each k in k_grid against gold."""
    out = {}
    for k in k_grid:
        top = set(retrieved[:k])
        out[k] = len(top & gold) / len(gold) if gold else 0.0
    return out


def _aggregate_recall(per_claim: list[dict[int, float]], gold_sizes: list[int],
                      hits_at_k: dict[int, int], k_grid: list[int]) -> dict[int, dict]:
    """Aggregate per-claim recalls into macro/micro at each k."""
    out = {}
    total_gold = sum(gold_sizes)
    for k in k_grid:
        macro = sum(r[k] for r in per_claim) / len(per_claim) if per_claim else 0.0
        micro = hits_at_k[k] / total_gold if total_gold else 0.0
        out[k] = {"macro": macro, "micro": micro}
    return out


def _measure_config(cfg: RetrievalConfig, gold: dict, evidence: dict,
                    bm25, dense, reranker, *, retrieve_k: int = 100,
                    progress: str = "") -> dict:
    """Run a config across all claims, return aggregated recall@k for k_grid.

    `retrieve_k` is the depth at which we collect predictions; we slice it
    at every value in K_GRID for reporting. Setting cfg.final_k = retrieve_k
    so the pipeline returns the deep list.
    """
    cfg_deep = replace(cfg, final_k=retrieve_k, label_conditioned_k=False)
    pipeline = RetrievalPipeline(
        evidence_corpus=evidence, bm25=bm25, dense=dense, reranker=reranker, cfg=cfg_deep,
    )

    per_claim: list[dict[int, float]] = []
    gold_sizes: list[int] = []
    hits_at_k: dict[int, int] = {k: 0 for k in K_GRID}
    per_label_macro: dict[str, dict[int, list[float]]] = {
        l: {k: [] for k in K_GRID} for l in LABELS
    }
    n_skipped = 0
    t0 = time.time()
    items = list(gold.items())
    for i, (cid, g) in enumerate(items):
        gold_ev = set(g.get("evidences") or [])
        if not gold_ev:
            n_skipped += 1
            continue
        retrieved = pipeline.retrieve(g["claim_text"])
        ids = [eid for eid, _ in retrieved]
        rec_at_k = _recall_at_k_list(ids, gold_ev, K_GRID)
        per_claim.append(rec_at_k)
        gold_sizes.append(len(gold_ev))
        for k in K_GRID:
            hits_at_k[k] += len(set(ids[:k]) & gold_ev)
        for k in K_GRID:
            per_label_macro[g["claim_label"]][k].append(rec_at_k[k])

        if progress and (i + 1) % 20 == 0:
            eta = (time.time() - t0) * (len(items) - i - 1) / (i + 1)
            print(f"    {progress}  {i+1}/{len(items)}  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    agg = _aggregate_recall(per_claim, gold_sizes, hits_at_k, K_GRID)
    per_label_summary = {
        l: {k: (sum(v) / len(v)) if v else 0.0 for k, v in d.items()}
        for l, d in per_label_macro.items()
    }
    return {
        "metrics_at_k": agg,
        "per_label_macro": per_label_summary,
        "n_claims": len(per_claim),
        "n_skipped": n_skipped,
        "elapsed_sec": elapsed,
    }


# -- Multi-query (synonym expand) ------------------------------------------

def _retrieve_multi_query(
    claim: str, queries: list[str], bm25, dense, reranker, evidence: dict,
    *, top_each: int = 200, fuse_top: int = 150, rerank_top: int = 50,
    final_k: int = 100, use_rerank: bool = True,
) -> list[str]:
    """Retrieve once per query variant, fuse via RRF, then optionally rerank.

    Returns a flat list of evidence_ids (deepest-first).
    """
    all_lists: list[list[tuple[str, float]]] = []
    if bm25:
        for hits in bm25.search_batch(queries, k=top_each):
            all_lists.append(hits)
    if dense:
        for hits in dense.search_batch(queries, k=top_each):
            all_lists.append(hits)
    if not all_lists:
        return []
    fused = rrf_fuse(*all_lists, top_k=fuse_top)

    if use_rerank and reranker is not None:
        cands = [(eid, evidence.get(eid, "")) for eid, _ in fused[:rerank_top]]
        reranked = reranker.rerank(claim, cands)
        ranked = reranked + fused[rerank_top:]
    else:
        ranked = fused
    return [eid for eid, _ in ranked[:final_k]]


def _measure_synonym_expand(gold: dict, evidence: dict, bm25, dense, reranker,
                            *, use_synonym: bool, use_rerank: bool) -> dict:
    """Measure recall for either: original claim only, or claim + WordNet synonyms."""
    from src.query_rewrite import synonym_expand

    per_claim: list[dict[int, float]] = []
    gold_sizes: list[int] = []
    hits_at_k: dict[int, int] = {k: 0 for k in K_GRID}
    per_label_macro: dict[str, dict[int, list[float]]] = {
        l: {k: [] for k in K_GRID} for l in LABELS
    }
    n_skipped = 0
    t0 = time.time()
    items = list(gold.items())
    for i, (cid, g) in enumerate(items):
        gold_ev = set(g.get("evidences") or [])
        if not gold_ev:
            n_skipped += 1
            continue
        if use_synonym:
            queries = synonym_expand(g["claim_text"])
        else:
            queries = [g["claim_text"]]
        ids = _retrieve_multi_query(
            g["claim_text"], queries, bm25, dense, reranker, evidence,
            final_k=100, use_rerank=use_rerank,
        )
        rec_at_k = _recall_at_k_list(ids, gold_ev, K_GRID)
        per_claim.append(rec_at_k)
        gold_sizes.append(len(gold_ev))
        for k in K_GRID:
            hits_at_k[k] += len(set(ids[:k]) & gold_ev)
        for k in K_GRID:
            per_label_macro[g["claim_label"]][k].append(rec_at_k[k])
        if (i + 1) % 20 == 0:
            eta = (time.time() - t0) * (len(items) - i - 1) / (i + 1)
            tag = "syn" if use_synonym else "orig"
            print(f"    synonym_expand[{tag}]  {i+1}/{len(items)}  ETA {eta:.0f}s")

    agg = _aggregate_recall(per_claim, gold_sizes, hits_at_k, K_GRID)
    per_label_summary = {
        l: {k: (sum(v) / len(v)) if v else 0.0 for k, v in d.items()}
        for l, d in per_label_macro.items()
    }
    return {
        "metrics_at_k": agg,
        "per_label_macro": per_label_summary,
        "n_claims": len(per_claim),
        "n_skipped": n_skipped,
        "elapsed_sec": time.time() - t0,
    }


# -- Mode runners ----------------------------------------------------------

def run_mode_final_k(gold, evidence, bm25, dense, reranker, *, use_rerank: bool):
    """Single full-pipeline run to top-100; slice for each k. Cheapest mode."""
    cfg = RetrievalConfig(
        use_bm25=True, use_dense=dense is not None,
        use_rerank=use_rerank and reranker is not None,
        use_rule_reorder=False,
    )
    print("  [final_k] one pipeline run, slicing top-100...")
    res = _measure_config(cfg, gold, evidence, bm25, dense, reranker,
                          retrieve_k=100, progress="final_k")
    # Each k in K_GRID is a "config" row in the report.
    rows = []
    for k in K_GRID:
        m = res["metrics_at_k"][k]
        rows.append({
            "config": f"full pipeline, final_k={k}",
            "k": k,
            "macro": m["macro"], "micro": m["micro"],
            "per_label_at_k": {l: res["per_label_macro"][l][k] for l in LABELS},
            "n": res["n_claims"], "elapsed": res["elapsed_sec"],
        })
    return rows


def run_mode_retriever(gold, evidence, bm25, dense, reranker, *, use_rerank: bool):
    """Ablate retrieval components: BM25-only, dense-only, fused, full."""
    rerank_on = use_rerank and reranker is not None
    configs = [
        ("BM25 only",        RetrievalConfig(use_bm25=True,  use_dense=False, use_rerank=False, use_rule_reorder=False)),
        ("dense only",       RetrievalConfig(use_bm25=False, use_dense=dense is not None, use_rerank=False, use_rule_reorder=False)),
        ("fused (no rerank)",RetrievalConfig(use_bm25=True,  use_dense=dense is not None, use_rerank=False, use_rule_reorder=False)),
    ]
    if rerank_on:
        configs.append(
            ("full (fused + rerank)", RetrievalConfig(use_bm25=True, use_dense=dense is not None, use_rerank=True, use_rule_reorder=False))
        )
    rows = []
    for label, cfg in configs:
        print(f"  [retriever] {label}...")
        # Use the active reranker only when the cfg asks for it.
        rer = reranker if cfg.use_rerank else None
        res = _measure_config(cfg, gold, evidence, bm25, dense, rer,
                              retrieve_k=100, progress=label)
        # Report at K_GRID[0] = 5 by default (matches current production)
        # but include all k values in the per-row dict for completeness.
        for k in K_GRID:
            m = res["metrics_at_k"][k]
            rows.append({
                "config": f"{label}, final_k={k}",
                "k": k,
                "macro": m["macro"], "micro": m["micro"],
                "per_label_at_k": {l: res["per_label_macro"][l][k] for l in LABELS},
                "n": res["n_claims"], "elapsed": res["elapsed_sec"],
            })
    return rows


def run_mode_fusion_w(gold, evidence, bm25, dense, reranker, *, use_rerank: bool):
    """Sweep w_bm25 ∈ {0.1, 0.3, 0.5, 0.7, 0.9}; full pipeline."""
    rerank_on = use_rerank and reranker is not None
    rows = []
    for w in (0.1, 0.3, 0.5, 0.7, 0.9):
        cfg = RetrievalConfig(
            use_bm25=True, use_dense=dense is not None,
            w_bm25=w, w_dense=1.0 - w,
            use_rerank=rerank_on, use_rule_reorder=False,
        )
        print(f"  [fusion_w] w_bm25={w}, w_dense={1.0 - w:.1f}...")
        rer = reranker if rerank_on else None
        res = _measure_config(cfg, gold, evidence, bm25, dense, rer,
                              retrieve_k=100, progress=f"w_bm25={w}")
        for k in K_GRID:
            m = res["metrics_at_k"][k]
            rows.append({
                "config": f"w_bm25={w}, w_dense={1.0 - w:.1f}, final_k={k}",
                "k": k,
                "macro": m["macro"], "micro": m["micro"],
                "per_label_at_k": {l: res["per_label_macro"][l][k] for l in LABELS},
                "n": res["n_claims"], "elapsed": res["elapsed_sec"],
            })
    return rows


def run_mode_synonym_expand(gold, evidence, bm25, dense, reranker, *, use_rerank: bool):
    """Original claim vs claim + WordNet synonyms (RRF across query variants)."""
    rows = []
    for use_syn in (False, True):
        label = "claim + WordNet synonyms" if use_syn else "claim only (baseline)"
        print(f"  [synonym_expand] {label}...")
        res = _measure_synonym_expand(
            gold, evidence, bm25, dense, reranker,
            use_synonym=use_syn, use_rerank=use_rerank,
        )
        for k in K_GRID:
            m = res["metrics_at_k"][k]
            rows.append({
                "config": f"{label}, final_k={k}",
                "k": k,
                "macro": m["macro"], "micro": m["micro"],
                "per_label_at_k": {l: res["per_label_macro"][l][k] for l in LABELS},
                "n": res["n_claims"], "elapsed": res["elapsed_sec"],
            })
    return rows


def _load_llm_rewrites() -> dict[str, dict]:
    """Load cached HyDE + sub-claim rewrites produced by scripts.rewrite_queries."""
    cache_path = OUTPUTS_DIR / "query_rewrite" / "claim_rewrites.jsonl"
    if not cache_path.exists():
        raise SystemExit(
            f"LLM rewrite cache not found at {cache_path}. "
            f"Run `python -m scripts.rewrite_queries --splits diag_test` first."
        )
    out: dict[str, dict] = {}
    import json
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["claim_id"]] = rec
    return out


def run_mode_llm_rewrite(gold, evidence, bm25, dense, reranker, *, use_rerank: bool):
    """LLM-driven multi-query (HyDE + sub-claim) vs baseline claim-only.

    Requires `outputs/query_rewrite/claim_rewrites.jsonl` from
    `scripts.rewrite_queries`. Each claim gets:
      - baseline: 1 query (the claim itself)
      - hyde_only: 2 queries (claim + HyDE hypothetical passage)
      - sub_only: 1 + N queries (claim + 1-3 sub-claims)
      - full: 1 + 1 + N queries (claim + HyDE + sub-claims)

    All four configs run through retrieve_multi_query with RRF fusion.
    """
    cache = _load_llm_rewrites()
    missing = [cid for cid in gold if cid not in cache]
    if missing:
        print(f"  WARN: {len(missing)}/{len(gold)} claims missing from rewrite "
              f"cache; they will fall back to claim-only. Run "
              f"`scripts.rewrite_queries` to fill.")

    rows = []
    configs = [
        ("baseline (claim only)", False, False),
        ("HyDE only", True, False),
        ("sub-claims only", False, True),
        ("HyDE + sub-claims", True, True),
    ]
    for label, use_hyde, use_sub in configs:
        print(f"  [llm_rewrite] {label}...")
        per_claim: list[dict[int, float]] = []
        gold_sizes: list[int] = []
        hits_at_k: dict[int, int] = {k: 0 for k in K_GRID}
        per_label_macro: dict[str, dict[int, list[float]]] = {
            l: {k: [] for k in K_GRID} for l in LABELS
        }
        n_skipped = 0
        t0 = time.time()
        items = list(gold.items())
        for i, (cid, g) in enumerate(items):
            gold_ev = set(g.get("evidences") or [])
            if not gold_ev:
                n_skipped += 1
                continue
            queries = [g["claim_text"]]
            rec = cache.get(cid)
            if rec:
                if use_hyde and rec.get("hyde"):
                    queries.append(rec["hyde"])
                if use_sub and rec.get("sub_claims"):
                    queries.extend(s for s in rec["sub_claims"] if s and s != g["claim_text"])
            ids = _retrieve_multi_query(
                g["claim_text"], queries, bm25, dense, reranker, evidence,
                final_k=100, use_rerank=use_rerank,
            )
            rec_at_k = _recall_at_k_list(ids, gold_ev, K_GRID)
            per_claim.append(rec_at_k)
            gold_sizes.append(len(gold_ev))
            for k in K_GRID:
                hits_at_k[k] += len(set(ids[:k]) & gold_ev)
            for k in K_GRID:
                per_label_macro[g["claim_label"]][k].append(rec_at_k[k])
            if (i + 1) % 20 == 0:
                eta = (time.time() - t0) * (len(items) - i - 1) / (i + 1)
                print(f"    llm_rewrite[{label}]  {i+1}/{len(items)}  ETA {eta:.0f}s")

        agg = _aggregate_recall(per_claim, gold_sizes, hits_at_k, K_GRID)
        per_label_summary = {
            l: {k: (sum(v) / len(v)) if v else 0.0 for k, v in d.items()}
            for l, d in per_label_macro.items()
        }
        for k in K_GRID:
            m = agg[k]
            rows.append({
                "config": f"{label}, final_k={k}",
                "k": k,
                "macro": m["macro"], "micro": m["micro"],
                "per_label_at_k": {l: per_label_summary[l][k] for l in LABELS},
                "n": len(per_claim), "elapsed": time.time() - t0,
            })
    return rows


# -- Reporting -------------------------------------------------------------

def _render_table(rows: list[dict], *, k_focus: int = 5) -> str:
    """Per-mode table at a given k (default k=5 to match production)."""
    lines = [
        f"| config | n | macro recall@{k_focus} | micro recall@{k_focus} | "
        f"S | R | NEI | D |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows_at_k = [r for r in rows if r["k"] == k_focus]
    rows_at_k.sort(key=lambda r: -r["macro"])
    for r in rows_at_k:
        per = r["per_label_at_k"]
        lines.append(
            f"| {r['config']} | {r['n']} | {r['macro']:.4f} | {r['micro']:.4f} | "
            f"{per['SUPPORTS']:.3f} | {per['REFUTES']:.3f} | "
            f"{per['NOT_ENOUGH_INFO']:.3f} | {per['DISPUTED']:.3f} |"
        )
    return "\n".join(lines)


def _render_k_curve(rows: list[dict], best_config_name: str | None = None) -> str:
    """Multi-row table showing recall@k across K_GRID for each unique base config."""
    by_base: dict[str, dict[int, dict]] = {}
    for r in rows:
        base = r["config"].rsplit(", final_k=", 1)[0]
        by_base.setdefault(base, {})[r["k"]] = r
    lines = [
        "| base config | " + " | ".join(f"r@{k}" for k in K_GRID) + " |",
        "|---" * (len(K_GRID) + 1) + "|",
    ]
    for base, by_k in by_base.items():
        marker = "**" if base == best_config_name else ""
        cells = " | ".join(f"{by_k[k]['macro']:.3f}" for k in K_GRID if k in by_k)
        lines.append(f"| {marker}{base}{marker} | {cells} |")
    return "\n".join(lines)


def _pick_best_overall(all_rows: dict[str, list[dict]],
                       *, k_focus: int = 5) -> tuple[str, dict] | None:
    """Best (mode, row) at k_focus by macro recall."""
    best = None
    for mode, rows in all_rows.items():
        for r in rows:
            if r["k"] != k_focus:
                continue
            if best is None or r["macro"] > best[1]["macro"]:
                best = (mode, r)
    return best


def write_report(all_rows: dict[str, list[dict]], dataset: str,
                 *, modes_ran: list[str], used_rerank: bool) -> Path:
    parts: list[str] = [
        f"# Retrieval ceiling audit — {dataset}",
        "",
        f"Modes run: {', '.join(modes_ran)}.  "
        f"Rerank: {'on' if used_rerank else 'off'}.  "
        f"k_grid = {K_GRID}.",
        "",
        "Reference: Track 2 v1 macro recall@5 was **0.1090** in "
        "`outputs/eval_phase1/diagnose_diag_test.md`. Numbers below "
        "should reproduce that for the matching config (full pipeline, "
        "w_bm25=0.3, final_k=5, rerank on).",
        "",
    ]

    # Best overall callout.
    best = _pick_best_overall(all_rows, k_focus=5)
    if best:
        mode, row = best
        parts.extend([
            "## 🏆 Best overall at recall@5",
            "",
            f"- mode: `{mode}`",
            f"- config: `{row['config']}`",
            f"- macro recall@5: **{row['macro']:.4f}**  "
            f"(micro {row['micro']:.4f})",
            f"- per-label macro: S {row['per_label_at_k']['SUPPORTS']:.3f} / "
            f"R {row['per_label_at_k']['REFUTES']:.3f} / "
            f"NEI {row['per_label_at_k']['NOT_ENOUGH_INFO']:.3f} / "
            f"D {row['per_label_at_k']['DISPUTED']:.3f}",
            "",
        ])
        # Pinpoint best base config across k for the curve.
        best_base = row["config"].rsplit(", final_k=", 1)[0]
    else:
        best_base = None

    for mode in modes_ran:
        rows = all_rows.get(mode, [])
        if not rows:
            continue
        parts.append(f"## Mode: `{mode}`")
        parts.append("")
        parts.append(f"Elapsed: {sum({r['config']: r['elapsed'] for r in rows}.values()):.0f}s")
        parts.append("")
        parts.append(f"### recall@5 (production k)")
        parts.append("")
        parts.append(_render_table(rows, k_focus=5))
        parts.append("")
        parts.append(f"### recall@k curve (k ∈ {K_GRID})")
        parts.append("")
        parts.append(_render_k_curve(rows, best_config_name=best_base))
        parts.append("")

    parts.append("---")
    parts.append("")
    parts.append("Next actions:")
    parts.append("1. Lock the best `RetrievalConfig` into `src/retrieval/pipeline.py` "
                 "(default) and re-run `phase1_eval --tracks 2 --prompts v1` to "
                 "confirm Track 2 HM lifts.")
    parts.append("2. Rebuild SFT data with the new retrieval config: "
                 "`python -m src.build_stage0 --force`.")
    parts.append("3. If recall@100 is still < 0.30, escalate to LLM-driven "
                 "rewrite (HyDE / sub-claim, see `optimization_plan.md` §3.5.4).")

    out_path = EVAL_DIR / OUT_PATH_TEMPLATE.format(dataset=dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


# -- Main ------------------------------------------------------------------

MODE_RUNNERS = {
    "final_k": run_mode_final_k,
    "retriever": run_mode_retriever,
    "fusion_w": run_mode_fusion_w,
    "synonym_expand": run_mode_synonym_expand,
    "llm_rewrite": run_mode_llm_rewrite,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="diag_test",
                   choices=["diag_test", "dev_holdout"])
    p.add_argument("--mode", default="final_k",
                   help=f"Comma-separated. Available: {','.join(MODE_RUNNERS)},all")
    p.add_argument("--no-rerank", action="store_true",
                   help="Skip cross-encoder rerank everywhere (cuts ~50%% runtime).")
    args = p.parse_args()

    use_rerank = not args.no_rerank
    raw_modes = [m.strip() for m in args.mode.split(",")]
    if "all" in raw_modes:
        modes = list(MODE_RUNNERS.keys())
    else:
        modes = []
        for m in raw_modes:
            if m not in MODE_RUNNERS:
                raise SystemExit(f"unknown mode: {m}; available: {list(MODE_RUNNERS)} (or 'all')")
            modes.append(m)

    print(f"=== Retrieval ceiling audit on {args.dataset} ===")
    print(f"  modes: {modes}")
    print(f"  rerank: {'on' if use_rerank else 'off'}")

    print("\n[1/3] loading gold + evidence corpus...")
    gold = _load_gold(args.dataset)
    evidence = load_evidence(show_progress=True)
    print(f"  {len(gold)} claims, {len(evidence):,} passages")

    print("\n[2/3] loading retrievers...")
    bm25, dense, reranker = _build_components(use_rerank=use_rerank)
    print(f"  bm25={'on' if bm25 else 'off'}, dense={'on' if dense else 'off'}, "
          f"reranker={'on' if reranker else 'off'}")

    print(f"\n[3/3] running {len(modes)} mode(s)...")
    all_rows: dict[str, list[dict]] = {}
    for mode in modes:
        print(f"\n--- mode: {mode} ---")
        runner = MODE_RUNNERS[mode]
        all_rows[mode] = runner(gold, evidence, bm25, dense, reranker, use_rerank=use_rerank)

    out_path = write_report(all_rows, args.dataset,
                            modes_ran=modes, used_rerank=use_rerank)
    print(f"\n=== Report written to {out_path} ===\n")
    print(out_path.read_text(encoding="utf-8").split("## Mode:", 1)[0])  # head only


if __name__ == "__main__":
    main()
