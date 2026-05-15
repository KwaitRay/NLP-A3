"""Build reranker fine-tuning data — hard negatives from the production retriever.

Why this exists:
    `bge-reranker-base` is a net negative on climate (×1.68 worse recall@5,
    see `outputs/eval_phase1/retrieval_ceiling_diag_test.md`). The fix is
    domain fine-tuning. For that we need (claim, positive_ev, hard_negs)
    rows where the negatives come from the SAME distribution the reranker
    sees at inference: the production retriever in no-rerank mode.

    This script materialises that data once so training runs are I/O cheap.
    See `reranker_finetune_plan.md` §2 for the data design rationale.

Two outputs:
    1. `train.jsonl` — one row per (claim, positive_ev_id) pair, with N
       hard negatives. Each row is a list-wise training instance for
       InfoNCE. ~2,070 rows from 986 claims × ~2.1 gold ev / claim.
    2. `eval.jsonl` — one row per eval claim with all gold ev ids + the
       full top-50 fused candidate list. Used by the training-time
       recall@k callback to score intermediate checkpoints. ~121 rows.

Hard-neg quality rules (strict, see plan §2.2):
    - Source: BM25+dense fused (no rerank, no rule_reorder), top-50 only —
      matches the inference distribution.
    - Exclude every gold ev id for that claim (no false negatives).
    - Per-candidate global cap: each ev id used as hard-neg at most
      `--neg-global-cap` times across the dataset (prevents generic
      filler like "climate change is real" from dominating training).
    - Claim filter: drop the claim if fewer than `--n-negs` valid hard
      negatives remain after filtering (should be < 1%).

Runtime:
    ~12 min on AutoDL 4080 SUPER for train+eval combined (BM25 ~30s,
    dense ~10 min for 986+121 queries at bs=64, hard-neg sampling <30s).

Usage::

    python -m scripts.build_reranker_ft_data \\
        --train outputs/splits/train_split.jsonl \\
        --eval  outputs/splits/diag_test.jsonl \\
        --out-dir outputs/reranker_ft_data \\
        --top-k 50 --n-negs 7 --neg-global-cap 5 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import load_evidence, read_jsonl  # noqa: E402
from src.paths import OUTPUTS_DIR, SPLITS_DIR  # noqa: E402
from src.retrieval.pipeline import RetrievalConfig, RetrievalPipeline  # noqa: E402


def _build_components():
    """Load BM25 + dense once. Reranker NOT loaded — we want raw fused output."""
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import DenseRetriever

    bm25_dir = OUTPUTS_DIR / "bm25_index"
    dense_dir = OUTPUTS_DIR / "dense_index"
    if not bm25_dir.exists():
        raise FileNotFoundError(
            f"BM25 index missing at {bm25_dir}. "
            f"Run: python -m scripts.build_indexes --skip-dense"
        )
    if not (dense_dir / "faiss.index").exists():
        raise FileNotFoundError(
            f"Dense index missing at {dense_dir}. Run: python -m scripts.build_indexes"
        )
    bm25 = BM25Retriever.load(bm25_dir)
    dense = DenseRetriever.load(dense_dir, max_seq_length=256, fp16=True)
    return bm25, dense


def _retrieve_top_k(pipeline: RetrievalPipeline, claim_text: str, k: int) -> list[tuple[str, float]]:
    """Run the pipeline up to top-k. Returns [(ev_id, fused_score), ...].

    `RetrievalPipeline.retrieve` only returns (ev_id, text) without the score;
    we want the score for inspection in the dump. So we replicate the
    fused-no-rerank stages inline. This stays in sync with pipeline.py
    because we construct the same config (no rerank, no rule_reorder).
    """
    from src.retrieval.fuse import weighted_fuse, rrf_fuse

    cfg = pipeline.cfg
    bm = pipeline.bm25.search(claim_text, k=cfg.bm25_top) if (cfg.use_bm25 and pipeline.bm25) else []
    de = pipeline.dense.search(claim_text, k=cfg.dense_top) if (cfg.use_dense and pipeline.dense) else []
    if bm and de:
        if cfg.fuse_strategy == "rrf":
            fused = rrf_fuse(bm, de, top_k=cfg.fuse_top)
        else:
            fused = weighted_fuse(bm, de, w_bm25=cfg.w_bm25, w_dense=cfg.w_dense, top_k=cfg.fuse_top)
    else:
        fused = bm or de
    return fused[:k]


def _sample_train_rows(
    claims: list[dict],
    evidence: dict[str, str],
    pipeline: RetrievalPipeline,
    *,
    top_k: int,
    n_negs: int,
    neg_global_cap: int,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    """For each (claim, positive_ev) pair: pick `n_negs` hard negs from top-k fused.

    Returns (rows, stats). `rows` is the list of training samples.
    """
    rows: list[dict] = []
    neg_usage: Counter = Counter()  # ev_id → times used as hard-neg
    dropped_no_pos_text = 0
    dropped_too_few_negs = 0
    n_claims = len(claims)

    t0 = time.time()
    for ci, c in enumerate(claims):
        gold_ev_ids = c.get("evidences") or []
        if not gold_ev_ids:
            continue

        # Retrieve once per claim (the candidate pool is the same for every gold ev).
        fused = _retrieve_top_k(pipeline, c["claim_text"], k=top_k)
        gold_set = set(gold_ev_ids)
        candidate_pool = [
            (eid, score) for eid, score in fused
            if eid not in gold_set and eid in evidence
        ]

        for pos_ev_id in gold_ev_ids:
            pos_text = evidence.get(pos_ev_id)
            if not pos_text:
                dropped_no_pos_text += 1
                continue

            # Filter pool by global cap, keep score order (higher = harder neg).
            filtered = [
                (eid, score) for eid, score in candidate_pool
                if neg_usage[eid] < neg_global_cap
            ]
            if len(filtered) < n_negs:
                # Fall back to ignoring cap rather than dropping the row.
                filtered = candidate_pool
            if len(filtered) < n_negs:
                dropped_too_few_negs += 1
                continue

            chosen = filtered[:n_negs]
            for eid, _ in chosen:
                neg_usage[eid] += 1

            rows.append({
                "claim_id": c["id"],
                "claim_text": c["claim_text"],
                "claim_label": c["claim_label"],
                "positive_ev_id": pos_ev_id,
                "positive_text": pos_text,
                "negatives": [
                    {"ev_id": eid, "text": evidence[eid], "fused_score": float(score)}
                    for eid, score in chosen
                ],
                "_meta": {
                    "all_gold_ev_ids": list(gold_ev_ids),
                    "n_pool": len(candidate_pool),
                    "domain": c.get("domain"),
                    "scenario": c.get("scenario"),
                    "difficulty": (c.get("difficulty") or {}).get("level"),
                },
            })

        if (ci + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (n_claims - ci - 1) / (ci + 1)
            print(f"    train {ci+1}/{n_claims}  rows={len(rows)}  ETA {eta:.0f}s")

    stats = {
        "n_input_claims": n_claims,
        "n_train_rows": len(rows),
        "dropped_no_pos_text": dropped_no_pos_text,
        "dropped_too_few_negs": dropped_too_few_negs,
        "unique_negs": len(neg_usage),
        "most_used_negs": neg_usage.most_common(10),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    return rows, stats


def _build_eval_rows(
    claims: list[dict],
    evidence: dict[str, str],
    pipeline: RetrievalPipeline,
    *,
    top_k: int,
) -> tuple[list[dict], dict]:
    """For each eval claim: keep the full top-k candidate pool + gold ev ids.

    The training-time callback in `finetune_reranker.py` reranks these
    candidates and computes recall@{5,10,20,50}.
    """
    rows: list[dict] = []
    n_no_gold = 0
    t0 = time.time()
    for c in claims:
        gold_ev_ids = c.get("evidences") or []
        if not gold_ev_ids:
            n_no_gold += 1
            continue
        fused = _retrieve_top_k(pipeline, c["claim_text"], k=top_k)
        rows.append({
            "claim_id": c["id"],
            "claim_text": c["claim_text"],
            "claim_label": c["claim_label"],
            "gold_ev_ids": list(gold_ev_ids),
            "candidates": [
                {"ev_id": eid, "text": evidence.get(eid, ""), "fused_score": float(score)}
                for eid, score in fused
            ],
        })
    stats = {
        "n_input_claims": len(claims),
        "n_eval_rows": len(rows),
        "n_no_gold": n_no_gold,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    return rows, stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, default=SPLITS_DIR / "train_split.jsonl",
                   help="Source claims for training (default: train_split.jsonl, 986 claims)")
    p.add_argument("--eval", type=Path, default=SPLITS_DIR / "diag_test.jsonl",
                   help="Source claims for training-time eval (default: diag_test.jsonl)")
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR / "reranker_ft_data")
    p.add_argument("--top-k", type=int, default=50,
                   help="Train-side pool depth from which to draw hard negs "
                       "(default 50, matches inference rerank_top).")
    p.add_argument("--eval-top-k", type=int, default=None,
                   help="Eval-side candidate pool depth. Defaults to --top-k. "
                       "Set higher (e.g. 100) to give the reranker access to "
                       "deeper candidates — raises the recall@k ceiling at "
                       "eval (baseline recall@100=0.579 vs recall@50=0.485). "
                       "Costs ~2× eval wall-time per checkpoint.")
    p.add_argument("--n-negs", type=int, default=7,
                   help="Hard negatives per training row (so list size = 1+N, default 7 → 8)")
    p.add_argument("--neg-global-cap", type=int, default=5,
                   help="Max times any ev_id can appear as hard-neg across the dataset")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    print("=" * 70)
    print("reranker fine-tune data build")
    print("=" * 70)
    print(f"  train source : {args.train}")
    print(f"  eval source  : {args.eval}")
    print(f"  out dir      : {args.out_dir}")
    print(f"  top-k pool   : {args.top_k}")
    print(f"  n hard-negs  : {args.n_negs}")
    print(f"  neg global cap: {args.neg_global_cap}")
    print(f"  seed         : {args.seed}")

    print("\n[1/4] loading evidence corpus...")
    evidence = load_evidence(show_progress=True)
    print(f"  {len(evidence):,} passages")

    print("\n[2/4] loading retrievers (no rerank — matches inference top-50 stage)...")
    bm25, dense = _build_components()
    cfg = RetrievalConfig(
        use_bm25=True, use_dense=True,
        use_rerank=False, use_rule_reorder=False,
        # Keep candidate pool deeper than top-k so re-fusion is stable.
        bm25_top=200, dense_top=200, fuse_top=max(150, args.top_k * 2),
        final_k=args.top_k, label_conditioned_k=False,
    )
    pipeline = RetrievalPipeline(
        evidence_corpus=evidence, bm25=bm25, dense=dense, reranker=None, cfg=cfg,
    )

    print(f"\n[3/4] building train rows ({args.train.name})...")
    train_claims = list(read_jsonl(args.train))
    train_rows, train_stats = _sample_train_rows(
        train_claims, evidence, pipeline,
        top_k=args.top_k, n_negs=args.n_negs,
        neg_global_cap=args.neg_global_cap, rng=rng,
    )

    eval_top_k = args.eval_top_k or args.top_k
    print(f"\n[4/4] building eval rows ({args.eval.name}, top-{eval_top_k})...")
    eval_claims = list(read_jsonl(args.eval))
    # Eval pipeline may use a different (deeper) pool. The pipeline's
    # `fuse_top` already accommodates `max(150, args.top_k * 2)`, but if
    # eval_top_k > that we widen `final_k`/`fuse_top` on the fly.
    if eval_top_k > args.top_k:
        from dataclasses import replace as _replace
        deep_cfg = _replace(cfg, final_k=eval_top_k,
                            fuse_top=max(cfg.fuse_top, eval_top_k * 2))
        eval_pipeline = RetrievalPipeline(
            evidence_corpus=evidence, bm25=bm25, dense=dense, reranker=None, cfg=deep_cfg,
        )
    else:
        eval_pipeline = pipeline
    eval_rows, eval_stats = _build_eval_rows(
        eval_claims, evidence, eval_pipeline, top_k=eval_top_k,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    eval_path = args.out_dir / "eval.jsonl"
    meta_path = args.out_dir / "meta.json"

    with train_path.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with eval_path.open("w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train": train_stats,
        "eval": eval_stats,
        "retrieval_config": {
            "bm25_top": cfg.bm25_top, "dense_top": cfg.dense_top,
            "fuse_top": cfg.fuse_top, "fuse_strategy": cfg.fuse_strategy,
            "w_bm25": cfg.w_bm25, "w_dense": cfg.w_dense,
            "use_rerank": cfg.use_rerank, "use_rule_reorder": cfg.use_rule_reorder,
            "top_k": args.top_k,
            "eval_top_k": eval_top_k,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print(f"  train rows : {len(train_rows):,}  →  {train_path}")
    print(f"  eval rows  : {len(eval_rows):,}  →  {eval_path}")
    print(f"  meta       : {meta_path}")
    print(f"  unique hard-negs: {train_stats['unique_negs']:,}")
    print(f"  dropped (no pos text)   : {train_stats['dropped_no_pos_text']}")
    print(f"  dropped (too few negs)  : {train_stats['dropped_too_few_negs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
