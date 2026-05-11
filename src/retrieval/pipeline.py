"""End-to-end retrieval orchestration: BM25 + dense → fuse → rerank → rule.

This is the function that the SFT dataset builder, the inference loop, and
the ablation harness all call. By isolating it here, we can swap individual
stages on/off via the ``cfg`` flags without touching downstream code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .fuse import rrf_fuse, weighted_fuse
from .rerank import rule_reorder


@dataclass
class RetrievalConfig:
    use_bm25: bool = True
    use_dense: bool = True
    fuse_strategy: str = "weighted"  # "weighted" | "rrf"
    w_bm25: float = 0.3
    w_dense: float = 0.7
    bm25_top: int = 200
    dense_top: int = 200
    fuse_top: int = 150
    use_rerank: bool = True
    rerank_top: int = 50
    use_rule_reorder: bool = True
    rule_top: int = 20
    # Phase 3.5 lock (2026-05-12): final_k=20 chosen over 5 because the audit
    # (`scripts.retrieval_ceiling --mode final_k`) showed macro recall@20=0.333
    # vs recall@5=0.119 — gold evidence sits at ranks 6-20 for most claims.
    # End-to-end Track 2 v1 HM lifted from 0.183→0.203. See
    # optimization_plan.md §10 decision log + outputs/eval_phase1/
    # retrieval_ceiling_diag_test.md for the curve.
    final_k: int = 20
    label_conditioned_k: bool = True
    # NEI-class evidence count is always 5; non-NEI median 2-3.
    nei_k: int = 5
    nei_predictor: Optional[callable] = None  # claim_text → bool (is NEI?)
    claim_entities_lookup: Optional[callable] = field(default=None, repr=False)


class RetrievalPipeline:
    def __init__(
        self,
        *,
        evidence_corpus: dict[str, str],
        bm25=None,
        dense=None,
        reranker=None,
        cfg: RetrievalConfig | None = None,
    ) -> None:
        self.evidence = evidence_corpus
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.cfg = cfg or RetrievalConfig()

    def retrieve(self, claim_text: str) -> list[tuple[str, str]]:
        cfg = self.cfg

        # Stage 1: candidate generation.
        bm = self.bm25.search(claim_text, k=cfg.bm25_top) if (cfg.use_bm25 and self.bm25) else []
        de = self.dense.search(claim_text, k=cfg.dense_top) if (cfg.use_dense and self.dense) else []

        # Stage 2: fusion.
        if bm and de:
            if cfg.fuse_strategy == "rrf":
                fused = rrf_fuse(bm, de, top_k=cfg.fuse_top)
            else:
                fused = weighted_fuse(bm, de, w_bm25=cfg.w_bm25, w_dense=cfg.w_dense, top_k=cfg.fuse_top)
        else:
            fused = bm or de

        # Stage 3: rerank.
        if cfg.use_rerank and self.reranker:
            cands = [(eid, self.evidence.get(eid, "")) for eid, _ in fused[: cfg.rerank_top]]
            reranked = self.reranker.rerank(claim_text, cands)
            # Stitch tail back unchanged so we keep depth for downstream rule step.
            tail = fused[cfg.rerank_top:]
            ranked = reranked + tail
        else:
            ranked = fused

        # Stage 4: rule-based reorder + dedup.
        if cfg.use_rule_reorder:
            entities = []
            if cfg.claim_entities_lookup is not None:
                entities = list(cfg.claim_entities_lookup(claim_text) or [])
            ranked = rule_reorder(
                ranked,
                evidence_corpus=self.evidence,
                claim_entities=entities,
                keep_top_k=cfg.rule_top,
            )

        # Stage 5: top-k selection (label-conditioned if a predictor is wired).
        k = cfg.final_k
        if cfg.label_conditioned_k and cfg.nei_predictor is not None:
            try:
                if cfg.nei_predictor(claim_text):
                    k = cfg.nei_k
            except Exception:
                pass
        chosen = ranked[:k]
        return [(eid, self.evidence.get(eid, "")) for eid, _ in chosen]
