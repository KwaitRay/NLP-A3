"""Cross-encoder reranking + rule-based reorder (Plan §1.3 + §1.4).

The reranker scores (claim, candidate) pairs and replaces the fused score.
Default model: ``BAAI/bge-reranker-base`` (278M, fits Colab T4).

The rule-based step then:
- boosts passages with NER-entity overlap with the claim (from spaCy)
- suppresses near-duplicates (cos > 0.95 within top-k of the reranked list)
- applies a diversity cap so no source/topic dominates

Heavy deps imported lazily.
"""
from __future__ import annotations

from typing import Sequence


DEFAULT_RERANKER = "BAAI/bge-reranker-base"
HEAVY_RERANKER = "BAAI/bge-reranker-v2-m3"  # 568M, better but slower


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str = "cuda") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            from ..paths import resolve_model_path
            # Prefer pre-downloaded local copy under models/<basename>/.
            path = resolve_model_path(self.model_name)
            self._model = CrossEncoder(path, device=self.device, max_length=512)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        *,
        batch_size: int = 32,
    ) -> list[tuple[str, float]]:
        """Score every (query, candidate_text) pair, return ev_ids sorted desc."""
        if not candidates:
            return []
        model = self._load()
        pairs = [[query, txt] for _, txt in candidates]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        out = list(zip([eid for eid, _ in candidates], (float(s) for s in scores)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out


# -- Rule-based reorder ------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def rule_reorder(
    ranked: list[tuple[str, float]],
    *,
    evidence_corpus: dict[str, str],
    claim_entities: Sequence[str] | None = None,
    entity_boost: float = 0.05,
    dedup_jaccard_threshold: float = 0.85,
    keep_top_k: int = 20,
) -> list[tuple[str, float]]:
    """Apply NER-overlap boost + near-duplicate suppression on top-k.

    - Entity boost: if a candidate text contains any of ``claim_entities``,
      its score is bumped by ``entity_boost`` per entity (capped at 5×).
    - Dedup: walking the ranked list top-down, drop a candidate if its token-
      Jaccard with any kept candidate exceeds ``dedup_jaccard_threshold``.

    Operates on ``keep_top_k`` so it doesn't slow down the long tail.
    """
    if not ranked:
        return ranked
    head = ranked[:keep_top_k]
    tail = ranked[keep_top_k:]

    # Apply entity boost.
    boosted: list[tuple[str, float]] = []
    if claim_entities:
        ent_lower = [e.lower() for e in claim_entities if e]
        for ev_id, sc in head:
            txt = evidence_corpus.get(ev_id, "").lower()
            hits = sum(1 for e in ent_lower if e and e in txt)
            boosted.append((ev_id, sc + min(5, hits) * entity_boost))
        boosted.sort(key=lambda x: x[1], reverse=True)
    else:
        boosted = list(head)

    # Greedy dedup.
    kept: list[tuple[str, float]] = []
    kept_token_sets: list[set] = []
    for ev_id, sc in boosted:
        toks = set(evidence_corpus.get(ev_id, "").lower().split())
        if any(_jaccard(toks, prev) >= dedup_jaccard_threshold for prev in kept_token_sets):
            continue
        kept.append((ev_id, sc))
        kept_token_sets.append(toks)

    return kept + tail
