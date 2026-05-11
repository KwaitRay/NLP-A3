"""Build SFT training jsonl in ms-swift format.

ms-swift standard schema (the messages-list form; confirmed in
materials/训练数据格式.docx §2.1 — this is the unambiguous official format,
recommended over query-response which goes through AutoPreprocessor mapping):

    {"messages": [
        {"role": "system",    "content": "<SYSTEM_PROMPT>"},
        {"role": "user",      "content": "<query>"},
        {"role": "assistant", "content": "<chosen response>"},
    ]}

We additionally carry ``id`` and ``_meta`` fields per record for downstream
processing (curriculum sort, DPO pair construction, ablation slicing).
ms-swift ignores unknown top-level keys, so leaving ``id``/``_meta`` in the
training file is harmless — `data_io.write_jsonl` does NOT strip them.

Two construction modes:
  - ``gold_only`` (default): use the claim's gold evidences directly. Trains
    the model to attend to ideal evidence, but at inference time the retrieval
    might surface different (and noisier) candidates.
  - ``retrieval``: use top-k retrieved evidences (a mix of correct and
    distractors). Better matches inference distribution, recommended once the
    retriever is in place. Falls back to gold when retrieval missing.

Hard-negative augmentation (`hard_negatives` param) appends extra rows that
re-use the same claim but pair it with non-gold evidences and relabel as
NOT_ENOUGH_INFO. Disabled by default (set ``n_hard_neg > 0`` to enable).
"""
from __future__ import annotations

import random
from typing import Callable

from .prompt import SYSTEM_PROMPT, build_target_response, build_user_query

# Cache the evidence-corpus key list keyed by object identity so we don't
# rebuild a 1.2M-item list per claim during random sampling. Using id(...)
# keeps identity-based caching cheap without weak references.
_ALL_IDS_CACHE: dict[int, list[str]] = {}


def _all_ids(evidence_corpus: dict[str, str]) -> list[str]:
    cid = id(evidence_corpus)
    out = _ALL_IDS_CACHE.get(cid)
    if out is None:
        out = list(evidence_corpus.keys())
        _ALL_IDS_CACHE[cid] = out
    return out

# Type alias for the retrieval callback. Returns ordered list of
# (evidence_id, evidence_text). When None, gold-only mode is used.
RetrievalFn = Callable[[str, int], list[tuple[str, str]]]


def _truncate_evidence(text: str, max_chars: int = 800) -> str:
    """Cap individual evidence text length to keep total prompt under 1024 toks.

    Climate corpus passages are mostly short, but a few are long Wikipedia
    paragraphs. Truncating at the character level is fine since we don't need
    word boundaries — the model still sees enough signal."""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _select_evidences(
    claim_id: str,
    claim_label: str,
    gold_ev_ids: list[str],
    evidence_corpus: dict[str, str],
    *,
    retrieval: RetrievalFn | None,
    k: int,
    pad_with_random: bool,
    rng: random.Random,
) -> list[tuple[str, str]]:
    if retrieval is not None:
        retrieved = retrieval(claim_id, k)
        # Always make sure gold is present in the prompt (oracle-augmented),
        # otherwise the SFT signal degenerates. Drop dups, preserve order.
        seen: set[str] = set()
        merged: list[tuple[str, str]] = []
        for ev_id, ev_text in retrieved:
            if ev_id in seen:
                continue
            seen.add(ev_id)
            merged.append((ev_id, ev_text))
        for g in gold_ev_ids:
            if g not in seen and g in evidence_corpus:
                merged.insert(0, (g, evidence_corpus[g]))
                seen.add(g)
        return merged[:k]

    # gold_only mode
    selected = [(g, evidence_corpus[g]) for g in gold_ev_ids if g in evidence_corpus]
    if pad_with_random and len(selected) < k:
        # Sample by index from the cached id list to avoid rebuilding a 1M-item
        # pool per claim. Reject samples that collide with already-selected ids.
        all_ids = _all_ids(evidence_corpus)
        chosen_ids = {g for g, _ in selected}
        need = k - len(selected)
        attempts = 0
        while need > 0 and attempts < need * 10:
            ev_id = all_ids[rng.randrange(len(all_ids))]
            if ev_id not in chosen_ids:
                selected.append((ev_id, evidence_corpus[ev_id]))
                chosen_ids.add(ev_id)
                need -= 1
            attempts += 1
    return selected[:k]


def build_sft_record(
    tagged: dict,
    evidence_corpus: dict[str, str],
    *,
    retrieval: RetrievalFn | None = None,
    k: int = 5,
    pad_with_random: bool = False,
    rng: random.Random | None = None,
    max_evidence_chars: int = 800,
) -> dict | None:
    """Build a single ms-swift SFT record from a tagged claim row.

    Returns ``None`` when no usable evidences could be assembled (rare but
    possible if gold IDs aren't in the corpus and pad_with_random=False).
    """
    rng = rng or random.Random(42)
    claim_id = tagged["id"]
    claim_text = tagged["claim_text"]
    claim_label = tagged["claim_label"]
    gold = list(tagged.get("evidences") or [])

    chosen = _select_evidences(
        claim_id, claim_label, gold, evidence_corpus,
        retrieval=retrieval, k=k, pad_with_random=pad_with_random, rng=rng,
    )
    if not chosen:
        return None

    chosen = [(eid, _truncate_evidence(t, max_evidence_chars)) for eid, t in chosen]
    shown_ids = [eid for eid, _ in chosen]
    query = build_user_query(claim_text, chosen)
    response = build_target_response(claim_label, gold, shown_ids)

    return {
        "id": claim_id,
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": query},
            {"role": "assistant", "content": response},
        ],
        "_meta": {
            "domain": tagged.get("domain"),
            "scenario": tagged.get("scenario"),
            "difficulty": tagged.get("difficulty", {}).get("level"),
            "n_gold": len(gold),
            "n_shown": len(shown_ids),
            "shown": shown_ids,
        },
    }


def build_hard_negative_record(
    tagged: dict,
    evidence_corpus: dict[str, str],
    *,
    k: int,
    rng: random.Random,
    max_evidence_chars: int = 800,
) -> dict | None:
    """Re-pair the same claim with k random non-gold evidences and relabel NEI.

    Trains the model to recognise that off-topic / unrelated evidence yields
    NOT_ENOUGH_INFO regardless of the original gold label.
    """
    claim_id = tagged["id"]
    claim_text = tagged["claim_text"]
    gold = set(tagged.get("evidences") or [])
    all_ids = _all_ids(evidence_corpus)
    if len(all_ids) - len(gold) < k:
        return None
    sample: list[str] = []
    seen: set[str] = set(gold)
    attempts = 0
    while len(sample) < k and attempts < k * 20:
        ev_id = all_ids[rng.randrange(len(all_ids))]
        if ev_id not in seen:
            sample.append(ev_id)
            seen.add(ev_id)
        attempts += 1
    if len(sample) < k:
        return None
    chosen = [(eid, _truncate_evidence(evidence_corpus[eid], max_evidence_chars)) for eid in sample]
    shown_ids = [eid for eid, _ in chosen]
    query = build_user_query(claim_text, chosen)
    # No gold IDs are present → fallback citation = all shown.
    response = build_target_response("NOT_ENOUGH_INFO", [], shown_ids)
    return {
        "id": f"{claim_id}__hn",
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": query},
            {"role": "assistant", "content": response},
        ],
        "_meta": {
            "domain": tagged.get("domain"),
            "scenario": "nei_topic_off",  # synthetic: by construction off-topic
            "difficulty": "medium",
            "n_gold": 0,
            "n_shown": len(shown_ids),
            "shown": shown_ids,
            "augmented": "hard_negative",
        },
    }


def curriculum_sort_key(rec: dict) -> tuple[int, int]:
    """Sort records easy → medium → hard within an epoch (Plan §0.5)."""
    rank = {"easy": 0, "medium": 1, "hard": 2}.get(rec.get("_meta", {}).get("difficulty"), 1)
    # Stable secondary key: hash of id, but keep curriculum primary.
    return (rank, hash(rec.get("id", "")))


def _bucket_value(row: dict, axis: str):
    """Read ``row[axis]`` accounting for the nested ``difficulty.level`` schema."""
    if axis == "difficulty":
        d = row.get("difficulty")
        return d.get("level") if isinstance(d, dict) else d
    return row.get(axis)


def _bucket_factor(row: dict, weak_buckets: dict[tuple[str, str], int] | None) -> int:
    """Return the max oversample factor for this row across matching keys.

    Default 1 (no duplication). When ``weak_buckets`` is e.g.
    ``{("scenario", "nei_underspec"): 4, ("difficulty", "hard"): 2}``, a row
    whose scenario is "nei_underspec" *and* difficulty is "hard" still gets
    factor 4 — we take the max, not the product, to avoid runaway when a row
    matches several weak axes.
    """
    if not weak_buckets:
        return 1
    factor = 1
    for (axis, bucket), f in weak_buckets.items():
        if _bucket_value(row, axis) == bucket:
            if f > factor:
                factor = f
    return factor


def build_dataset(
    tagged_rows: list[dict],
    evidence_corpus: dict[str, str],
    *,
    retrieval: RetrievalFn | None = None,
    k: int = 5,
    pad_with_random: bool = False,
    n_hard_neg: int = 0,
    seed: int = 42,
    apply_curriculum: bool = True,
    weak_buckets: dict[tuple[str, str], int] | None = None,
) -> list[dict]:
    """Top-level builder. ``tagged_rows`` should be from claims_tagged.jsonl.

    ``weak_buckets``: Phase 4 weighted oversampling per
    ``optimization_plan.md §4``. Pass ``{(axis, bucket_name): factor}`` to
    duplicate each matching row's records (both real and hard-negative) by
    ``factor``. Multiple matches → max factor, not product. Random padding
    (`pad_with_random=True`) and hard-neg sampling consume the shared
    ``rng`` across duplicates, so duplicate copies do see different
    distractor / random evidence — they aren't bit-identical clones.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for row in tagged_rows:
        factor = _bucket_factor(row, weak_buckets)
        for _ in range(factor):
            rec = build_sft_record(
                row, evidence_corpus,
                retrieval=retrieval, k=k,
                pad_with_random=pad_with_random, rng=rng,
            )
            if rec is not None:
                out.append(rec)
        for _ in range(n_hard_neg * factor):
            hn = build_hard_negative_record(
                row, evidence_corpus, k=k, rng=rng,
            )
            if hn is not None:
                out.append(hn)
    if apply_curriculum:
        out.sort(key=curriculum_sort_key)
    return out
