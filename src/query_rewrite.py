"""Stage 2 — query rewriting / expansion (Plan §1.5 + §2).

Three transformations, each callable in isolation so the pipeline can mix
and match. None of them require a GPU: synonym expansion uses NLTK WordNet,
the LLM-driven ones build prompt strings only — generation happens elsewhere.

Returned form for ``decompose_subclaims`` and ``hyde_prompt`` is just the
*prompt string*. The notebook wires these to a Qwen call (zero-shot before
SFT, SFT'd afterwards) — see notebook section 2.7.
"""
from __future__ import annotations

import re
from typing import Iterable

# WordNet imports are lazy: NLTK is in our requirements but downloading the
# corpus has side effects, so we trigger it on first call.
_WORDNET_READY = False


def _ensure_wordnet() -> None:
    global _WORDNET_READY
    if _WORDNET_READY:
        return
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    _WORDNET_READY = True


def synonym_expand(claim_text: str, *, max_per_token: int = 2) -> list[str]:
    """Generate alternate phrasings by swapping in WordNet synonyms.

    Conservative by default: ``max_per_token=2`` and only single-word
    replacements at noun/verb POS, so the output stays readable. Returns the
    original claim plus a small set of variants (deduped).

    Useful when wired into multi-query retrieval: encode each variant
    separately, union the candidates, then rerank.
    """
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn

    tokens = re.findall(r"[A-Za-z][A-Za-z'-]+|\d+|[^\w\s]+", claim_text)
    out: list[str] = [claim_text]
    seen: set[str] = {claim_text}

    for i, tok in enumerate(tokens):
        if not tok.isalpha() or len(tok) <= 3:
            continue
        candidates: list[str] = []
        for syn in wn.synsets(tok.lower()):
            if syn.pos() not in {"n", "v"}:
                continue
            for lem in syn.lemmas():
                w = lem.name().replace("_", " ")
                if w.lower() == tok.lower() or " " in w:
                    continue
                if w not in candidates:
                    candidates.append(w)
                if len(candidates) >= max_per_token:
                    break
            if len(candidates) >= max_per_token:
                break
        for cand in candidates:
            new_tokens = tokens[:i] + [cand] + tokens[i + 1:]
            variant = "".join(
                (" " + t if (t.isalnum() and j > 0 and tokens[j - 1].isalnum()) else t)
                for j, t in enumerate(new_tokens)
            )
            if variant not in seen:
                seen.add(variant)
                out.append(variant)
    return out


# -- LLM-driven prompts ------------------------------------------------------

_SUBCLAIM_PROMPT = """\
You are a climate fact-checking analyst. Decompose the following claim into
atomic sub-claims, one per line. Each sub-claim should make a single factual
assertion that can be verified independently. Output 1 to 3 sub-claims.

Output format (no other text):
1. <sub-claim>
2. <sub-claim>
...

Claim: {claim}
Decomposition:"""


def decompose_subclaims_prompt(claim: str) -> str:
    """Build the prompt that asks the LLM to break a compound claim into atoms.

    The retrieval pipeline encodes each sub-claim separately, unions the
    candidates, and reranks. This boosts recall on multi-hop / conjunctive
    claims like "X causes Y and Z increased due to W"."""
    return _SUBCLAIM_PROMPT.format(claim=claim.strip())


_SUBCLAIM_LINE_RE = re.compile(r"^\s*(?:\d+\.|-|\*)\s*(.+?)\s*$", re.MULTILINE)


def parse_subclaims(text: str, *, fallback: str | None = None) -> list[str]:
    """Extract enumerated sub-claims from the LLM response.

    Resilient: lines without numbering are kept if they look like full
    sentences. Returns ``[fallback]`` if nothing parses (so the caller can
    always proceed)."""
    matches = [m.group(1).strip() for m in _SUBCLAIM_LINE_RE.finditer(text)]
    matches = [m for m in matches if 5 <= len(m) <= 200]
    if matches:
        return matches[:3]
    if fallback is not None:
        return [fallback]
    return []


_HYDE_PROMPT = """\
You are a climate-science encyclopaedia. Write a short, factual passage
(1-2 sentences) that, if true, would directly support or refute the
following claim. The passage should resemble Wikipedia: declarative,
neutral, with concrete numbers or named entities where possible.

Claim: {claim}
Passage:"""


def hyde_prompt(claim: str) -> str:
    """Build a HyDE prompt that asks the LLM to invent a hypothetical evidence
    sentence. The hypothesis is then embedded by the dense retriever and
    blended with the claim embedding (Plan §1.5).

    HyDE shines when the claim's surface form differs lexically from the
    likely evidence (e.g. claim is journalese but evidence is academic)."""
    return _HYDE_PROMPT.format(claim=claim.strip())


def blend_query_text(
    claim: str, hypothesis: str, *, alpha: float = 0.6
) -> str:
    """Cheap text-level blend used when we don't yet have a dense retriever:
    repeat the claim ``alpha``-weighted to bias BM25 toward the actual claim
    while still allowing the hypothesis terms to broaden recall.

    Once the dense retriever is in place, prefer ``blend_embeddings`` (does
    a vector-level convex combination)."""
    n_claim_repeats = max(1, int(round(alpha * 4)))
    n_hyp_repeats = max(1, int(round((1 - alpha) * 4)))
    return " ".join([claim] * n_claim_repeats + [hypothesis] * n_hyp_repeats)


def blend_embeddings(
    claim_emb, hyp_emb, *, alpha: float = 0.6
):
    """Convex combination of claim and hypothesis embeddings (numpy arrays)."""
    return alpha * claim_emb + (1.0 - alpha) * hyp_emb


# -- Multi-query retrieval interface ----------------------------------------

def expand_query(
    claim: str,
    *,
    use_synonym: bool = True,
    use_subclaim: Iterable[str] | None = None,
    use_hyde: str | None = None,
) -> list[str]:
    """One-stop query expansion that returns the list of strings the
    retriever should encode separately. Always includes the original claim.

    - ``use_synonym``: include synonym-substituted variants.
    - ``use_subclaim``: pre-decomposed sub-claims from a prior LLM call.
    - ``use_hyde``: a hypothetical passage from a prior LLM call.

    The pipeline can run all three in parallel and union candidates.
    """
    queries: list[str] = [claim]
    if use_synonym:
        queries.extend(q for q in synonym_expand(claim) if q != claim)
    if use_subclaim:
        queries.extend(s for s in use_subclaim if s and s != claim)
    if use_hyde:
        queries.append(blend_query_text(claim, use_hyde))
    # Dedupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out
