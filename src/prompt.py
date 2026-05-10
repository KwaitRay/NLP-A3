"""SFT prompt template + response parser.

The format is identical between training (SFT/DPO targets) and inference,
so the same builder is reused. Output schema, with example::

    SUPPORTS ##[1,3]##

Parser is robust to whitespace, case (label normalised to upper), and
missing citation block (falls back to all evidence indices).
"""
from __future__ import annotations

import re
from typing import Sequence

from .paths import LABELS

SYSTEM_PROMPT = (
    "You are a climate fact-checking expert. Given a claim and several numbered "
    "evidence passages, decide whether the claim is SUPPORTED, REFUTED, has "
    "NOT_ENOUGH_INFO, or is DISPUTED, based on the evidence."
)

NO_RAG_SYSTEM_PROMPT = (
    "You are a climate fact-checking expert. Given a claim, decide whether it is "
    "SUPPORTED, REFUTED, has NOT_ENOUGH_INFO, or is DISPUTED, based on your own "
    "knowledge."
)

_NO_RAG_OUTPUT_RULES = (
    "Output rules:\n"
    "1. Output exactly one label as the only token: SUPPORTS / REFUTES / "
    "NOT_ENOUGH_INFO / DISPUTED.\n"
    "2. Do not output anything else."
)


def build_no_rag_query(claim_text: str) -> str:
    """Track 1 prompt — claim only, no evidence."""
    return f"{_NO_RAG_OUTPUT_RULES}\n\nClaim: {claim_text}\n\nAnswer:"

_OUTPUT_RULES = (
    "Output rules:\n"
    "1. Output exactly one label as the first token: SUPPORTS / REFUTES / "
    "NOT_ENOUGH_INFO / DISPUTED.\n"
    "2. After the label, list the evidence numbers you relied on, in the form "
    "##[1,3]##.\n"
    "3. Do not output anything else."
)


def build_user_query(claim_text: str, evidences: Sequence[tuple[str, str]]) -> str:
    """Compose the user-facing query string.

    ``evidences`` is a sequence of (evidence_id, evidence_text). Numbering is
    1-based and stable across the same call (so the response can refer to
    [1], [2] etc. unambiguously).
    """
    lines: list[str] = [_OUTPUT_RULES, "", f"Claim: {claim_text}", "", "Evidence:"]
    for i, (_, text) in enumerate(evidences, start=1):
        text_clean = text.replace("\n", " ").strip()
        lines.append(f"[{i}] {text_clean}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def build_target_response(claim_label: str, gold_evidence_ids: Sequence[str],
                          shown_evidence_ids: Sequence[str]) -> str:
    """Produce the gold response string for SFT.

    ``shown_evidence_ids`` is the ordered list of evidence IDs as numbered in
    the prompt; the cited indices in the response are computed against this
    list, retaining only those gold IDs that actually appear (others are
    silently dropped — the retriever didn't surface them).
    """
    if claim_label not in LABELS:
        raise ValueError(f"unknown claim_label: {claim_label!r}")
    shown_idx = {ev_id: i for i, ev_id in enumerate(shown_evidence_ids, start=1)}
    cited = sorted({shown_idx[g] for g in gold_evidence_ids if g in shown_idx})
    if not cited:
        cited = list(range(1, len(shown_evidence_ids) + 1))
    return f"{claim_label} ##[{','.join(str(c) for c in cited)}]##"


_LABEL_RE = re.compile(
    r"\b(SUPPORTS|REFUTES|NOT[_\s]?ENOUGH[_\s]?INFO|DISPUTED)\b",
    re.IGNORECASE,
)
_CITE_RE = re.compile(r"##\s*\[\s*([\d,\s]+?)\s*\]\s*##")


def parse_response(
    text: str, shown_evidence_ids: Sequence[str], default_label: str = "NOT_ENOUGH_INFO",
) -> tuple[str, list[str]]:
    """Parse a generated response into (label, evidence_id_list).

    - Label: first match of the four canonical strings, case-insensitive.
      ``NOT ENOUGH INFO`` and ``NOT_ENOUGH_INFO`` are both accepted.
      Falls back to ``default_label`` if no label is found.
    - Citation indices outside ``[1, len(shown)]`` are dropped silently. If no
      valid index survives, returns all shown evidence IDs (so the prediction
      JSON always carries at least one — eval.py rejects empty lists).
    """
    label = default_label
    m = _LABEL_RE.search(text)
    if m:
        norm = m.group(1).upper().replace(" ", "_")
        if norm == "NOTENOUGHINFO":  # malformed but recoverable
            norm = "NOT_ENOUGH_INFO"
        if norm in LABELS:
            label = norm

    cited: list[str] = []
    for cm in _CITE_RE.finditer(text):
        for tok in cm.group(1).split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                idx = int(tok)
            except ValueError:
                continue
            if 1 <= idx <= len(shown_evidence_ids):
                ev = shown_evidence_ids[idx - 1]
                if ev not in cited:
                    cited.append(ev)
    if not cited:
        cited = list(shown_evidence_ids)
    return label, cited
