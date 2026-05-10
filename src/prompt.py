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


_OUTPUT_RULES = (
    "Output rules:\n"
    "1. Output exactly one label as the first token: SUPPORTS / REFUTES / "
    "NOT_ENOUGH_INFO / DISPUTED.\n"
    "2. After the label, list the evidence numbers you relied on, in the form "
    "##[1,3]##.\n"
    "3. Do not output anything else."
)


# -- Prompt variants registry (Phase 2, D-015) ------------------------------
#
# Each variant overrides one or more of {system, no_rag_system, rules,
# no_rag_rules, few_shot_examples}. v1 is the production baseline, v2-v4 are
# Phase 2 candidates for the diagnostic-driven prompt iteration.
#
# Few-shot examples are injected between the rules and "Claim:" line. They
# are intentionally short and label-balanced (one example per behaviour the
# variant is trying to induce).

_NEI_EXPLICIT_RULES = (
    "Output rules:\n"
    "1. Output exactly one label as the first token: SUPPORTS / REFUTES / "
    "NOT_ENOUGH_INFO / DISPUTED.\n"
    "2. **Use NOT_ENOUGH_INFO when the evidence is not directly about the claim, "
    "or does not provide enough information to either support or refute it.** "
    "Do not guess REFUTES just because the claim sounds wrong.\n"
    "3. After the label, list the evidence numbers you relied on, in the form "
    "##[1,3]##.\n"
    "4. Do not output anything else."
)

_DISPUTED_EXPLICIT_RULES = (
    "Output rules:\n"
    "1. Output exactly one label as the first token: SUPPORTS / REFUTES / "
    "NOT_ENOUGH_INFO / DISPUTED.\n"
    "2. **Use DISPUTED when different evidence pieces contradict each other "
    "on the same claim** (e.g. one supports while another refutes). Do NOT "
    "default to SUPPORTS or REFUTES when evidence is split.\n"
    "3. **Use NOT_ENOUGH_INFO when the evidence is not directly about the claim.** "
    "Do not guess REFUTES just because the claim sounds wrong.\n"
    "4. After the label, list the evidence numbers you relied on, in the form "
    "##[1,3]##.\n"
    "5. Do not output anything else."
)

# Few-shot demonstrations (one per label). Kept short (~25 tokens of evidence
# each) so the budget overhead is < 200 tokens. Designed to teach behaviour:
# - SUPPORTS: clear single-evidence agreement
# - REFUTES: clear contradiction
# - NEI: evidence off-topic (this is the failure mode v2 targets)
# - DISPUTED: two evidences disagree (this is the failure mode v3 targets)
_FEW_SHOT_BLOCK = (
    "Examples:\n"
    "---\n"
    "Claim: Sea level has risen since 1900.\n"
    "Evidence:\n"
    "[1] Tide gauge records show global mean sea level has risen ~20 cm since 1900.\n"
    "Answer: SUPPORTS ##[1]##\n"
    "---\n"
    "Claim: There has been no warming since 1998.\n"
    "Evidence:\n"
    "[1] Each decade since 1980 has been warmer than the previous one, with the 2010s the warmest on record.\n"
    "Answer: REFUTES ##[1]##\n"
    "---\n"
    "Claim: Vanilla ice cream causes glacier melt.\n"
    "Evidence:\n"
    "[1] She made guest appearances at the Edinburgh Festival in 1957.\n"
    "[2] Antarctic ice shelves are losing mass annually.\n"
    "Answer: NOT_ENOUGH_INFO ##[1,2]##\n"
    "---\n"
    "Claim: Climate sensitivity is exactly 3 °C per CO2 doubling.\n"
    "Evidence:\n"
    "[1] CMIP6 models converge on a central estimate near 3 °C with confidence interval 2-5 °C.\n"
    "[2] Recent observational reanalyses suggest the true value may be closer to 2 °C.\n"
    "Answer: DISPUTED ##[1,2]##\n"
    "---\n"
)


PROMPT_VARIANTS: dict[str, dict[str, str]] = {
    "v1": {
        "name": "baseline",
        "description": "Original production prompt; minimal output rules, no examples.",
        "system": SYSTEM_PROMPT,
        "no_rag_system": NO_RAG_SYSTEM_PROMPT,
        "rules": _OUTPUT_RULES,
        "no_rag_rules": _NO_RAG_OUTPUT_RULES,
        "few_shot": "",
    },
    "v2": {
        "name": "nei_explicit",
        "description": "v1 + explicit NEI trigger condition (don't guess REFUTES on weird claims).",
        "system": SYSTEM_PROMPT,
        "no_rag_system": NO_RAG_SYSTEM_PROMPT,
        "rules": _NEI_EXPLICIT_RULES,
        "no_rag_rules": _NO_RAG_OUTPUT_RULES,  # No-RAG can't see evidence, NEI rule N/A
        "few_shot": "",
    },
    "v3": {
        "name": "disputed_explicit",
        "description": "v2 + explicit DISPUTED trigger condition (recognize contradicting evidence).",
        "system": SYSTEM_PROMPT,
        "no_rag_system": NO_RAG_SYSTEM_PROMPT,
        "rules": _DISPUTED_EXPLICIT_RULES,
        "no_rag_rules": _NO_RAG_OUTPUT_RULES,
        "few_shot": "",
    },
    "v4": {
        "name": "few_shot",
        "description": "v3 + 4 few-shot examples (one per label class).",
        "system": SYSTEM_PROMPT,
        "no_rag_system": NO_RAG_SYSTEM_PROMPT,
        "rules": _DISPUTED_EXPLICIT_RULES,
        "no_rag_rules": _NO_RAG_OUTPUT_RULES,
        "few_shot": _FEW_SHOT_BLOCK,
    },
    # v5 (chain-of-thought) deferred — needs parser update to scan last
    # label match instead of first, otherwise CoT prose like "we should
    # check if it SUPPORTS..." would short-circuit parse_response.
}


def get_variant_system(version: str = "v1", *, no_rag: bool = False) -> str:
    """Return the system prompt for a given variant."""
    v = PROMPT_VARIANTS[version]
    return v["no_rag_system"] if no_rag else v["system"]


def build_no_rag_query(claim_text: str, *, version: str = "v1") -> str:
    """Track 1 prompt — claim only, no evidence.

    ``version`` selects a variant from ``PROMPT_VARIANTS``; default ``"v1"``
    matches the historical production behaviour.
    """
    rules = PROMPT_VARIANTS[version]["no_rag_rules"]
    return f"{rules}\n\nClaim: {claim_text}\n\nAnswer:"


def build_user_query(
    claim_text: str,
    evidences: Sequence[tuple[str, str]],
    *,
    version: str = "v1",
) -> str:
    """Compose the user-facing query string.

    ``evidences`` is a sequence of (evidence_id, evidence_text). Numbering is
    1-based and stable across the same call (so the response can refer to
    [1], [2] etc. unambiguously). ``version`` selects a prompt variant from
    ``PROMPT_VARIANTS`` (default ``"v1"`` matches the historical behaviour).
    """
    variant = PROMPT_VARIANTS[version]
    lines: list[str] = [variant["rules"]]
    if variant["few_shot"]:
        lines.extend(["", variant["few_shot"].rstrip()])
    lines.extend(["", f"Claim: {claim_text}", "", "Evidence:"])
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
