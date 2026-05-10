"""Three-axis tagging for SFT data: scenario × difficulty × domain.

All taggers operate on claim-side metadata only (no evidence text). They use
heuristics so they can run without a GPU and without evidence.json.

Refinements that need evidence text (e.g. claim-evidence cosine for
scenario disambiguation, model-loss difficulty) are deferred to Colab and
will overwrite the heuristic fields where appropriate.
"""
from __future__ import annotations

import re
from typing import Literal

DomainCode = Literal[
    "temperature",
    "co2_atmospheric",
    "sea_level",
    "extreme_weather",
    "paleoclimate",
    "models_attribution",
    "policy_economics",
    "general_other",
]

ScenarioCode = Literal[
    "supports_clear",
    "supports_aggregated",
    "refutes_clear",
    "refutes_aggregated",
    "nei_topic_off",
    "nei_underspec",
    "disputed_conflict",
]

DifficultyLevel = Literal["easy", "medium", "hard"]


# -- Domain taxonomy ---------------------------------------------------------

# Patterns ordered by specificity. Each rule contributes a score to its
# domain; the highest-scoring domain wins, tiebreak by order.
_DOMAIN_PATTERNS: list[tuple[DomainCode, list[re.Pattern]]] = [
    (
        "paleoclimate",
        [re.compile(p, re.IGNORECASE) for p in [
            r"\bice\s*core",
            r"\bproxy\b",
            r"\bholocene\b",
            r"\bpleistocene\b",
            r"\bmedieval\s+warm\s+period\b|\bMWP\b",
            r"\blittle\s+ice\s+age\b|\bLIA\b",
            r"\bdendro|\btree[-\s]ring\b",
            r"\bstomata\b",
            r"\bpaleoclim",
            r"\bmillennium\b",
        ]],
    ),
    (
        "sea_level",
        [re.compile(p, re.IGNORECASE) for p in [
            r"\bsea\s*level\b",
            r"\bice\s*sheet\b",
            r"\bglacier\b|\bglacial\b",
            r"\bantarctic\w*",
            r"\bgreenland\b",
            r"\barctic\w*\s+(ice|melt)",
            r"\bocean\s+(level|rise)",
        ]],
    ),
    (
        "co2_atmospheric",
        [re.compile(p, re.IGNORECASE) for p in [
            r"\bco2\b|\bcarbon\s+dioxide\b",
            r"\bppm\b|\bparts\s+per\s+million\b",
            r"\bemission",
            r"\bcarbon\s+cycle\b",
            r"\bgreenhouse\s+gas\b|\bGHG\b",
            r"\bmethane\b|\bCH4\b",
        ]],
    ),
    (
        "extreme_weather",
        [re.compile(p, re.IGNORECASE) for p in [
            r"\bhurricane\b|\bcyclone\b|\btyphoon\b",
            r"\bflood",
            r"\bdrought",
            r"\bwildfire\b|\bforest\s+fire\b",
            r"\bstorm",
            r"\btornado",
            r"\bheat\s*wave\b|\bheatwave\b",
            r"\bextreme\s+weather\b",
        ]],
    ),
    (
        "models_attribution",
        [re.compile(p, re.IGNORECASE) for p in [
            r"\bclimate\s+model",
            r"\bGCM\b|\bgeneral\s+circulation\s+model\b",
            r"\bsensitivity\b",
            r"\bIPCC\b",
            r"\bRCP\b|\bSSP\b",
            r"\battribution\b",
            r"\bsimulation\b",
            r"\bensemble\b",
            r"\bforcing\b",
        ]],
    ),
    (
        "policy_economics",
        [re.compile(p, re.IGNORECASE) for p in [
            r"\brenewable",
            r"\bsubsid",
            r"\bcarbon\s+(tax|trading|market|price|pricing)\b",
            r"\bkyoto\b",
            r"\bparis\s+(agreement|accord)\b",
            r"\bUNFCCC\b",
            r"\bnet[-\s]?zero\b",
            r"\bclean\s+energy\b",
            r"\bsolar\s+(panel|power|energy)\b",
            r"\bwind\s+(power|farm|energy)\b",
            r"\bnuclear\s+(power|energy|reactor)\b",
        ]],
    ),
    (
        "temperature",
        [re.compile(p, re.IGNORECASE) for p in [
            r"\bwarming\b",
            r"\bcooling\b",
            r"°\s*C\b|\bdegrees?\s+celsius\b|\bdeg\s*C\b",
            r"\bhiatus\b|\bpause\b",
            r"\b(temperature|temp)\s+(record|anomaly|trend|rise|increase|decrease)\b",
            r"\bglobal\s+(warming|temperature)\b",
            r"\b(record|hottest|warmest|coldest)\s+(year|decade|month)\b",
        ]],
    ),
]


def tag_domain(claim_text: str) -> DomainCode:
    """Assign one of 8 climate-science domain codes via keyword matching.

    Highest-scoring domain wins. If no rule matches, returns ``general_other``.
    Order of evaluation favours specific topics (paleo, sea level) over
    broader ones (temperature) so generic ``warming`` claims about ice cores
    map to ``paleoclimate`` rather than ``temperature``.
    """
    best: tuple[DomainCode, int] | None = None
    for code, patterns in _DOMAIN_PATTERNS:
        score = sum(1 for p in patterns if p.search(claim_text))
        if score > 0 and (best is None or score > best[1]):
            best = (code, score)
    return best[0] if best else "general_other"


# -- Scenario tagging --------------------------------------------------------

def tag_scenario(claim_label: str, n_evidence: int) -> ScenarioCode:
    """Coarse scenario from (label, n_evidence). Refined later with embeddings.

    Heuristic:
      - SUPPORTS/REFUTES with 1-2 ev → ``*_clear``
      - SUPPORTS/REFUTES with >=3 ev → ``*_aggregated`` (multi-source needed)
      - NEI: arbitrary split based on n_ev (always 5 in train) — refined later
        via claim-evidence cosine similarity. Default to ``nei_underspec``.
      - DISPUTED → ``disputed_conflict``.
    """
    if claim_label == "DISPUTED":
        return "disputed_conflict"
    if claim_label == "NOT_ENOUGH_INFO":
        return "nei_underspec"
    if claim_label == "SUPPORTS":
        return "supports_clear" if n_evidence <= 2 else "supports_aggregated"
    if claim_label == "REFUTES":
        return "refutes_clear" if n_evidence <= 2 else "refutes_aggregated"
    raise ValueError(f"unknown label: {claim_label!r}")


# -- Difficulty heuristic ----------------------------------------------------

def _difficulty_score_heuristic(claim_text: str, claim_label: str, n_evidence: int) -> float:
    """Compose a difficulty score in [0, 1] from claim metadata only.

    Components (each 0-1):
      - length_norm  : longer claims → harder
      - n_ev_norm    : more required evidences → harder (multi-hop)
      - label_prior  : DISPUTED hardest, NEI medium-hard, REFUTES medium,
                       SUPPORTS easiest
    """
    n_tokens = len(claim_text.split())
    length_norm = min(1.0, n_tokens / 40.0)
    n_ev_norm = min(1.0, n_evidence / 5.0)
    label_prior = {
        "SUPPORTS": 0.20,
        "REFUTES": 0.50,
        "NOT_ENOUGH_INFO": 0.65,
        "DISPUTED": 0.85,
    }[claim_label]
    score = 0.30 * length_norm + 0.25 * n_ev_norm + 0.45 * label_prior
    return round(min(1.0, max(0.0, score)), 4)


def tag_difficulty(
    claim_text: str, claim_label: str, n_evidence: int
) -> dict:
    score = _difficulty_score_heuristic(claim_text, claim_label, n_evidence)
    if score < 0.40:
        level: DifficultyLevel = "easy"
    elif score < 0.65:
        level = "medium"
    else:
        level = "hard"
    return {"level": level, "score": score, "source": "heuristic"}


# -- High-level wrapper ------------------------------------------------------

def tag_claim(claim_id: str, claim: dict) -> dict:
    """Apply all three taggers to a single labelled claim.

    Returns a dict suitable for SFT-style jsonl with diagnostic tags.
    """
    text = claim["claim_text"]
    label = claim.get("claim_label")
    evidences: list[str] = claim.get("evidences", [])
    n_ev = len(evidences)

    out = {
        "id": claim_id,
        "claim_text": text,
        "claim_label": label,
        "evidences": evidences,
        "n_evidence": n_ev,
        "domain": tag_domain(text),
    }
    if label is not None:  # train/dev: full tagging
        out["scenario"] = tag_scenario(label, n_ev)
        out["difficulty"] = tag_difficulty(text, label, n_ev)
    return out
