"""Stage 5 — inference orchestration.

Three flavours, all feeding ``eval.py``-compatible prediction dicts:

1. ``ModelInferer``  — full pipeline: retrieval → SFT/DPO model with
   self-consistency sampling (5 samples @ T=0.7) → majority-vote label,
   max-confidence sample's evidence list.
2. ``ZeroShotInferer`` — retrieval + zero-shot Qwen (no SFT). Used for
   ablation rows A1-A4 (retrieval-only configurations).
3. ``RetrievalOnlyInferer`` — retrieval + heuristic label (predicts the
   majority class on dev: SUPPORTS). Lets us isolate retrieval F-score
   without an LLM, useful for early Stage 1 sanity checking.

All three implement ``predict(claim_text) -> {"claim_label", "evidences"}``.
``predict_all(claims, out_path)`` writes the JSON.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Protocol

from .data_io import write_predictions
from .paths import LABELS
from .prompt import (
    NO_RAG_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_no_rag_query,
    build_user_query,
    get_variant_system,
    parse_response,
)


def _apply_template_to_device(tokenizer, msgs, device):
    """Run apply_chat_template and return a 2-D tensor on `device`.

    transformers 5.x returns a BatchEncoding (dict-like) from
    apply_chat_template even with return_tensors="pt"; older versions
    returned a bare tensor. Treating BatchEncoding as a tensor raises an
    empty AttributeError when we hit `.shape` (BatchEncoding.__getattr__
    proxies to self.data which has no 'shape' key). Normalize both forms
    to a tensor so callers can keep using `.shape[1]` and `[0][n:]`.
    """
    import torch
    encoded = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = encoded if torch.is_tensor(encoded) else encoded["input_ids"]
    return prompt_ids.to(device)


class _Retriever(Protocol):
    def retrieve(self, claim_text: str) -> list[tuple[str, str]]: ...


class Inferer(Protocol):
    """Anything that can predict a single claim. Used by predict_all."""
    def predict(self, claim_text: str) -> dict[str, Any]: ...


# -- 1. Model-driven self-consistency ----------------------------------------

class ModelInferer:
    """Self-consistency sampling on top of retrieval.

    The expensive parts (model + tokenizer) are injected, so the same class
    is used for SFT-only, DPO-aligned, and the 9B int4 ablation."""

    def __init__(
        self,
        retriever: _Retriever,
        model: Any,
        tokenizer: Any,
        *,
        n_samples: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 32,
        default_label: str = "NOT_ENOUGH_INFO",
        prompt_version: str = "v1",
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.default_label = default_label
        self.prompt_version = prompt_version

    def _generate_one(self, prompt_ids):
        """One sampled completion. Returns the decoded continuation only."""
        out = self.model.generate(
            prompt_ids,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        # Strip prompt; decode only the new tokens.
        new_ids = out[0][prompt_ids.shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def predict(self, claim_text: str) -> dict[str, Any]:
        retrieved = self.retriever.retrieve(claim_text)
        if not retrieved:
            return {"claim_label": self.default_label, "evidences": []}
        shown_ids = [eid for eid, _ in retrieved]
        user = build_user_query(claim_text, retrieved, version=self.prompt_version)
        msgs = [
            {"role": "system", "content": get_variant_system(self.prompt_version)},
            {"role": "user", "content": user},
        ]
        prompt_ids = _apply_template_to_device(self.tokenizer, msgs, self.model.device)

        labels: list[str] = []
        ev_lists: list[list[str]] = []
        for _ in range(self.n_samples):
            txt = self._generate_one(prompt_ids)
            lbl, evs = parse_response(txt, shown_ids, default_label=self.default_label)
            labels.append(lbl)
            ev_lists.append(evs)

        # Majority vote on label.
        final_label = Counter(labels).most_common(1)[0][0]
        # Among samples that voted for the winning label, pick the one with
        # the longest evidence list as a confidence proxy. This biases toward
        # samples that committed to specific citations rather than defaulted
        # to "all shown".
        winners = [i for i, l in enumerate(labels) if l == final_label]
        best = max(winners, key=lambda i: len(ev_lists[i]))
        return {"claim_label": final_label, "evidences": list(ev_lists[best])}


# -- 2. Zero-shot model (no SFT) --------------------------------------------

class ZeroShotInferer(ModelInferer):
    """Same as ``ModelInferer`` but ``predict`` uses greedy decoding.

    Slightly hacky: zero-shot benefits less from sampling diversity, and
    using the same code path for both makes ablation comparisons fairer.
    """

    def _generate_one(self, prompt_ids):
        out = self.model.generate(
            prompt_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)

    def predict(self, claim_text: str) -> dict[str, Any]:
        # n_samples=1 for greedy — overrides parent.
        old_n = self.n_samples
        self.n_samples = 1
        try:
            return super().predict(claim_text)
        finally:
            self.n_samples = old_n


# -- 3. No-RAG (Track 1: pure base model, claim only) ----------------------

class NoRagInferer:
    """Track 1 baseline: base LLM sees claim only, no retrieval at all.

    Evidence list is a stub (``["evidence-0"]``) since ``eval.py`` rejects
    empty lists; this gives F=0 by construction. Read this track's metric
    as **label accuracy in isolation** — it isolates how much of the task
    the LLM can solve from its parametric knowledge alone, motivating the
    other tracks' use of RAG.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 16,
        default_label: str = "NOT_ENOUGH_INFO",
        prompt_version: str = "v1",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.default_label = default_label
        self.prompt_version = prompt_version

    def predict(self, claim_text: str) -> dict[str, Any]:
        msgs = [
            {"role": "system", "content": get_variant_system(self.prompt_version, no_rag=True)},
            {"role": "user", "content": build_no_rag_query(claim_text, version=self.prompt_version)},
        ]
        prompt_ids = _apply_template_to_device(self.tokenizer, msgs, self.model.device)
        out = self.model.generate(
            prompt_ids,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(
            out[0][prompt_ids.shape[1]:], skip_special_tokens=True
        )
        label, _ = parse_response(text, shown_evidence_ids=[], default_label=self.default_label)
        return {"claim_label": label, "evidences": ["evidence-0"]}


# -- 4. Retrieval-only (no LLM) ---------------------------------------------

class RetrievalOnlyInferer:
    """Outputs retrieved evidences and a fixed label. For Stage 1 sanity
    checks: lets us measure retrieval F-score independently of any model.

    ``label_strategy`` options:
    - ``"majority"`` (default): predict SUPPORTS (the majority class on dev).
    - ``"random"``: deterministic per-claim using ``hash(claim_id)``. Useful
      to verify the eval pipeline. Not a sensible model.
    - ``str`` matching one of LABELS: predict that label for every claim.
    """

    def __init__(
        self,
        retriever: _Retriever,
        *,
        label_strategy: str = "majority",
    ) -> None:
        self.retriever = retriever
        if label_strategy not in {"majority", "random", *LABELS}:
            raise ValueError(f"unknown label_strategy: {label_strategy!r}")
        self.label_strategy = label_strategy

    def predict(self, claim_text: str) -> dict[str, Any]:
        retrieved = self.retriever.retrieve(claim_text)
        ev_ids = [eid for eid, _ in retrieved] or ["evidence-0"]
        if self.label_strategy == "majority":
            label = "SUPPORTS"
        elif self.label_strategy == "random":
            # Deterministic mock for tests.
            label = LABELS[hash(claim_text) % len(LABELS)]
        else:
            label = self.label_strategy
        return {"claim_label": label, "evidences": ev_ids}


# -- Batch driver ------------------------------------------------------------

def predict_all(
    claims: dict[str, dict],
    inferer: Inferer,
    out_path: str | Path | None = None,
    *,
    progress: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run inference over every claim in ``claims`` (dict claim_id → claim).

    ``progress`` prints a tqdm-style line every 25 claims to stderr if tqdm
    isn't available. Result is the prediction dict; if ``out_path`` is given,
    also written via ``write_predictions`` (validates the schema)."""
    preds: dict[str, dict[str, Any]] = {}
    items = list(claims.items())
    iterator: Iterable
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(items, desc="predict")
    except Exception:
        import sys
        def _bar(seq):
            for i, x in enumerate(seq):
                if progress and (i % 25 == 0 or i == len(items) - 1):
                    print(f"  predict {i + 1}/{len(items)}", file=sys.stderr)
                yield x
        iterator = _bar(items)

    import traceback as _tb
    _err_traces_shown = 0
    for cid, claim in iterator:
        try:
            preds[cid] = inferer.predict(claim["claim_text"])
        except Exception as e:  # robust to per-claim failures during long runs
            preds[cid] = {"claim_label": "NOT_ENOUGH_INFO", "evidences": ["evidence-0"]}
            # Print full traceback for the first 3 failures so silent
            # errors (e.g. empty AttributeError) are diagnosable.
            if _err_traces_shown < 3:
                print(f"  WARN {cid}: {e!r}")
                _tb.print_exc()
                _err_traces_shown += 1
            else:
                print(f"  WARN {cid}: {e!r}")

    if out_path is not None:
        write_predictions(preds, out_path)
    return preds


def load_predictions(path: str | Path) -> dict[str, dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
