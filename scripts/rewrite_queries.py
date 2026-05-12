"""LLM-driven query rewrite for Phase 3.5b retrieval ceiling escape.

Why
---
Phase 3.5b audit closed the param-tuning lever at recall@20 = 0.360
(fused no-rerank, best non-LLM config). Target is recall@20 ≥ 0.50.
The remaining lever is LLM-driven multi-query expansion:
  - HyDE (Hypothetical Document Embeddings): ask the LLM to write a
    short factual passage that would support/refute the claim; embed
    that and use it as an additional query.
  - Sub-claim decomposition: break compound claims into 1-3 atomic
    sub-claims, each retrieved separately.

`src/query_rewrite.py` already has the prompt builders + parse helpers.
This script wraps them with a real Qwen3.5-4B inference loop + caching.

What this does
--------------
1. Loads claims from one or more split files (default: all splits +
   official dev + test-unlabelled — all claim texts the retrieval will
   ever see).
2. For each claim, runs base Qwen3.5-4B (cache-first, same loader as
   phase1_eval) once for HyDE and once for sub-claim decomposition.
3. Caches results to outputs/query_rewrite/claim_rewrites.jsonl in the
   form `{claim_id, claim_text, hyde, sub_claims}`. Idempotent — re-runs
   skip claims already in the cache (`--force` to redo).
4. Prints a small sample so the user can eyeball quality.

The rewrites are CONSUMED by `scripts.retrieval_ceiling --mode
llm_rewrite` (added separately) which fuses (claim, hyde, sub_claim_1,
sub_claim_2) queries via RRF and re-measures recall@k.

Performance budget
------------------
On 4080 SUPER 4-bit Qwen3.5-4B:
  - HyDE: ~64 tokens generation, ~3-4 s/claim greedy
  - Sub-claim: ~96 tokens, ~4-5 s/claim greedy
  - 2 calls × ~4 s × ~1500 unique claims (train+dev+diag+official+test)
    ≈ 200 min one-time. Cached forever after.

Run
---

.. code-block:: bash

    # source /etc/network_turbo on AutoDL
    cd ~/autodl-tmp/NLP-A3

    # Smoke (5 claims, no model load if --dry-run):
    python -m scripts.rewrite_queries --splits diag_test --limit 5

    # Full diag_test (121 claims, ~10 min):
    python -m scripts.rewrite_queries --splits diag_test

    # All splits (one-time ~3.5h, then cache used forever):
    python -m scripts.rewrite_queries --splits all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import load_dev, load_test_unlabelled, read_jsonl  # noqa: E402
from src.paths import OUTPUTS_DIR, SPLITS_DIR  # noqa: E402
from src.query_rewrite import (  # noqa: E402
    decompose_subclaims_prompt, hyde_prompt, parse_subclaims,
)

CACHE_DIR = OUTPUTS_DIR / "query_rewrite"
CACHE_PATH = CACHE_DIR / "claim_rewrites.jsonl"


# -- Claim loading ---------------------------------------------------------

def load_claims(splits: list[str]) -> dict[str, str]:
    """Return {claim_id: claim_text}, deduplicated across all requested splits."""
    out: dict[str, str] = {}
    if "all" in splits:
        splits = ["train_split", "dev_holdout", "diag_test", "official_dev", "test"]

    for name in splits:
        if name == "official_dev":
            gold = load_dev()
            for cid, g in gold.items():
                out.setdefault(cid, g["claim_text"])
        elif name == "test":
            test = load_test_unlabelled()
            for cid, t in test.items():
                out.setdefault(cid, t["claim_text"])
        elif name in {"train_split", "dev_holdout", "diag_test"}:
            path = SPLITS_DIR / f"{name}.jsonl"
            if not path.exists():
                print(f"  WARN: split {path} not found — skipping")
                continue
            for row in read_jsonl(path):
                out.setdefault(row["id"], row["claim_text"])
        else:
            raise SystemExit(f"unknown split: {name}")
    return out


def load_cache() -> dict[str, dict]:
    """Existing cached rewrites: {claim_id: record}."""
    if not CACHE_PATH.exists():
        return {}
    cache: dict[str, dict] = {}
    for row in read_jsonl(CACHE_PATH):
        cache[row["claim_id"]] = row
    return cache


def write_cache(records: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -- Qwen3.5-4B inference loop --------------------------------------------

def load_model_and_tokenizer(model_dir: str | None):
    """Mirror scripts.phase1_eval.load_model_and_tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if model_dir is None:
        from src.paths import MODELS_DIR
        local = MODELS_DIR / "Qwen3.5-4B"
        if (local / "config.json").exists():
            model_dir = str(local)
            print(f"  [cache] using {model_dir}")
        else:
            from modelscope import snapshot_download
            print("  models/Qwen3.5-4B/ not found — downloading via ModelScope...")
            model_dir = snapshot_download(
                "Qwen/Qwen3.5-4B",
                cache_dir=str(OUTPUTS_DIR / "model_cache"),
            )
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, quantization_config=bnb_cfg, device_map="auto",
        trust_remote_code=True, torch_dtype=compute_dtype,
    )
    model.eval()
    return model, tok


def generate(model, tokenizer, system: str, user: str, max_new_tokens: int) -> str:
    """Greedy generation; thinking disabled (Qwen3.5 VL specifics)."""
    import torch
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    encoded = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = encoded if torch.is_tensor(encoded) else encoded["input_ids"]
    prompt_ids = prompt_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            prompt_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_ids = out[0][prompt_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


SYS_REWRITE = (
    "You are a climate fact-checking analyst. Help retrieve relevant "
    "evidence by reformulating claims into search-friendly forms."
)


# -- Main ------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--splits", default="diag_test",
                   help="Comma-separated splits to rewrite. Options: "
                        "train_split, dev_holdout, diag_test, official_dev, "
                        "test, all.")
    p.add_argument("--model-dir", default=None,
                   help="Local base-model dir; auto-resolves to "
                        "models/Qwen3.5-4B/ if present.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap claims for quick smoke (e.g. 5 for sanity).")
    p.add_argument("--force", action="store_true",
                   help="Re-rewrite claims already in cache.")
    p.add_argument("--hyde-tokens", type=int, default=80,
                   help="Max new tokens for HyDE generation.")
    p.add_argument("--sub-tokens", type=int, default=120,
                   help="Max new tokens for sub-claim decomposition.")
    p.add_argument("--dry-run", action="store_true",
                   help="Load claims + cache; don't load model or generate.")
    args = p.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]
    print(f"=== rewrite_queries.py — splits={splits} ===")

    print("\n[1/4] loading claims...")
    claims = load_claims(splits)
    if args.limit:
        claims = dict(list(claims.items())[: args.limit])
    print(f"  {len(claims)} unique claim ids")

    print("\n[2/4] loading cache...")
    cache = load_cache()
    print(f"  {len(cache)} existing rewrites in {CACHE_PATH}")

    to_do = [cid for cid in claims if args.force or cid not in cache]
    print(f"  {len(to_do)} claims need rewriting "
          f"(skip {len(claims) - len(to_do)} cached)")

    if args.dry_run or not to_do:
        print("\n[dry-run / nothing-to-do] exiting.")
        return

    print("\n[3/4] loading Qwen3.5-4B (4-bit)...")
    model, tok = load_model_and_tokenizer(args.model_dir)

    print(f"\n[4/4] rewriting {len(to_do)} claims...")
    t0 = time.time()
    new_records = 0
    for i, cid in enumerate(to_do):
        text = claims[cid]
        try:
            hyde_text = generate(
                model, tok, SYS_REWRITE, hyde_prompt(text),
                max_new_tokens=args.hyde_tokens,
            )
        except Exception as e:
            print(f"  WARN {cid}: HyDE failed ({type(e).__name__}: {e})")
            hyde_text = ""
        try:
            sub_raw = generate(
                model, tok, SYS_REWRITE, decompose_subclaims_prompt(text),
                max_new_tokens=args.sub_tokens,
            )
            sub_list = parse_subclaims(sub_raw, fallback=text)
        except Exception as e:
            print(f"  WARN {cid}: sub-claim failed ({type(e).__name__}: {e})")
            sub_list = [text]

        cache[cid] = {
            "claim_id": cid,
            "claim_text": text,
            "hyde": hyde_text,
            "sub_claims": sub_list,
        }
        new_records += 1

        # Periodic save so a crash doesn't lose progress.
        if (i + 1) % 25 == 0 or (i + 1) == len(to_do):
            write_cache(cache)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(to_do) - i - 1)
            print(f"  {i+1}/{len(to_do)}  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    write_cache(cache)
    elapsed = time.time() - t0
    print(f"\n[done] wrote {new_records} new rewrites in {elapsed/60:.1f} min")
    print(f"  cache: {CACHE_PATH}  ({len(cache)} total)")

    # Sample print so user can eyeball quality.
    print("\n=== Sample rewrites (first 2) ===")
    for cid in list(claims.keys())[:2]:
        rec = cache.get(cid)
        if rec is None:
            continue
        print(f"\nclaim_id: {cid}")
        print(f"  claim     : {rec['claim_text'][:120]}")
        print(f"  hyde      : {rec['hyde'][:200]}")
        print(f"  sub_claims: {rec['sub_claims']}")


if __name__ == "__main__":
    main()
