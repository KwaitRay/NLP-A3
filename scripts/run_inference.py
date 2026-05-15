"""End-to-end inference CLI — produces a predictions JSON ready for build_submission.

Cache-aware: relies on outputs/{bm25_index, dense_index}/ and the local
Qwen3.5-4B snapshot under models/. Never rebuilds — prints a clear error
that points at the notebook cell which produces the missing artifact.

Reuses phase1_eval's pipeline / model loaders so cache & quantisation
behaviour stays identical between leaderboard runs and ablation runs.

Run::

    # Final leaderboard run on test (default = full self-consistency, k=5):
    python -m scripts.run_inference --target test --tag v1-sft \\
        --sft-merged-dir outputs/sft-merged

    # Quick greedy sanity on diag_test:
    python -m scripts.run_inference --target diag_test --tag smoke \\
        --decoding greedy --limit 30

    # Pipeline only (no LLM) — useful when the model isn't ready but you
    # want to feel out the retrieval-only ceiling:
    python -m scripts.run_inference --target test --tag retrieval-only \\
        --decoding retrieval-only

Output::
    outputs/predictions/<tag>__<target>.json
        — same schema as build_submission expects (claim_text injected).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import (  # noqa: E402
    load_dev,
    load_evidence,
    load_test_unlabelled,
    read_jsonl,
)
from src.inference import (  # noqa: E402
    ModelInferer,
    RetrievalOnlyInferer,
    ZeroShotInferer,
    predict_all,
)
from src.paths import LABELS, MODELS_DIR, OUTPUTS_DIR, SPLITS_DIR  # noqa: E402

# Reuse phase1_eval's loaders so all submission-time inference picks up the
# same cache lookup + quantisation as the ablation runs.
from scripts.phase1_eval import (  # noqa: E402
    build_pipeline,
    load_model_and_tokenizer,
)


PRED_DIR = OUTPUTS_DIR / "predictions"


# ---- pretty printing -------------------------------------------------------

def _info(msg: str) -> None:
    print(f"  [info] {msg}")


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _warn(msg: str) -> None:
    print(f"  [warn] {msg}")


# ---- target loading -------------------------------------------------------

def load_target(name: str) -> dict[str, dict]:
    """Return {claim_id: {claim_text, ...}} for the requested target.

    test / dev / official_dev come from data/. diag_test / dev_holdout come
    from outputs/splits/ (built by Stage 0).
    """
    if name == "test":
        return load_test_unlabelled()
    if name in ("dev", "official_dev"):
        return load_dev()
    if name in ("diag_test", "dev_holdout"):
        path = SPLITS_DIR / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"split file missing: {path}\n"
                f"Run notebook cell 1.3 (or `python -m scripts.dry_run`) to build it."
            )
        return {
            r["id"]: {
                "claim_text": r["claim_text"],
                "claim_label": r.get("claim_label"),
                "evidences": r.get("evidences", []),
            }
            for r in read_jsonl(path)
        }
    raise ValueError(f"unknown target: {name!r}")


# ---- cache audit ----------------------------------------------------------

def audit_caches(*, need_pipeline: bool, need_model: bool, model_dir_arg: str | None) -> None:
    """Fail fast with actionable messages if a required cache is missing."""
    print("\n[cache audit]")
    bm25_dir = OUTPUTS_DIR / "bm25_index"
    dense_dir = OUTPUTS_DIR / "dense_index"
    qwen_local = MODELS_DIR / "Qwen3.5-4B"

    if need_pipeline:
        if not bm25_dir.exists():
            raise SystemExit(
                f"  [fail] BM25 index missing: {bm25_dir}\n"
                f"          → run notebook cell 2.1 (or `python -m scripts.build_indexes --skip-dense`)."
            )
        _ok(f"BM25 index   : {bm25_dir}")
        if (dense_dir / "faiss.index").exists():
            _ok(f"dense index  : {dense_dir}/faiss.index")
        else:
            _warn(f"dense index missing at {dense_dir}; pipeline will run BM25-only (degraded)")

    if need_model:
        if model_dir_arg:
            if not Path(model_dir_arg).exists():
                raise SystemExit(f"  [fail] --model-dir not found: {model_dir_arg}")
            _ok(f"model dir    : {model_dir_arg} (explicit)")
        elif (qwen_local / "config.json").exists():
            _ok(f"model dir    : {qwen_local} (cached local copy)")
        else:
            _warn(
                f"no local Qwen3.5-4B at {qwen_local} — phase1_eval will "
                f"fall back to ModelScope download (~8 GB)."
            )


# ---- inferer factory ------------------------------------------------------

def build_inferer(
    decoding: str,
    *,
    pipeline,
    model,
    tokenizer,
    n_samples: int,
    temperature: float,
    top_p: float,
    prompt_version: str,
    label_strategy: str,
):
    if decoding == "retrieval-only":
        if pipeline is None:
            raise SystemExit("retrieval-only decoding requires a pipeline; run with --use-rag.")
        return RetrievalOnlyInferer(pipeline, label_strategy=label_strategy)
    if decoding == "greedy":
        if pipeline is None or model is None:
            raise SystemExit("greedy decoding needs both pipeline + model.")
        return ZeroShotInferer(pipeline, model, tokenizer, prompt_version=prompt_version)
    if decoding == "self_consistency":
        if pipeline is None or model is None:
            raise SystemExit("self_consistency decoding needs both pipeline + model.")
        return ModelInferer(
            pipeline, model, tokenizer,
            n_samples=n_samples, temperature=temperature, top_p=top_p,
            prompt_version=prompt_version,
        )
    raise ValueError(f"unknown --decoding: {decoding}")


# ---- main ----------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target", required=True,
                   choices=["test", "dev", "official_dev", "diag_test", "dev_holdout"],
                   help="claim set to predict over. 'test' → leaderboard.")
    p.add_argument("--tag", required=True,
                   help="short id for this run; used in output filename and "
                        "later passed to scripts.build_submission.")

    # Decoding strategy
    p.add_argument("--decoding", default="self_consistency",
                   choices=["self_consistency", "greedy", "retrieval-only"],
                   help="self_consistency = ModelInferer (5×T=0.7 sampled, majority vote). "
                        "greedy = ZeroShotInferer. retrieval-only = no LLM, fixed label.")
    p.add_argument("--n-samples", type=int, default=5,
                   help="self-consistency sample count (only used with --decoding self_consistency).")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--prompt-version", default="v1",
                   help="prompt template variant (see src/prompt.py PROMPT_VARIANTS).")
    p.add_argument("--label-strategy", default="majority",
                   choices=["majority", "random", *LABELS],
                   help="only used with --decoding retrieval-only.")

    # Retrieval pipeline
    p.add_argument("--final-k", type=int, default=5,
                   help="top-k passages shown to the LLM. Default 5 (matches the "
                        "≤5-evidences leaderboard cap; build_submission rejects more).")
    p.add_argument("--rerank", action="store_true",
                   help="enable bge-reranker-base. OFF by default — Phase 3.5b audit "
                        "showed it cuts recall@5 by ~×1.68 on climate domain.")

    # Model
    p.add_argument("--model-dir", default=None,
                   help="explicit local model snapshot. Default: auto (models/Qwen3.5-4B "
                        "if present, else download via ModelScope).")
    p.add_argument("--sft-adapter", default=None,
                   help="LoRA SFT adapter dir. NB: ms-swift Qwen3.5 adapters typically "
                        "fail to attach via peft — prefer --sft-merged-dir.")
    p.add_argument("--sft-merged-dir", default=None,
                   help="swift-merged base dir (output of `swift export --merge_lora true`). "
                        "Loaded as a regular base model. Recommended for Qwen3.5 + ms-swift.")

    # Misc
    p.add_argument("--limit", type=int, default=None,
                   help="cap claims for quick smoke (e.g. --limit 30).")
    p.add_argument("--out", default=None, type=Path,
                   help="output path; default outputs/predictions/<tag>__<target>.json")
    args = p.parse_args()

    if args.sft_adapter and args.sft_merged_dir:
        raise SystemExit("--sft-adapter and --sft-merged-dir are mutually exclusive.")

    if args.target == "test" and args.final_k > 5:
        _warn(
            f"--final-k={args.final_k} > 5; build_submission will reject the resulting "
            f"file unless you also pass --max-evidences {args.final_k}. The leaderboard "
            f"penalises precision, so this is rarely what you want."
        )

    print("=" * 70)
    print(f"run_inference: target={args.target} decoding={args.decoding} tag={args.tag}")
    print("=" * 70)

    need_pipeline = args.decoding != "retrieval-only" or True  # retrieval-only also needs pipeline
    need_model = args.decoding in ("self_consistency", "greedy")
    audit_caches(
        need_pipeline=True,  # always — every decoding mode needs retrieval
        need_model=need_model,
        model_dir_arg=args.model_dir,
    )

    # ---- claims --------------------------------------------------------
    print("\n[1/4] loading claims…")
    claims = load_target(args.target)
    if args.limit:
        claims = dict(list(claims.items())[: args.limit])
    _info(f"{args.target}: {len(claims)} claims")

    # ---- model ---------------------------------------------------------
    model = tokenizer = None
    if need_model:
        print("\n[2/4] loading base model + tokenizer…")
        model, tokenizer = load_model_and_tokenizer(args.model_dir)

        if args.sft_adapter:
            from peft import PeftModel
            print(f"  loading SFT LoRA adapter: {args.sft_adapter}")
            model = PeftModel.from_pretrained(model, args.sft_adapter)
            model.eval()
            n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
            _info(f"LoRA params: {n_lora / 1e6:.2f} M")
            if n_lora < 1e6:
                _warn(
                    "LoRA params suspiciously low — adapter likely not applied. "
                    "Re-export with `swift export --merge_lora true` and use --sft-merged-dir."
                )
        elif args.sft_merged_dir:
            print(f"\n[2b/4] loading SFT-merged base from {args.sft_merged_dir}…")
            model, tokenizer = load_model_and_tokenizer(args.sft_merged_dir)
    else:
        print("\n[2/4] skipping model load (decoding=retrieval-only)")

    # ---- pipeline ------------------------------------------------------
    print("\n[3/4] loading evidence + retrieval pipeline…")
    evidence = load_evidence(show_progress=True)
    pipeline = build_pipeline(evidence, final_k=args.final_k, use_rerank=args.rerank)

    # ---- inference -----------------------------------------------------
    print("\n[4/4] running inference…")
    inferer = build_inferer(
        args.decoding,
        pipeline=pipeline,
        model=model,
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_version=args.prompt_version,
        label_strategy=args.label_strategy,
    )

    out_path = args.out or (PRED_DIR / f"{args.tag}__{args.target}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # predict_all needs {cid: {"claim_text": ..., ...}} — load_target already
    # returns that shape. It also injects claim_text into every output record
    # (per the post-2026-05-15 patch in src/inference.py).
    t0 = time.time()
    preds = predict_all(claims, inferer, out_path, progress=True)
    elapsed = time.time() - t0

    _ok(f"wrote {len(preds)} predictions → {out_path}")
    _info(f"elapsed: {elapsed:.1f}s ({elapsed / max(len(preds), 1):.2f}s/claim)")

    if args.target == "test":
        print()
        print(f"next: python -m scripts.build_submission \\")
        print(f"          --preds {out_path.relative_to(OUTPUTS_DIR.parent)} \\")
        print(f"          --tag {args.tag} --phase 1")
    else:
        print()
        print(f"next: python eval.py --predictions {out_path.relative_to(OUTPUTS_DIR.parent)} \\")
        print(f"          --groundtruth data/dev-claims.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
