"""Phase 1 evaluation harness — base + RAG with prompt-variant sweep.

Implements the workflow from design.md §11.1b / D-015:
- Phase 1a: Track 1 (no-RAG, base model) — establishes parametric baseline.
- Phase 1b: Track 2 (base + full RAG) — RAG-only contribution.
- Across both tracks, sweep prompt variants v1..v4 from src.prompt.
- Per run, compute (F, Acc, HM) overall + per-bucket diagnostic slices
  (domain × scenario × difficulty) using outputs/splits/{dataset}.jsonl
  for the bucket lookup.

The main use case is "find the worst buckets so SFT data can target them".

Usage::

    # quickest sanity: v1 only, no-RAG only, on diag_test (~30 sec)
    python -m scripts.phase1_eval --tracks 1 --prompts v1 --dataset diag_test

    # full sweep (Track 1 + Track 2, all 4 prompt variants, on diag_test)
    python -m scripts.phase1_eval --tracks 1,2 --prompts v1,v2,v3,v4 \\
                                  --dataset diag_test

    # final report (with the locked-best prompt) on official dev — burns a
    # "look at dev" budget, see D-006:
    python -m scripts.phase1_eval --tracks 1,2 --prompts v3 --dataset official_dev

Output structure::

    outputs/eval_phase1/
        track{N}_{prompt}_{dataset}.json   # raw predictions
        track{N}_{prompt}_{dataset}.md     # per-bucket diagnostic table
        summary_{dataset}.md               # cross-prompt comparison table

Datasets:
    diag_test     — 121 claims from outputs/splits/diag_test.jsonl
                    (the safe default — does not consume a "look at dev" budget)
    dev_holdout   — 121 claims from outputs/splits/dev_holdout.jsonl
                    (reserved for DPO; using it here is a soft pollution)
    official_dev  — 154 claims from data/dev-claims.json
                    (use sparingly, see D-006)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import load_dev, load_evidence, read_jsonl  # noqa: E402
from src.eval_helpers import score_per_bucket, score_predictions  # noqa: E402
from src.paths import OUTPUTS_DIR, SPLITS_DIR  # noqa: E402
from src.prompt import PROMPT_VARIANTS  # noqa: E402

OUT_DIR = OUTPUTS_DIR / "eval_phase1"


# -- Dataset loading --------------------------------------------------------

def load_dataset(name: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """Return (gold, tag_lookup). Both keyed by claim_id.

    gold[cid] = {"claim_label": str, "evidences": [str], "claim_text": str}
    tag_lookup[cid] = full tagged row from splits/*.jsonl with domain /
    scenario / difficulty fields. May be empty for official_dev (no tags).
    """
    if name == "official_dev":
        gold = load_dev()  # {cid: {claim_label, claim_text, evidences}}
        # Official dev has no tags. Bucket-by-domain etc will be empty.
        return gold, {}

    if name not in {"diag_test", "dev_holdout"}:
        raise ValueError(f"unknown dataset: {name!r}")

    path = SPLITS_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            f"Run notebook cell 1.3 (or scripts.dry_run) to produce it."
        )
    rows = list(read_jsonl(path))
    gold = {
        r["id"]: {
            "claim_label": r["claim_label"],
            "claim_text": r["claim_text"],
            "evidences": r.get("evidences", []),
        }
        for r in rows
    }
    tag_lookup = {r["id"]: r for r in rows}
    return gold, tag_lookup


# -- Inference per track ---------------------------------------------------

def run_track1(model, tokenizer, gold: dict, prompt_version: str) -> dict:
    """Track 1 = no RAG, base model, greedy."""
    from src.inference import NoRagInferer, predict_all
    inferer = NoRagInferer(model, tokenizer, prompt_version=prompt_version)
    return predict_all(
        {cid: {"claim_text": g["claim_text"]} for cid, g in gold.items()},
        inferer,
    )


def run_track2(model, tokenizer, gold: dict, prompt_version: str, pipeline) -> dict:
    """Track 2 = full RAG (BM25+dense+rerank) → base model, greedy."""
    from src.inference import ZeroShotInferer, predict_all
    inferer = ZeroShotInferer(pipeline, model, tokenizer, prompt_version=prompt_version)
    return predict_all(
        {cid: {"claim_text": g["claim_text"]} for cid, g in gold.items()},
        inferer,
    )


# -- Bucket reporting -------------------------------------------------------

def render_per_bucket(
    preds: dict, gold: dict, tag_lookup: dict, axis: str
) -> str:
    """Render a markdown table sliced by `axis`."""
    if not tag_lookup:
        return f"\n_No tag info for axis '{axis}' (likely official_dev)._\n"

    def lookup(cid):
        rec = tag_lookup.get(cid)
        if rec is None:
            return None
        if axis == "difficulty":
            d = rec.get("difficulty")
            return d.get("level") if isinstance(d, dict) else d
        return rec.get(axis)

    sliced = score_per_bucket(preds, gold, lookup)
    if not sliced:
        return f"\n_No buckets produced for axis '{axis}'._\n"

    lines = [f"\n#### Per-{axis}\n", "| bucket | n | F | Acc | HM |", "|---|---|---|---|---|"]
    # Sort by HM ascending so worst buckets are at the top (the actionable end).
    sorted_buckets = sorted(sliced.items(), key=lambda kv: kv[1]["harmonic_mean"])
    for bucket, m in sorted_buckets:
        lines.append(
            f"| {bucket} | {m['n']} | {m['f_score']:.3f} | "
            f"{m['accuracy']:.3f} | {m['harmonic_mean']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_run_report(
    track: int, prompt: str, dataset: str,
    preds: dict, gold: dict, tag_lookup: dict, elapsed: float,
) -> Path:
    overall = score_predictions(preds, gold)
    out_md = OUT_DIR / f"track{track}_{prompt}_{dataset}.md"
    parts = [
        f"# Track {track} — prompt {prompt} on {dataset}",
        "",
        f"- variant: **{PROMPT_VARIANTS[prompt]['name']}** ({PROMPT_VARIANTS[prompt]['description']})",
        f"- claims: {overall['n']}",
        f"- elapsed: {elapsed:.1f}s",
        "",
        "## Overall",
        "| F | Acc | HM |",
        "|---|---|---|",
        f"| {overall['f_score']:.4f} | {overall['accuracy']:.4f} | {overall['harmonic_mean']:.4f} |",
    ]
    for axis in ("domain", "scenario", "difficulty"):
        parts.append(render_per_bucket(preds, gold, tag_lookup, axis))
    out_md.write_text("\n".join(parts), encoding="utf-8")
    return out_md


# -- Cross-prompt summary ---------------------------------------------------

def write_summary(results: list[dict], dataset: str) -> Path:
    """One-table summary of all (track, prompt) combinations."""
    out_md = OUT_DIR / f"summary_{dataset}.md"
    lines = [
        f"# Phase 1 summary on {dataset}",
        "",
        "Prompt variant sweep (D-015 Phase 2). Higher HM is better.",
        "Track 1 = no-RAG (base model parametric only). Track 1 F is 0 by design.",
        "Track 2 = full RAG (BM25 + dense + rerank) → base model.",
        "",
        "| Track | Prompt | Variant | n | F | Acc | HM |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['track']} | {r['prompt']} | "
            f"{PROMPT_VARIANTS[r['prompt']]['name']} | "
            f"{r['metrics']['n']} | "
            f"{r['metrics']['f_score']:.4f} | "
            f"{r['metrics']['accuracy']:.4f} | "
            f"{r['metrics']['harmonic_mean']:.4f} |"
        )
    lines.extend([
        "",
        "## Phase 2 next step",
        "",
        "1. Pick the prompt with the highest Track-2 HM as the locked production prompt.",
        "2. Open the matching `track2_<prompt>_<dataset>.md` and inspect the per-bucket tables.",
        "3. Buckets with HM < 0.30 are the SFT-data-augmentation targets for Phase 4.",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


# -- Pipeline init ---------------------------------------------------------

def build_pipeline(evidence: dict):
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.pipeline import RetrievalPipeline, RetrievalConfig

    bm25_dir = OUTPUTS_DIR / "bm25_index"
    dense_dir = OUTPUTS_DIR / "dense_index"

    if not bm25_dir.exists():
        raise FileNotFoundError(
            f"BM25 index missing at {bm25_dir}\n"
            f"Run: python -m scripts.build_indexes --skip-dense"
        )
    bm25 = BM25Retriever.load(bm25_dir)

    dense = None
    reranker = None
    if (dense_dir / "faiss.index").exists():
        dense = DenseRetriever.load(dense_dir, max_seq_length=256, fp16=True)
        try:
            from src.retrieval.rerank import CrossEncoderReranker
            reranker = CrossEncoderReranker()
        except Exception as e:
            print(f"  WARN: reranker load failed ({type(e).__name__}: {e}); BM25+dense only")
    else:
        print(f"  WARN: dense index missing at {dense_dir}; BM25-only RAG (degraded)")

    cfg = RetrievalConfig(
        use_bm25=True, use_dense=dense is not None,
        use_rerank=reranker is not None,
        use_rule_reorder=False,  # rule_reorder needs spaCy; skip in eval
        final_k=5,
    )
    return RetrievalPipeline(
        evidence_corpus=evidence, bm25=bm25, dense=dense, reranker=reranker, cfg=cfg,
    )


def load_model_and_tokenizer(model_dir: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if model_dir is None:
        # 1. Prefer pre-downloaded local copy under models/Qwen3.5-4B/
        #    (via scripts.download_models)
        from src.paths import MODELS_DIR
        local = MODELS_DIR / "Qwen3.5-4B"
        if (local / "config.json").exists():
            model_dir = str(local)
            print(f"  [cache] using {model_dir}")
        else:
            # 2. Fall back to ModelScope download into outputs/model_cache/
            from modelscope import snapshot_download
            print("  models/Qwen3.5-4B/ not found — downloading via ModelScope...")
            model_dir = snapshot_download(
                "Qwen/Qwen3.5-4B",
                cache_dir=str(OUTPUTS_DIR / "model_cache"),
            )
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"  loading {model_dir} (dtype={compute_dtype}, 4-bit)...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, quantization_config=bnb_cfg, device_map="auto",
        trust_remote_code=True, torch_dtype=compute_dtype,
    )
    model.eval()
    return model, tokenizer


# -- Main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Phase 1 eval — Track 1/2 × prompt sweep")
    p.add_argument("--tracks", default="1,2",
                   help="Comma-separated track ids: 1 (no-RAG), 2 (RAG). Default 1,2.")
    p.add_argument("--prompts", default="v1",
                   help=f"Comma-separated prompt versions. Available: {','.join(PROMPT_VARIANTS)}.")
    p.add_argument("--dataset", default="diag_test",
                   choices=["diag_test", "dev_holdout", "official_dev"],
                   help="Eval set. diag_test is the safe default; official_dev consumes a 'look at dev' budget.")
    p.add_argument("--model-dir", default=None,
                   help="Local model snapshot. Omit to download from ModelScope.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap claims for quick smoke (e.g. --limit 30).")
    args = p.parse_args()

    tracks = [int(x) for x in args.tracks.split(",")]
    prompts = args.prompts.split(",")
    for v in prompts:
        if v not in PROMPT_VARIANTS:
            raise SystemExit(f"unknown prompt version: {v}; available: {list(PROMPT_VARIANTS)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Phase 1 eval: tracks={tracks} prompts={prompts} dataset={args.dataset} ===")

    print("\n[1/4] loading dataset...")
    gold, tag_lookup = load_dataset(args.dataset)
    if args.limit:
        gold = dict(list(gold.items())[: args.limit])
        tag_lookup = {k: tag_lookup[k] for k in gold if k in tag_lookup}
    print(f"  {len(gold)} claims; {len(tag_lookup)} tagged")

    print("\n[2/4] loading model + tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    pipeline = None
    if 2 in tracks:
        print("\n[3/4] loading evidence corpus + RAG pipeline...")
        evidence = load_evidence(show_progress=True)
        pipeline = build_pipeline(evidence)
        print(f"  evidence: {len(evidence):,} passages")

    print("\n[4/4] running track × prompt sweep...")
    results: list[dict] = []
    for track in tracks:
        for prompt in prompts:
            tag = f"track{track}_{prompt}_{args.dataset}"
            print(f"\n--- {tag} ---")
            t0 = time.time()
            if track == 1:
                preds = run_track1(model, tokenizer, gold, prompt)
            elif track == 2:
                preds = run_track2(model, tokenizer, gold, prompt, pipeline)
            else:
                raise SystemExit(f"unknown track: {track}")
            elapsed = time.time() - t0

            # Save raw predictions (eval.py compatible).
            json_path = OUT_DIR / f"{tag}.json"
            json_path.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")

            md_path = write_run_report(track, prompt, args.dataset, preds, gold, tag_lookup, elapsed)
            metrics = score_predictions(preds, gold)
            print(f"  → F={metrics['f_score']:.4f}  Acc={metrics['accuracy']:.4f}  "
                  f"HM={metrics['harmonic_mean']:.4f}  ({elapsed:.1f}s)")
            print(f"  → {md_path}")
            results.append({"track": track, "prompt": prompt, "metrics": metrics})

    summary_path = write_summary(results, args.dataset)
    print(f"\n=== Summary written to {summary_path} ===\n")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
