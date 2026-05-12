"""Phase 1 evaluation harness — base / SFT × RAG / no-RAG × prompt sweep.

Implements the workflow from design.md §11.1b / D-015:
- Track 1: no-RAG, base model — parametric baseline.
- Track 2: base + full RAG — RAG-only contribution.
- Track 3: base + SFT adapter + full RAG — what Phase 5 SFT delivers.
- Sweeps prompt variants v1..v4 from src.prompt across all enabled tracks.
- Per run, compute (F, Acc, HM) overall + per-bucket diagnostic slices
  (domain × scenario × difficulty) using outputs/splits/{dataset}.jsonl
  for the bucket lookup.

The main use case is "find the worst buckets so SFT data can target them",
then compare Track 2 → Track 3 to validate the SFT data design closed the gap.

Usage::

    # quickest sanity: v1 only, no-RAG only, on diag_test (~30 sec)
    python -m scripts.phase1_eval --tracks 1 --prompts v1 --dataset diag_test

    # full sweep (Track 1 + Track 2, all 4 prompt variants, on diag_test)
    python -m scripts.phase1_eval --tracks 1,2 --prompts v1,v2,v3,v4 \\
                                  --dataset diag_test

    # Track 3 (SFT) vs Track 2 (base+RAG) head-to-head, locked prompt v1.
    # For Qwen3.5 + ms-swift, the LoRA adapter target_modules + state_dict
    # are baked against the VL wrapper but AutoModelForCausalLM strips it,
    # so peft can't reattach. Merge LoRA into base first via swift export:
    #     swift export --adapters /path/to/sft-out/checkpoint-final \\
    #         --merge_lora true --output_dir /path/to/sft-merged
    # Then point phase1_eval at the merged base:
    python -m scripts.phase1_eval --tracks 2,3 --prompts v1 --dataset diag_test \\
        --sft-merged-dir /path/to/sft-merged

    # final report (locked-best prompt) on official dev — burns a "look at
    # dev" budget, see D-006:
    python -m scripts.phase1_eval --tracks 1,2 --prompts v3 --dataset official_dev

Output structure::

    outputs/eval_phase1/
        track{N}_{prompt}_{dataset}.json   # raw predictions (N=1/2/3)
        track{N}_{prompt}_{dataset}.md     # per-bucket diagnostic table
        summary_{dataset}.md               # cross-(track,prompt) comparison

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
    track: int, prompt: str, dataset_label: str,
    preds: dict, gold: dict, tag_lookup: dict, elapsed: float,
) -> Path:
    overall = score_predictions(preds, gold)
    out_md = OUT_DIR / f"track{track}_{prompt}_{dataset_label}.md"
    parts = [
        f"# Track {track} — prompt {prompt} on {dataset_label}",
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

def write_summary(results: list[dict], dataset_label: str) -> Path:
    """One-table summary of all (track, prompt) combinations."""
    out_md = OUT_DIR / f"summary_{dataset_label}.md"
    lines = [
        f"# Phase 1 summary on {dataset_label}",
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

def build_pipeline(evidence: dict, *, final_k: int = 20, use_rerank: bool = False):
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
        if use_rerank:
            # Phase 3.5b audit (2026-05-12 PM) showed bge-reranker-base
            # hurts recall@5 on climate domain (-0.081, ×1.68 worse than
            # fused alone). Default is now use_rerank=False; pass
            # --rerank to opt back in for ablation.
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
        final_k=final_k,
    )
    print(f"  pipeline: bm25={bm25 is not None} dense={dense is not None} "
          f"rerank={reranker is not None} final_k={final_k}")
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
    # AutoModelForCausalLM resolves Qwen3.5-4B to Qwen3_5ForCausalLM
    # (text-only, structure model.layers.X). We tried AutoModelForImageTextToText
    # to preserve the VL wrapper for adapter compat, but it (a) broke Track 2
    # baseline (HM 0.203 → 0.133, 3x slowdown) and (b) still didn't load the
    # adapter properly (0.0M params). The right fix is to *merge* the LoRA into
    # the base via `swift export --merge_lora true` and load the merged dir as
    # a normal base model via --model-dir. See debug_log 复用经验 31.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, quantization_config=bnb_cfg, device_map="auto",
        trust_remote_code=True, torch_dtype=compute_dtype,
    )
    model.eval()
    return model, tokenizer


# -- Main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Phase 1 eval — Track 1/2/3 × prompt sweep")
    p.add_argument("--tracks", default="1,2",
                   help="Comma-separated track ids: "
                        "1 (no-RAG base), 2 (RAG base), "
                        "3 (RAG + SFT adapter, requires --sft-adapter). "
                        "Default 1,2.")
    p.add_argument("--prompts", default="v1",
                   help=f"Comma-separated prompt versions. Available: {','.join(PROMPT_VARIANTS)}.")
    p.add_argument("--dataset", default="diag_test",
                   choices=["diag_test", "dev_holdout", "official_dev"],
                   help="Eval set. diag_test is the safe default; official_dev consumes a 'look at dev' budget.")
    p.add_argument("--model-dir", default=None,
                   help="Local base-model snapshot. Omit to download from ModelScope.")
    p.add_argument("--sft-adapter", default=None,
                   help="Path to a LoRA SFT adapter (e.g. outputs/sft-out/"
                        "checkpoint-final). Wraps the base model via "
                        "peft.PeftModel.from_pretrained. ⚠️ For ms-swift "
                        "trained adapters on Qwen3.5 (VL wrapper) this path "
                        "currently breaks — use --sft-merged-dir instead "
                        "(see debug_log 复用经验 31). Kept for non-VL models.")
    p.add_argument("--sft-merged-dir", default=None,
                   help="Path to a swift-merged SFT base model (output of "
                        "`swift export --adapters <lora> --merge_lora true "
                        "--output_dir <merged>`). Loaded as a regular base "
                        "model — no peft / adapter type mismatch. Recommended "
                        "path for Qwen3.5 + ms-swift SFT.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap claims for quick smoke (e.g. --limit 30).")
    p.add_argument("--final-k", type=int, default=20,
                   help="Top-k evidences shown to the model in Track 2/3 RAG. "
                        "Default 20 (Phase 3.5 lock, see optimization_plan.md "
                        "§10). Use --final-k 5 to reproduce the pre-Phase-3.5 "
                        "baseline; outputs get a `_k5` filename suffix to "
                        "preserve the current production tables.")
    p.add_argument("--rerank", action="store_true",
                   help="Enable bge-reranker-base cross-encoder reordering. "
                        "DEFAULT IS OFF since Phase 3.5b audit (2026-05-12 PM) "
                        "showed it cuts recall@5 by ~×1.68 on climate domain "
                        "(debug_log 复用经验 35). Use this flag only for "
                        "ablation; output filenames get a `_rerank` suffix.")
    args = p.parse_args()

    tracks = [int(x) for x in args.tracks.split(",")]
    prompts = args.prompts.split(",")
    for v in prompts:
        if v not in PROMPT_VARIANTS:
            raise SystemExit(f"unknown prompt version: {v}; available: {list(PROMPT_VARIANTS)}")
    if 3 in tracks and not (args.sft_adapter or args.sft_merged_dir):
        raise SystemExit(
            "--tracks 3 requires --sft-adapter PATH (LoRA dir) or "
            "--sft-merged-dir PATH (swift-merged base dir).")
    if args.sft_adapter and args.sft_merged_dir:
        raise SystemExit(
            "--sft-adapter and --sft-merged-dir are mutually exclusive; pick one.")
    if (args.sft_adapter or args.sft_merged_dir) and 3 not in tracks:
        print(f"  WARN: SFT model path set but track 3 not in --tracks; "
              f"SFT model will load but go unused.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Phase 3.5 lock: production final_k is 20. When --final-k differs we
    # suffix output filenames so the production tables aren't clobbered.
    # diagnose_phase1.py strips the suffix when looking up the gold split.
    # Phase 3.5b: --rerank off by default; if user opts in, also suffix.
    dataset_label = args.dataset
    if args.final_k != 20:
        dataset_label = f"{dataset_label}_k{args.final_k}"
    if args.rerank:
        dataset_label = f"{dataset_label}_rerank"
    print(f"=== Phase 1 eval: tracks={tracks} prompts={prompts} "
          f"dataset={args.dataset} final_k={args.final_k} ===")
    if dataset_label != args.dataset:
        print(f"  outputs will be written as track*_*_{dataset_label}.* "
              f"(non-default final_k → suffixed to preserve baseline)")

    print("\n[1/4] loading dataset...")
    gold, tag_lookup = load_dataset(args.dataset)
    if args.limit:
        gold = dict(list(gold.items())[: args.limit])
        tag_lookup = {k: tag_lookup[k] for k in gold if k in tag_lookup}
    print(f"  {len(gold)} claims; {len(tag_lookup)} tagged")

    print("\n[2/4] loading base model + tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    sft_model = None
    if args.sft_adapter:
        from peft import PeftModel
        print(f"  loading SFT LoRA adapter from {args.sft_adapter}...")
        sft_model = PeftModel.from_pretrained(model, args.sft_adapter)
        sft_model.eval()
        # Count actual LoRA params (regardless of requires_grad — peft sets
        # it to False at eval-time load, so the "trainable" count is 0 even
        # when LoRA is correctly applied).
        n_lora_params = sum(
            p.numel() for n, p in sft_model.named_parameters() if "lora_" in n
        )
        print(f"  SFT adapter loaded ({n_lora_params / 1e6:.2f}M LoRA params).")
        if n_lora_params < 1e6:
            print(f"  ⚠️ LoRA params suspiciously low — see debug_log 复用经验 31. "
                  f"If Track 3 numbers match Track 2 exactly, the adapter "
                  f"isn't applied. Try --sft-merged-dir instead.")
    elif args.sft_merged_dir:
        print(f"\n[2b/4] loading SFT-merged base from {args.sft_merged_dir}...")
        sft_model, _ = load_model_and_tokenizer(args.sft_merged_dir)
        print(f"  SFT-merged base loaded as {type(sft_model).__name__}.")

    pipeline = None
    if 2 in tracks or 3 in tracks:
        print("\n[3/4] loading evidence corpus + RAG pipeline...")
        evidence = load_evidence(show_progress=True)
        pipeline = build_pipeline(evidence, final_k=args.final_k, use_rerank=args.rerank)
        print(f"  evidence: {len(evidence):,} passages")
        print(f"  RAG final_k = {args.final_k}, rerank = {args.rerank}")

    print("\n[4/4] running track × prompt sweep...")
    results: list[dict] = []
    for track in tracks:
        for prompt in prompts:
            tag = f"track{track}_{prompt}_{dataset_label}"
            print(f"\n--- {tag} ---")
            t0 = time.time()
            if track == 1:
                preds = run_track1(model, tokenizer, gold, prompt)
            elif track == 2:
                preds = run_track2(model, tokenizer, gold, prompt, pipeline)
            elif track == 3:
                # Same RAG pipeline + ZeroShotInferer as Track 2, but with the
                # SFT-adapted model. The PeftModel wraps base in-place, so we
                # don't need a fresh base copy.
                preds = run_track2(sft_model, tokenizer, gold, prompt, pipeline)
            else:
                raise SystemExit(f"unknown track: {track}")
            elapsed = time.time() - t0

            # Save raw predictions (eval.py compatible).
            json_path = OUT_DIR / f"{tag}.json"
            json_path.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")

            md_path = write_run_report(track, prompt, dataset_label, preds, gold, tag_lookup, elapsed)
            metrics = score_predictions(preds, gold)
            print(f"  → F={metrics['f_score']:.4f}  Acc={metrics['accuracy']:.4f}  "
                  f"HM={metrics['harmonic_mean']:.4f}  ({elapsed:.1f}s)")
            print(f"  → {md_path}")
            results.append({"track": track, "prompt": prompt, "metrics": metrics})

    summary_path = write_summary(results, dataset_label)
    print(f"\n=== Summary written to {summary_path} ===\n")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
