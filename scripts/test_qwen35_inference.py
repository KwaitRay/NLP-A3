"""Standalone Qwen3.5-4B inference smoke-test.

Designed to run on AutoDL / Colab GPU (not local Windows). Validates:

1. **Model loading** — QLoRA 4-bit base (matches SFT config), auto-detects
   bf16 (Ampere+) vs fp16 (Turing T4); prints VRAM after load.
2. **System prompt + tokenizer wiring** — confirms chat_template exists,
   probes whether `enable_thinking=False` is accepted by the template,
   and detects whether `apply_chat_template` returns a tensor or
   `BatchEncoding` (the transformers 5.x change that bit us in
   `src/inference.py`).
3. **Query construction** — uses the *actual* prompts from `src/prompt.py`
   (NO_RAG_SYSTEM_PROMPT / SYSTEM_PROMPT, build_no_rag_query /
   build_user_query) so this test mirrors the real Track 1-4 paths.
4. **Inference deployment** — runs greedy + sampled generation on three
   sample claims (no-RAG and RAG variants), then parses with
   `parse_response`. Prints raw output, parsed label, and per-claim
   latency.

Run::

    cd /root/Assignment3   # or wherever the repo lives
    python -m scripts.test_qwen35_inference \\
        --model-dir models/Qwen3.5-4B   # if pre-downloaded via scripts.download_models
        # OR omit --model-dir to auto-download from ModelScope into outputs/model_cache/

Exit code: 0 if all four sections complete; non-zero on any hard error.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Repo-relative imports work whether you run as a module or a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompt import (  # noqa: E402
    NO_RAG_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_no_rag_query,
    build_user_query,
    parse_response,
)


def _h(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def _kv(k: str, v) -> None:
    print(f"  {k:<32} {v}")


# --- Section 1: env --------------------------------------------------------

def dump_env() -> None:
    _h("1. Environment")
    import torch
    import transformers
    _kv("python", sys.version.split()[0])
    _kv("torch", torch.__version__)
    _kv("transformers", transformers.__version__)
    try:
        import peft; _kv("peft", peft.__version__)
    except ImportError:
        _kv("peft", "NOT INSTALLED")
    try:
        import bitsandbytes as bnb; _kv("bitsandbytes", bnb.__version__)
    except ImportError:
        _kv("bitsandbytes", "NOT INSTALLED (4-bit will fail)")
    try:
        import qwen_vl_utils; _kv("qwen_vl_utils", "installed (Qwen3.5 needs this)")
    except ImportError:
        _kv("qwen_vl_utils", "NOT INSTALLED — will likely warn at load time")
    try:
        import fla; _kv("flash-linear-attention", getattr(fla, "__version__", "installed"))
    except ImportError:
        _kv("flash-linear-attention", "NOT INSTALLED — GatedDeltaNet may be slow")

    _kv("cuda available", torch.cuda.is_available())
    if torch.cuda.is_available():
        _kv("device", torch.cuda.get_device_name(0))
        _kv("compute capability", torch.cuda.get_device_capability(0))
        _kv("bf16 supported", torch.cuda.is_bf16_supported())
        _kv("total VRAM (GB)", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))


# --- Section 2: model load -------------------------------------------------

def load_model(model_dir: str | None, quantize: bool):
    _h("2. Model load")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if model_dir is None:
        # Prefer pre-downloaded local copy under models/Qwen3.5-4B/
        repo_root = Path(__file__).resolve().parent.parent
        local = repo_root / "models" / "Qwen3.5-4B"
        if (local / "config.json").exists():
            model_dir = str(local)
            print(f"  using local copy: {model_dir}")
        else:
            from modelscope import snapshot_download
            print("  models/Qwen3.5-4B/ not found — downloading from ModelScope...")
            model_dir = snapshot_download(
                "Qwen/Qwen3.5-4B",
                cache_dir=str(repo_root / "outputs" / "model_cache"),
            )
    _kv("model dir", model_dir)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _kv("chosen dtype", compute_dtype)

    kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        _kv("quantization", "nf4 4-bit (QLoRA-style)")
    else:
        _kv("quantization", "none (full precision)")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    _kv("load time (s)", round(time.time() - t0, 1))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        _kv("VRAM after load (GB)", round(torch.cuda.memory_allocated() / 2**30, 2))
    return model, tokenizer


# --- Section 3: tokenizer probe -------------------------------------------

def probe_tokenizer(tokenizer):
    _h("3. Tokenizer / chat-template probe")
    import torch
    _kv("type", type(tokenizer).__name__)
    _kv("chat_template is None?", tokenizer.chat_template is None)
    _kv("pad_token_id", tokenizer.pad_token_id)
    _kv("eos_token_id", tokenizer.eos_token_id)

    msgs = [{"role": "user", "content": "hello"}]
    # 3a. With enable_thinking
    try:
        out = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=False,
        )
        ret_type = type(out).__name__
        _kv("apply_chat_template return type", ret_type)
        if torch.is_tensor(out):
            _kv("  → shape", tuple(out.shape))
        elif hasattr(out, "keys"):
            _kv("  → keys", list(out.keys()))
            _kv("  → input_ids shape", tuple(out["input_ids"].shape))
        _kv("enable_thinking=False accepted?", "yes")
    except Exception as e:
        _kv("enable_thinking=False accepted?", f"NO — {type(e).__name__}: {e}")
        _kv("  → falling back without enable_thinking for the rest of the test", "")
        return False
    return True


# --- Section 4: inference --------------------------------------------------

def _to_input_ids(tokenizer, msgs, device):
    """Mirror the helper in src/inference.py — handle BatchEncoding vs Tensor."""
    import torch
    encoded = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = encoded if torch.is_tensor(encoded) else encoded["input_ids"]
    return ids.to(device)


def run_inference(model, tokenizer, n_samples: int):
    _h("4. Inference deployment")

    sample_claims = [
        ("c-supports", "Global temperatures have risen by approximately 1°C since 1880."),
        ("c-refutes",  "There has been no warming of the global atmosphere since 1998."),
        ("c-nei",      "Vanilla ice cream consumption causes glacial melt in the Alps."),
    ]

    fake_evidences = [
        ("ev-1", "NASA records show global mean surface temperature has increased by about 1.1 degrees Celsius since the late 19th century."),
        ("ev-2", "The 2010s decade was the warmest on record, with each successive year ranking among the top warmest globally."),
        ("ev-3", "She made guest appearances at the Edinburgh Festival in 1957 and recorded several solo albums in the 1960s."),
    ]

    import torch

    # 4a. NO-RAG (Track 1 style)
    print("\n  --- 4a. No-RAG (Track 1 style) ---")
    for cid, claim in sample_claims:
        msgs = [
            {"role": "system", "content": NO_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": build_no_rag_query(claim)},
        ]
        prompt_ids = _to_input_ids(tokenizer, msgs, model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                prompt_ids, do_sample=False, max_new_tokens=24,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        dt = time.time() - t0
        text = tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        label, _ = parse_response(text, shown_evidence_ids=[])
        print(f"\n  [{cid}] claim: {claim}")
        print(f"    raw:    {text!r}")
        print(f"    parsed: label={label}  ({dt:.2f}s)")

    # 4b. RAG (Track 2/3 style)
    print("\n  --- 4b. With RAG evidences (Track 2/3 style, greedy) ---")
    for cid, claim in sample_claims:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_query(claim, fake_evidences)},
        ]
        prompt_ids = _to_input_ids(tokenizer, msgs, model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                prompt_ids, do_sample=False, max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        dt = time.time() - t0
        text = tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        label, ev_ids = parse_response(text, shown_evidence_ids=[e for e, _ in fake_evidences])
        print(f"\n  [{cid}] claim: {claim}")
        print(f"    raw:    {text!r}")
        print(f"    parsed: label={label}  evidences={ev_ids}  ({dt:.2f}s)")

    # 4c. Self-consistency sampling (Track 4 style) — DISPUTED claim + mixed
    # evidence. The earlier version used an easy SUPPORTS claim where 5/5
    # samples agreed (T=0.7 didn't perturb a confident model), so SC was
    # trivially "useful". This version stress-tests SC on a genuinely
    # ambiguous case where evidence is split: ev-A supports, ev-B refutes.
    # We *want* to see disagreement across the 5 samples; if they all still
    # agree, SC adds no value on this claim either.
    print(f"\n  --- 4c. Self-consistency on DISPUTED claim (n={n_samples}, T=0.7) ---")
    disputed_claim = "Cloud feedback significantly amplifies global warming."
    mixed_evidences = [
        ("ev-pro", "Climate model intercomparison projects (CMIP6) generally show net positive cloud feedback that amplifies warming by 0.3 to 0.8 K per doubling of CO2."),
        ("ev-con", "Recent satellite-based observational studies (Ceppi 2024) suggest the cloud feedback may be smaller and more uncertain than CMIP6 models predict."),
        ("ev-off", "Eurasian dust transport patterns have been studied via Lidar networks since the 1990s."),
    ]
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_query(disputed_claim, mixed_evidences)},
    ]
    prompt_ids = _to_input_ids(tokenizer, msgs, model.device)
    print(f"\n  [c-disputed] claim: {disputed_claim}")
    print(f"  evidences: ev-pro (supports) / ev-con (refutes) / ev-off (irrelevant)")
    t0 = time.time()
    samples = []
    for i in range(n_samples):
        with torch.no_grad():
            out = model.generate(
                prompt_ids, do_sample=True, temperature=0.7, top_p=0.9,
                max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        lbl, evs = parse_response(text, shown_evidence_ids=[e for e, _ in mixed_evidences])
        samples.append((lbl, evs, text))
        print(f"    sample {i+1}: label={lbl}  evidences={evs}  raw={text!r}")
    dt = time.time() - t0
    from collections import Counter
    counts = Counter(s[0] for s in samples)
    final_label = counts.most_common(1)[0][0]
    print(f"\n    → majority label: {final_label}  (vote distribution: {dict(counts)})")
    print(f"    → {dt:.2f}s total, {dt/n_samples:.2f}s/sample")
    if len(counts) == 1:
        print("    NOTE: 5/5 agreement — SC adds no value on this claim. Try a harder one.")
    else:
        print(f"    NOTE: {len(counts)} distinct labels emerged → SC has value here.")


def run_real_rag(model, tokenizer, n_claims: int = 3) -> None:
    """4d. End-to-end Track-2 path: real BM25(+dense+rerank) → model.

    Gated on cached indices existing on disk. Keeps the smoke test fast for
    the common case (model + prompt validation only); only opt in via
    --with-real-rag when you've already built the indices on this box.
    """
    _h("4d. Real RAG pipeline (Track 2 end-to-end)")
    repo_root = Path(__file__).resolve().parent.parent
    bm25_dir  = repo_root / "outputs" / "bm25_index"
    dense_dir = repo_root / "outputs" / "dense_index"
    evidence_path = repo_root / "data" / "evidence.json"

    missing = []
    if not evidence_path.exists():
        missing.append(f"evidence corpus ({evidence_path})")
    if not bm25_dir.exists():
        missing.append(f"BM25 index ({bm25_dir})")
    if missing:
        print("  SKIPPED — required artifacts missing:")
        for m in missing:
            print(f"    - {m}")
        print("  Build them first via the notebook cells 2.1 (BM25) and optionally 2.2 (dense),")
        print("  or copy a pre-built index into outputs/. Re-run with --with-real-rag once ready.")
        return

    print(f"  loading evidence corpus from {evidence_path} ...")
    from src.data_io import load_evidence, load_dev
    evidence = load_evidence(evidence_path, show_progress=True)
    print(f"  evidence: {len(evidence):,} passages")

    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.pipeline import RetrievalPipeline, RetrievalConfig
    bm25 = BM25Retriever.load(bm25_dir)
    print(f"  BM25 loaded from {bm25_dir}")

    dense = None
    reranker = None
    if dense_dir.exists() and (dense_dir / "faiss.index").exists():
        try:
            from src.retrieval.dense import DenseRetriever
            dense = DenseRetriever.load(dense_dir, max_seq_length=256, fp16=True)
            print(f"  dense loaded from {dense_dir}")
            from src.retrieval.rerank import CrossEncoderReranker
            reranker = CrossEncoderReranker()
            print(f"  cross-encoder reranker loaded")
        except Exception as e:
            print(f"  dense/reranker load failed ({type(e).__name__}: {e}); using BM25 only")

    cfg = RetrievalConfig(
        use_bm25=True, use_dense=dense is not None,
        use_rerank=reranker is not None,
        use_rule_reorder=False,  # rule_reorder needs spaCy; skip in smoke test
        final_k=5,
    )
    pipe = RetrievalPipeline(
        evidence_corpus=evidence, bm25=bm25, dense=dense, reranker=reranker, cfg=cfg,
    )

    dev = load_dev()
    sample_ids = list(dev.keys())[:n_claims]
    print(f"\n  running {len(sample_ids)} dev claims through Track 2 (RAG → model greedy)\n")

    import torch
    for cid in sample_ids:
        claim = dev[cid]
        gold_label = claim["claim_label"]
        gold_evs = set(claim["evidences"])
        t_ret = time.time()
        retrieved = pipe.retrieve(claim["claim_text"])
        ret_dt = time.time() - t_ret
        shown_ids = [eid for eid, _ in retrieved]
        hits = sum(1 for eid in shown_ids if eid in gold_evs)

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_query(claim["claim_text"], retrieved)},
        ]
        prompt_ids = _to_input_ids(tokenizer, msgs, model.device)
        t_gen = time.time()
        with torch.no_grad():
            out = model.generate(
                prompt_ids, do_sample=False, max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_dt = time.time() - t_gen
        text = tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        pred_label, pred_evs = parse_response(text, shown_evidence_ids=shown_ids)
        label_ok = "✓" if pred_label == gold_label else "✗"
        print(f"  [{cid}] claim: {claim['claim_text'][:80]}")
        print(f"    gold:     {gold_label}  evidences={list(gold_evs)[:3]}{'...' if len(gold_evs)>3 else ''}")
        print(f"    retrieved {len(shown_ids)} (gold-hit {hits}/{len(gold_evs)}, {ret_dt:.2f}s)")
        print(f"    raw:      {text!r}")
        print(f"    pred:     {label_ok} {pred_label}  evidences={pred_evs}  ({gen_dt:.2f}s)\n")


# --- main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Qwen3.5-4B inference smoke test")
    p.add_argument(
        "--model-dir", default=None,
        help="Local path to model snapshot. Omit to auto-download from ModelScope.",
    )
    p.add_argument(
        "--no-quantize", action="store_true",
        help="Skip 4-bit quantization (use only on >=24 GB GPUs).",
    )
    p.add_argument("--n-samples", type=int, default=5, help="Self-consistency sample count.")
    p.add_argument(
        "--with-real-rag", action="store_true",
        help="Run section 4d (real BM25+dense+rerank pipeline). Requires "
             "outputs/bm25_index/ (and optionally outputs/dense_index/) to exist "
             "and data/evidence.json to be downloaded.",
    )
    p.add_argument(
        "--rag-claims", type=int, default=3,
        help="Number of dev claims to run through real RAG (only with --with-real-rag).",
    )
    args = p.parse_args()

    dump_env()
    model, tokenizer = load_model(args.model_dir, quantize=not args.no_quantize)
    probe_tokenizer(tokenizer)
    run_inference(model, tokenizer, args.n_samples)
    if args.with_real_rag:
        run_real_rag(model, tokenizer, n_claims=args.rag_claims)
    _h("Done")
    print("  All sections completed without hard error.")


if __name__ == "__main__":
    main()
