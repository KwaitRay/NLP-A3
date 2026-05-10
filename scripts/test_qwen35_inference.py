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
        --model-dir outputs/model_cache/Qwen/Qwen3___5-4B   # if pre-downloaded
        # OR omit --model-dir to auto-download from ModelScope

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
        from modelscope import snapshot_download
        print("  downloading from ModelScope...")
        model_dir = snapshot_download(
            "Qwen/Qwen3.5-4B",
            cache_dir=str(Path(__file__).resolve().parent.parent / "outputs" / "model_cache"),
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

    # 4c. Self-consistency sampling (Track 4 style)
    print(f"\n  --- 4c. Self-consistency sampling (n={n_samples}, T=0.7) ---")
    cid, claim = sample_claims[0]
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_query(claim, fake_evidences)},
    ]
    prompt_ids = _to_input_ids(tokenizer, msgs, model.device)
    print(f"\n  [{cid}] claim: {claim}")
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
        lbl, evs = parse_response(text, shown_evidence_ids=[e for e, _ in fake_evidences])
        samples.append((lbl, evs, text))
        print(f"    sample {i+1}: label={lbl}  evidences={evs}  raw={text!r}")
    dt = time.time() - t0
    from collections import Counter
    final_label = Counter(s[0] for s in samples).most_common(1)[0][0]
    print(f"\n    → majority label: {final_label}  ({dt:.2f}s total, {dt/n_samples:.2f}s/sample)")


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
    args = p.parse_args()

    dump_env()
    model, tokenizer = load_model(args.model_dir, quantize=not args.no_quantize)
    probe_tokenizer(tokenizer)
    run_inference(model, tokenizer, args.n_samples)
    _h("Done")
    print("  All four sections completed without hard error.")


if __name__ == "__main__":
    main()
