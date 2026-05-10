"""Pre-download all third-party model weights to ``models/<basename>/``.

Why this exists
---------------
We use four pretrained models across the project:

  - Qwen3.5-4B           (~8 GB)   SFT base model
  - bge-m3               (~2 GB)   dense retriever
  - bge-reranker-base    (~1 GB)   cross-encoder reranker
  - bge-small-en-v1.5    (~130 MB) lightweight dense fallback (optional)

By default sentence-transformers / transformers cache them under
``~/.cache/huggingface/`` (or ModelScope's home cache). That works but
makes them invisible to the project — you can't ``ls models/`` and see
what's pulled, and switching machines means re-downloading.

This script downloads all four into one place — ``<project_root>/models/`` —
so they're co-located with the code, easy to inspect / `du -sh`, and
trivially `scp`-able to another box. ``src/paths.py:resolve_model_path()``
makes ``DenseRetriever`` and ``CrossEncoderReranker`` prefer this local
copy automatically (falls back to network fetch if missing).

Usage
-----

    # download everything (skips already-present models)  ~11 GB total
    python -m scripts.download_models

    # only the lightweight ones (no Qwen)  ~3 GB
    python -m scripts.download_models --only bge-m3,bge-reranker-base

    # only Qwen
    python -m scripts.download_models --only qwen

    # skip Qwen (useful when you already have it via test_qwen35_inference)
    python -m scripts.download_models --skip qwen

    # force re-download
    python -m scripts.download_models --only bge-m3 --force

    # use HuggingFace instead of ModelScope (e.g. outside China)
    python -m scripts.download_models --source hf

OOM / disk notes
----------------
Downloads themselves don't consume RAM (streamed to disk). On Windows
local boxes, the only constraint is **disk space**:
  - Qwen3.5-4B alone: ~8 GB
  - All four: ~11 GB

If you don't have space locally, run this on AutoDL / Colab and
optionally `scp` the result back. Loading the models for inference
DOES use VRAM — that's a separate concern handled in the eval scripts.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import MODELS_DIR  # noqa: E402


# Registry: CLI short_name → spec.
# `folder` is the on-disk folder name under models/ — MUST match the repo
# basename (after the slash) so paths.resolve_model_path() can find the
# local copy by stripping the org prefix from the HF repo_id.
MODELS: dict[str, dict[str, str]] = {
    "qwen": {
        "folder": "Qwen3.5-4B",            # matches resolve_model_path('Qwen/Qwen3.5-4B')
        "ms_id":  "Qwen/Qwen3.5-4B",
        "hf_id":  "Qwen/Qwen3.5-4B",
        "size":   "~8 GB",
        "purpose": "SFT base model (mixed-thinking VL, used text-only)",
    },
    "bge-m3": {
        "folder": "bge-m3",
        "ms_id":  "BAAI/bge-m3",
        "hf_id":  "BAAI/bge-m3",
        "size":   "~2 GB",
        "purpose": "Dense retriever (1024-d, multilingual)",
    },
    "bge-reranker-base": {
        "folder": "bge-reranker-base",
        "ms_id":  "Xorbits/bge-reranker-base",   # ModelScope mirror
        "hf_id":  "BAAI/bge-reranker-base",
        "size":   "~1 GB",
        "purpose": "Cross-encoder reranker",
    },
    "bge-small-en-v1.5": {
        "folder": "bge-small-en-v1.5",
        "ms_id":  "Xorbits/bge-small-en-v1.5",
        "hf_id":  "BAAI/bge-small-en-v1.5",
        "size":   "~130 MB",
        "purpose": "Lightweight dense fallback (33M, 384-d)",
    },
}


def _h(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def _kv(k: str, v) -> None:
    print(f"  {k:<24} {v}")


def _du(path: Path) -> str:
    if not path.exists():
        return "(missing)"
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def _is_complete(target: Path) -> bool:
    """A directory counts as 'downloaded' if it has config.json + at least
    one weights file (.safetensors / .bin)."""
    if not target.exists():
        return False
    if not (target / "config.json").exists():
        return False
    has_weights = any(target.glob("*.safetensors")) or any(target.glob("*.bin"))
    return has_weights


def _download_via_modelscope(repo_id: str, target: Path) -> None:
    """ModelScope's snapshot_download stores under cache_dir/<owner>/<name>/.
    We move the contents into ``target`` so the layout is uniform with HF."""
    from modelscope import snapshot_download
    # Stage into a temp cache dir under target's parent so we know what to move.
    tmp_root = target.parent / "_ms_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    actual_dir = snapshot_download(repo_id, cache_dir=str(tmp_root))
    # Move contents up to target.
    target.mkdir(parents=True, exist_ok=True)
    for child in Path(actual_dir).iterdir():
        dest = target / child.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(child), str(dest))
    # Clean up the modelscope intermediate dirs.
    shutil.rmtree(tmp_root, ignore_errors=True)


def _download_via_hf(repo_id: str, target: Path) -> None:
    """HuggingFace snapshot_download with explicit local_dir → no cache layout."""
    from huggingface_hub import snapshot_download as hf_snapshot
    target.mkdir(parents=True, exist_ok=True)
    hf_snapshot(repo_id=repo_id, local_dir=str(target), local_dir_use_symlinks=False)


def download_one(short_name: str, source: str, force: bool) -> bool:
    info = MODELS[short_name]
    target = MODELS_DIR / info["folder"]
    _h(f"{short_name}  ({info['size']})")
    _kv("purpose", info["purpose"])
    _kv("target", target)

    if _is_complete(target) and not force:
        _kv("status", f"already present ({_du(target)}) — skip (use --force to redownload)")
        return True

    if force and target.exists():
        _kv("note", "removing existing dir (--force)")
        shutil.rmtree(target)

    repo_id = info["ms_id"] if source == "modelscope" else info["hf_id"]
    _kv("source", f"{source} ({repo_id})")

    t0 = time.time()
    try:
        if source == "modelscope":
            _download_via_modelscope(repo_id, target)
        else:
            _download_via_hf(repo_id, target)
    except Exception as e:
        _kv("ERROR", f"{type(e).__name__}: {e}")
        # If ModelScope failed, suggest HF fallback (and vice versa)
        other = "hf" if source == "modelscope" else "modelscope"
        print(f"  hint: try --source {other} for this model")
        return False

    elapsed = time.time() - t0
    if not _is_complete(target):
        _kv("ERROR", "download finished but config.json or weights missing")
        return False
    _kv("done", f"{elapsed:.1f}s → {_du(target)}")
    return True


def main():
    p = argparse.ArgumentParser(description="Download all project model weights to models/")
    p.add_argument("--only", default=None,
                   help=f"Comma-separated subset to download. Available: {','.join(MODELS)}")
    p.add_argument("--skip", default=None,
                   help="Comma-separated subset to skip.")
    p.add_argument("--source", choices=["modelscope", "hf", "auto"], default="auto",
                   help="Download source. 'auto' = ModelScope for Qwen, HF for the rest "
                        "(empirically faster combo from a CN box).")
    p.add_argument("--force", action="store_true", help="Re-download even if present")
    args = p.parse_args()

    selected = list(MODELS) if not args.only else args.only.split(",")
    if args.skip:
        skip = set(args.skip.split(","))
        selected = [s for s in selected if s not in skip]
    for s in selected:
        if s not in MODELS:
            raise SystemExit(f"unknown model: {s}; available: {list(MODELS)}")

    _h("Plan")
    _kv("target dir", MODELS_DIR)
    _kv("source", args.source)
    _kv("force", args.force)
    print(f"  models to fetch: {selected}")
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True)
        print(f"  (created {MODELS_DIR})")

    results = {}
    t_total = time.time()
    for short_name in selected:
        # 'auto' source: ModelScope for Qwen (CN-fast), HF for the bge-* trio
        # (ModelScope mirrors are inconsistent for bge models).
        if args.source == "auto":
            src = "modelscope" if short_name == "qwen" else "hf"
        else:
            src = args.source
        results[short_name] = download_one(short_name, src, args.force)

    _h("Summary")
    for short_name, ok in results.items():
        target = MODELS_DIR / MODELS[short_name]["folder"]
        status = "[OK]" if ok else "[FAIL]"
        size = _du(target) if target.exists() else "(none)"
        print(f"  {status} {short_name:<22} {size}  -> {target}")
    print(f"\n  total time: {time.time() - t_total:.1f}s")
    print(f"  total disk: {_du(MODELS_DIR)}")
    print(f"\n  src/paths.py:resolve_model_path() will now route DenseRetriever /")
    print(f"  CrossEncoderReranker / Qwen-loading code to these local copies.")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
