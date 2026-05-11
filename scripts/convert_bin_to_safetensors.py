"""Convert pytorch_model.bin → model.safetensors under models/*/

Why this exists:
    transformers' recent CVE-2025-32434 mitigation refuses to load via
    `torch.load` unless torch >= 2.6. AutoDL is pinned at torch 2.5.1+cu124
    for flash-attn / bitsandbytes / flash-linear-attention compatibility
    (see debug_log.md Issue 13), so upgrading torch is high-risk. The
    ModelScope mirror for `BAAI/bge-m3` (and likely the smaller bge models)
    serves only `pytorch_model.bin` without a `.safetensors` counterpart,
    which trips the safety check inside SentenceTransformer.load.

    `torch.load` itself works fine in 2.5 — the restriction is only in
    transformers' wrapper. Calling `torch.load` directly from user code is
    permitted, so we can convert offline once and let sentence-transformers
    pick up the safetensors version on subsequent loads.

Behavior:
    - Walks `models/*/` subdirectories.
    - For each that contains exactly `pytorch_model.bin` (single-file layout)
      AND no existing `*.safetensors`, converts in place.
    - Renames the original `.bin` to `.bin.bak` so users can roll back.
    - Skips directories already carrying safetensors (e.g. Qwen3.5-4B).
    - Skips sharded `pytorch_model-*-of-*.bin` layouts with a warning —
      converting those correctly also requires regenerating the index.json,
      which is fragile; redownload via a safetensors-aware fetcher instead.

Run::

    python -m scripts.convert_bin_to_safetensors
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _any(glob_iter) -> bool:
    return next(iter(glob_iter), None) is not None


def has_safetensors(d: Path) -> bool:
    return _any(d.glob("*.safetensors"))


def has_sharded_bin(d: Path) -> bool:
    return _any(d.glob("pytorch_model-*-of-*.bin"))


def convert_single(model_dir: Path) -> None:
    src = model_dir / "pytorch_model.bin"
    dst = model_dir / "model.safetensors"
    bak = model_dir / "pytorch_model.bin.bak"

    print(f"  [load] {src}  ({src.stat().st_size / 1e9:.2f} GB)", flush=True)
    sd = torch.load(src, map_location="cpu", weights_only=False)
    # safetensors requires contiguous, non-shared-storage tensors.
    sd = {k: v.contiguous().clone() for k, v in sd.items()}

    print(f"  [save] {len(sd)} tensors → {dst}", flush=True)
    save_file(sd, dst)

    print(
        f"  [done] {dst.name} = {dst.stat().st_size / 1e9:.2f} GB; "
        f"renaming .bin → .bin.bak",
        flush=True,
    )
    src.rename(bak)


def main() -> int:
    if not MODELS_DIR.exists():
        print(f"models/ missing at {MODELS_DIR}", file=sys.stderr)
        return 1

    n_converted = 0
    n_skipped_safetensors = 0
    n_skipped_nobin = 0
    n_sharded = 0

    for d in sorted(p for p in MODELS_DIR.iterdir() if p.is_dir()):
        print(f"\n=== {d.name} ===")
        if has_sharded_bin(d) and not has_safetensors(d):
            print(
                "  [warn] sharded pytorch_model-*-of-*.bin layout — index "
                "regeneration is fragile; redownload via a safetensors-aware "
                "fetcher (HF_ENDPOINT=https://hf-mirror.com) instead."
            )
            n_sharded += 1
            continue
        if has_safetensors(d):
            print("  [skip] *.safetensors already present")
            n_skipped_safetensors += 1
            continue
        if not (d / "pytorch_model.bin").exists():
            print("  [skip] no pytorch_model.bin in this directory")
            n_skipped_nobin += 1
            continue
        convert_single(d)
        n_converted += 1

    print(
        f"\nSummary: converted={n_converted}, "
        f"skipped-already-safetensors={n_skipped_safetensors}, "
        f"skipped-no-bin={n_skipped_nobin}, sharded-warn={n_sharded}"
    )
    if n_converted:
        print(
            "Tip: once you confirm retrieval works end-to-end, "
            "`find models -name '*.bin.bak' -delete` to reclaim disk."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
