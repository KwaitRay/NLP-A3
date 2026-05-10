"""Standalone retrieval index builder (BM25 + dense bge-m3).

Same logic as notebook cells 2.1 / 2.2 but runnable from the CLI, so the
indexes can be built once on AutoDL/local and the notebook just consumes
the cache. The notebook cells already detect cached indexes and skip
build, so running this script first makes the notebook execution path
"load only".

Usage::

    # build both (skips already-cached)
    python -m scripts.build_indexes

    # only BM25 (faster — ~3 min) for an initial smoke test
    python -m scripts.build_indexes --skip-dense

    # use lighter dense model (bge-small-en-v1.5, 33M, 384-d) when bge-m3 OOMs
    python -m scripts.build_indexes --light-dense

    # rebuild even if cache exists
    python -m scripts.build_indexes --force

Outputs (gitignored)::

    outputs/bm25_index/   ~200 MB
    outputs/dense_index/  ~5 GB (bge-m3) or ~1.6 GB (bge-small)

Hardware notes:
- BM25 build is CPU-only, ~3 min on any modern box, ~1.5 GB RAM peak.
- Dense build needs GPU. Defaults are T4-safe (fp16, max_seq=256, bs=32).
  On 4080/A100 you can crank batch_size to 64-128 for ~2x speedup.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import load_evidence  # noqa: E402
from src.paths import EVIDENCE_JSON, OUTPUTS_DIR  # noqa: E402


def _h(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def _kv(k: str, v) -> None:
    print(f"  {k:<28} {v}")


def _du(path: Path) -> str:
    """Human-readable directory size."""
    if not path.exists():
        return "(missing)"
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def build_bm25(evidence: dict[str, str], force: bool) -> None:
    _h("BM25 (sparse)")
    save_dir = OUTPUTS_DIR / "bm25_index"
    if save_dir.exists() and not force:
        _kv("status", f"cached at {save_dir} ({_du(save_dir)}) — skip (use --force to rebuild)")
        return

    from src.retrieval.bm25 import BM25Retriever
    if force and save_dir.exists():
        import shutil
        shutil.rmtree(save_dir)
        _kv("note", "removed existing cache (--force)")

    bm25 = BM25Retriever()
    t0 = time.time()
    bm25.build(evidence, save_dir=save_dir)
    _kv("built", f"{time.time() - t0:.1f}s → {save_dir} ({_du(save_dir)})")


def build_dense(evidence: dict[str, str], *, force: bool, light: bool, batch_size: int) -> None:
    _h("Dense (bge-m3)" if not light else "Dense (bge-small-en-v1.5, light)")
    save_dir = OUTPUTS_DIR / "dense_index"
    if (save_dir / "faiss.index").exists() and not force:
        _kv("status", f"cached at {save_dir} ({_du(save_dir)}) — skip (use --force to rebuild)")
        return

    from src.retrieval.dense import DenseRetriever, DEFAULT_MODEL, LIGHT_MODEL
    model_name = LIGHT_MODEL if light else DEFAULT_MODEL
    _kv("model", model_name)
    _kv("batch_size", batch_size)
    _kv("max_seq_length", 256)
    _kv("fp16", True)

    if force and save_dir.exists():
        import shutil
        shutil.rmtree(save_dir)
        _kv("note", "removed existing cache (--force)")

    dense = DenseRetriever(model_name=model_name, max_seq_length=256, fp16=True)
    t0 = time.time()
    dense.build(evidence, save_dir=save_dir, batch_size=batch_size)
    _kv("built", f"{time.time() - t0:.1f}s → {save_dir} ({_du(save_dir)})")


def main():
    p = argparse.ArgumentParser(description="Build BM25 + dense retrieval indexes")
    p.add_argument("--skip-bm25", action="store_true", help="Skip BM25 build")
    p.add_argument("--skip-dense", action="store_true", help="Skip dense build (fast smoke test)")
    p.add_argument("--light-dense", action="store_true",
                   help="Use bge-small-en-v1.5 (33M, 384-d) instead of bge-m3 (568M, 1024-d)")
    p.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Dense encoder batch size (T4: 32, 4080+: 64-128)")
    args = p.parse_args()

    _h("Environment")
    if not EVIDENCE_JSON.exists():
        print(f"  ERROR: evidence corpus not found at {EVIDENCE_JSON}")
        print("  Download per data/README and rerun.")
        sys.exit(1)

    _kv("evidence", f"{EVIDENCE_JSON} ({_du(EVIDENCE_JSON.parent / EVIDENCE_JSON.name)})")
    _kv("outputs", OUTPUTS_DIR)

    t0 = time.time()
    print("\n  loading evidence corpus (174 MB JSON, ~30s on first call) ...")
    evidence = load_evidence(show_progress=True)
    _kv("evidence loaded", f"{len(evidence):,} passages ({time.time() - t0:.1f}s)")

    if not args.skip_bm25:
        build_bm25(evidence, force=args.force)
    else:
        _h("BM25 skipped (--skip-bm25)")

    if not args.skip_dense:
        build_dense(evidence, force=args.force, light=args.light_dense, batch_size=args.batch_size)
    else:
        _h("Dense skipped (--skip-dense)")

    _h("Done")
    print(f"  total: {time.time() - t0:.1f}s")
    print("  notebook cells 2.1/2.2 will now skip build and load these caches directly.")


if __name__ == "__main__":
    main()
