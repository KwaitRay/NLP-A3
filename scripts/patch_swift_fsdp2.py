"""Patch ms-swift's FSDP2 import so the library works on torch < 2.6.

Why this exists
---------------
ms-swift 4.2.0 introduced an `ActivationCpuOffloadCallBack` callback that
imports `torch.distributed.fsdp.FSDPModule` (a torch 2.6+ API). The
import sits at module-load time in
`swift/callbacks/activation_cpu_offload.py`, so even running
`swift sft` (which never uses that callback) explodes with::

    ImportError: cannot import name 'FSDPModule' from
    'torch.distributed.fsdp' (/.../torch/distributed/fsdp/__init__.py)

AutoDL pins us to torch 2.5.1+cu124 because flash-attn 2.x /
bitsandbytes / flash-linear-attention all want that combination
(debug_log Issue 13). Upgrading torch to 2.6+ is high-blast-radius;
downgrading ms-swift drops the `--tuner_type` / thinking-trio flags we
just spent a session getting right (debug_log Issue 9 / 复用经验 12).

Fix
---
Wrap that one import in `try/except ImportError` with a stub class for
the offending name. The callback never runs in our training path (we
don't pass any FSDP flag), so the stub is never instantiated — purely
defensive.

The same fix is applied to any other ms-swift modules that import
`FSDPModule` at module-load time. As of ms-swift 4.2.0 there's only
the one, but we scan generically.

Idempotent: re-running on an already-patched install is a no-op.

Run
---
    python -m scripts.patch_swift_fsdp2

Exit codes:
    0 — patched (or already patched / no patch needed)
    1 — couldn't locate swift package, or patch pattern not found and
        un-patched (suggests ms-swift version drift; manual inspection)
"""
from __future__ import annotations

import sys
from pathlib import Path


PATCH_NEW = """\
try:
    from torch.distributed.fsdp import FSDPModule as FSDP2
except ImportError:
    # torch < 2.6 fallback. The class is only used by
    # ActivationCpuOffloadCallBack, which our training command never
    # activates — see scripts/patch_swift_fsdp2.py for the rationale.
    class FSDP2:  # type: ignore[no-redef]
        pass\
"""

PATCH_OLD = "from torch.distributed.fsdp import FSDPModule as FSDP2"

# Sentinel comment dropped by an already-patched file so we don't re-wrap.
SENTINEL = "torch < 2.6 fallback. The class is only used by"


def find_swift_package() -> Path | None:
    """Locate the installed swift package directory."""
    try:
        import swift  # noqa: F401
    except ImportError:
        print("error: ms-swift not installed in this environment")
        return None
    swift_dir = Path(sys.modules["swift"].__file__).resolve().parent
    if not swift_dir.is_dir():
        print(f"error: swift package path is not a directory: {swift_dir}")
        return None
    return swift_dir


def patch_file(path: Path) -> tuple[bool, str]:
    """Patch one file. Returns (was_modified, status_message)."""
    try:
        src = path.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"read error: {e}"
    if SENTINEL in src:
        return False, "already patched (sentinel found)"
    if PATCH_OLD not in src:
        return False, "pattern not present"
    new_src = src.replace(PATCH_OLD, PATCH_NEW, 1)
    path.write_text(new_src, encoding="utf-8")
    return True, "patched"


def main() -> int:
    swift_dir = find_swift_package()
    if swift_dir is None:
        return 1
    print(f"[scan] swift package at {swift_dir}")

    # Generic scan: any .py file that imports FSDPModule at module load.
    candidates: list[Path] = []
    for path in swift_dir.rglob("*.py"):
        try:
            head = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if PATCH_OLD in head and SENTINEL not in head:
            candidates.append(path)

    if not candidates:
        # Either already-patched everywhere, or the offending import is
        # gone (e.g. ms-swift fixed upstream). Both are fine.
        print("[ok] no unpatched FSDPModule imports found — nothing to do")
        return 0

    print(f"[scan] {len(candidates)} file(s) to patch:")
    for p in candidates:
        rel = p.relative_to(swift_dir)
        modified, status = patch_file(p)
        marker = "✓" if modified else "·"
        print(f"  {marker} {rel}  — {status}")

    # Sanity-check by attempting to import the most common offender.
    try:
        import importlib
        # Force a fresh import in case swift.callbacks was imported earlier
        # in this process with the un-patched form.
        for name in list(sys.modules):
            if name == "swift.callbacks" or name.startswith("swift.callbacks."):
                del sys.modules[name]
        importlib.import_module("swift.callbacks")
        print("[verify] swift.callbacks now imports cleanly")
    except Exception as e:
        print(f"[verify] still failing: {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
