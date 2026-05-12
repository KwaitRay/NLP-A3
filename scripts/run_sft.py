"""Standalone SFT training entry — bypasses JupyterLab.

Why this exists
---------------
JupyterLab's kernel + autosave + browser-state combo is fragile for
long-running training:
  - Browser tab idles → kernel disconnects after a while
  - Autosave + git pull deadlock (debug_log 复用经验 29)
  - Kernel hangs / runs out of memory leaving zombie state
  - DataLoader workers crashing (复用经验 33) leave the notebook cell
    in a half-broken state

This script is the same `swift sft` invocation the notebook builds,
but runs from a plain terminal (or `tmux` / `nohup`) so we don't need
notebook state to train.

What this does
--------------
1. Auto-detects GPU VRAM and picks `--per_device_train_batch_size` /
   `--gradient_accumulation_steps` / `--quant_bits` (mirrors notebook
   autodl-setup-gpu logic).
2. Resolves model dir via cache-first (`models/Qwen3.5-4B/` if present,
   otherwise ModelScope snapshot_download).
3. Builds the full swift sft command with the locked resilience config
   (`--save_steps 50`, `--dataloader_num_workers 0`, thinking trio,
   liger, group_by_length, save_total_limit=3 — debug_log 复用经验 33).
4. Conditionally adds `--resume_from_checkpoint` only if the target
   path exists (avoids the ms-swift first-run crash, 复用经验 27).
5. Prints the command before running so the user can sanity-check;
   `--dry-run` to only print without launching.

Recommended invocation for long runs
------------------------------------

.. code-block:: bash

    tmux new -s sft
    source /etc/network_turbo
    python -m scripts.run_sft 2>&1 | tee outputs/sft_train.log
    # Ctrl+B then D to detach; training keeps running.
    # Reattach with: tmux attach -t sft

CLI reference
-------------
See `python -m scripts.run_sft --help` for the full flag list.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT, SFT_DIR  # noqa: E402


# -- Default paths ---------------------------------------------------------

DEFAULT_MODEL_DIR = MODELS_DIR / "Qwen3.5-4B"
DEFAULT_DATA = SFT_DIR / "sft_train_v2.jsonl"
DEFAULT_VAL_DATA = SFT_DIR / "sft_dev_holdout_v2.jsonl"

# On AutoDL we point output at the NVMe data disk; locally fall back to
# outputs/sft-out so dry-runs work without /root/autodl-tmp.
_AUTODL_CACHE = Path("/root/autodl-tmp/nlp_a3_cache/sft-out")
DEFAULT_OUTPUT_DIR = _AUTODL_CACHE if _AUTODL_CACHE.parent.exists() else OUTPUTS_DIR / "sft-out"


# -- GPU / VRAM detection --------------------------------------------------

def detect_gpu_config() -> dict[str, object]:
    """Mirror notebook autodl-setup-gpu logic: pick BS/GA/quant by VRAM.

    Returns dict with: vram_gb, sft_bs, sft_ga, use_4bit, max_len, quant_flag.
    """
    try:
        import torch
    except ImportError:
        return {
            "vram_gb": 0, "sft_bs": 1, "sft_ga": 16, "use_4bit": True,
            "max_len": 1024, "quant_flag": "--quant_bits 4 --bnb_4bit_compute_dtype bfloat16",
            "device": "cpu",
        }
    if not torch.cuda.is_available():
        return {
            "vram_gb": 0, "sft_bs": 1, "sft_ga": 16, "use_4bit": True,
            "max_len": 1024, "quant_flag": "--quant_bits 4 --bnb_4bit_compute_dtype bfloat16",
            "device": "cpu",
        }
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory // (1024 ** 3)

    if vram >= 40:          # A100/A800 (40 GB+): drop 4-bit
        bs, ga, q4 = 4, 4, False
    elif vram >= 24:        # RTX 3090/4080/4090 (24-32 GB): standard
        bs, ga, q4 = 2, 8, True
    else:                    # T4 / RTX 3080 (≤16 GB): tighten
        bs, ga, q4 = 1, 16, True

    has_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = "bfloat16" if has_bf16 else "float16"
    quant_flag = (
        f"--quant_bits 4 --bnb_4bit_compute_dtype {compute_dtype}" if q4 else ""
    )
    return {
        "device": p.name, "vram_gb": vram,
        "sft_bs": bs, "sft_ga": ga, "use_4bit": q4,
        "max_len": 1536, "quant_flag": quant_flag,
        "bf16": has_bf16,
    }


# -- Model path resolution -------------------------------------------------

def resolve_model_dir(explicit: str | None) -> str:
    """Cache-first: prefer models/Qwen3.5-4B/ over ModelScope download."""
    if explicit:
        if not Path(explicit).exists():
            raise SystemExit(f"--model-dir {explicit} does not exist")
        return str(explicit)
    if (DEFAULT_MODEL_DIR / "config.json").exists():
        print(f"  [cache] using {DEFAULT_MODEL_DIR}")
        return str(DEFAULT_MODEL_DIR)
    # Fall back to ModelScope into outputs/model_cache/
    from modelscope import snapshot_download
    print("  models/Qwen3.5-4B/ not found — downloading via ModelScope...")
    return snapshot_download(
        "Qwen/Qwen3.5-4B",
        cache_dir=str(OUTPUTS_DIR / "model_cache"),
    )


# -- Command building ------------------------------------------------------

def build_swift_cmd(args: argparse.Namespace, gpu_cfg: dict) -> list[str]:
    """Compose the full `swift sft ...` command as a list of tokens."""
    model_dir = resolve_model_dir(args.model_dir)
    data_path = Path(args.data).resolve()
    val_path = Path(args.val_data).resolve() if args.val_data else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise SystemExit(f"--data {data_path} not found. Did you run "
                         f"`python -m src.build_stage0 --force`?")

    bs = args.batch_size or gpu_cfg["sft_bs"]
    ga = args.grad_accum or gpu_cfg["sft_ga"]
    max_len = args.max_len or gpu_cfg["max_len"]
    quant_flag = args.quant_flag if args.quant_flag is not None else gpu_cfg["quant_flag"]

    parts: list[str] = ["swift", "sft",
        "--model", str(model_dir),
        "--use_hf", "false",
        "--tuner_type", "lora",
        "--target_modules", "all-linear",
    ]
    if quant_flag:
        parts += shlex.split(quant_flag)
    parts += [
        "--enable_thinking", "false",
        "--add_non_thinking_prefix", "true",
        "--loss_scale", "ignore_empty_think",
        "--use_liger_kernel", "true",
        "--group_by_length", "true",
        "--dataset", str(data_path),
    ]
    if val_path:
        parts += ["--val_dataset", str(val_path)]
    parts += [
        "--output_dir", str(output_dir),
        "--num_train_epochs", str(args.epochs),
        "--per_device_train_batch_size", str(bs),
        "--gradient_accumulation_steps", str(ga),
        "--learning_rate", str(args.lr),
        "--warmup_ratio", str(args.warmup_ratio),
        "--max_length", str(max_len),
        "--gradient_checkpointing", "true",
        "--bf16" if gpu_cfg.get("bf16", True) else "--fp16", "true",
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--save_steps", str(args.save_steps),
        "--eval_steps", str(args.save_steps),     # mirror save_steps unless overridden
        "--save_total_limit", str(args.save_total_limit),
        "--dataloader_num_workers", str(args.num_workers),
    ]

    # Resume: explicit `--resume-from PATH` or `--auto-resume`. Default is
    # FRESH start — auto-resuming an old run when training data changed
    # (e.g. pad_with_random toggled) loads a stale-LoRA + stale-optimizer
    # state and ALSO triggers transformers CVE-2025-32434 (optimizer.pt uses
    # torch.load which is rejected on torch < 2.6). Safer to explicitly
    # opt into resume when you mean it.
    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    elif args.auto_resume and not args.no_resume:
        # Auto-detect: look for output_dir/last or output_dir/v*/checkpoint-*
        last_sym = output_dir / "last"
        if last_sym.exists():
            resume_path = last_sym
        else:
            run_dirs = sorted(output_dir.glob("v*-*"), key=lambda p: p.stat().st_mtime)
            if run_dirs:
                latest_run = run_dirs[-1]
                ckpts = sorted(latest_run.glob("checkpoint-*"),
                               key=lambda p: int(p.name.split("-")[-1]))
                if ckpts:
                    resume_path = ckpts[-1]
    if resume_path:
        if resume_path.exists():
            parts += ["--resume_from_checkpoint", str(resume_path)]
            print(f"  resume from: {resume_path}")
        else:
            print(f"  WARN: --resume-from {resume_path} does not exist; starting fresh")

    return parts


# -- Main ------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Standalone SFT training (bypass JupyterLab)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default=str(DEFAULT_DATA),
                   help="SFT train jsonl (ms-swift messages format).")
    p.add_argument("--val-data", default=str(DEFAULT_VAL_DATA),
                   help="SFT eval jsonl. Pass empty string '' to skip.")
    p.add_argument("--model-dir", default=None,
                   help=f"Base model dir. Auto-resolves to {DEFAULT_MODEL_DIR} "
                        f"if present, else downloads via ModelScope.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                   help="Where ms-swift writes v<N>-<timestamp>/checkpoint-*/.")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Per-device train batch size. Default: auto from VRAM.")
    p.add_argument("--grad-accum", type=int, default=None,
                   help="Gradient accumulation. Default: auto from VRAM.")
    p.add_argument("--max-len", type=int, default=None,
                   help="Max sequence length. Default: 1536 on 24+ GB, 1024 on smaller.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--save-steps", type=int, default=50,
                   help="Checkpoint frequency. 50 caps max loss-on-crash to "
                        "~19 min (debug_log 复用经验 33).")
    p.add_argument("--save-total-limit", type=int, default=3,
                   help="Max kept checkpoints (older ones auto-deleted).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. 0 = single-process (default, "
                        "resilient to worker SIGABRT — 复用经验 33).")
    p.add_argument("--quant-flag", default=None,
                   help="Override the auto-detected QLoRA flag string. "
                        "Pass empty string '' to disable 4-bit on A100+.")

    p.add_argument("--resume-from", default=None,
                   help="Explicit checkpoint path for --resume_from_checkpoint. "
                        "Takes precedence over --auto-resume.")
    p.add_argument("--auto-resume", action="store_true",
                   help="Auto-detect and resume from the latest checkpoint "
                        "under --output-dir/v*-*/checkpoint-*. DEFAULT IS "
                        "OFF — auto-resuming after training data changed "
                        "(e.g. pad_with_random toggled) loads stale state "
                        "and may trip the transformers CVE-2025-32434 "
                        "torch.load check on optimizer.pt. Only use this "
                        "when continuing a previously interrupted training "
                        "on the SAME data.")
    p.add_argument("--no-resume", action="store_true",
                   help="Force fresh start; overrides --auto-resume.")

    p.add_argument("--dry-run", action="store_true",
                   help="Print the command and exit without launching.")

    args = p.parse_args()
    if not args.val_data:
        args.val_data = None  # empty string = skip val

    print("=" * 70)
    print("  run_sft.py — standalone SFT training")
    print("=" * 70)
    gpu_cfg = detect_gpu_config()
    print(f"  device: {gpu_cfg.get('device', 'cpu')}  VRAM: {gpu_cfg['vram_gb']} GB")
    print(f"  auto config: BS={gpu_cfg['sft_bs']}  GA={gpu_cfg['sft_ga']}  "
          f"max_len={gpu_cfg['max_len']}  4bit={gpu_cfg['use_4bit']}")

    cmd = build_swift_cmd(args, gpu_cfg)
    print()
    print("  command:")
    # Pretty-print with line continuation
    pretty = " \\\n    ".join(shlex.quote(c) for c in cmd)
    print("    " + pretty)
    print()

    if args.dry_run:
        print("[dry-run] not launching.")
        return

    print("[launch] starting swift sft (Ctrl+C to abort)...")
    t0 = time.time()
    rc = subprocess.run(cmd, env={**os.environ}).returncode
    elapsed = time.time() - t0
    print(f"\n[done] swift sft exited with code {rc} after {elapsed / 60:.1f} min")
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
