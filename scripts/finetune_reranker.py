"""Fine-tune bge-reranker-base on climate claims with LoRA + InfoNCE.

Prereq: run `scripts.build_reranker_ft_data` first to produce
`outputs/reranker_ft_data/{train,eval}.jsonl`. See `reranker_finetune_plan.md`
for the full design (data, loss, hyperparameters, decision gates).

What this script does:
    1. Loads bge-reranker-base (XLM-RoBERTa, num_labels=1, fp16).
    2. Wraps with LoRA (r=8, α=16, target_modules=Q/K/V/dense,
       `modules_to_save=["classifier"]` so the 1-d regression head trains).
    3. Trains with list-wise InfoNCE (cross-entropy over 8 candidates per
       row, positive at index 0) + label smoothing 0.05.
    4. Every `--eval-steps`, reranks the cached top-50 candidates of every
       eval claim and logs recall@{5,10,20,50}.
    5. Saves the best-by-recall@20 LoRA adapter under
       `models/bge-reranker-base-ft/lora-seed-{N}/`.
    6. Optionally merges adapter into base for production inference.

Runtime: ~12 min per seed on AutoDL 4080 SUPER (2070 lists × 3 epochs at
BS 8, fp16). See plan §5 for hyperparameter rationale.

Usage::

    # Single seed (smoke / Gate A)
    python -m scripts.finetune_reranker --seed 42

    # Multi-seed (after Gate A passes)
    for s in 1337 2024; do
        python -m scripts.finetune_reranker --seed $s
    done

    # Merge LoRA into base for production deployment
    python -m scripts.finetune_reranker --seed 42 --merge-only

Gate criteria (plan §11):
    Gate A — diag_test recall@20 ≥ 0.40 at any checkpoint.
    Gate B — end-to-end Track 2 HM ≥ 0.235 (run via phase1_eval).
    Gate C — dev_holdout recall@20 not falling > 0.05 vs diag_test.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import MODELS_DIR, OUTPUTS_DIR  # noqa: E402

FT_DATA_DIR = OUTPUTS_DIR / "reranker_ft_data"
BASE_MODEL_NAME = "bge-reranker-base"
DEFAULT_OUT_DIR = MODELS_DIR / "bge-reranker-base-ft"


# -- Data loading ----------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# -- Listwise collator -----------------------------------------------------

class ListwiseCollator:
    """Tokenize a batch of {claim, positive, negatives} rows into a flat tensor.

    For B rows × (1 pos + N negs) = B × (1+N) (claim, candidate) pairs.
    Returns input_ids/attention_mask of shape [B*(1+N), seq_len] and a
    `labels` tensor of shape [B] (all zeros — positive is always at index
    0 within each list, by construction).
    """

    def __init__(self, tokenizer, *, max_length: int, n_cands: int):
        self.tok = tokenizer
        self.max_length = max_length
        self.n_cands = n_cands  # 1 positive + (n_cands - 1) negatives

    def __call__(self, batch: list[dict]) -> dict:
        import torch
        claims: list[str] = []
        cands: list[str] = []
        for row in batch:
            negs = row["negatives"][: self.n_cands - 1]
            if len(negs) < self.n_cands - 1:
                # Pad with the same negs cycled — shouldn't happen post-prep,
                # but defensive. (build_reranker_ft_data drops short rows.)
                pad = negs + negs * ((self.n_cands - 1) // max(1, len(negs)))
                negs = pad[: self.n_cands - 1]
            claims.append(row["claim_text"])
            cands.append(row["positive_text"])
            for n in negs:
                claims.append(row["claim_text"])
                cands.append(n["text"])
        enc = self.tok(
            claims, cands,
            truncation="only_second",
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.zeros(len(batch), dtype=torch.long)
        return enc


# -- Recall@k on eval set --------------------------------------------------

def _score_candidates(model, tokenizer, claim: str, cand_texts: list[str],
                       *, max_length: int, batch_size: int, device: str) -> list[float]:
    """Score every (claim, cand) pair; returns scalar logits in candidate order."""
    import torch
    scores: list[float] = []
    for i in range(0, len(cand_texts), batch_size):
        chunk = cand_texts[i:i + batch_size]
        enc = tokenizer(
            [claim] * len(chunk), chunk,
            truncation="only_second", max_length=max_length,
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc).logits.squeeze(-1)
        scores.extend(out.detach().float().cpu().tolist())
    return scores


def evaluate_recall_at_k(model, tokenizer, eval_rows: list[dict], *,
                         k_list=(5, 10, 20, 50), max_length: int = 512,
                         batch_size: int = 32, device: str = "cuda") -> dict:
    """Rerank each eval row's candidates by the model, compute macro recall@k.

    Returns {"recall@5": x, ..., "n_claims": n}.
    """
    model.eval()
    macro_acc = {k: 0.0 for k in k_list}
    micro_hits = {k: 0 for k in k_list}
    total_gold = 0
    n = 0
    for row in eval_rows:
        gold = set(row.get("gold_ev_ids") or [])
        if not gold:
            continue
        cands = row["candidates"]
        scores = _score_candidates(
            model, tokenizer, row["claim_text"], [c["text"] for c in cands],
            max_length=max_length, batch_size=batch_size, device=device,
        )
        order = sorted(range(len(cands)), key=lambda i: -scores[i])
        ranked_ids = [cands[i]["ev_id"] for i in order]
        n += 1
        total_gold += len(gold)
        for k in k_list:
            hits = len(set(ranked_ids[:k]) & gold)
            macro_acc[k] += hits / len(gold)
            micro_hits[k] += hits
    macro = {f"recall@{k}": macro_acc[k] / n if n else 0.0 for k in k_list}
    micro = {f"micro_recall@{k}": micro_hits[k] / total_gold if total_gold else 0.0 for k in k_list}
    return {**macro, **micro, "n_claims": n}


# -- Training driver -------------------------------------------------------

def train_one_seed(args) -> dict:
    """Train one seed end-to-end. Returns best-eval metrics + ckpt path."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("\n[1/5] loading data...")
    train_rows = _load_jsonl(FT_DATA_DIR / "train.jsonl")
    eval_rows = _load_jsonl(FT_DATA_DIR / "eval.jsonl")
    print(f"  train rows : {len(train_rows):,}")
    print(f"  eval rows  : {len(eval_rows):,}")

    print("\n[2/5] loading bge-reranker-base + LoRA...")
    from src.paths import resolve_model_path
    base_path = resolve_model_path(BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    # XLM-R + LoRA + InfoNCE in pure fp16 produces NaN logits within 20
    # steps (attention softmax overflows, classifier head saturates). bf16
    # has fp32-equivalent dynamic range so the same recipe is stable.
    # 4080 SUPER (Ampere+) supports bf16 natively; fall back to fp32 on
    # pre-Ampere GPUs.
    train_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    print(f"  train dtype: {train_dtype}")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_path, num_labels=1, dtype=train_dtype,
    )
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.1, bias="none",
        task_type=TaskType.SEQ_CLS,
        # The classifier head is num_labels=1 linear; it MUST train or
        # the score scale stays at pre-trained init and InfoNCE drifts
        # the LoRA deltas in useless directions. (plan §3.1)
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("\n[3/5] dataloader + optimizer...")
    n_cands = 1 + args.n_negs
    collator = ListwiseCollator(tokenizer, max_length=args.max_length, n_cands=n_cands)
    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    train_loader = DataLoader(
        train_rows, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=0,
    )
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps,
    )

    print(f"\n[4/5] training {args.epochs} epoch(s) × {len(train_loader)} batches "
          f"= {total_steps} optimizer steps (GA={args.grad_accum})...")
    history: list[dict] = []
    best = {"recall@20": -1.0, "step": -1, "metrics": None}
    patience_left = args.early_stop_patience
    out_dir = args.out_dir / f"lora-seed-{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"train_log_seed-{args.seed}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")

    step = 0
    t0 = time.time()
    model.train()
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits.squeeze(-1)  # [B*(1+N)]
            B = labels.size(0)
            scores = logits.view(B, n_cands)
            loss = F.cross_entropy(scores, labels, label_smoothing=0.05)
            # Hard fail on NaN — once AdamW state is poisoned by a NaN
            # grad, the run never recovers. Faster to abort and surface
            # the root cause (dtype/lr/data) than to chew through 3 min
            # of zero-progress steps.
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"NaN/Inf loss at step {step + 1} (batch {batch_idx}). "
                    f"Common causes: fp16 dtype (use bf16), too-high LR, "
                    f"or empty/bad input batch."
                )
            (loss / args.grad_accum).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_steps == 0:
                    rec = {"event": "train", "step": step, "epoch": epoch,
                           "loss": float(loss.item()),
                           "lr": scheduler.get_last_lr()[0],
                           "elapsed_sec": round(time.time() - t0, 1)}
                    history.append(rec)
                    log_f.write(json.dumps(rec) + "\n")
                    print(f"    step {step}/{total_steps}  loss {loss.item():.4f}  "
                          f"lr {rec['lr']:.2e}  ({rec['elapsed_sec']}s)")

                if step % args.eval_steps == 0 or step == total_steps:
                    metrics = evaluate_recall_at_k(
                        model, tokenizer, eval_rows,
                        max_length=args.max_length, batch_size=args.eval_batch_size,
                        device=device,
                    )
                    rec = {"event": "eval", "step": step, "epoch": epoch, **metrics,
                           "elapsed_sec": round(time.time() - t0, 1)}
                    history.append(rec)
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    print(f"    [eval@{step}] recall@5={metrics['recall@5']:.4f}  "
                          f"@10={metrics['recall@10']:.4f}  "
                          f"@20={metrics['recall@20']:.4f}  "
                          f"@50={metrics['recall@50']:.4f}")

                    if metrics["recall@20"] > best["recall@20"]:
                        best = {"recall@20": metrics["recall@20"], "step": step,
                                "metrics": metrics}
                        # Save adapter to a "best" symlink-like folder
                        model.save_pretrained(str(out_dir))
                        print(f"    new best @ step {step}; adapter → {out_dir}")
                        patience_left = args.early_stop_patience
                    else:
                        patience_left -= 1
                        print(f"    no improvement; patience left = {patience_left}")
                        if patience_left <= 0:
                            print(f"    early stopping at step {step}")
                            log_f.close()
                            return {"best": best, "history": history, "out_dir": str(out_dir)}
                    model.train()

    log_f.close()
    return {"best": best, "history": history, "out_dir": str(out_dir)}


# -- Merge -----------------------------------------------------------------

def merge_lora(args) -> Path:
    """Load saved LoRA adapter, merge into base, save full model for production."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel
    from src.paths import resolve_model_path

    src = args.out_dir / f"lora-seed-{args.seed}"
    dst = args.out_dir / f"merged-seed-{args.seed}"
    print(f"[merge] loading adapter from {src}")
    base_path = resolve_model_path(BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    # bf16 to match training dtype; fall back to fp32 on pre-Ampere.
    merge_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    base = AutoModelForSequenceClassification.from_pretrained(
        base_path, num_labels=1, dtype=merge_dtype,
    )
    model = PeftModel.from_pretrained(base, str(src))
    merged = model.merge_and_unload()
    dst.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(dst))
    tokenizer.save_pretrained(str(dst))
    # The reranker is loaded via sentence_transformers.CrossEncoder, which
    # needs config.json present — save_pretrained writes it automatically.
    print(f"[merge] merged model → {dst}")
    return dst


# -- Main ------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    # data
    p.add_argument("--n-negs", type=int, default=7,
                   help="hard-negs per row; must match build_reranker_ft_data output")
    p.add_argument("--max-length", type=int, default=512)
    # optim
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=8,
                   help="number of LISTS per batch; total pairs = bs × (1+n_negs)")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    # eval / early stop
    p.add_argument("--log-steps", type=int, default=20)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--early-stop-patience", type=int, default=3,
                   help="number of evals without recall@20 improvement before stopping")
    # lora
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    # control
    p.add_argument("--merge-only", action="store_true",
                   help="skip training; just merge an existing adapter into base")
    args = p.parse_args()

    if not args.merge_only:
        if not (FT_DATA_DIR / "train.jsonl").exists() or not (FT_DATA_DIR / "eval.jsonl").exists():
            raise SystemExit(
                f"FT data not found at {FT_DATA_DIR}/{{train,eval}}.jsonl. "
                f"Run: python -m scripts.build_reranker_ft_data"
            )

        print("=" * 70)
        print(f"reranker fine-tune  seed={args.seed}")
        print("=" * 70)
        for k, v in sorted(vars(args).items()):
            print(f"  {k}: {v}")

        result = train_one_seed(args)
        best = result["best"]
        print("\n=== Training done ===")
        print(f"  best recall@20: {best['recall@20']:.4f} at step {best['step']}")
        print(f"  full metrics  : {best['metrics']}")
        print(f"  adapter saved : {result['out_dir']}")
        print(f"  log file      : {args.out_dir}/train_log_seed-{args.seed}.jsonl")

        # Gate A check
        if best["recall@20"] >= 0.40:
            print(f"\n  ✅ Gate A PASS (recall@20 {best['recall@20']:.4f} ≥ 0.40)")
        else:
            print(f"\n  ⚠️  Gate A NOT MET (recall@20 {best['recall@20']:.4f} < 0.40); "
                  f"see plan §11 for next-step options.")

    print("\n[5/5] merging LoRA into base...")
    dst = merge_lora(args)
    print(f"  merged model: {dst}")
    print("\nNext steps (plan §7.2-7.4):")
    print(f"  python -m scripts.retrieval_ceiling --dataset diag_test --mode retriever "
          f"--reranker-path {dst}")
    print(f"  python -m scripts.phase1_eval --tracks 2 --prompts v1 --dataset diag_test "
          f"--reranker-path {dst} --use-rerank")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
