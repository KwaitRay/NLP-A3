"""Profile evidence.json — length distribution + structure markers.

Decision aid for chunking strategy. The headline question is "are passages
long enough that chunking even matters". Outputs a single markdown report
at outputs/evidence_profile.md.

Run::

    python -m scripts.profile_evidence            # ~30 s on 174 MB corpus
    python -m scripts.profile_evidence --sample 50000   # quick subset

Reports:
  - char & token length distribution (bge-m3 tokenizer when available)
  - sentences/newlines per passage (proxy for internal structure)
  - long-tail samples (top-N longest with text snippets)
  - bucket counts (0-50 / 51-100 / 101-256 / 257-512 / 513-1024 / 1024+ tokens)
  - decision verdict — "chunking unlikely to help" vs "long-tail warrants chunking"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import load_evidence  # noqa: E402
from src.paths import MODELS_DIR, OUTPUTS_DIR  # noqa: E402


REPORT_PATH = OUTPUTS_DIR / "evidence_profile.md"

# Token bucket boundaries chosen to mirror practical thresholds:
# - 50: very short fragments — probably already atomic
# - 256: bge-m3 default max_seq_length for dense indexing
# - 512: default cross-encoder reranker max_length
# - 1024: anything above starts to be its own document
TOKEN_BUCKETS = [(0, 50), (51, 100), (101, 256), (257, 512), (513, 1024), (1025, float("inf"))]
CHAR_BUCKETS = [(0, 200), (201, 500), (501, 1000), (1001, 2000), (2001, 5000), (5001, float("inf"))]

# Cheap sentence splitter — climate corpora aren't pathological enough to
# need spaCy. Splits on sentence-final punctuation followed by space + capital.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


def _percentiles(values: list[int], pcts=(0, 25, 50, 75, 90, 95, 99, 100)) -> dict[int, int]:
    if not values:
        return {p: 0 for p in pcts}
    s = sorted(values)
    n = len(s)
    out = {}
    for p in pcts:
        idx = max(0, min(n - 1, int(round((p / 100) * (n - 1)))))
        out[p] = s[idx]
    return out


def _bucket_counts(values: list[int], buckets: list[tuple[float, float]]) -> list[tuple[str, int, float]]:
    """Returns [(label, count, pct), ...] for human display."""
    n = len(values) or 1
    out = []
    for lo, hi in buckets:
        if hi == float("inf"):
            label = f"{lo}+"
            c = sum(1 for v in values if v >= lo)
        else:
            label = f"{lo}-{int(hi)}"
            c = sum(1 for v in values if lo <= v <= hi)
        out.append((label, c, c / n))
    return out


def _try_load_tokenizer():
    """Prefer the bge-m3 tokenizer (it's what indexes the corpus). Fall back to
    whitespace word count if not present.
    """
    try:
        from transformers import AutoTokenizer
        path = MODELS_DIR / "bge-m3"
        if (path / "config.json").exists():
            tok = AutoTokenizer.from_pretrained(str(path))
            print(f"  [info] using bge-m3 tokenizer from {path}")
            return tok
        print("  [info] models/bge-m3 not found, falling back to whitespace word count")
    except ImportError:
        print("  [info] transformers not installed, using whitespace word count")
    return None


def _count_tokens(text: str, tok) -> int:
    if tok is None:
        return len(text.split())
    # Fast path — return only token count, no IDs.
    return len(tok.encode(text, add_special_tokens=False))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=int, default=None,
                   help="randomly sample N passages instead of full corpus (for quick iteration)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-long", type=int, default=10,
                   help="how many long-tail samples to include in the report")
    p.add_argument("--out", type=Path, default=REPORT_PATH,
                   help=f"output markdown path (default {REPORT_PATH})")
    args = p.parse_args()

    print("=" * 70)
    print("evidence corpus profile")
    print("=" * 70)

    t0 = time.time()
    corpus = load_evidence(show_progress=True)
    print(f"  loaded {len(corpus):,} passages in {time.time() - t0:.1f}s")

    items = list(corpus.items())
    if args.sample and args.sample < len(items):
        import random
        random.seed(args.seed)
        items = random.sample(items, args.sample)
        print(f"  sampled {len(items):,} passages (seed={args.seed})")

    tok = _try_load_tokenizer()
    using_real_tokens = tok is not None

    char_lens: list[int] = []
    token_lens: list[int] = []
    sent_counts: list[int] = []
    newline_counts: list[int] = []
    longest: list[tuple[int, str, str]] = []  # (token_len, eid, text)

    print("\n[1/2] scanning passages…")
    t0 = time.time()
    for i, (eid, text) in enumerate(items):
        if not isinstance(text, str):
            continue
        cl = len(text)
        tl = _count_tokens(text, tok)
        char_lens.append(cl)
        token_lens.append(tl)
        sent_counts.append(len(SENTENCE_SPLIT_RE.split(text)))
        newline_counts.append(text.count("\n"))

        # Maintain top-N longest with a tiny manual bounded list (cheap on 1M items).
        if len(longest) < args.top_long:
            longest.append((tl, eid, text))
            longest.sort(key=lambda x: x[0])
        elif tl > longest[0][0]:
            longest[0] = (tl, eid, text)
            longest.sort(key=lambda x: x[0])

        if i and i % 200_000 == 0:
            print(f"    {i:,}/{len(items):,}  ({time.time() - t0:.1f}s)")
    print(f"  scanned {len(items):,} in {time.time() - t0:.1f}s")

    # ---- summary stats -------------------------------------------------
    char_pcts = _percentiles(char_lens)
    token_pcts = _percentiles(token_lens)
    sent_pcts = _percentiles(sent_counts, pcts=(50, 90, 95, 99, 100))
    newline_pcts = _percentiles(newline_counts, pcts=(50, 90, 95, 99, 100))

    char_buckets = _bucket_counts(char_lens, CHAR_BUCKETS)
    token_buckets = _bucket_counts(token_lens, TOKEN_BUCKETS)

    # Decision verdict — based on token-length distribution.
    pct_over_256 = sum(1 for v in token_lens if v > 256) / len(token_lens)
    pct_over_512 = sum(1 for v in token_lens if v > 512) / len(token_lens)
    median_tokens = token_pcts[50]
    if pct_over_256 < 0.01 and median_tokens < 100:
        verdict = (
            "**Chunking unlikely to help.** >99% of passages fit in a single 256-token "
            "encoding window, and median is short. Pivot to reranker fine-tuning instead."
        )
    elif pct_over_256 < 0.05:
        verdict = (
            f"**Marginal.** {pct_over_256:.1%} passages exceed 256 tokens. "
            "Chunking would only affect a small slice; expected recall@k improvement is small."
        )
    elif pct_over_256 < 0.20:
        verdict = (
            f"**Worth trying.** {pct_over_256:.1%} of passages exceed 256 tokens, "
            f"{pct_over_512:.1%} exceed 512. Run fixed-token (256, overlap 64) and "
            "sentence-grouping; compare on dev_holdout recall@5/20."
        )
    else:
        verdict = (
            f"**Strongly recommended.** {pct_over_256:.1%} of passages exceed 256 tokens; "
            f"the dense encoder is silently truncating most of them. Chunking will likely "
            f"yield a meaningful recall lift."
        )

    # ---- write report --------------------------------------------------
    lines = [
        "# Evidence corpus profile",
        "",
        f"_Generated by `scripts/profile_evidence.py` over "
        f"{'full corpus' if not args.sample else f'{args.sample:,}-passage sample'}._",
        "",
        f"- corpus size: **{len(corpus):,}** passages",
        f"- tokenizer: **{'bge-m3' if using_real_tokens else 'whitespace (fallback)'}**",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Length distribution",
        "",
        "### Characters",
        "",
        "| pct | min | p25 | p50 | p75 | p90 | p95 | p99 | max |",
        "|---|---|---|---|---|---|---|---|---|",
        "| chars | "
        + " | ".join(f"{char_pcts[p]:,}" for p in (0, 25, 50, 75, 90, 95, 99, 100))
        + " |",
        "",
        "### Tokens" + (" (bge-m3)" if using_real_tokens else " (whitespace word count)"),
        "",
        "| pct | min | p25 | p50 | p75 | p90 | p95 | p99 | max |",
        "|---|---|---|---|---|---|---|---|---|",
        "| tokens | "
        + " | ".join(f"{token_pcts[p]:,}" for p in (0, 25, 50, 75, 90, 95, 99, 100))
        + " |",
        "",
        f"- **% over 256 tokens** (dense encoder cap): `{pct_over_256:.2%}`",
        f"- **% over 512 tokens** (reranker cap):       `{pct_over_512:.2%}`",
        "",
        "### Buckets",
        "",
        "| char range | count | % |",
        "|---|---|---|",
    ]
    for label, c, pct in char_buckets:
        lines.append(f"| {label} | {c:,} | {pct:.2%} |")
    lines += [
        "",
        "| token range | count | % |",
        "|---|---|---|",
    ]
    for label, c, pct in token_buckets:
        lines.append(f"| {label} | {c:,} | {pct:.2%} |")

    # Internal structure
    lines += [
        "",
        "## Internal structure",
        "",
        "Sentences per passage (rough — split on `[.!?] + space + capital`):",
        "",
        "| p50 | p90 | p95 | p99 | max |",
        "|---|---|---|---|---|",
        "| " + " | ".join(f"{sent_pcts[p]:,}" for p in (50, 90, 95, 99, 100)) + " |",
        "",
        "Newlines per passage (proxy for paragraph breaks):",
        "",
        "| p50 | p90 | p95 | p99 | max |",
        "|---|---|---|---|---|",
        "| " + " | ".join(f"{newline_pcts[p]:,}" for p in (50, 90, 95, 99, 100)) + " |",
        "",
    ]

    # Long-tail samples
    lines += [
        f"## Top-{len(longest)} longest passages",
        "",
        "_(Longest first — eyeball these to see if they have internal topic drift.)_",
        "",
    ]
    for tl, eid, text in reversed(longest):
        snippet = text.replace("\n", " ⏎ ")[:300]
        lines.append(f"### `{eid}` — {tl:,} tokens, {len(text):,} chars")
        lines.append("")
        lines.append(f"> {snippet}{'…' if len(text) > 300 else ''}")
        lines.append("")

    # Strategy hints
    lines += [
        "## Suggested chunking strategies (only if verdict says it's worth it)",
        "",
        "Based on the distribution above:",
        "",
        "1. **Fixed-token + overlap** — `chunk_size=256, overlap=64`. Cheap, deterministic, no extra deps. Best when long passages lack clear internal structure.",
        "2. **Sentence grouping** — group N sentences per chunk (N chosen so chunk ≈ 256 tokens). Preserves natural boundaries. Best when sentences are well-formed.",
        "3. **Paragraph-aware** — split on double newlines first, then fixed-token within each. Best when newline counts above are non-trivial (>1 in p90).",
        "",
        "All three should write to `outputs/{bm25,dense}_index_<strategy>/` so the baseline indexes stay untouched and `eval_chunking.py` can sweep all three.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  wrote report → {args.out}")
    print(f"\n  VERDICT: {verdict.splitlines()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
