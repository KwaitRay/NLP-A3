"""Package a predictions JSON into a Codabench leaderboard submission zip.

Pipeline (per BENCHMARK_SUBMISSION.md §3):
  1. Load predictions JSON + test-claims-unlabelled.json.
  2. Merge claim_text from the test file into every entry (defensive — even
     if predict_all already injected it, we re-merge from the source of
     truth so a stale preds file can never overwrite the canonical text).
  3. Validate schema & content:
       - all 153 test claim_ids present
       - claim_label in LABELS
       - evidences is a non-empty list, len <= --max-evidences (default 5)
       - every evidence id matches r"^evidence-\\d+$"
       - every evidence id exists in evidence.json (skippable via flag)
  4. Write benchmark/runs/<TAG>/test-output.json (UTF-8, indent=2).
  5. Zip into benchmark/runs/<TAG>/submission.zip — single file at
     archive root, no directory layer.
  6. Append benchmark/ledger.jsonl with quota guards:
       - Phase 1: <5 today AEST, <100 ever-in-phase-1
       - Phase 2: <3 ever-in-phase-2
     Use --force to override (logged in the ledger row).

Run::
    python -m scripts.build_submission \\
        --preds outputs/predictions/test_run_v1.json \\
        --tag v1-sft-only \\
        --phase 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import load_test_unlabelled  # noqa: E402
from src.paths import BENCHMARK_DIR, EVIDENCE_JSON, LABELS, OUTPUTS_DIR  # noqa: E402


# All Codabench artifacts live under benchmark/. Each run is self-contained
# at benchmark/runs/<TAG>/ (preds.json + test-output.json + submission.zip +
# config.json snapshot). Cross-run state — ledger + evidence-id cache —
# sits at folder root so it's shared between runs.
RUNS_DIR = BENCHMARK_DIR / "runs"
LEDGER = BENCHMARK_DIR / "ledger.jsonl"
EVIDENCE_ID_CACHE = BENCHMARK_DIR / ".evidence_ids.txt"
EVIDENCE_ID_RE = re.compile(r"^evidence-\d+$")
AEST = timezone(timedelta(hours=10))
PHASE_DAILY_CAP = {1: 5, 2: None}
PHASE_TOTAL_CAP = {1: 100, 2: 3}


# ---- pretty printing -------------------------------------------------------

def _info(msg: str) -> None:
    print(f"  [info] {msg}")


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _warn(msg: str) -> None:
    print(f"  [warn] {msg}")


def _fail(msg: str) -> None:
    print(f"  [fail] {msg}", file=sys.stderr)


# ---- evidence id corpus ----------------------------------------------------

def load_evidence_ids(*, refresh: bool) -> set[str]:
    """Return the full set of evidence-ids in the corpus.

    Cached at outputs/submissions/.evidence_ids.txt — rebuilt when the cache
    is older than evidence.json or when --refresh-evidence-cache is set.
    Reading the 174 MB JSON takes ~30s; the cache reads in <1s.
    """
    if (
        not refresh
        and EVIDENCE_ID_CACHE.exists()
        and EVIDENCE_JSON.exists()
        and EVIDENCE_ID_CACHE.stat().st_mtime > EVIDENCE_JSON.stat().st_mtime
    ):
        _info(f"using cached evidence ids: {EVIDENCE_ID_CACHE}")
        return set(EVIDENCE_ID_CACHE.read_text(encoding="utf-8").splitlines())

    if not EVIDENCE_JSON.exists():
        raise FileNotFoundError(
            f"{EVIDENCE_JSON} not found — needed to validate evidence ids. "
            "Either download evidence.json (see data/evidence.md) or pass "
            "--skip-evidence-check."
        )

    _info(f"loading {EVIDENCE_JSON} (174 MB) to build evidence-id cache…")
    with open(EVIDENCE_JSON, encoding="utf-8") as f:
        corpus = json.load(f)
    ids = sorted(corpus.keys())
    EVIDENCE_ID_CACHE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_ID_CACHE.write_text("\n".join(ids), encoding="utf-8")
    _ok(f"cached {len(ids):,} evidence ids → {EVIDENCE_ID_CACHE}")
    return set(ids)


# ---- validation ------------------------------------------------------------

def validate_and_merge(
    preds: dict[str, dict],
    test_claims: dict[str, dict],
    evidence_ids: set[str] | None,
    *,
    max_evidences: int,
    truncate_to_max: bool = False,
) -> dict[str, dict]:
    """Return a fresh dict with claim_text merged in and schema enforced.

    Raises ValueError listing every problem found (so a single run surfaces
    everything wrong, not just the first claim that breaks).

    When `truncate_to_max=True`, claims with > max_evidences citations get
    truncated to the first `max_evidences` instead of rejected. Citation
    order = LLM's `##[i,j,k]##` order ≈ retrieval rank, so first-k keeps
    the highest-confidence picks. Useful when final_k > 5 and the LLM
    sometimes over-cites; see debug_log #40 (2026-05-16).
    """
    errors: list[str] = []
    warnings: list[str] = []
    truncated_count = 0
    out: dict[str, dict] = {}

    expected = set(test_claims.keys())
    got = set(preds.keys())
    missing = expected - got
    extra = got - expected
    if missing:
        errors.append(f"{len(missing)} test claim_ids missing from preds (e.g. {sorted(missing)[:5]})")
    if extra:
        # Extra IDs are non-fatal (we drop them) but worth flagging since
        # they suggest the preds file targets the wrong split.
        warnings.append(f"{len(extra)} extra claim_ids in preds will be dropped (e.g. {sorted(extra)[:5]})")

    for cid in sorted(expected):
        if cid not in preds:
            continue  # already reported above
        rec = preds[cid]
        label = rec.get("claim_label")
        evs = rec.get("evidences")

        if label not in LABELS:
            errors.append(f"{cid}: invalid claim_label {label!r}")
            continue
        if not isinstance(evs, list) or len(evs) == 0:
            errors.append(f"{cid}: evidences must be a non-empty list, got {evs!r}")
            continue
        if len(evs) > max_evidences:
            if truncate_to_max:
                evs = list(evs)[:max_evidences]
                truncated_count += 1
            else:
                errors.append(f"{cid}: {len(evs)} evidences > cap {max_evidences}")
                continue

        bad_format = [e for e in evs if not (isinstance(e, str) and EVIDENCE_ID_RE.match(e))]
        if bad_format:
            errors.append(f"{cid}: malformed evidence ids {bad_format!r}")
            continue

        if evidence_ids is not None:
            unknown = [e for e in evs if e not in evidence_ids]
            if unknown:
                errors.append(f"{cid}: evidence ids not in corpus {unknown!r}")
                continue

        # claim_text: prefer the canonical test file over whatever the preds
        # file carries (defends against stale preds with edited claim text).
        out[cid] = {
            "claim_text": test_claims[cid]["claim_text"],
            "claim_label": label,
            "evidences": list(evs),
        }

    for w in warnings:
        _warn(w)
    if errors:
        # Cap noise at 20 lines so a totally broken file doesn't drown the log.
        head = errors[:20]
        more = f"\n  …and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError("validation failed:\n  - " + "\n  - ".join(head) + more)

    if truncated_count:
        _warn(f"truncated {truncated_count} claim(s) to top-{max_evidences} evidences "
              f"(first-k by LLM citation order, ≈ retrieval rank)")
    _ok(f"validated {len(out)}/{len(expected)} claims, no schema errors")
    return out


# ---- ledger / quota --------------------------------------------------------

def _read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def _today_aest_iso() -> str:
    return datetime.now(AEST).date().isoformat()


def check_quota(phase: int, *, force: bool) -> None:
    """Refuse to build if the ledger says we've blown the cap. --force overrides."""
    rows = _read_ledger()
    in_phase = [r for r in rows if r.get("phase") == phase]
    today = _today_aest_iso()
    today_in_phase = [r for r in in_phase if r.get("ts_aest_date") == today]

    daily_cap = PHASE_DAILY_CAP.get(phase)
    total_cap = PHASE_TOTAL_CAP.get(phase)

    msgs = [
        f"phase {phase}: {len(in_phase)} prior submissions on record",
        f"today (AEST {today}): {len(today_in_phase)} submission(s) so far",
    ]
    if daily_cap is not None:
        msgs.append(f"daily cap: {daily_cap}")
    if total_cap is not None:
        msgs.append(f"phase total cap: {total_cap}")
    for m in msgs:
        _info(m)

    breaches = []
    if daily_cap is not None and len(today_in_phase) >= daily_cap:
        breaches.append(f"daily cap ({daily_cap}) reached for {today}")
    if total_cap is not None and len(in_phase) >= total_cap:
        breaches.append(f"phase {phase} total cap ({total_cap}) reached")

    if breaches:
        joined = "; ".join(breaches)
        if force:
            _warn(f"QUOTA OVERRIDE via --force: {joined}")
        else:
            raise SystemExit(f"refusing to build — {joined}. Re-run with --force to override.")


def append_ledger_row(
    *,
    tag: str,
    phase: int,
    git_sha: str | None,
    preds_sha256: str,
    submission_path: Path,
    dev_holdout_hmean: float | None,
    forced: bool,
) -> None:
    row = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ts_aest_date": _today_aest_iso(),
        "tag": tag,
        "phase": phase,
        "git_sha": git_sha,
        "preds_sha256": preds_sha256,
        "submission_path": str(submission_path.relative_to(OUTPUTS_DIR.parent)),
        "dev_holdout_hmean": dev_holdout_hmean,
        "codabench_hmean": None,  # backfill manually after upload
        "forced": forced,
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")
    _ok(f"appended ledger row → {LEDGER}")


# ---- helpers ---------------------------------------------------------------

def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(OUTPUTS_DIR.parent),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(OUTPUTS_DIR.parent),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def write_submission(
    merged: dict[str, dict], tag: str
) -> tuple[Path, Path, str]:
    """Write test-output.json + submission.zip under benchmark/runs/<TAG>/.

    Co-locates with preds.json (written by run_inference) and config.json
    (snapshot of the run flags) so the entire submission is self-contained
    in one folder, easy to migrate or audit independently.
    """
    out_dir = RUNS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "test-output.json"
    zip_path = out_dir / "submission.zip"

    json_text = json.dumps(merged, ensure_ascii=False, indent=2)
    json_bytes = json_text.encode("utf-8")
    json_path.write_bytes(json_bytes)
    sha = _sha256(json_bytes)

    # ZIP_DEFLATED for compression; arcname kept flat so unzip yields the
    # file at the archive root, matching the Codabench example.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="test-output.json")

    _ok(f"wrote {json_path} ({len(json_bytes):,} B, sha256={sha[:12]}…)")
    _ok(f"wrote {zip_path} ({zip_path.stat().st_size:,} B)")
    return json_path, zip_path, sha


# ---- main ------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", required=True, type=Path, help="path to predictions JSON")
    p.add_argument("--tag", required=True, help="short identifier for this submission (used in folder name + ledger)")
    p.add_argument("--phase", required=True, type=int, choices=[1, 2], help="Codabench phase (1=Ongoing, 2=Final)")
    p.add_argument("--max-evidences", type=int, default=5, help="cap any claim at this many evidences (default 5, matches leaderboard precision sweet spot)")
    p.add_argument("--truncate-to-max", action="store_true",
                   help="instead of rejecting claims with > --max-evidences citations, "
                        "truncate to the first N (≈ retrieval-rank order). Use when "
                        "final_k > 5 and the LLM over-cites on uncertain claims "
                        "(common with NEI/DISPUTED). See debug_log #40.")
    p.add_argument("--dev-hmean", type=float, default=None, help="optional: H_FA on dev_holdout for the same model, recorded in ledger")
    p.add_argument("--skip-evidence-check", action="store_true", help="don't validate evidence ids against evidence.json (use only when corpus unavailable)")
    p.add_argument("--refresh-evidence-cache", action="store_true", help="rebuild benchmark/.evidence_ids.txt from evidence.json")
    p.add_argument("--force", action="store_true", help="override quota refusal — logged as forced=true in ledger")
    args = p.parse_args()

    print("=" * 70)
    print(f"build_submission: tag={args.tag} phase={args.phase}")
    print("=" * 70)

    # 1. Quota check up-front so we don't waste evidence-corpus load on a
    #    submission we won't be allowed to ship.
    check_quota(args.phase, force=args.force)

    # 2. Load inputs.
    if not args.preds.exists():
        _fail(f"preds file not found: {args.preds}")
        return 2
    preds = json.loads(args.preds.read_text(encoding="utf-8"))
    test_claims = load_test_unlabelled()
    _info(f"preds:       {args.preds} ({len(preds)} entries)")
    _info(f"test claims: {len(test_claims)} ids in test-claims-unlabelled.json")

    evidence_ids = None
    if not args.skip_evidence_check:
        evidence_ids = load_evidence_ids(refresh=args.refresh_evidence_cache)
    else:
        _warn("evidence-id corpus check SKIPPED — make sure preds came from a real retriever")

    # 3. Validate + merge.
    try:
        merged = validate_and_merge(
            preds, test_claims, evidence_ids, max_evidences=args.max_evidences,
            truncate_to_max=args.truncate_to_max
        )
    except ValueError as e:
        _fail(str(e))
        return 3

    # 4. Write artifact.
    _, zip_path, preds_sha = write_submission(merged, args.tag)

    # 5. Record git state + ledger row.
    sha = _git_sha()
    if sha is None:
        _warn("could not read git SHA — submission not tied to a commit")
    elif _git_dirty():
        _warn(f"git working tree is DIRTY at {sha[:12]} — uncommitted changes won't be reproducible")
    else:
        _info(f"git: clean at {sha[:12]}")

    append_ledger_row(
        tag=args.tag,
        phase=args.phase,
        git_sha=sha,
        preds_sha256=preds_sha,
        submission_path=zip_path,
        dev_holdout_hmean=args.dev_hmean,
        forced=args.force,
    )

    print()
    print(f"OK submission ready: {zip_path}")
    print(f"  upload to Codabench -> My Submissions -> Phase {args.phase}")
    print(f"  remember to backfill 'codabench_hmean' in {LEDGER} after the platform scores it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
