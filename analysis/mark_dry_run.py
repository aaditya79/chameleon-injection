"""
Backfill a `dry_run` boolean onto every record in results/trials.jsonl.

WHY THIS EXISTS
---------------
342 llama3.1 trials in results/trials.jsonl are dry-run MOCK data, not real
inference. They were written into the production results file and are not
distinguishable by any existing field, because TrialLogger.log_trial() records
`config.agent_provider` ("ollama") rather than the provider string the client
actually returned ("ollama[dry_run]") -- so the dry-run marker was discarded at
logging time.

DETECTION SIGNATURE
-------------------
A trial is marked dry_run=True if EITHER:

  (A) its final_answer or any of its agent_responses is a verbatim entry of
      _MOCK_RESPONSES in src/utils/llm_client.py; or

  (B) it belongs to a "timestamp collapse" second -- a wall-clock second in which
      more trials were logged than real sequential inference could produce.
      TrialLogger stamps datetime.now(timezone.utc) PER TRIAL (logger.py:68), and
      a single llama3.1 completion takes seconds, so >50 trials sharing one second
      is only possible for dry-run, which returns instantly.

(A) is the primary, content-based signal. (B) catches detection-only trials
(exp3/exp4) from the same batch, which have no agent response to match against.
Both are recorded separately so the overlap can be audited.

USAGE
-----
    python analysis/mark_dry_run.py            # report only, no write
    python analysis/mark_dry_run.py --write    # rewrite trials.jsonl in place
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.llm_client import _MOCK_RESPONSES

TRIALS_PATH = Path("results/trials.jsonl")

# A wall-clock second holding more than this many trials cannot be real inference.
# The real runs log at most a handful of trials per second; the mock batch logs 342.
COLLAPSE_THRESHOLD = 50

_MOCK_SET = {m.strip() for m in _MOCK_RESPONSES}


def load_trials(path: Path = TRIALS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def _has_mock_text(trial: dict) -> bool:
    """True if the trial's answer text is a verbatim _MOCK_RESPONSES entry."""
    candidates: list[str] = []
    if trial.get("final_answer"):
        candidates.append(trial["final_answer"])
    for r in trial.get("agent_responses") or []:
        if r:
            candidates.append(r)
    return any(c.strip() in _MOCK_SET for c in candidates)


def _collapsed_seconds(trials: list[dict]) -> set[str]:
    """Wall-clock seconds holding an impossible number of trials."""
    per_second = Counter(t["timestamp"][:19] for t in trials)
    return {sec for sec, n in per_second.items() if n > COLLAPSE_THRESHOLD}


def mark(trials: list[dict]) -> tuple[list[dict], dict]:
    """
    Annotate each trial with dry_run plus the two evidence flags.

    Returns:
        (annotated_trials, stats_dict)
    """
    collapsed = _collapsed_seconds(trials)

    n_text = n_time = n_both = 0
    for t in trials:
        by_text = _has_mock_text(t)
        by_time = t["timestamp"][:19] in collapsed
        t["dry_run"] = bool(by_text or by_time)
        t["dry_run_evidence"] = {
            "mock_response_text": by_text,
            "timestamp_collapse": by_time,
        }
        n_text += by_text
        n_time += by_time
        n_both += by_text and by_time

    flagged = [t for t in trials if t["dry_run"]]
    stats = {
        "total_trials": len(trials),
        "flagged_dry_run": len(flagged),
        "collapsed_seconds": sorted(collapsed),
        "by_mock_text": n_text,
        "by_timestamp_collapse": n_time,
        "by_both": n_both,
        "by_text_only": n_text - n_both,
        "by_time_only": n_time - n_both,
        "flagged_by_model": dict(Counter(t["model"] for t in flagged)),
        "flagged_by_experiment": dict(Counter(t["experiment"] for t in flagged)),
        "flagged_by_domain": dict(Counter(t["domain"] for t in flagged)),
        "flagged_by_task": dict(Counter(t["task_id"] for t in flagged)),
    }
    return trials, stats


def contamination_table(trials: list[dict]) -> list[tuple]:
    """Per (experiment, model) contamination counts, for reporting."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        buckets[(t["experiment"], t["model"])].append(t)
    rows = []
    for key in sorted(buckets):
        group = buckets[key]
        n_dry = sum(1 for t in group if t["dry_run"])
        if n_dry:
            rows.append((*key, n_dry, len(group)))
    return rows


def main(write: bool = False) -> None:
    trials = load_trials()
    trials, stats = mark(trials)

    print("=" * 68)
    print("DRY-RUN MOCK TRIAL DETECTION")
    print("=" * 68)
    print(f"  Total trials          : {stats['total_trials']}")
    print(f"  Flagged as dry_run    : {stats['flagged_dry_run']}")
    print(f"    by mock text        : {stats['by_mock_text']}")
    print(f"    by timestamp collapse: {stats['by_timestamp_collapse']}")
    print(f"    by both             : {stats['by_both']}")
    print(f"    text-only           : {stats['by_text_only']}")
    print(f"    time-only           : {stats['by_time_only']}")
    print(f"  Collapsed seconds     : {stats['collapsed_seconds']}")
    print(f"  Flagged by model      : {stats['flagged_by_model']}")
    print(f"  Flagged by domain     : {stats['flagged_by_domain']}")
    print(f"  Flagged task_ids      : {stats['flagged_by_task']}")

    print("\n  Contamination by (experiment, model):")
    print(f"  {'experiment':<8} {'model':<32} {'dry':>6} {'total':>7}")
    print(f"  {'-'*58}")
    for exp, model, n_dry, n_tot in contamination_table(trials):
        print(f"  {exp:<8} {model:<32} {n_dry:>6} {n_tot:>7}")

    if write:
        backup = TRIALS_PATH.with_suffix(".jsonl.pre_dryrun_backup")
        if not backup.exists():
            shutil.copy2(TRIALS_PATH, backup)
            print(f"\n  Backup written to {backup}")
        with TRIALS_PATH.open("w") as f:
            for t in trials:
                f.write(json.dumps(t) + "\n")
        print(f"  Rewrote {TRIALS_PATH} with dry_run field on {len(trials)} records")
    else:
        print("\n  (dry report only -- pass --write to persist the dry_run field)")

    Path("results/dry_run_audit.json").write_text(json.dumps(stats, indent=2))
    print("  Audit written to results/dry_run_audit.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Flag dry-run mock trials in trials.jsonl")
    p.add_argument("--write", action="store_true", help="persist the dry_run field")
    main(**vars(p.parse_args()))
