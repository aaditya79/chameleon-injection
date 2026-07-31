"""
Recompute DAF using the regenerated, all-45-task single-agent baseline.

    DAF = ASR_debate / ASR_single

The submitted DAF divided a 45-task debate ASR by a single-agent ASR that only
ever covered 5 financial tasks (an artefact of the shared resume cache in
run_all.py:215), and 21 of those 56 baseline trials were dry-run mock records.
This script recomputes DAF against the phase-(g) baseline, which covers all 45
tasks for both models and carries dry_run=False explicitly.

Baselines compared, so the effect of each fix is visible:
  published  -- exp2 single-agent slice, mock included   (what the paper reported)
  mock-excl  -- exp2 single-agent slice, mock excluded
  exp1       -- exp1 single-agent slice, all 45 tasks, mock excluded
  v2         -- phase-(g) regenerated baseline, all 45 tasks   <-- the honest one

    python analysis/daf_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

OLD = Path("results/trials.jsonl")
NEW = Path("results/trials_v2.jsonl")
OUT = Path("results/daf_v2.json")

LLAMA = "llama3.1"
LLAMA_V2 = "llama3.1:latest"
GEMINI = "google/gemini-2.0-flash-001"


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def asr(rows: list[dict]) -> tuple[int, int, float]:
    judged = [r for r in rows if r.get("asr_judgment")]
    if not judged:
        return 0, 0, float("nan")
    n = sum(1 for r in judged if r["asr_judgment"].get("followed_injection"))
    return n, len(judged), n / len(judged)


def fmt(t) -> str:
    n, d, v = t
    return f"{n}/{d}={v:.4f}" if d else "n/a"


def ratio(num, den) -> float:
    return num[2] / den[2] if den[1] and den[2] else float("nan")


def main() -> None:
    old, new = read(OLD), read(NEW)
    if not new:
        raise SystemExit("results/trials_v2.jsonl not found -- run phase (g) first.")

    report: dict = {}
    print("=" * 74)
    print("DAF RECOMPUTED AGAINST THE ALL-45-TASK SINGLE-AGENT BASELINE")
    print("=" * 74)

    for model_old, model_new, label in [
        (LLAMA, LLAMA_V2, "Llama 3.1 8B"),
        (GEMINI, GEMINI, "Gemini 2.0 Flash"),
    ]:
        print(f"\n{'-'*74}\n{label}\n{'-'*74}")
        m: dict = {}

        for pt in ("static", "camouflage"):
            debate = [
                r for r in old
                if r["model"] == model_old and r["experiment"] == "exp2"
                and r["architecture"] == "debate" and r["payload_type"] == pt
                and not r.get("dry_run", False)
            ]
            base_pub = [
                r for r in old
                if r["model"] == model_old and r["experiment"] == "exp2"
                and r["architecture"] == "single_agent" and r["payload_type"] == pt
            ]
            base_nomock = [r for r in base_pub if not r.get("dry_run", False)]
            base_exp1 = [
                r for r in old
                if r["model"] == model_old and r["experiment"] == "exp1"
                and r["architecture"] == "single_agent" and r["payload_type"] == pt
                and not r.get("dry_run", False)
            ]
            base_v2 = [
                r for r in new
                if r["model"] == model_new and r["architecture"] == "single_agent"
                and r["payload_type"] == pt
            ]

            d, bp, bn, b1, b2 = (asr(debate), asr(base_pub), asr(base_nomock),
                                 asr(base_exp1), asr(base_v2))
            ntasks = lambda rs: len({r["task_id"] for r in rs})

            m[pt] = {
                "asr_debate": d, "n_debate_tasks": ntasks(debate),
                "baseline_published": bp, "n_tasks_published": ntasks(base_pub),
                "baseline_mock_excluded": bn,
                "baseline_exp1": b1, "n_tasks_exp1": ntasks(base_exp1),
                "baseline_v2": b2, "n_tasks_v2": ntasks(base_v2),
                "daf_published": ratio(d, bp),
                "daf_mock_excluded": ratio(d, bn),
                "daf_vs_exp1": ratio(d, b1),
                "daf_v2": ratio(d, b2),
            }

            print(f"  {pt}:")
            print(f"    ASR debate (all tasks, mock-excl) : {fmt(d):>16}  tasks={ntasks(debate)}")
            print(f"    baseline published (mock incl)    : {fmt(bp):>16}  tasks={ntasks(base_pub)}")
            print(f"    baseline mock-excluded            : {fmt(bn):>16}")
            print(f"    baseline exp1 (45 tasks)          : {fmt(b1):>16}  tasks={ntasks(base_exp1)}")
            print(f"    baseline v2  (45 tasks, phase g)  : {fmt(b2):>16}  tasks={ntasks(base_v2)}")
            print(f"      DAF published      : {m[pt]['daf_published']:.3f}")
            print(f"      DAF mock-excluded  : {m[pt]['daf_mock_excluded']:.3f}")
            print(f"      DAF vs exp1        : {m[pt]['daf_vs_exp1']:.3f}")
            print(f"      DAF v2  <-- report : {m[pt]['daf_v2']:.3f}")

        report[label] = m

    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWritten to {OUT}")
    print("\nNOTE: the debate arm still uses the original (unpersisted) payload draw,")
    print("while the v2 baseline uses the regenerated corpus. Both are drawn from the")
    print("same generator under identical settings, but they are not the same strings.")
    print("State this when reporting DAF v2.")


if __name__ == "__main__":
    main()
