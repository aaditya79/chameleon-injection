"""
Item 5b offline detection: run the 7 item-3c detectors over the injected field texts in the banking traces.
No agent is run and nothing is generated here; this reads the injected text each episode already recorded and
scores it. The only paid detector is gpt4o (capped by ITEM5B_DETECT_CAP, default $1.00); the rest are local (ollama).

    ~/research/agentdojo-venv/bin/python experiments/item5b_detect.py [TRACE_ROOT] [--detectors ...]

TRACE_ROOT defaults to results/item5b. The injected text is read from each trace's `injections` map; the detector's
"instruction" argument is the banking user task's prompt. Payload text is never printed or inlined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from experiments.item3c_detect import CLASSIFIERS, DETECTORS

OUT = Path("results/item5b_detections.jsonl")
CAP = float(os.environ.get("ITEM5B_DETECT_CAP", "1.00"))


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.open()] if p.exists() else []


def spent() -> float:
    return sum(r.get("cost_usd", 0.0) for r in jl(OUT))


def slot_id(arm: str, field_id: str, text: str, instr: str) -> str:
    h = hashlib.sha1(f"{arm}\x00{field_id}\x00{text}\x00{instr}".encode()).hexdigest()[:16]
    return f"{arm}:{field_id}:{h}"


def items(trace_root: Path) -> list[dict]:
    """One row per distinct (arm, field, injected text, user-task prompt) seen in the traces."""
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite("v1.2.2", "banking")
    prompts = {u: t.PROMPT for u, t in suite.user_tasks.items()}
    seen: dict[str, dict] = {}
    for f in trace_root.glob("**/banking/*/*/*.json"):
        t = json.loads(f.read_text())
        arm = t.get("attack_type") or "clean"
        instr = prompts.get(t["user_task_id"], "")
        for field_id, text in (t.get("injections") or {}).items():
            if not isinstance(text, str) or not text.strip():
                continue
            sid = slot_id(arm, field_id, text, instr)
            seen.setdefault(sid, {"slot_id": sid, "arm": arm, "field_id": field_id, "text": text,
                                  "instruction": instr, "user_task": t["user_task_id"],
                                  "injection_task": t["injection_task_id"]})
    return sorted(seen.values(), key=lambda p: p["slot_id"])


def run(trace_root: Path, names: list[str]) -> None:
    from experiments.mechanism_run import make

    rows = items(trace_root)
    print(f"[5b detect] {len(rows)} distinct injected-text slots under {trace_root}")
    for name in names:
        modes = ("embedded", "alone") if name in CLASSIFIERS else ("embedded",)
        done = {(r["detector"], r["mode"], r["slot_id"]) for r in jl(OUT) if r.get("status") == "ok"}
        todo = [(p, m) for p in rows for m in modes if (name, m, p["slot_id"]) not in done]
        print(f"[5b detect:{name}] {len(todo)} to run" + (f" (gpt4o spent ${spent():.4f} of ${CAP})" if name == "gpt4o" else ""))
        if not todo:
            continue
        fn = make(name)
        for p, m in tqdm(todo, desc=name, unit="slot"):
            if name == "gpt4o" and spent() >= CAP:
                raise SystemExit(f"gpt4o detect cap reached (${spent():.4f} of ${CAP})")
            # embedded and alone are the same text here (the injected text is the whole agent-visible field);
            # the classifiers keep both modes for parity with item 3c's detections schema.
            rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "detector": name, "mode": m,
                   "slot_id": p["slot_id"], "arm": p["arm"], "field_id": p["field_id"],
                   "injection_task": p["injection_task"], "user_task": p["user_task"]}
            for attempt in range(3):
                try:
                    rec.update(status="ok", attempts=attempt + 1, **fn(p["text"], p["instruction"]))
                    break
                except Exception as e:  # noqa: BLE001 - recorded and excluded
                    rec.update(status="error", attempts=attempt + 1, error=f"{type(e).__name__}: {str(e)[:200]}")
                    time.sleep(2 * (attempt + 1))
            with OUT.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_root", nargs="?", default="results/item5b")
    ap.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    a = ap.parse_args()
    run(Path(a.trace_root), a.detectors)


if __name__ == "__main__":
    main()
