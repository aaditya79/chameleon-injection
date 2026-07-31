"""
Re-run injection detection over the regenerated, refusal-screened corpus.

WHY
---
The submitted CDG was computed on a camouflage set in which roughly 44% of the
payloads were attacker refusals rather than payloads (see
analysis/refusal_impact.py). A refusal contains no injection, so a correct
detector calls it CLEAN -- and scored against a ground truth of "injected", that
CLEAN is recorded as a miss. Refusals therefore depress IDR_camouflage and
INFLATE CDG. Combined with the dry-run mock contamination, which pushed CDG the
other way, the net direction of the headline number is unresolved without this
run.

WHAT IT DOES
------------
Runs the StaticDetector over:
  * 900 static contexts   (20 template payloads x 45 tasks; deterministic)
  * 135 camouflage contexts (the phase-(f) corpus, refusal flags carried through)
= 1035 detection calls, all local.

Every record carries `is_refusal`, so IDR and CDG can be computed on the full set
and on the refusal-screened subset, and the difference between them IS the
measurement of the bug's effect.

NOTE ON SCOPE: this runs the StaticDetector only -- the detector that defines
CDG. The AugmentedDetector arm (Exp 4) would need a further ~1035 calls and is
not run here, so Exp 4's augmentation figures remain on the unscreened corpus.

    python experiments/redetect_v2.py
    python experiments/redetect_v2.py --limit 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from config import CONFIG
from src.utils.llm_client import LLMClient
from src.detection.static_detector import StaticDetector
from src.detection.augmented_detector import AugmentedDetector
from src.attacks.payload_bank import PayloadBank
from src.attacks.static_payloads import instantiate_payloads

TASKS_PATH = Path("data/tasks.json")
CORPUS_PATH = Path("data/camouflage_corpus_v2.json")
OUT_PATH = Path("results/detections_v2.jsonl")

MODEL = "llama3.1:latest"


def resume_index() -> set[tuple]:
    """
    Signatures already in the output file. Self-contained; no cross-experiment
    sharing. detector_type is part of the key so the static and augmented passes
    over the same contexts do not suppress one another -- the exact failure mode
    that broke the Exp 2 baseline (run_all.py:215).
    """
    if not OUT_PATH.exists():
        return set()
    return {
        (r["task_id"], r["payload_type"], r["payload_id"], r["detector_type"])
        for r in (json.loads(l) for l in OUT_PATH.open())
    }


def build_work(tasks: list[dict], corpus: dict, limit: int | None) -> list[tuple]:
    """(task, payload_type, payload_id, payload_text, is_refusal) for every context."""
    store = corpus["generated"]["llama"]
    work: list[tuple] = []
    for task in (tasks[:limit] if limit else tasks):
        for p in store.get(task["task_id"], []):
            work.append((task, "camouflage", p["payload_id"], p["payload_text"],
                         bool(p.get("is_refusal"))))
        for sp in instantiate_payloads(task["malicious_goal"], CONFIG.data_dir):
            work.append((task, "static", sp.payload_id, sp.instantiated_text, False))
    return work


def main(limit: int | None = None, detector_kind: str = "static") -> None:
    if not CORPUS_PATH.exists():
        raise SystemExit("corpus missing -- run phase (f) first")

    tasks = json.loads(TASKS_PATH.read_text())
    corpus = json.loads(CORPUS_PATH.read_text())

    client = LLMClient("ollama", MODEL, dry_run=False, cost_alert_usd=5.0)
    assert client.dry_run is False, "refusing to run with dry_run enabled"
    detector = StaticDetector(client) if detector_kind == "static" else AugmentedDetector(client)
    bank = PayloadBank()

    work = build_work(tasks, corpus, limit)
    done = resume_index()
    pending = [w for w in work
               if (w[0]["task_id"], w[1], w[2], detector.DETECTOR_TYPE) not in done]

    n_static = sum(1 for w in work if w[1] == "static")
    n_cam = sum(1 for w in work if w[1] == "camouflage")
    n_ref = sum(1 for w in work if w[4])
    print("=" * 70)
    print(f"DETECTION RE-RUN over regenerated corpus  [{detector.DETECTOR_TYPE}]")
    print("=" * 70)
    print(f"  contexts: {len(work)}  (static {n_static}, camouflage {n_cam}, "
          f"of which refusals {n_ref})")
    print(f"  already done {len(work)-len(pending)} | to run {len(pending)}")

    t0, errors = time.monotonic(), 0
    for task, ptype, pid, ptext, is_ref in tqdm(pending, desc="detect", unit="ctx"):
        ctx = bank.inject(task["clean_context"], ptext, "end")
        try:
            res = detector.detect(ctx, task["instruction"], ground_truth=True)
        except Exception as e:                       # noqa: BLE001 - keep the run alive
            errors += 1
            tqdm.write(f"    ! {task['task_id']}/{pid}: {type(e).__name__}: {str(e)[:110]}")
            continue
        with OUT_PATH.open("a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": task["task_id"],
                "domain": task["domain"],
                "payload_type": ptype,
                "payload_id": pid,
                "is_refusal": is_ref,
                "verdict": res.verdict,
                "confidence": res.confidence,
                "reason": res.reason,
                "detector_type": res.detector_type,
                "model": MODEL,
                "dry_run": False,
                "corpus_version": "v2",
            }) + "\n")

    print(f"\n  ran {len(pending)-errors} ({errors} errors) in {(time.monotonic()-t0)/60:.1f} min")
    print(f"  written to {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Detection re-run on the v2 corpus")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--detector", dest="detector_kind",
                    choices=["static", "augmented"], default="static")
    main(**vars(ap.parse_args()))
