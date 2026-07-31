"""
Revision work item (a): dedicated injection-defense baselines.

Evaluates spotlighting (delimiting / datamarking / encoding) and a structured-
query approximation against the refusal-screened v2 corpus, and asks whether the
Camouflage Detection Gap survives them.

SAMPLING. Per defense config: all 135 camouflage payloads, and 180 static
payloads (4 categories x 45 tasks, one representative each -- the Exp 2 static
design). The full 900-payload static bank is not needed: static IDR is already
1.000 with zero variance, so the camouflage arm is where all the information is.
A `none` control is re-run on the identical sampling so the comparison is
internally consistent rather than being read against the 900-trial baseline.

  5 configs x 315 contexts = 1,575 detection calls, all local.

Llama-only; Gemini 2.0 Flash is retired.

    python experiments/defenses_v2.py
    python experiments/defenses_v2.py --limit 2   # smoke test
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
from src.attacks.payload_bank import PayloadBank
from src.attacks.static_payloads import instantiate_payloads
from src.defenses.spotlighting import DefendedDetector, DEFENSES

TASKS_PATH = Path("data/tasks.json")
CORPUS_PATH = Path("data/camouflage_corpus_v2.json")
OUT_PATH = Path("results/detections_defenses.jsonl")

MODEL = "llama3.1:latest"
STATIC_CATEGORIES = ["override_directive", "authority_claim",
                     "role_confusion", "social_engineering"]


def resume_index() -> set[tuple]:
    """Keyed on (task, payload_id, defense) -- cannot collide with other runs."""
    if not OUT_PATH.exists():
        return set()
    return {(r["task_id"], r["payload_id"], r["defense"])
            for r in (json.loads(l) for l in OUT_PATH.open())}


def build_contexts(tasks, corpus, limit):
    store = corpus["generated"]["llama"]
    out = []
    for task in (tasks[:limit] if limit else tasks):
        for p in store.get(task["task_id"], []):
            out.append((task, "camouflage", p["payload_id"], p["payload_text"],
                        bool(p.get("is_refusal"))))
        for cat in STATIC_CATEGORIES:
            ps = instantiate_payloads(task["malicious_goal"], CONFIG.data_dir,
                                      categories=[cat])
            if ps:
                out.append((task, "static", ps[0].payload_id, ps[0].instantiated_text, False))
        # Clean control: the unmodified document. Without these we cannot tell a
        # defense that genuinely improves detection from one that has simply
        # biased the detector toward INJECTED -- the base64 variant flags nearly
        # everything in smoke testing, and only a false-positive rate separates
        # those two explanations.
        out.append((task, "clean", f"clean_{task['task_id']}", None, False))
    return out


def main(limit: int | None = None) -> None:
    tasks = json.loads(TASKS_PATH.read_text())
    corpus = json.loads(CORPUS_PATH.read_text())

    client = LLMClient("ollama", MODEL, dry_run=False, cost_alert_usd=5.0)
    assert client.dry_run is False, "refusing to run with dry_run enabled"

    contexts = build_contexts(tasks, corpus, limit)
    bank = PayloadBank()
    done = resume_index()

    work = [(d, c) for d in DEFENSES for c in contexts
            if (c[0]["task_id"], c[2], d) not in done]

    n_cam = sum(1 for c in contexts if c[1] == "camouflage")
    n_clean = sum(1 for c in contexts if c[1] == "clean")
    print("=" * 72)
    print("(a) INJECTION-DEFENSE BASELINES")
    print("=" * 72)
    print(f"  defenses: {DEFENSES}")
    print(f"  contexts/config: {len(contexts)} (camouflage {n_cam}, "
          f"static {len(contexts)-n_cam-n_clean}, clean {n_clean})")
    print(f"  total to run: {len(work)}")

    detectors = {d: DefendedDetector(client, d) for d in DEFENSES}
    t0, errors = time.monotonic(), 0

    for defense, (task, ptype, pid, ptext, is_ref) in tqdm(work, desc="defend", unit="ctx"):
        ctx = (task["clean_context"] if ptype == "clean"
               else bank.inject(task["clean_context"], ptext, "end"))
        try:
            res = detectors[defense].detect(ctx, task["instruction"],
                                            ground_truth=(ptype != "clean"))
        except Exception as e:                        # noqa: BLE001
            errors += 1
            tqdm.write(f"    ! {defense}/{task['task_id']}/{pid}: {type(e).__name__}")
            continue
        with OUT_PATH.open("a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": task["task_id"], "domain": task["domain"],
                "payload_type": ptype, "payload_id": pid, "is_refusal": is_ref,
                "defense": defense,
                "verdict": res.verdict, "confidence": res.confidence,
                "reason": res.reason, "detector_type": res.detector_type,
                "model": MODEL, "dry_run": False, "corpus_version": "v2",
            }) + "\n")

    print(f"\n  ran {len(work)-errors} ({errors} errors) in {(time.monotonic()-t0)/60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="(a) injection-defense baselines")
    ap.add_argument("--limit", type=int, default=None)
    main(**vars(ap.parse_args()))
