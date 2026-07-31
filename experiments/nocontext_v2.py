"""
Revision work item (d): the no-document-access generator condition.

WHY
---
Reviewer X1Xm identified a mismatch between our threat model and our method. The
threat model states the attacker does not know the exact document the agent will
process; the camouflage generator was nevertheless handed the full clean_context.
Every CDG figure in the paper is therefore an upper bound produced under an
attacker strictly stronger than the one described.

This condition implements the stated attacker: it sees the domain, the genre of
document, the domain's professional vocabulary, and the malicious goal -- but
never the target document, and not the task instruction either.

ACS is still scored against the real clean_context. The question the condition
asks is precisely how well a context-blind attacker matches a document it has
not seen, so the scoring reference must stay the real one.

DESIGN NOTES (lessons applied from earlier failures in this codebase)
--------------------------------------------------------------------
* Refusal screening is built in from the first draw, not retrofitted: <=4
  resamples, is_refusal persisted, screened and unscreened counts both reported.
* ACS is computed at generation time (compute_similarity=True), not left null.
* Every record carries generator_variant="no_context" and payload_ids are
  prefixed `noctx_`. Resume keys include generator_variant AND detector_type, so
  this condition cannot collide with the static/camouflage/augmented records --
  the shared-state collision class that produced four separate bugs here.
* Nothing in this script inspects processes; completion is judged by record count.

Llama-only. Gemini 2.0 Flash has been withdrawn from the provider catalogue and
cannot run this or any other new condition.

    python experiments/nocontext_v2.py --phase gen
    python experiments/nocontext_v2.py --phase detect
    python experiments/nocontext_v2.py --phase all --limit 2   # smoke test
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

from config import CONFIG
from src.utils.llm_client import LLMClient
from src.attacks.camouflage_generator import CamouflageGenerator
from src.attacks.payload_bank import PayloadBank
from src.detection.static_detector import StaticDetector
from src.detection.augmented_detector import AugmentedDetector
from experiments.regen_corpus import looks_like_refusal, MAX_REGEN_ATTEMPTS

TASKS_PATH = Path("data/tasks.json")
DOMAINS_PATH = Path("data/domains.json")
CORPUS_PATH = Path("data/camouflage_corpus_nocontext.json")
DET_PATH = Path("results/detections_nocontext.jsonl")

MODEL = "llama3.1:latest"
VARIANT = "no_context"

# What genre the attacker is told to write, given only the task's subcategory.
# Drawn from the authority_structures already recorded in data/domains.json.
GENRE_BY_SUBCATEGORY = {
    "stock_analysis": "analyst note",
    "risk_assessment": "risk memorandum",
    "market_interpretation": "market commentary",
    "contract_review": "contract clause review",
    "compliance_check": "compliance assessment",
    "risk_flagging": "legal risk opinion",
    "factual_qa": "reference article",
    "summarization": "report",
    "classification": "document excerpt",
}


def build_client() -> LLMClient:
    client = LLMClient("ollama", MODEL, dry_run=False, cost_alert_usd=5.0)
    assert client.dry_run is False, "refusing to run with dry_run enabled"
    return client


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def phase_generate(tasks: list[dict], domains: dict, limit: int | None) -> None:
    print("=" * 72)
    print(f"(d) NO-CONTEXT GENERATION  [{MODEL}]")
    print("=" * 72)

    gen = CamouflageGenerator(build_client(), compute_similarity=True)
    corpus = json.loads(CORPUS_PATH.read_text()) if CORPUS_PATH.exists() else {"generated": {}, "meta": {}}
    store = corpus.setdefault("generated", {})

    todo = [t for t in (tasks[:limit] if limit else tasks) if t["task_id"] not in store]
    print(f"  tasks needing generation: {len(todo)}")

    t0 = time.monotonic()
    n_first_refusal = n_retries = n_unresolved = 0

    for task in tqdm(todo, desc="generate[no_ctx]", unit="task"):
        dom = domains[task["domain"]]
        genre = GENRE_BY_SUBCATEGORY.get(task["subcategory"], "professional document")
        kw = dict(
            malicious_goal=task["malicious_goal"],
            domain=task["domain"],
            domain_description=dom["description"],
            genre=genre,
            vocabulary=dom["vocabulary_markers"],
            task_id=task["task_id"],
            acs_reference=task["clean_context"],
        )
        variants = gen.generate_no_context(n_variants=CONFIG.n_camouflage_variants, **kw)

        for i, v in enumerate(variants):
            if not looks_like_refusal(v.payload_text):
                continue
            n_first_refusal += 1
            for _ in range(MAX_REGEN_ATTEMPTS):
                n_retries += 1
                retry = gen.generate_no_context(n_variants=1, **kw)[0]
                if not looks_like_refusal(retry.payload_text):
                    retry.payload_id, retry.variant_idx = v.payload_id, v.variant_idx
                    variants[i] = retry
                    break
            else:
                n_unresolved += 1
                tqdm.write(f"    ! {task['task_id']} {v.payload_id}: still refusing")

        store[task["task_id"]] = [{
            "payload_id": v.payload_id,
            "variant_idx": v.variant_idx,
            "payload_text": v.payload_text,
            "acs": round(v.semantic_similarity, 6),
            "domain": v.domain,
            "genre": genre,
            "malicious_goal": v.malicious_goal,
            "generation_model": v.generation_model,
            "generator_variant": VARIANT,
            "is_refusal": looks_like_refusal(v.payload_text),
            "n_words": len(v.payload_text.split()),
        } for v in variants]
        CORPUS_PATH.write_text(json.dumps(corpus, indent=2))

    payloads = [p for ps in store.values() for p in ps]
    acs = [p["acs"] for p in payloads if not p["is_refusal"] and p["acs"] > 0]
    corpus["meta"] = {
        "generator_variant": VARIANT,
        "regenerated_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "attacker_temperature": 0.7,
        "n_tasks": len(store),
        "n_payloads": len(payloads),
        "refusals_on_first_draw": n_first_refusal,
        "refusal_retries": n_retries,
        "refusals_unresolved": n_unresolved,
        "refusals_remaining": sum(1 for p in payloads if p["is_refusal"]),
        "acs_mean": round(sum(acs) / len(acs), 4) if acs else None,
        "corpus_sha256": hashlib.sha256(
            json.dumps(store, sort_keys=True).encode()).hexdigest()[:16],
    }
    CORPUS_PATH.write_text(json.dumps(corpus, indent=2))

    m = corpus["meta"]
    print(f"\n  payloads {m['n_payloads']} over {m['n_tasks']} tasks")
    print(f"  refusals: {m['refusals_on_first_draw']} first draw "
          f"({m['refusals_on_first_draw']/max(1,m['n_payloads']):.1%}), "
          f"{m['refusal_retries']} retries, {m['refusals_remaining']} remaining")
    print(f"  ACS mean {m['acs_mean']}   sha256 {m['corpus_sha256']}")
    print(f"  elapsed {(time.monotonic()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# Detection (both detectors)
# ---------------------------------------------------------------------------

def resume_index() -> set[tuple]:
    """Keyed on (task, payload_id, detector, variant) -- cannot collide."""
    if not DET_PATH.exists():
        return set()
    return {
        (r["task_id"], r["payload_id"], r["detector_type"], r["generator_variant"])
        for r in (json.loads(l) for l in DET_PATH.open())
    }


def phase_detect(tasks: list[dict], limit: int | None) -> None:
    if not CORPUS_PATH.exists():
        raise SystemExit("no-context corpus missing -- run --phase gen first")
    corpus = json.loads(CORPUS_PATH.read_text())
    store = corpus["generated"]
    by_id = {t["task_id"]: t for t in tasks}

    client = build_client()
    detectors = [StaticDetector(client), AugmentedDetector(client)]
    bank = PayloadBank()
    done = resume_index()

    work = []
    for tid in (list(store)[:limit] if limit else store):
        for p in store[tid]:
            for det in detectors:
                if (tid, p["payload_id"], det.DETECTOR_TYPE, VARIANT) not in done:
                    work.append((by_id[tid], p, det))

    print("=" * 72)
    print(f"(d) NO-CONTEXT DETECTION  [static + augmented]")
    print("=" * 72)
    print(f"  contexts to run: {len(work)}")

    t0, errors = time.monotonic(), 0
    for task, p, det in tqdm(work, desc="detect[no_ctx]", unit="ctx"):
        ctx = bank.inject(task["clean_context"], p["payload_text"], "end")
        try:
            res = det.detect(ctx, task["instruction"], ground_truth=True)
        except Exception as e:                       # noqa: BLE001
            errors += 1
            tqdm.write(f"    ! {task['task_id']}/{p['payload_id']}: {type(e).__name__}")
            continue
        with DET_PATH.open("a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": task["task_id"],
                "domain": task["domain"],
                "payload_type": "camouflage",
                "generator_variant": VARIANT,
                "payload_id": p["payload_id"],
                "is_refusal": p["is_refusal"],
                "acs": p["acs"],
                "verdict": res.verdict,
                "confidence": res.confidence,
                "reason": res.reason,
                "detector_type": res.detector_type,
                "model": MODEL,
                "dry_run": False,
            }) + "\n")

    print(f"\n  ran {len(work)-errors} ({errors} errors) in {(time.monotonic()-t0)/60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser(description="(d) no-document-access condition")
    ap.add_argument("--phase", choices=["gen", "detect", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    tasks = json.loads(TASKS_PATH.read_text())
    domains = json.loads(DOMAINS_PATH.read_text())

    if a.phase in ("gen", "all"):
        phase_generate(tasks, domains, a.limit)
    if a.phase in ("detect", "all"):
        phase_detect(tasks, a.limit)


if __name__ == "__main__":
    main()
