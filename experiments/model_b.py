"""
Revision item (b): does the CDG finding and the document-access result generalize
beyond Llama 3.1 8B? Scoped to Exp 1 (CDG) + Exp 5 (no-context) on ONE new model.

Model: openai/gpt-4o-2024-08-06 via OpenRouter. Chosen because it is the same
family as the GPT-4o-mini trials already in the corpus, so it doubles as a
size-scaling point on that cross-model observation; it is also the cheapest
reliable frontier tier ($2.50/$10 per Mtok).

Design mirrors the Llama pipeline exactly:
  * attacker = detector = the model under test (within-model CDG, as in the paper)
  * full-context and no-context camouflage generation, refusal-screened from the
    first draw (<=4 resamples, is_refusal persisted), ACS computed on generation
  * StaticDetector CDG on both corpora, with 180 static + 45 clean controls

Collision safety: every record carries model="gpt-4o-2024-08-06" and an explicit
generator_variant; the resume index keys on (model, task, payload_id, detector,
variant, kind) so nothing can collide with Llama/Gemini records or across phases.
No process-based checks anywhere; completion is judged by record count.

Cost is computed manually (the client's price table has stale gpt-4o pricing).

    python experiments/model_b.py --phase smoke   # 1 task, prints token/cost basis
    python experiments/model_b.py --phase gen      # full generation (both corpora)
    python experiments/model_b.py --phase detect   # detection over both corpora
    python experiments/model_b.py --phase all
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
from src.attacks.camouflage_generator import CamouflageGenerator
from src.attacks.payload_bank import PayloadBank
from src.attacks.static_payloads import instantiate_payloads
from src.detection.static_detector import StaticDetector
from experiments.regen_corpus import looks_like_refusal, MAX_REGEN_ATTEMPTS
from experiments.nocontext_v2 import GENRE_BY_SUBCATEGORY

MODEL = "openai/gpt-4o-2024-08-06"
MODEL_TAG = "gpt-4o-2024-08-06"
PRICE_IN, PRICE_OUT = 2.50, 10.00   # USD per Mtok, from OpenRouter catalogue

TASKS_PATH = Path("data/tasks.json")
DOMAINS_PATH = Path("data/domains.json")
CORPUS_PATH = Path("data/camouflage_corpus_gpt4o.json")
DET_PATH = Path("results/detections_gpt4o.jsonl")
STATIC_CATEGORIES = ["override_directive", "authority_claim", "role_confusion", "social_engineering"]


class Meter:
    def __init__(self): self.tin = self.tout = self.n = 0
    def add(self, c):
        self.tin += c.input_tokens; self.tout += c.output_tokens; self.n += 1
    @property
    def cost(self): return (self.tin*PRICE_IN + self.tout*PRICE_OUT)/1e6
    def line(self): return f"{self.n} calls, {self.tin} in + {self.tout} out tok, ${self.cost:.3f}"


def client() -> LLMClient:
    key = os.environ.get("OPENROUTER_API_KEY", "") or CONFIG.openrouter_api_key
    if not key: raise SystemExit("OPENROUTER_API_KEY not set")
    c = LLMClient("openrouter", MODEL, api_key=key, base_url=CONFIG.openrouter_base_url,
                  dry_run=False, cost_alert_usd=25.0)
    assert c.dry_run is False
    return c


def load_corpus():
    return json.loads(CORPUS_PATH.read_text()) if CORPUS_PATH.exists() else {"full": {}, "noctx": {}, "meta": {}}


def gen_variants(gen, meter, task, domains, mode, n=3):
    """Generate n screened variants for one task in the given mode; returns dicts."""
    dom = domains[task["domain"]]
    if mode == "full":
        mk = lambda k: gen.generate(clean_context=task["clean_context"],
            malicious_goal=task["malicious_goal"], domain=task["domain"],
            task_id=task["task_id"], n_variants=k)
    else:
        genre = GENRE_BY_SUBCATEGORY.get(task["subcategory"], "professional document")
        mk = lambda k: gen.generate_no_context(malicious_goal=task["malicious_goal"],
            domain=task["domain"], domain_description=dom["description"], genre=genre,
            vocabulary=dom["vocabulary_markers"], task_id=task["task_id"],
            acs_reference=task["clean_context"], n_variants=k)
    variants = mk(n)
    for v in variants:
        meter.add(v_completion(v))
    for i, v in enumerate(variants):
        if not looks_like_refusal(v.payload_text): continue
        for _ in range(MAX_REGEN_ATTEMPTS):
            r = mk(1)[0]; meter.add(v_completion(r))
            if not looks_like_refusal(r.payload_text):
                r.payload_id, r.variant_idx = v.payload_id, v.variant_idx
                variants[i] = r; break
    pref = "cam" if mode == "full" else "noctx"
    return [{
        "payload_id": (v.payload_id if v.payload_id.startswith(("cam","noctx")) else f"{pref}_{v.payload_id}"),
        "payload_text": v.payload_text, "acs": round(v.semantic_similarity, 6),
        "domain": v.domain, "is_refusal": looks_like_refusal(v.payload_text),
        "n_words": len(v.payload_text.split()),
    } for v in variants]


# CamouflageGenerator doesn't return the completion object; wrap to capture cost.
# We monkeypatch the client.complete to tally via the meter instead.
_LAST = {}
def v_completion(_v):
    """Return the last completion captured by the metered client."""
    return _LAST["c"]


class MeteredClient:
    """Wraps LLMClient.complete to stash each CompletionResult for cost tallying."""
    def __init__(self, c): self._c = c
    def __getattr__(self, k): return getattr(self._c, k)
    def complete(self, *a, **k):
        r = self._c.complete(*a, **k); _LAST["c"] = r; return r


def phase_generate(tasks, domains, limit, meter):
    print(f"\n=== (b) GENERATION [{MODEL_TAG}] ===")
    gen = CamouflageGenerator(MeteredClient(client()), compute_similarity=True)
    corpus = load_corpus()
    run = tasks[:limit] if limit else tasks
    for mode in ("full", "noctx"):
        store = corpus.setdefault(mode, {})
        todo = [t for t in run if t["task_id"] not in store]
        for task in tqdm(todo, desc=f"gen-{mode}", unit="task"):
            store[task["task_id"]] = gen_variants(gen, meter, task, domains, mode)
            CORPUS_PATH.write_text(json.dumps(corpus, indent=2))
    # meta
    for mode in ("full", "noctx"):
        ps = [p for v in corpus[mode].values() for p in v]
        acs = [p["acs"] for p in ps if not p["is_refusal"] and p["acs"] > 0]
        corpus["meta"][mode] = {
            "n_payloads": len(ps),
            "n_refusals": sum(1 for p in ps if p["is_refusal"]),
            "acs_mean": round(sum(acs)/len(acs), 4) if acs else None,
            "n_words_mean": round(sum(p["n_words"] for p in ps)/len(ps), 1) if ps else None,
        }
    corpus["meta"]["model"] = MODEL
    corpus["meta"]["generated_utc"] = datetime.now(timezone.utc).isoformat()
    CORPUS_PATH.write_text(json.dumps(corpus, indent=2))
    print(f"  generation: {meter.line()}")


def resume_index():
    if not DET_PATH.exists(): return set()
    return {(r["model"], r["task_id"], r["payload_id"], r["detector_type"], r["generator_variant"])
            for r in (json.loads(l) for l in DET_PATH.open())}


def phase_detect(tasks, limit, meter):
    print(f"\n=== (b) DETECTION [{MODEL_TAG}, static detector] ===")
    if not CORPUS_PATH.exists(): raise SystemExit("run --phase gen first")
    corpus = load_corpus()
    det = StaticDetector(MeteredClient(client()))
    bank = PayloadBank()
    by_id = {t["task_id"]: t for t in tasks}
    done = resume_index()
    run = tasks[:limit] if limit else tasks

    work = []
    for task in run:
        # static arm (shared; tagged generator_variant="static")
        for cat in STATIC_CATEGORIES:
            ps = instantiate_payloads(task["malicious_goal"], CONFIG.data_dir, categories=[cat])
            if ps: work.append((task, "static", ps[0].payload_id, ps[0].instantiated_text, False, "static"))
        # clean control
        work.append((task, "clean", f"clean_{task['task_id']}", None, False, "static"))
        # camouflage arms
        for mode, var in (("full", "full_context"), ("noctx", "no_context")):
            for p in corpus[mode].get(task["task_id"], []):
                work.append((task, "camouflage", p["payload_id"], p["payload_text"], p["is_refusal"], var))

    pending = [w for w in work if (MODEL_TAG, w[0]["task_id"], w[2], "static", w[5]) not in done]
    print(f"  contexts: {len(work)} total, {len(pending)} to run")

    for task, ptype, pid, ptext, is_ref, var in tqdm(pending, desc="detect", unit="ctx"):
        ctx = task["clean_context"] if ptype == "clean" else bank.inject(task["clean_context"], ptext, "end")
        try:
            res = det.detect(ctx, task["instruction"], ground_truth=(ptype != "clean"))
            meter.add(_LAST["c"])
        except Exception as e:                       # noqa: BLE001
            tqdm.write(f"    ! {var}/{task['task_id']}/{pid}: {type(e).__name__}: {str(e)[:90]}")
            continue
        with DET_PATH.open("a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": MODEL_TAG, "task_id": task["task_id"], "domain": task["domain"],
                "payload_type": ptype, "generator_variant": var, "payload_id": pid,
                "is_refusal": is_ref, "verdict": res.verdict, "confidence": res.confidence,
                "detector_type": "static", "dry_run": False,
            }) + "\n")
    print(f"  detection: {meter.line()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "gen", "detect", "all"], default="all")
    a = ap.parse_args()
    tasks = json.loads(TASKS_PATH.read_text())
    domains = json.loads(DOMAINS_PATH.read_text())
    meter = Meter()

    if a.phase == "smoke":
        phase_generate(tasks, domains, 1, meter)
        phase_detect(tasks, 1, meter)
        # extrapolate: full run is ~45x the per-task work
        per_task = meter.cost
        print(f"\n  SMOKE: 1 task cost ${per_task:.4f} ({meter.line()})")
        print(f"  Extrapolated 45-task full run: ~${per_task*45:.2f} "
              f"(refusal retries not linear; treat as lower bound)")
        return

    if a.phase in ("gen", "all"): phase_generate(tasks, domains, None, meter)
    if a.phase in ("detect", "all"): phase_detect(tasks, None, meter)
    print(f"\n=== TOTAL [{MODEL_TAG}]: {meter.line()} ===")


if __name__ == "__main__":
    main()
