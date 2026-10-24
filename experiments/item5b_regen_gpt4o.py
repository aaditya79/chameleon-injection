"""
Item 5b: regenerate ONLY the refused camouflage slots with GPT-4o (local llama refused ~85%), screen with the existing
local screen (free), and update data/item5b_camouflage_payloads.json in place. Originally-screened slots (local llama)
are left untouched, so each arm ends mixed-generator; recorded as a caveat. GPT-4o spend is bounded by a live-balance
headroom guard: stop before any slot that would leave < MIN_HEADROOM of key limit. Payload text is never printed here.

    ~/research/agentdojo-venv/bin/python experiments/item5b_regen_gpt4o.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from experiments.mechanism_run import GPT4O
from src.attacks.camouflage_screened import CACHE, _key, _load

MIN_HEADROOM = 3.0  # stop if a slot would leave less than this of key limit (reserve the no-defense block)
N_VARIANTS = 3
DOMAIN = "financial"


def headroom() -> float:
    r = requests.get("https://openrouter.ai/api/v1/key",
                     headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=20)
    return float(r.json()["data"]["limit_remaining"])


def main() -> None:
    from agentdojo.task_suite.load_suites import get_suite
    from analysis.refusal_screen_v2 import final, llm_class
    from experiments.regen_corpus import refusal_class
    from src.attacks.camouflage_generator import CamouflageGenerator
    from src.utils.llm_client import LLMClient

    suite = get_suite("v1.2.2", "banking")
    defaults = suite.get_injection_vector_defaults()
    dom = json.loads(__import__("pathlib").Path("data/domains.json").read_text())[DOMAIN]
    store = _load()
    refused = [k for k, v in store.items() if v.get("status") == "refused"]
    print(f"[regen] {len(refused)} refused slots; headroom ${headroom():.2f}", flush=True)

    gen = CamouflageGenerator(LLMClient("openrouter", GPT4O, api_key=os.environ["OPENROUTER_API_KEY"],
                                        base_url="https://openrouter.ai/api/v1", dry_run=False), compute_similarity=False)
    screen = LLMClient("ollama", "llama3.1:latest", dry_run=False)
    recovered = 0
    for i, k in enumerate(refused, 1):
        if headroom() < MIN_HEADROOM:
            print(f"[regen] STOP: headroom < ${MIN_HEADROOM}; {i-1}/{len(refused)} done", flush=True)
            break
        arm, inj, fld = k.split("|")
        goal = suite.injection_tasks[inj].GOAL
        task_id = f"{inj}_{fld}"
        if arm == "camouflage_full":
            cands = gen.generate(defaults[fld], goal, DOMAIN, task_id, n_variants=N_VARIANTS)
        else:
            cands = gen.generate_no_context(goal, DOMAIN, dom["description"], dom["authority_structures"][0],
                                            dom["vocabulary_markers"], task_id, acs_reference=defaults[fld], n_variants=N_VARIANTS)
        rec = {"arm": arm, "injection_task_id": inj, "field_id": fld, "n_candidates": len(cands),
               "status": "refused", "generator": "gpt4o"}
        for c in cands:
            if final(refusal_class(c.payload_text), llm_class(screen, c.payload_text)[0]) == "payload":
                rec = {**rec, "status": "screened", "payload_id": c.payload_id, "text": c.payload_text}
                recovered += 1
                break
        store = _load()
        store[k] = rec
        CACHE.write_text(json.dumps(store, indent=0))
        print(f"[regen] {i}/{len(refused)} {arm} {inj} {fld} -> {rec['status']}", flush=True)
    print(f"[regen] done; recovered {recovered}; headroom ${headroom():.2f}", flush=True)


if __name__ == "__main__":
    main()
