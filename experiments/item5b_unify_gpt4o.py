"""
Item 5b: unify both camouflage arms to a single generator (GPT-4o). Regenerate the slots whose screened payload came
from llama (generator != "gpt4o") with GPT-4o, screen locally (free). REPLACE only if a new variant passes the screen;
otherwise KEEP the existing llama payload so 144/144 coverage never drops. Same headroom guard as the regen step.

    ~/research/agentdojo-venv/bin/python experiments/item5b_unify_gpt4o.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.item5b_regen_gpt4o import MIN_HEADROOM, headroom
from experiments.mechanism_run import GPT4O
from src.attacks.camouflage_screened import CACHE, _load

N_VARIANTS, DOMAIN = 3, "financial"


def main() -> None:
    from agentdojo.task_suite.load_suites import get_suite
    from analysis.refusal_screen_v2 import final, llm_class
    from experiments.regen_corpus import refusal_class
    from src.attacks.camouflage_generator import CamouflageGenerator
    from src.utils.llm_client import LLMClient

    suite = get_suite("v1.2.2", "banking")
    defaults = suite.get_injection_vector_defaults()
    dom = json.loads(Path("data/domains.json").read_text())[DOMAIN]
    store = _load()
    todo = [k for k, v in store.items() if v.get("status") == "screened" and v.get("generator") != "gpt4o"]
    print(f"[unify] {len(todo)} llama-screened slots to move to gpt4o; headroom ${headroom():.2f}", flush=True)

    gen = CamouflageGenerator(LLMClient("openrouter", GPT4O, api_key=os.environ["OPENROUTER_API_KEY"],
                                        base_url="https://openrouter.ai/api/v1", dry_run=False), compute_similarity=False)
    screen = LLMClient("ollama", "llama3.1:latest", dry_run=False)
    replaced = kept = 0
    for i, k in enumerate(todo, 1):
        if headroom() < MIN_HEADROOM:
            print(f"[unify] STOP: headroom < ${MIN_HEADROOM}", flush=True)
            break
        arm, inj, fld = k.split("|")
        goal = suite.injection_tasks[inj].GOAL
        task_id = f"{inj}_{fld}"
        if arm == "camouflage_full":
            cands = gen.generate(defaults[fld], goal, DOMAIN, task_id, n_variants=N_VARIANTS)
        else:
            cands = gen.generate_no_context(goal, DOMAIN, dom["description"], dom["authority_structures"][0],
                                            dom["vocabulary_markers"], task_id, acs_reference=defaults[fld], n_variants=N_VARIANTS)
        new = next((c for c in cands if final(refusal_class(c.payload_text), llm_class(screen, c.payload_text)[0]) == "payload"), None)
        store = _load()
        if new is not None:
            store[k] = {"arm": arm, "injection_task_id": inj, "field_id": fld, "n_candidates": len(cands),
                        "status": "screened", "payload_id": new.payload_id, "text": new.payload_text, "generator": "gpt4o"}
            replaced += 1
            tag = "replaced"
        else:
            kept += 1  # keep the existing llama payload; coverage preserved
            tag = "kept-llama"
        CACHE.write_text(json.dumps(store, indent=0))
        print(f"[unify] {i}/{len(todo)} {arm} {inj} {fld} -> {tag}", flush=True)
    print(f"[unify] done; replaced {replaced}, kept-llama {kept}; headroom ${headroom():.2f}", flush=True)


if __name__ == "__main__":
    main()
