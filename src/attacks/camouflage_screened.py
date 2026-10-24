"""
Shared generate -> refusal-screen -> cache for the AgentDojo camouflage arms (item 5b). Delegates to the repo's
existing CamouflageGenerator and refusal screen (analysis/refusal_screen_v2). Every payload is written to
data/item5b_camouflage_payloads.json keyed by id and is NEVER printed or returned to logs in bulk. Generation and
screening both run on local Ollama (llama3.1), so producing the banking payloads costs no API credit.

  screened_payload(..., generate=False)  matrix time: cache-only lookup; raise NoScreenedPayload on a miss.
  screened_payload(..., generate=True)   pre-step: generate n variants, screen, cache the first that passes.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

CACHE = Path("data/item5b_camouflage_payloads.json")
N_VARIANTS = 3
GEN_MODEL = "llama3.1:latest"  # local attacker/generator, as in the item3c Llama arm (free)
_LOCK = threading.Lock()


def _load() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def _key(arm: str, injection_task_id: str, field_id: str) -> str:
    return f"{arm}|{injection_task_id}|{field_id}"


def screened_payload(arm: str, goal: str, surrounding: str | None, field_id: str, injection_task_id: str,
                     domain: str = "financial", generate: bool = False) -> str:
    """Return a screened camouflage payload text for one injection slot, or raise NoScreenedPayload."""
    from src.attacks.agentdojo_camouflage_attack import NoScreenedPayload

    key = _key(arm, injection_task_id, field_id)
    hit = _load().get(key)
    if hit is not None:
        if hit.get("status") == "screened":
            return hit["text"]
        raise NoScreenedPayload(key)
    if not generate:
        raise NoScreenedPayload(key)
    rec = _generate_and_screen(arm, goal, surrounding, field_id, injection_task_id, domain)
    with _LOCK:
        store = _load()
        store[key] = rec
        CACHE.write_text(json.dumps(store, indent=0))
    if rec["status"] == "screened":
        return rec["text"]
    raise NoScreenedPayload(key)


def _generate_and_screen(arm: str, goal: str, surrounding: str | None, field_id: str,
                         injection_task_id: str, domain: str) -> dict:
    """Run the existing CamouflageGenerator for this slot and keep the first variant that passes the refusal screen."""
    from analysis.refusal_screen_v2 import final, llm_class
    from experiments.regen_corpus import refusal_class
    from src.attacks.camouflage_generator import CamouflageGenerator
    from src.utils.llm_client import LLMClient

    gen = CamouflageGenerator(LLMClient("ollama", GEN_MODEL, dry_run=False), compute_similarity=False)
    task_id = f"{injection_task_id}_{field_id}"
    if arm == "camouflage_full":
        cands = gen.generate(surrounding or "", goal, domain, task_id, n_variants=N_VARIANTS)
    else:  # camouflage_no_context: generator never sees the surrounding text, only domain/genre/vocab
        dom = json.loads(Path("data/domains.json").read_text())[domain]
        cands = gen.generate_no_context(goal, domain, dom["description"], dom["authority_structures"][0],
                                        dom["vocabulary_markers"], task_id, acs_reference=surrounding, n_variants=N_VARIANTS)
    screen = LLMClient("ollama", GEN_MODEL, dry_run=False)
    base = {"arm": arm, "injection_task_id": injection_task_id, "field_id": field_id, "n_candidates": len(cands)}
    for c in cands:
        if final(refusal_class(c.payload_text), llm_class(screen, c.payload_text)[0]) == "payload":
            return {**base, "status": "screened", "payload_id": c.payload_id, "text": c.payload_text}
    return {**base, "status": "refused"}
