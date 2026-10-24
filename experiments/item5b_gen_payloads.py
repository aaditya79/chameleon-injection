"""
Item 5b pre-step: generate the banking camouflage payloads for both arms and screen them, BEFORE the agent matrix.
Generation + refusal screen run on local Ollama (no API credit). Payloads are written by id to
data/item5b_camouflage_payloads.json (via src.attacks.camouflage_screened); this script prints only aggregate
counts. Run with the AgentDojo venv (for the suite):

    ~/research/agentdojo-venv/bin/python experiments/item5b_gen_payloads.py gen      # generate + screen + cache
    ~/research/agentdojo-venv/bin/python experiments/item5b_gen_payloads.py report   # per-arm counts only (no gen)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VERSION, SUITE, ARMS = "v1.2.2", "banking", ("camouflage_full", "camouflage_no_context")


def _suite_and_attacks():
    import src.attacks.agentdojo_camouflage_attack  # registers the arms
    from agentdojo.attacks.attack_registry import ATTACKS
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite(VERSION, SUITE)
    return suite, {a: ATTACKS[a](suite, None) for a in ARMS}


def _slots(suite, attack):
    """Distinct (injection_task, field) slots over the 144 pairs, with goal and surrounding field text."""
    defaults = suite.get_injection_vector_defaults()
    k = lambda x: int(x.split("_")[-1])
    seen = {}
    for u in sorted(suite.user_tasks, key=k):
        for field in attack.get_injection_candidates(suite.user_tasks[u]):
            for inj in sorted(suite.injection_tasks, key=k):
                seen[(inj, field)] = (suite.injection_tasks[inj].GOAL, defaults[field])
    return seen


def _pairs(suite, attack):
    k = lambda x: int(x.split("_")[-1])
    return [(u, inj, attack.get_injection_candidates(suite.user_tasks[u]))
            for u in sorted(suite.user_tasks, key=k) for inj in sorted(suite.injection_tasks, key=k)]


def gen() -> None:
    from src.attacks.agentdojo_camouflage_attack import NoScreenedPayload
    from src.attacks.camouflage_screened import screened_payload

    suite, attacks = _suite_and_attacks()
    for arm in ARMS:
        slots = _slots(suite, attacks[arm])
        print(f"[gen:{arm}] {len(slots)} distinct (injection_task, field) slots", flush=True)
        for i, ((inj, field), (goal, surrounding)) in enumerate(slots.items(), 1):
            try:
                screened_payload(arm, goal, surrounding, field, inj, generate=True)
            except NoScreenedPayload:
                pass
            print(f"[gen:{arm}] {i}/{len(slots)} {inj} {field}", flush=True)
    report()


def report() -> None:
    import json
    from pathlib import Path

    from src.attacks.camouflage_screened import _key

    suite, attacks = _suite_and_attacks()
    store = json.loads(Path("data/item5b_camouflage_payloads.json").read_text()) if Path(
        "data/item5b_camouflage_payloads.json").exists() else {}
    print("\n## Item 5b banking camouflage payloads — per-arm counts (144 pairs)")
    for arm in ARMS:
        pairs = _pairs(suite, attacks[arm])
        scr = ref = 0
        for u, inj, fields in pairs:
            recs = [store.get(_key(arm, inj, f)) for f in fields]
            ok = recs and all(r and r.get("status") == "screened" for r in recs)
            scr += bool(ok)
            ref += not ok
        slots = _slots(suite, attacks[arm])
        slot_scr = sum(1 for (inj, field) in slots if (store.get(_key(arm, inj, field)) or {}).get("status") == "screened")
        print(f"- {arm}: pairs screened {scr}/{len(pairs)}, refused {ref}/{len(pairs)}; "
              f"slots screened {slot_scr}/{len(slots)}")


if __name__ == "__main__":
    (gen if len(sys.argv) > 1 and sys.argv[1] == "gen" else report)()
