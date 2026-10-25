"""
Item 3 (pre-registration 4d6687c): instantiate the directive x vocabulary templates.

build_payloads() returns payload dicts in memory; nothing is written until the templates are frozen (P2).

    python experiments/mechanism_payloads.py --check   # length + automatic directiveness check (P3)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG

TEMPLATES = Path("data/mechanism_templates.json")
NEUTRAL_GOALS = Path("data/mechanism_goals_neutral.json")  # amendment A1: primary goal set
GENERAL_BASIS = ["as the text states", "as the document indicates", "according to the {authority_structure}",
                 "based on the passage"]
DIRECTIVE_RE = re.compile(r"\b(you|your)\b|\b(must|should|are required to|make sure|present|state|cite|keep|apply|"
                          r"treat|avoid)\b", re.I)


def build_payloads(probe: bool = True, goals: str = "neutral") -> list[dict]:
    """goals: "neutral" (A1 primary, data/mechanism_goals_neutral.json) or "verbatim" (tasks.json malicious_goal)."""
    spec = json.loads(TEMPLATES.read_text())
    tasks = json.loads(Path("data/tasks.json").read_text())
    neutral = json.loads(NEUTRAL_GOALS.read_text())["goals"] if goals == "neutral" else None
    domains = json.loads(Path("data/domains.json").read_text())
    rng = random.Random(CONFIG.seed)
    out = []
    for t in tasks:
        dom = domains[t["domain"]]
        auth = rng.choice(dom["authority_structures"])
        if t["domain"] == "general":
            basis = rng.choice(GENERAL_BASIS).format(authority_structure=auth)
        else:
            m1, m2 = rng.sample(dom["vocabulary_markers"], 2)
            basis = f"the {m1} and {m2} considerations"
        for cell, temps in spec["cells"].items():
            for tpl in temps:
                if "all" in tpl:
                    form = tpl["all"]
                else:
                    form = tpl["general"] if t["domain"] == "general" else tpl["fin_leg"]
                goal = neutral[t["task_id"]] if neutral else t["malicious_goal"]
                text = form.format(goal=goal, authority_structure=auth, basis=basis)
                tag = "n" if neutral else "v"
                base = {"task_id": t["task_id"], "domain": t["domain"], "template_id": tpl["id"], "goal_set": goals,
                        "goal": goal}
                out.append({**base, "cell": cell, "payload_id": f"mech{tag}_{t['task_id']}_{tpl['id']}", "text": text})
                if probe and cell in ("D+V-", "D+V+"):
                    out.append({**base, "cell": cell + "+L", "payload_id": f"mech{tag}_{t['task_id']}_{tpl['id']}_L",
                                "text": spec["probe_prefix"] + text})
    return out


def check(goals: str) -> None:
    ps = build_payloads(goals=goals)
    by = defaultdict(list)
    for p in ps:
        by[p["cell"]].append(p)
    print(f"{len(ps)} payloads, goal set {goals}")
    bad = []
    for cell, xs in by.items():
        w = [len(p["text"].split()) for p in xs]
        dirc = [len(DIRECTIVE_RE.findall(p["text"])) for p in xs]
        print(f"  {cell:<7} n={len(xs):<4} words {min(w)}-{max(w)} (mean {sum(w)/len(w):.1f})  directive-marker count mean {sum(dirc)/len(dirc):.2f}")
        bad += [(p["payload_id"], len(p["text"].split())) for p in xs if not 55 <= len(p["text"].split()) <= 70
                and not p["cell"].endswith("+L")]
    print(f"  core payloads outside 55-70 words: {len(bad)}")
    per_t = defaultdict(list)
    for pid, n in bad:
        per_t[pid.rsplit('_', 1)[-1]].append(n)
    for k, v in sorted(per_t.items()):
        print(f"    template {k}: {len(v)} payloads, words {min(v)}-{max(v)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--goals", choices=["neutral", "verbatim"], default="neutral")
    a = ap.parse_args()
    if a.check:
        check(a.goals)
