"""
Item 3b analysis, part 1 of 3: data loading (pre-registration 654a4cf; A-log amendments incl. 8).

Everything is read by id from the frozen files. Texts are used only to compute word counts and specificity with the
frozen extractor (experiments/item3b_rewrite.py); they are never printed or written out.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = ("O", "a", "a_prime", "f", "b", "c", "d", "e")
DETECTORS = ("static", "augmented", "gpt4o", "llamaguard3_injonly", "protectai_v2", "promptguard2_86m",
             "promptguard2_22m")
CLASSIFIERS = ("protectai_v2", "promptguard2_86m", "promptguard2_22m")


def jl(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).open() if x.strip()]


def source_meta() -> dict:
    """{source payload id: {task_id, domain, accepted arms}}."""
    src = json.loads(Path("data/item3b_payloads.json").read_text())["payloads"]
    return {pid: {"task_id": r["task_id"], "domain": r["domain"],
                  "arms": [a for a in ARMS if r["arms"][a]["status"] == "accepted"]} for pid, r in src.items()}


def analysis_sets(meta: dict) -> dict:
    """primary = amendment 8 exclusions removed; sensitivity = all source payloads."""
    excl = {x["payload_id"] for x in json.loads(Path("data/item3b_primary_exclusions.json").read_text())
            ["excluded_source_payloads"]}
    assert excl <= set(meta)
    return {"primary": sorted(p for p in meta if p not in excl), "sensitivity": sorted(meta)}


def phrase_type_by_task() -> dict:
    return {t: v["type"] for t, v in json.loads(Path("data/item3b_intent_phrases.json").read_text())["phrases"].items()
            if v}


def text_features() -> dict:
    """{(pid, arm): {"words", "spec"}} from the frozen extractor; texts stay inside this function."""
    from experiments.item3b_rewrite import specificity, words

    src = json.loads(Path("data/item3b_payloads.json").read_text())["payloads"]
    ctx = {t["task_id"]: t["clean_context"] for t in json.loads(Path("data/tasks.json").read_text())}
    out = {}
    for pid, r in src.items():
        for arm in ARMS:
            a = r["arms"][arm]
            if a["status"] == "accepted":
                out[(pid, arm)] = {"words": words(a["text"]), "spec": specificity(a["text"], ctx[r["task_id"]])}
    return out


def detections(path: str | Path) -> tuple[dict, dict]:
    """({(detector, mode): {(pid, arm): 0/1}}, error counts). Records with status != "ok" are excluded (E1)."""
    y, err = {}, Counter()
    for r in jl(path):
        if r.get("status") != "ok":
            err[f"{r['detector']}|{r['mode']}"] += 1
            continue
        y.setdefault((r["detector"], r["mode"]), {})[(r["template_id"], r["cell"])] = int(r["verdict"] == "INJECTED")
    return y, dict(err)


def asr(path: str | Path) -> tuple[dict, int]:
    """({(pid, arm): 0/1 agent followed}, n excluded)."""
    v, n_err = {}, 0
    for r in jl(path):
        if r.get("status", "ok") != "ok" or r.get("valid") is None:
            n_err += 1
            continue
        v[(r["template_id"], r["cell"])] = int(bool(r["valid"]))
    return v, n_err
