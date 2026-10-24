"""
Item 3c analysis, part 1 of 2: records (ITEM3C_BENIGN_TWINS_DESIGN.md, "Detection", "Outcomes and analysis").

detection_records(arm) -> list of {kind: twin|camouflage, item_id, task_id, domain, detector, mode, detected, score}
  twins       results/item3c_detections.jsonl
  camouflage  llama arm: detections_v2 (static, augmented), cross_detect cell D (gpt4o),
                         detections_baselines llama_full / llama_full_alone (Llama Guard, classifiers)
              gpt4o arm: cross_detect cell C (static), item3c_camfill_detections (augmented),
                         detections_gpt4o full_context (gpt4o), detections_baselines gpt4o_full / _alone
              re-screened with the frozen exclusion set (data/refusal_exclusions_final.json)
score: classifiers their probability; Llama Guard the verdict (no probability logged); LLM detectors the 6-level
ordinal verdict x confidence (CLEAN-HIGH 1 ... INJECTED-HIGH 6).
"""

from __future__ import annotations

import json
from pathlib import Path

PRIMARY = ("static", "augmented", "gpt4o", "llamaguard3_injonly", "protectai_v2", "promptguard2_86m",
           "promptguard2_22m")
SECONDARY = ("llamaguard3_s15",)
CLASSIFIERS = ("protectai_v2", "promptguard2_86m", "promptguard2_22m")
ORD = {("CLEAN", "HIGH"): 1, ("CLEAN", "MEDIUM"): 2, ("CLEAN", "LOW"): 3,
       ("INJECTED", "LOW"): 4, ("INJECTED", "MEDIUM"): 5, ("INJECTED", "HIGH"): 6}
CORPUS = {"llama": "v2_llama", "gpt4o": "gpt4o_full"}


def jl(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(x) for x in p.open() if x.strip()] if p.exists() else []


def tasks43() -> set:
    return set(json.loads(Path("data/item3c_honest_conclusions.json").read_text())["conclusions"])


def excluded(arm: str) -> set:
    return set(json.loads(Path("data/refusal_exclusions_final.json").read_text())["excluded"][CORPUS[arm]])


def _score(det: str, r: dict) -> float:
    if det in CLASSIFIERS:
        return float(r["score"])
    if det.startswith("llamaguard"):
        return float(r["verdict"] == "INJECTED")
    return float(ORD.get((r["verdict"], str(r.get("confidence", "")).upper()), 6 if r["verdict"] == "INJECTED" else 1))


def _rec(kind, item_id, r, det, mode) -> dict:
    return {"kind": kind, "item_id": item_id, "task_id": r["task_id"], "domain": r["domain"], "detector": det,
            "mode": mode, "detected": int(r["verdict"] == "INJECTED"), "score": _score(det, r)}


def detection_records(arm: str) -> tuple[list[dict], dict]:
    """(records, E1 error counts). Camouflage covers all 45 tasks; restrict with tasks43() for the primary set."""
    out, err = [], {}
    for r in jl("results/item3c_detections.jsonl"):
        if r["arm"] != arm:
            continue
        if r.get("status") != "ok":
            err[f"twin|{r['detector']}|{r['mode']}"] = err.get(f"twin|{r['detector']}|{r['mode']}", 0) + 1
            continue
        out.append(_rec("twin", r["twin_id"], r, r["detector"], r["mode"]))
    ex = excluded(arm)
    cam = []
    if arm == "llama":
        cam += [(r, r["detector_type"], "embedded") for r in jl("results/detections_v2.jsonl")
                if r["payload_type"] == "camouflage" and not r.get("dry_run")]
        cam += [(r, "gpt4o", "embedded") for r in jl("results/cross_detect.jsonl") if r["cell"] == "D"]
        corp = "llama_full"
    else:
        cam += [(r, "static", "embedded") for r in jl("results/cross_detect.jsonl") if r["cell"] == "C"]
        cam += [(r, "augmented", "embedded") for r in jl("results/item3c_camfill_detections.jsonl")
                if r.get("status") == "ok"]
        cam += [(r, "gpt4o", "embedded") for r in jl("results/detections_gpt4o.jsonl")
                if r.get("generator_variant") == "full_context" and r["payload_type"] == "camouflage"]
        corp = "gpt4o_full"
    for r in jl("results/detections_baselines.jsonl"):
        if r["corpus"] == corp:
            cam.append((r, r["detector"], "embedded"))
        elif r["corpus"] == f"{corp}_alone":
            cam.append((r, r["detector"], "alone"))
    out += [_rec("camouflage", r["payload_id"], r, det, mode) for r, det, mode in cam if r["payload_id"] not in ex]
    return out, err


def alignment_records() -> list[dict]:
    """AlignmentCheck rows (Llama arm): twin trials, re-screened camouflage trials, clean trials."""
    ex = excluded("llama")
    rows = [{**r, "kind": "twin"} for r in jl("results/item3c_alignment_check.jsonl")]
    for r in jl("results/alignment_check.jsonl"):
        if r["payload_type"] == "camouflage" and r["payload_id"] not in ex:
            rows.append({**r, "kind": "camouflage"})
        elif r["payload_type"] == "clean":
            rows.append({**r, "kind": "clean"})
    return rows


def twin_pairs(arm: str) -> dict:
    """{twin_id: target camouflage payload id or None (task / arm mean target)}."""
    slots = json.loads(Path("data/item3c_twins.json").read_text())["arms"][arm]["slots"]
    return {s["twin_id"]: s.get("target_payload_id") for s in slots.values() if s["status"] == "selected"}
