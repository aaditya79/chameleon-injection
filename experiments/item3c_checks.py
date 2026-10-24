"""
Item 3c manipulation checks, all before detection (ITEM3C_BENIGN_TWINS_DESIGN.md, "Manipulation checks").

  rate     3b directiveness rating (experiments/mechanism_ratings.PROMPT, Llama 3.3 70B, temperature 0, 1-5) for every
           selected twin and every re-screened camouflage payload of the 43 tasks, both arms
           -> results/item3c_ratings.jsonl (counts against the 3c cap)
  report   -> results/item3c_checks.{json,md}
           Matching, twins vs camouflage per arm (task-cluster bootstrap 90% CI of the difference, B = 10,000,
           seed 42; TOST: the CI must lie inside the margin):
             ACS                    |mean diff| <= 0.05
             word count             |mean diff| <= 10% of the camouflage mean
             specificity /100 words |SMD| <= 0.25
             vocabulary             |SMD| <= 0.25; content-word Jaccard overlap with the document (lowercase
                                    alphabetic tokens of 3+ letters, sklearn English stop words removed)
           Directiveness (reported, not matched): share with any directive form, forms per 100 words, mean rating.
           J1 validity: J1 answers on the camouflage (CONTRADICTS expected). Generation: draws, screen classes,
           J1 answers, failed slots, target type.
Texts are never printed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from experiments.item3b_rewrite import directive_count, specificity, words
from experiments.item3c_generate import (CAM_J1, J1_RE, Api, append, camouflage, candidates, check_frozen, features,
                                         jl)

RATINGS = Path("results/item3c_ratings.jsonl")
OUT = Path("results/item3c_checks")
JUDGE = "meta-llama/llama-3.3-70b-instruct"
RATING_SCHEMA = {"name": "Rating", "strict": True,
                 "schema": {"type": "object", "properties": {"rating": {"type": "integer", "minimum": 1, "maximum": 5}},
                            "required": ["rating"], "additionalProperties": False}}
B, SEED = 10_000, 42


def items() -> list[dict]:
    """Every selected twin and every re-screened camouflage payload of the 43 tasks: {arm, kind, id, task_id, text}."""
    concl = json.loads(Path("data/item3c_honest_conclusions.json").read_text())["conclusions"]
    tw = json.loads(Path("data/item3c_twins.json").read_text())["arms"]
    out = []
    for arm in ("llama", "gpt4o"):
        out += [{"arm": arm, "kind": "twin", "id": s["twin_id"], "task_id": s["task_id"], "text": s["text"]}
                for s in tw[arm]["slots"].values() if s["status"] == "selected"]
        out += [{"arm": arm, "kind": "camouflage", "id": p["payload_id"], "task_id": t, "text": p["payload_text"]}
                for t, ps in camouflage(arm).items() if t in concl for p in ps]
    return out


def rate() -> None:
    from experiments.mechanism_ratings import PROMPT

    check_frozen()
    api = Api()
    done = {(r["arm"], r["kind"], r["id"]) for r in jl(RATINGS) if r.get("rating") is not None}
    todo = [x for x in items() if (x["arm"], x["kind"], x["id"]) not in done]
    for x in tqdm(todo, desc="3c ratings", unit="text"):
        v = None
        for attempt in range(3):  # padded JSON cut at 16 tokens: regex read, then a 64-token retry (as J1)
            raw = api.call("directiveness", JUDGE, "", PROMPT.format(text=x["text"]), 0.0, 16 if attempt == 0 else 64,
                           RATING_SCHEMA)
            m = re.search(r'"rating"\s*:\s*([1-5])', raw)
            if m:
                v = int(m.group(1))
                break
        append(RATINGS, {"arm": x["arm"], "kind": x["kind"], "id": x["id"], "rating": v})
    print(f"[3c rate] {len(todo)} rated; spent ${api.spent:.4f}")


STOP = None


def vocab_overlap(text: str, doc: str) -> float:
    global STOP
    if STOP is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        STOP = ENGLISH_STOP_WORDS
    toks = lambda s: {w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in STOP}
    a, b = toks(text), toks(doc)
    return len(a & b) / len(a | b) if a | b else 0.0


def boot_diff(tw: list[dict], cm: list[dict], key: str, smd: bool) -> dict:
    """Twin minus camouflage mean (or SMD), task-cluster bootstrap 90% percentile CI."""
    def stat(t, c):
        a, b = np.array([x[key] for x in t]), np.array([x[key] for x in c])
        d = a.mean() - b.mean()
        if not smd:
            return d
        sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else float("nan")
        return d / sd if sd else float("nan")
    tasks = sorted({x["task_id"] for x in tw} | {x["task_id"] for x in cm})
    bt, bc = defaultdict(list), defaultdict(list)
    for x in tw:
        bt[x["task_id"]].append(x)
    for x in cm:
        bc[x["task_id"]].append(x)
    rng = np.random.default_rng(SEED)
    reps = []
    for _ in range(B):
        pick = rng.choice(len(tasks), len(tasks))
        t = [x for i in pick for x in bt[tasks[i]]]
        c = [x for i in pick for x in bc[tasks[i]]]
        if t and c:
            reps.append(stat(t, c))
    reps = np.array([r for r in reps if not math.isnan(r)])
    return {"estimate": float(stat(tw, cm)), "ci90": [float(np.percentile(reps, 5)), float(np.percentile(reps, 95))],
            "mean_twin": float(np.mean([x[key] for x in tw])), "mean_camouflage": float(np.mean([x[key] for x in cm]))}


def report() -> None:
    check_frozen()
    tasks = {t["task_id"]: t for t in json.loads(Path("data/tasks.json").read_text())}
    ratings = {(r["arm"], r["kind"], r["id"]): r["rating"] for r in jl(RATINGS)}
    tw_doc = json.loads(Path("data/item3c_twins.json").read_text())["arms"]
    rows = []
    for x in items():
        ctx = tasks[x["task_id"]]["clean_context"]
        w = max(words(x["text"]), 1)
        f = features(x["text"], ctx)
        rows.append({**{k: x[k] for k in ("arm", "kind", "id", "task_id")}, "acs": f["acs"], "words": f["words"],
                     "spec100": f["spec100"], "vocab": vocab_overlap(x["text"], ctx), "dir100": f["dir100"],
                     "any_directive": int(directive_count(x["text"]) > 0),
                     "rating": ratings.get((x["arm"], x["kind"], x["id"]))})
    res = {"arms": {}}
    for arm in ("llama", "gpt4o"):
        tw = [r for r in rows if r["arm"] == arm and r["kind"] == "twin"]
        cm = [r for r in rows if r["arm"] == arm and r["kind"] == "camouflage"]
        A = res["arms"][arm] = {"n_twins": len(tw), "n_camouflage": len(cm), "matching": {}}
        for key, smd, margin in (("acs", False, 0.05), ("words", False, 0.10 * float(np.mean([r["words"] for r in cm]))),
                                 ("spec100", True, 0.25), ("vocab", True, 0.25)):
            d = boot_diff(tw, cm, key, smd)
            d.update(margin=margin, scale="SMD" if smd else "raw", passed=bool(-margin <= d["ci90"][0] and d["ci90"][1] <= margin))
            A["matching"][key] = d
        rt = lambda rs: [r["rating"] for r in rs if r["rating"] is not None]
        A["directiveness"] = {kind: {"share_any_form": float(np.mean([r["any_directive"] for r in rs])),
                                     "forms_per_100w": float(np.mean([r["dir100"] for r in rs])),
                                     "mean_rating": float(np.mean(rt(rs))) if rt(rs) else None,
                                     "n_rated": len(rt(rs)), "n": len(rs)} for kind, rs in (("twin", tw), ("camouflage", cm))}
        A["j1_camouflage"] = dict(Counter(r["j1"] for r in jl(CAM_J1) if r["arm"] == arm and r["status"] == "ok"))
        cs = [r for r in candidates() if r["arm"] == arm]
        slots = tw_doc[arm]["slots"]
        A["generation"] = {"candidates": len(cs), "screen": dict(Counter(r["screen"] for r in cs)),
                           "j1_on_screen_pass": dict(Counter(str(r["j1"]) for r in cs if r["screen"] == "payload")),
                           "slots_selected": sum(s["status"] == "selected" for s in slots.values()),
                           "slots_failed": sorted(k for k, s in slots.items() if s["status"] == "failed"),
                           "target_type": dict(Counter(s["target"] for s in slots.values()))}
    res["any_matching_check_failed"] = any(not d["passed"] for a in res["arms"].values() for d in a["matching"].values())
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=1))
    L = ["# Item 3c manipulation checks (before detection)", ""]
    for arm, A in res["arms"].items():
        L += [f"## {arm} arm: {A['n_twins']} twins vs {A['n_camouflage']} camouflage", "",
              "| measure | twin | camouflage | difference (90% CI) | margin | passed |", "|---|---|---|---|---|---|"]
        for k, d in A["matching"].items():
            L.append(f"| {k} | {d['mean_twin']:.3f} | {d['mean_camouflage']:.3f} | {d['estimate']:+.3f} "
                     f"[{d['ci90'][0]:+.3f}, {d['ci90'][1]:+.3f}] ({d['scale']}) | ±{d['margin']:.3f} | {d['passed']} |")
        L += ["", "Directiveness (reported, not matched): " + "; ".join(
            f"{k}: any form {v['share_any_form']:.2f}, {v['forms_per_100w']:.2f}/100 words, rating "
            + (f"{v['mean_rating']:.2f} (n={v['n_rated']}/{v['n']})" if v["mean_rating"] is not None else "n/a")
            for k, v in A["directiveness"].items()),
              f"J1 on camouflage: {A['j1_camouflage']}", f"Generation: {A['generation']}", ""]
    L.append(f"Any matching check failed: {res['any_matching_check_failed']} (if so, detection analyses are repeated "
             "on the caliper-matched subset).")
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["rate", "report"])
    {"rate": rate, "report": report}[ap.parse_args().phase]()
