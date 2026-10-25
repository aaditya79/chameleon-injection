"""
Item 3 Amendment B (POST-UNBLINDING; committed 1dd022e before this script existed): Firth-penalized logistic models
(B1, analysis/mechanism_firth.R) and task-cluster sign-flip permutation tests (B2) beside the P6 results, and the
probe-L analysis (B3).

Core (default):
  inputs  results/mechanism_detections.jsonl, results/mechanism_trials.jsonl, results/mechanism_detection_stats.json (P6)
  outputs results/mechanism_amendment_b.json, results/mechanism_amendment_b_report.md
Probe (--probe):
  inputs  the core files plus results/mechanism_probe_detections.jsonl, results/mechanism_probe_trials.jsonl
  outputs results/mechanism_probe_stats.json, results/mechanism_probe_report.md
  Within the D+ cells: L effect by the P6-style mixed model (analysis/mechanism_glmm.R, with the L code passed in
  its D column; see probe()), Firth, and permutation; Holm across the 6 detectors run on the probe (no GPT-4o, P5).

B4 reporting rule: "supported in the post-unblinding sensitivity analyses" iff both B1 and B2 give Holm p < 0.05
in the predicted direction (and, for H1, D+ > D- at both V levels). It never replaces the P6 verdict.

    python analysis/mechanism_amendment_b.py [--probe]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.bootstrap_ci import cluster_bca
from analysis.mechanism_detection_stats import (CELLS, CLASSIFIERS, DETECTORS, GOAL_SETS, LABEL, LLM, RD, SCOPES,
                                                TEST_SCOPE, holm, load_rows, pct, pv)
from config import CONFIG

WORK = Path("results/mechanism_glmm")
P6 = Path("results/mechanism_detection_stats.json")
N_PERM = 10_000
COLS = ["detector", "goal_set", "task", "domain", "template", "cell", "D", "V", "L", "detected", "success",
        "acs_z", "words_z"]
PROBE_CELLS = ["D+V-", "D+V+", "D+V-+L", "D+V++L"]
L_WEIGHTS = {"D+V-+L": 0.5, "D+V-": -0.5, "D+V++L": 0.5, "D+V+": -0.5}
FIRTH_SPEC = {  # test -> (a, b, term, a_dummy)
    "D": ("D", "V", "a", False), "V": ("D", "V", "b", False), "DxV": ("D", "V", "a:b", False),
    "V|D+": ("D", "V", "b", True),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        fh.write(",".join(COLS) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, 0)) for c in COLS) + "\n")


def run_r(script: str, rows: list[dict], specs: list[dict], tag: str) -> dict:
    WORK.mkdir(exist_ok=True)
    write_csv(WORK / f"{tag}_input.csv", rows)
    (WORK / f"{tag}_specs.json").write_text(json.dumps(specs))
    subprocess.run(["Rscript", script, str(WORK / f"{tag}_input.csv"), str(WORK / f"{tag}_specs.json"),
                    str(WORK / f"{tag}_fits.json")], check=True)
    return {f["id"]: f for f in json.loads((WORK / f"{tag}_fits.json").read_text())}


def sign_flip(rows: list[dict], weights: dict, rng: np.random.Generator) -> dict:
    """Mean of per-task contrasts and its two-sided sign-flip permutation p (B2)."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["task"]][r["cell"]].append(r["detected"])
    cs = np.array([sum(w * np.mean(cells[c]) for c, w in weights.items())
                   for cells in by.values() if all(cells[c] for c in weights)])
    if len(cs) == 0:
        return {"stat": None, "p": 1.0, "n_tasks": 0}
    obs = cs.mean()
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(cs)))
    null = (flips * cs).mean(axis=1)
    p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + N_PERM)
    return {"stat": float(obs), "p": float(p), "n_tasks": int(len(cs))}


def sel(rows, gs, det, gating, scope, cells):
    return [r for r in rows if r["goal_set"] == gs and r["detector"] == det and SCOPES[scope](r)
            and r["cell"] in cells and (gating == "ungated" or r["success"] == 1)]


def core() -> None:
    emb, _ = load_rows()
    p6 = json.loads(P6.read_text())["goal_sets"]
    specs = [{"id": f"{gs}|{det}|{g}|{t}", "detector": det, "goal_set": gs, "gated": g == "gated",
              "scope": TEST_SCOPE[t], "cells": CELLS, "a": a, "b": b, "term": term, "a_dummy": dummy}
             for gs in GOAL_SETS for det in DETECTORS for g in ("gated", "ungated")
             for t, (a, b, term, dummy) in FIRTH_SPEC.items()]
    firth = run_r("analysis/mechanism_firth.R", emb, specs, "firth_core")
    rng = np.random.default_rng(CONFIG.seed)
    out = {}
    for gs in GOAL_SETS:
        for g in ("gated", "ungated"):
            res = out.setdefault(gs, {}).setdefault(g, {"tests": {}, "hypotheses": {}})
            for t in RD:
                T = res["tests"][t] = {}
                for det in DETECTORS:
                    rows = sel(emb, gs, det, g, TEST_SCOPE[t], CELLS)
                    x = p6[gs][g]["detectors"][det]["tests"][t]
                    T[det] = {"p6": {"coef": x["coef"], "p": x["p"], "p_holm": x["p_holm"], "model": x["model"],
                                     "note": x.get("note")},
                              "rd_pooled": p6[gs][g]["detectors"][det]["rd"][t],
                              "firth": firth[f"{gs}|{det}|{g}|{t}"],
                              "perm": sign_flip(rows, RD[t], rng)}
                for k in ("firth", "perm"):
                    adj = holm({det: T[det][k]["p"] if T[det][k].get("p") is not None else 1.0 for det in DETECTORS})
                    for det in DETECTORS:
                        T[det][k]["p_holm"] = adj[det]
            for det in DETECTORS:
                cells = {c: v["k_n"][0] / v["k_n"][1] for c, v in p6[gs][g]["detectors"][det]["cells"]["all"].items()}

                def ok(t, sign):
                    f, q = res["tests"][t][det]["firth"], res["tests"][t][det]["perm"]
                    return bool(f["p_holm"] < 0.05 and q["p_holm"] < 0.05 and np.sign(f["coef"]) == sign
                                and np.sign(q["stat"] or 0) == sign)
                h = {"P6": p6[gs][g]["hypotheses"][det]}
                if det in LLM:
                    h["H1_B"] = ok("D", 1) and cells["D+V-"] > cells["D-V-"] and cells["D+V+"] > cells["D-V+"]
                h["H2_B"] = ok("V|D+", -1)
                h["DxV_B"] = {"firth_p_holm": res["tests"]["DxV"][det]["firth"]["p_holm"],
                              "perm_p_holm": res["tests"]["DxV"][det]["perm"]["p_holm"]}
                res["hypotheses"][det] = h
    Path("results/mechanism_amendment_b.json").write_text(json.dumps(out, indent=1, default=float))
    report_core(out)


def report_core(out: dict) -> None:
    L = ["# Item 3 Amendment B (POST-UNBLINDING sensitivity analyses; P6 remains the pre-registered record)\n",
         "Per test: P6 Wald (glmer/GEE; sep. = separation), B1 Firth (logistf, profile PL), B2 task-cluster sign-flip "
         "permutation on per-task risk differences (10,000). p shown as raw / Holm (7 detectors). "
         "D: all 45 tasks; V, DxV, V|D+: financial+legal.\n"]
    for gs in GOAL_SETS:
        for g in ("gated", "ungated"):
            res = out[gs][g]
            L.append(f"\n## {gs} goals, {g}\n")
            L.append("| Detector | test | P6 coef; p / Holm | Firth coef [95% CI]; p / Holm | perm mean RD (tasks); p / Holm |")
            L.append("|---|---|---|---|---|")
            for det in DETECTORS:
                for t in RD:
                    e = res["tests"][t][det]
                    p6, f, q = e["p6"], e["firth"], e["perm"]
                    p6s = ("—" if p6["coef"] is None else f"{p6['coef']:+.2f}{' sep.' if p6.get('note') else ''}") + \
                        f"; {pv(p6['p'])} / {pv(p6['p_holm'])}"
                    fs = (f"{f['coef']:+.2f} [{f['ci95'][0]:+.2f}, {f['ci95'][1]:+.2f}]; {pv(f['p'])} / {pv(f['p_holm'])}"
                          if f.get("coef") is not None else f.get("status", "—"))
                    qs = ("—" if q["stat"] is None else f"{100 * q['stat']:+.1f} ({q['n_tasks']})") + \
                        f"; {pv(q['p'])} / {pv(q['p_holm'])}"
                    L.append(f"| {LABEL[det]} | {t} | {p6s} | {fs} | {qs} |")
            L.append(f"\n### Hypotheses, {gs}, {g}: P6 verdict vs post-unblinding (B: Firth AND permutation, Holm)\n")
            for det in DETECTORS:
                h = res["hypotheses"][det]
                p6h = ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in h["P6"].items())
                b = ", ".join(f"{k} {'yes' if h[k] else 'no'}" for k in ("H1_B", "H2_B") if k in h)
                L.append(f"- {LABEL[det]}: P6 [{p6h}]; B [{b}]; DxV Holm p Firth {pv(h['DxV_B']['firth_p_holm'])}, "
                         f"perm {pv(h['DxV_B']['perm_p_holm'])}")
    Path("results/mechanism_amendment_b_report.md").write_text("\n".join(L) + "\n")
    print("wrote results/mechanism_amendment_b_report.md and results/mechanism_amendment_b.json")


# ---------------------------------------------------------------------------
# Probe (B3)
# ---------------------------------------------------------------------------

def probe() -> None:
    emb, alone = load_rows()
    success = {r["payload_id"]: int(bool(r["valid"])) for r in map(json.loads, open("results/mechanism_probe_trials.jsonl"))
               if r.get("status", "ok") == "ok"}
    p_emb, p_alone = [], []
    for r in map(json.loads, open("results/mechanism_probe_detections.jsonl")):
        if r["status"] != "ok" or r["payload_id"] not in success:
            continue
        row = {"detector": r["detector"], "goal_set": r["goal_set"], "payload_id": r["payload_id"],
               "task": r["task_id"], "domain": r["domain"], "template": r["template_id"], "cell": r["cell"],
               "detected": int(r["verdict"] == "INJECTED"), "success": success[r["payload_id"]]}
        (p_emb if r["mode"] == "embedded" else p_alone).append(row)
    dets = [d for d in DETECTORS if d != "gpt4o"]

    def lv(rows):  # L and V codes; mechanism_glmm.R fits detected ~ D * V, so its D column carries L here
        for r in rows:
            r["L"] = 0.5 if r["cell"].endswith("+L") else -0.5
            r["V"] = 0.5 if r["cell"].startswith("D+V+") else -0.5
            r["D"] = r["L"]
        return rows
    E = lv([dict(r) for r in emb if r["cell"] in ("D+V-", "D+V+") and r["detector"] in dets] + p_emb)
    A = lv([dict(r) for r in alone if r["cell"] in ("D+V-", "D+V+")] + p_alone)
    glmm_specs = [{"id": f"{gs}|{det}|{g}|all|primary", "goal_set": gs, "detector": det, "gated": g == "gated",
                   "scope": "all", "model": "primary"} for gs in GOAL_SETS for det in dets for g in ("gated", "ungated")]
    glmm = run_r("analysis/mechanism_glmm.R", E, glmm_specs, "probe_glmm")
    firth_specs = [{"id": f"{gs}|{det}|{g}", "detector": det, "goal_set": gs, "gated": g == "gated", "scope": "all",
                    "cells": PROBE_CELLS, "a": "L", "b": "V", "term": "a", "a_dummy": False}
                   for gs in GOAL_SETS for det in dets for g in ("gated", "ungated")]
    firth = run_r("analysis/mechanism_firth.R", E, firth_specs, "probe_firth")
    rng = np.random.default_rng(CONFIG.seed)
    out = {}
    for gs in GOAL_SETS:
        for g in ("gated", "ungated"):
            res = out.setdefault(gs, {}).setdefault(g, {})
            for det in dets:
                rows = sel(E, gs, det, g, "all", PROBE_CELLS)
                gf = glmm[f"{gs}|{det}|{g}|all|primary"]
                est = gf.get("estimates", {}).get("D") if gf["status"] == "ok" else None
                if est and any(isinstance(v, str) for v in est.values()):
                    est = None
                res[det] = {
                    "cells": {c: [sum(r["detected"] for r in rows if r["cell"] == c), sum(1 for r in rows if r["cell"] == c)]
                              for c in PROBE_CELLS},
                    "rd_L": cluster_bca([{"task_id": r["task"], "domain": r["domain"], "cell": r["cell"], "y": r["detected"]}
                                         for r in rows], lambda rs: _contrast(rs, L_WEIGHTS)),
                    "p6": {"model": gf.get("model_used", gf["status"]), "coef": est["coef"] if est else None,
                           "p": est["p"] if est else 1.0, "sep": bool(est and abs(est["coef"]) > 10)},
                    "firth": firth[f"{gs}|{det}|{g}"], "perm": sign_flip(rows, L_WEIGHTS, rng)}
            for k in ("p6", "firth", "perm"):
                adj = holm({det: res[det][k]["p"] if res[det][k].get("p") is not None else 1.0 for det in dets})
                for det in dets:
                    res[det][k]["p_holm"] = adj[det]
            for det in CLASSIFIERS:
                e = res[det]
                e["H3_L_P6"] = bool(e["p6"]["p_holm"] < 0.05 and (e["p6"]["coef"] or 0) > 0)
                e["H3_L_B"] = bool(e["firth"]["p_holm"] < 0.05 and e["perm"]["p_holm"] < 0.05
                                   and e["firth"]["coef"] > 0 and (e["perm"]["stat"] or 0) > 0)
            if g == "ungated":
                res["alone"] = {det: {c: [sum(r["detected"] for r in A if r["detector"] == det and r["goal_set"] == gs and r["cell"] == c),
                                          sum(1 for r in A if r["detector"] == det and r["goal_set"] == gs and r["cell"] == c)]
                                      for c in PROBE_CELLS} for det in CLASSIFIERS}
    Path("results/mechanism_probe_stats.json").write_text(json.dumps(out, indent=1, default=float))
    L = ["# Item 3 probe-L (P6 PROBE as pre-registered, made concrete in Amendment B3; B1/B2 are post-unblinding)\n",
         "Within D+ cells, all 45 tasks, embedded. L effect: P6-style glmer (L*V, task and template intercepts), "
         "Firth, sign-flip permutation; p raw / Holm (6 detectors; GPT-4o not run on the probe, P5).\n"]
    for gs in GOAL_SETS:
        for g in ("gated", "ungated"):
            res = out[gs][g]
            L.append(f"\n## {gs} goals, {g}\n")
            L.append("| Detector | " + " | ".join(PROBE_CELLS) + " | RD L [BCa] | P6 glmer L | Firth L | perm L | H3 (P6 / B) |")
            L.append("|---|" + "---|" * (len(PROBE_CELLS) + 5))
            for det in dets:
                e = res[det]
                rd = e["rd_L"]
                rds = (f"{100 * rd['point']:+.0f} [{100 * rd['ci'][0]:+.0f}, {100 * rd['ci'][1]:+.0f}]" if rd and rd.get("ci")
                       else (f"{100 * rd['point']:+.0f}" if rd else "—"))
                p6s = ("—" if e["p6"]["coef"] is None else f"{e['p6']['coef']:+.2f}{' sep.' if e['p6']['sep'] else ''}") + \
                    f"; {pv(e['p6']['p'])} / {pv(e['p6']['p_holm'])} ({e['p6']['model']})"
                f = e["firth"]
                fs = f"{f['coef']:+.2f}; {pv(f['p'])} / {pv(f['p_holm'])}" if f.get("coef") is not None else f.get("status")
                q = e["perm"]
                qs = ("—" if q["stat"] is None else f"{100 * q['stat']:+.1f} ({q['n_tasks']})") + f"; {pv(q['p'])} / {pv(q['p_holm'])}"
                h3 = f"{'yes' if e['H3_L_P6'] else 'no'} / {'yes' if e['H3_L_B'] else 'no'}" if det in CLASSIFIERS else ""
                L.append(f"| {LABEL[det]} | " + " | ".join(pct(e["cells"][c]) for c in PROBE_CELLS) + f" | {rds} | {p6s} | {fs} | {qs} | {h3} |")
        L.append(f"\n## {gs} goals: classifiers, payload alone (ungated, descriptive)\n")
        L.append("| Detector | " + " | ".join(PROBE_CELLS) + " |")
        L.append("|---|" + "---|" * len(PROBE_CELLS))
        for det in CLASSIFIERS:
            L.append(f"| {LABEL[det]} | " + " | ".join(pct(out[gs]["ungated"]["alone"][det][c]) for c in PROBE_CELLS) + " |")
    Path("results/mechanism_probe_report.md").write_text("\n".join(L) + "\n")
    print("wrote results/mechanism_probe_report.md and results/mechanism_probe_stats.json")


def _contrast(rs, weights):
    by = defaultdict(list)
    for r in rs:
        by[r["cell"]].append(r["y"])
    if any(not by[c] for c in weights):
        return float("nan")
    return float(sum(w * np.mean(by[c]) for c, w in weights.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    probe() if ap.parse_args().probe else core()
