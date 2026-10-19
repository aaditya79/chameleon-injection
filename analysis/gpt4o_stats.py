"""
Recompute every GPT-4o number reported in the paper from raw logs.

Inputs (all raw, append-only logs):
  results/detections_gpt4o.jsonl   within-model detection (experiments/model_b.py)
  results/cross_detect.jsonl       off-diagonal 2x2 cells C and D (experiments/cross_detect.py)
  results/detections_v2.jsonl      Llama x Llama cell (static detector, refusal-screened)
  data/camouflage_corpus_gpt4o.json, data/camouflage_corpus_v2.json   ACS and refusals

Outputs results/gpt4o_stats.json plus a console report covering Table 1's GPT-4o column, the frontier
paragraph, Table 3 (cross-model) and Appendix J.

Methods:
  * CDG CI: cluster BCa over tasks, stratified by domain (analysis/bootstrap_ci.cluster_bca).
  * McNemar: task-level majority vote, continuity-corrected (analysis/significance_tests).
  * Main effects: two-sided Fisher exact test on the 2x2 margins; OR = sample odds ratio.
  * Interaction: log odds ratio of odds ratios, Woolf SE, Haldane-Anscombe +0.5 correction on all
    four cells (the GPT-4o x GPT-4o cell has a single detection). The uncorrected value is reported too.
  * ACS: Mann-Whitney U, two-sided.
  * Refusals: two-sided Fisher exact test, unresolved refusals per 135 payloads.

    python analysis/gpt4o_stats.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import fisher_exact, mannwhitneyu

from analysis.bootstrap_ci import cluster_bca, cdg_stat
from analysis.significance_tests import mcnemar_test

DET_GPT4O = Path("results/detections_gpt4o.jsonl")
DET_CROSS = Path("results/cross_detect.jsonl")
DET_V2 = Path("results/detections_v2.jsonl")
CORPUS_GPT4O = Path("data/camouflage_corpus_gpt4o.json")
CORPUS_V2 = Path("data/camouflage_corpus_v2.json")
OUT = Path("results/gpt4o_stats.json")
DOMAINS = ["financial", "legal", "general"]


def jl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def hits(rows) -> tuple[int, int]:
    return sum(r["verdict"] == "INJECTED" for r in rows), len(rows)


def task_mcnemar(static_rows, cam_rows) -> dict:
    def maj(rows):
        by = {}
        for r in rows:
            by.setdefault(r["task_id"], []).append(r["verdict"] == "INJECTED")
        return {t: sum(v) > len(v) / 2 for t, v in by.items()}
    s, c = maj(static_rows), maj(cam_rows)
    tasks = sorted(set(s) & set(c))
    b = sum(s[t] and not c[t] for t in tasks)
    cc = sum((not s[t]) and c[t] for t in tasks)
    chi2, p = mcnemar_test(b, cc, continuity=True)
    return {"n_tasks": len(tasks), "b": b, "c": cc, "chi2": round(chi2, 3), "p": p}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def interaction(cells: dict, h: float) -> dict:
    lo = lambda k, n: math.log((k + h) / (n - k + h))
    var = lambda k, n: 1 / (k + h) + 1 / (n - k + h)
    est = (lo(*cells["GG"]) - lo(*cells["GL"])) - (lo(*cells["LG"]) - lo(*cells["LL"]))
    se = math.sqrt(sum(var(*v) for v in cells.values()))
    return {"log_or": round(est, 3), "ci": [round(est - 1.96 * se, 2), round(est + 1.96 * se, 2)],
            "correction": h}


def main() -> None:
    g = jl(DET_GPT4O)
    assert not any(r.get("dry_run") for r in g)
    static = [r for r in g if r["payload_type"] == "static"]
    clean = [r for r in g if r["payload_type"] == "clean"]
    report: dict = {"within_model": {}}

    s_k, s_n = hits(static)
    c_k, c_n = hits(clean)
    print(f"GPT-4o static IDR {s_k}/{s_n}; clean FPR {c_k}/{c_n}")
    report["within_model"]["idr_static"] = [s_k, s_n]
    report["within_model"]["fpr_clean"] = [c_k, c_n]

    for var in ("full_context", "no_context"):
        cam = [r for r in g if r["generator_variant"] == var]
        k, n = hits(cam)
        cdg = s_k / s_n - k / n
        ci = cluster_bca(static + cam, cdg_stat)
        mc = task_mcnemar(static, cam)
        dom = {}
        for d in DOMAINS:
            ds, dc = hits([r for r in static if r["domain"] == d]), hits([r for r in cam if r["domain"] == d])
            dom[d] = round(ds[0] / ds[1] - dc[0] / dc[1], 3)
        report["within_model"][var] = {"idr_camouflage": [k, n], "cdg": round(cdg, 4),
                                       "cdg_bca_ci": ci["ci"], "mcnemar": mc, "cdg_by_domain": dom}
        print(f"  {var:<13} cam {k}/{n}  CDG {cdg:.3f}  BCa {ci['ci']}  McNemar chi2 {mc['chi2']} "
              f"(b={mc['b']}, c={mc['c']}, p={mc['p']:.2g})  by domain {dom}")

    # ---- 2x2 ----
    v2 = [r for r in jl(DET_V2) if r["payload_type"] == "camouflage"
          and r["detector_type"] == "static" and not r["is_refusal"]]
    cross = jl(DET_CROSS)
    cells = {
        "LL": hits(v2),
        "LG": hits([r for r in cross if r["cell"] == "D"]),   # Llama payloads, GPT-4o detector
        "GL": hits([r for r in cross if r["cell"] == "C"]),   # GPT-4o payloads, Llama detector
        "GG": hits([r for r in g if r["generator_variant"] == "full_context"]),
    }
    print("\n2x2 (payload author, detector):", {k: f"{a}/{n} CDG {1 - a / n:.3f}" for k, (a, n) in cells.items()})

    def margin(a, b):
        return [[a[0] + b[0], a[1] + b[1] - a[0] - b[0]]]
    att = margin(cells["LL"], cells["LG"]) + margin(cells["GL"], cells["GG"])
    det = margin(cells["LL"], cells["GL"]) + margin(cells["LG"], cells["GG"])
    o_att, p_att = fisher_exact(att)
    o_det, p_det = fisher_exact(det)
    print(f"  attacker main effect {att}: OR {o_att:.3f}, Fisher p={p_att:.4f}")
    print(f"  detector main effect {det}: OR {o_det:.3f}, Fisher p={p_det:.4f}")
    inter_h = interaction(cells, 0.5)
    inter_0 = interaction(cells, 0.0)
    print(f"  interaction (Haldane +0.5): {inter_h}; uncorrected: {inter_0}")
    gg_wilson = wilson(*cells["GG"])
    print(f"  Wilson CI on GPT-4o x GPT-4o IDR: [{gg_wilson[0]:.3f}, {gg_wilson[1]:.3f}]")

    # ---- corroborating attacker measurements ----
    gc = json.loads(CORPUS_GPT4O.read_text())
    lc = json.loads(CORPUS_V2.read_text())["generated"]["llama"]
    acs_g = [p["acs"] for v in gc["full"].values() for p in v if not p["is_refusal"]]
    acs_l = [p["acs"] for v in lc.values() for p in v if not p.get("is_refusal")]
    mw = mannwhitneyu(acs_g, acs_l, alternative="two-sided")
    ref_g = sum(p["is_refusal"] for v in gc["full"].values() for p in v)
    ref_l = sum(bool(p.get("is_refusal")) for v in lc.values() for p in v)
    _, p_ref = fisher_exact([[ref_g, 135 - ref_g], [ref_l, 135 - ref_l]])
    print(f"\n  ACS GPT-4o {sum(acs_g) / len(acs_g):.3f} (n={len(acs_g)}) vs Llama {sum(acs_l) / len(acs_l):.3f} "
          f"(n={len(acs_l)}): Mann-Whitney two-sided p={mw.pvalue:.3f}")
    print(f"  unresolved refusals GPT-4o {ref_g}/135 vs Llama {ref_l}/135: Fisher p={p_ref:.2g}")

    report["cross"] = {
        "cells": {k: list(v) for k, v in cells.items()},
        "attacker": {"table": att, "or": round(o_att, 3), "fisher_p": p_att},
        "detector": {"table": det, "or": round(o_det, 3), "fisher_p": p_det},
        "interaction_haldane": inter_h, "interaction_uncorrected": inter_0,
        "wilson_gg": [round(x, 3) for x in gg_wilson],
        "acs": {"gpt4o_mean": round(sum(acs_g) / len(acs_g), 4), "llama_mean": round(sum(acs_l) / len(acs_l), 4),
                "mannwhitney_two_sided_p": mw.pvalue},
        "refusals_unresolved": {"gpt4o": ref_g, "llama": ref_l, "fisher_p": p_ref},
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
