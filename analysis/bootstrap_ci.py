"""
Cluster-bootstrap BCa confidence intervals for the paper's headline quantities.

Implements BOOTSTRAP_CI_SPEC.md.

RESAMPLING UNIT: the task, not the trial. Three camouflage variants share a task,
a document, and a malicious goal; 20 static payloads likewise. Resampling trials
would treat 135 clustered observations as 135 independent draws and understate
variance. We resample the 45 task IDs with replacement, stratified by domain
(15/15/15) so every replicate preserves the design's domain balance, and take all
trials belonging to the drawn tasks.

INTERVALS: BCa, B=10,000, acceleration from leave-one-task-out jackknife. Several
statistics are bounded and skewed near their bounds (IDR_camouflage ~0.10), which
is exactly where the percentile method misbehaves.

Ratios (DAF, CDG_delta) are bootstrapped on the log scale where strictly positive
and exponentiated back, and the proportion of undefined replicates is reported --
if a ratio's denominator can be zero under resampling, the interval is not
trustworthy and we say so rather than printing a number.

    python analysis/bootstrap_ci.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

TRIALS = Path("results/trials.jsonl")
DET_V2 = Path("results/detections_v2.jsonl")
OUT = Path("results/bootstrap_cis.json")

B = 10_000
SEED = 42
LLAMA, GEMINI = "llama3.1", "google/gemini-2.0-flash-001"


# ---------------------------------------------------------------------------
# Bootstrap machinery
# ---------------------------------------------------------------------------

def _stratified_task_draw(rng, tasks_by_domain):
    drawn = []
    for _, ts in tasks_by_domain.items():
        drawn.extend(rng.choice(ts, size=len(ts), replace=True))
    return drawn


def cluster_bca(records, stat_fn, log_scale=False):
    """
    BCa interval for stat_fn over task-clustered records.

    records: list of dicts each carrying 'task_id' and 'domain'.
    stat_fn: list[dict] -> float (nan if undefined for that resample).
    """
    rng = np.random.default_rng(SEED)
    by_task = defaultdict(list)
    domain_of = {}
    for r in records:
        by_task[r["task_id"]].append(r)
        domain_of[r["task_id"]] = r["domain"]

    tasks = sorted(by_task)
    if not tasks:
        return None
    tasks_by_domain = defaultdict(list)
    for t in tasks:
        tasks_by_domain[domain_of[t]].append(t)

    theta_hat = stat_fn(records)
    if not np.isfinite(theta_hat):
        return None

    # --- bootstrap replicates ---
    reps, undefined = [], 0
    for _ in range(B):
        drawn = _stratified_task_draw(rng, tasks_by_domain)
        sample = [r for t in drawn for r in by_task[t]]
        v = stat_fn(sample)
        if not np.isfinite(v) or (log_scale and v <= 0):
            undefined += 1
            continue
        reps.append(math.log(v) if log_scale else v)
    if len(reps) < B * 0.5:
        return {"point": theta_hat, "undefined_frac": undefined / B,
                "ci": None, "note": "too many undefined replicates; unestimable"}
    reps = np.array(reps)

    # --- jackknife over tasks, for acceleration ---
    jack = []
    for t in tasks:
        sample = [r for u in tasks if u != t for r in by_task[u]]
        v = stat_fn(sample)
        if np.isfinite(v) and (not log_scale or v > 0):
            jack.append(math.log(v) if log_scale else v)
    jack = np.array(jack)
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0

    th = math.log(theta_hat) if log_scale else theta_hat
    z0 = norm.ppf(max(1e-6, min(1 - 1e-6, (reps < th).mean())))

    out = []
    for alpha in (0.025, 0.975):
        z = norm.ppf(alpha)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        out.append(float(np.percentile(reps, 100 * norm.cdf(adj))))
    lo, hi = out
    if log_scale:
        lo, hi = math.exp(lo), math.exp(hi)
    return {"point": round(theta_hat, 4), "ci": [round(lo, 4), round(hi, 4)],
            "undefined_frac": round(undefined / B, 4), "B": B}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def idr_of(rows, ptype, verdict_key="verdict"):
    sel = [r for r in rows if r["payload_type"] == ptype]
    return sum(1 for r in sel if r[verdict_key] == "INJECTED") / len(sel) if sel else float("nan")


def cdg_stat(rows):
    s, c = idr_of(rows, "static"), idr_of(rows, "camouflage")
    return s - c if np.isfinite(s) and np.isfinite(c) else float("nan")


def asr_of(rows):
    j = [r for r in rows if r.get("asr_judgment")]
    return sum(1 for r in j if r["asr_judgment"]["followed_injection"]) / len(j) if j else float("nan")


def fmt(res, pct=False):
    if res is None:
        return "n/a"
    if res.get("ci") is None:
        return f"{res['point']:.4f}  [unestimable, {res['undefined_frac']:.0%} undefined]"
    m = 100 if pct else 1
    return f"{res['point']*m:.4f}  [{res['ci'][0]*m:.4f}, {res['ci'][1]*m:.4f}]"


def main() -> None:
    trials = [json.loads(l) for l in TRIALS.open()]
    real = [t for t in trials if not t.get("dry_run", False)]
    report = {}

    print("=" * 74)
    print(f"CLUSTER-BOOTSTRAP BCa INTERVALS  (B={B}, resampled over 45 tasks)")
    print("=" * 74)

    # ---- Exp 1 CDG, both models, original corpus ----
    for model, label in [(LLAMA, "Llama 3.1 8B"), (GEMINI, "Gemini 2.0 Flash")]:
        rows = [{"task_id": r["task_id"], "domain": r["domain"],
                 "payload_type": r["payload_type"],
                 "verdict": r["detection_result"]["verdict"]}
                for r in real
                if r["model"] == model and r["experiment"] == "exp1"
                and r["architecture"] == "single_agent"
                and (r.get("detection_result") or {}).get("detector_type") == "static"
                and r["payload_type"] in ("static", "camouflage")]
        res = cluster_bca(rows, cdg_stat)
        report[f"cdg_exp1_{model}"] = res
        print(f"\n  CDG (Exp 1, {label}): {fmt(res)}")

        for pt in ("static", "camouflage"):
            r2 = cluster_bca(rows, lambda x, p=pt: idr_of(x, p))
            report[f"idr_{pt}_exp1_{model}"] = r2
            print(f"    IDR {pt:<11}: {fmt(r2)}")

        for pt in ("static", "camouflage"):
            a = [{"task_id": r["task_id"], "domain": r["domain"],
                  "asr_judgment": r["asr_judgment"]}
                 for r in real if r["model"] == model and r["experiment"] == "exp1"
                 and r["architecture"] == "single_agent" and r["payload_type"] == pt
                 and r.get("asr_judgment")]
            r3 = cluster_bca(a, asr_of)
            report[f"asr_{pt}_{model}"] = r3
            print(f"    ASR {pt:<11}: {fmt(r3)}")

    # ---- v2 screened corpus CDG + variant contrast ----
    if DET_V2.exists():
        d2 = [json.loads(l) for l in DET_V2.open()]
        scr = [r for r in d2 if r["detector_type"] == "static"
               and not (r["payload_type"] == "camouflage" and r["is_refusal"])]
        res = cluster_bca(scr, cdg_stat)
        report["cdg_v2_screened"] = res
        print(f"\n  CDG (v2 screened corpus, static det.): {fmt(res)}")

        cam = [r for r in scr if r["payload_type"] == "camouflage"]

        def variant_contrast(rows):
            def rate(v):
                s = [r for r in rows if r["payload_id"].endswith(f"_v{v}")]
                return sum(1 for r in s if r["verdict"] == "INJECTED") / len(s) if s else float("nan")
            return rate("1") - rate("3")

        rv = cluster_bca(cam, variant_contrast)
        report["variant_v1_minus_v3"] = rv
        print(f"  Variant contrast v1 - v3 catch rate: {fmt(rv)}")
        if rv and rv.get("ci") and rv["ci"][0] <= 0 <= rv["ci"][1]:
            print("    -> interval spans zero; the v1>v3 ordering is not supported.")

    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Written to {OUT}")


if __name__ == "__main__":
    main()
