"""
Item 3b analysis, part 2 of 3: statistics (pre-registration 654a4cf, "Analysis (fixed before generation)").

paired()  risk difference X - Y over source payloads that have both records; task-cluster BCa 95% CI
          (analysis/bootstrap_ci.cluster_bca, B = 10,000); two-sided sign-flip permutation on per-task mean paired
          differences (10,000, seed 42); exact McNemar counts and p.
holm(), wilson(), firth() (exploratory: detected ~ log(words) + specificity, via analysis/item3b_firth.R).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import binomtest

from analysis.bootstrap_ci import cluster_bca

N_PERM, SEED = 10_000, 42
SCOPES = {"all": lambda d: True, "fin+legal": lambda d: d != "general", "general": lambda d: d == "general"}


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if not n:
        return [float("nan")] * 2
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [1.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def _mean_d(rs: list[dict]) -> float:
    return float(np.mean([r["d"] for r in rs])) if rs else float("nan")


def paired(y: dict, x_arm: str, y_arm: str, pids: list[str], meta: dict, scope: str = "all",
           keep=lambda pid: True) -> dict:
    """y: {(pid, arm): 0/1}. keep(pid) restricts the pairs further (gating, strata)."""
    use = [p for p in pids if (p, x_arm) in y and (p, y_arm) in y and SCOPES[scope](meta[p]["domain"]) and keep(p)]
    if not use:
        return {"n": 0}
    recs = [{"task_id": meta[p]["task_id"], "domain": meta[p]["domain"], "d": y[(p, x_arm)] - y[(p, y_arm)]}
            for p in use]
    d = np.array([r["d"] for r in recs], float)
    by = defaultdict(list)
    for r in recs:
        by[r["task_id"]].append(r["d"])
    tm = np.array([np.mean(v) for v in by.values()])
    rng = np.random.default_rng(SEED)
    null = (rng.choice([-1.0, 1.0], size=(N_PERM, len(tm))) * tm).mean(axis=1)
    p_perm = float((1 + np.sum(np.abs(null) >= abs(tm.mean()) - 1e-12)) / (1 + N_PERM))
    b, c = int((d == 1).sum()), int((d == -1).sum())
    ci = cluster_bca(recs, _mean_d) if d.any() else {"ci": [0.0, 0.0]}
    return {"n": len(use), "n_tasks": int(len(tm)), "k_x": int(sum(y[(p, x_arm)] for p in use)),
            "k_y": int(sum(y[(p, y_arm)] for p in use)), "rd": float(d.mean()), "ci": (ci or {}).get("ci"),
            "p_perm": p_perm, "mcnemar_b": b, "mcnemar_c": c,
            "p_mcnemar_exact": float(binomtest(b, b + c).pvalue) if b + c else 1.0}


def firth(rows: list[dict], work: Path) -> dict:
    """rows: {key, detected, logw, spec}. One logistf fit per key."""
    work.mkdir(parents=True, exist_ok=True)
    with (work / "firth_input.csv").open("w") as fh:
        fh.write("key,detected,logw,spec\n")
        for r in rows:
            fh.write(f"{r['key']},{r['detected']},{r['logw']},{r['spec']}\n")
    subprocess.run(["Rscript", "analysis/item3b_firth.R", str(work / "firth_input.csv"), str(work / "firth.json")],
                   check=True)
    return {f["key"]: f for f in json.loads((work / "firth.json").read_text())}
