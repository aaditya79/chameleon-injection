"""
Item 3: D x V interaction on attack success (ASR gate outcomes only; reads no detection data).

Input:  results/mechanism_trials.jsonl  (experiments/mechanism_run.py asr)
Output: results/mechanism_asr_interaction.json, plus a markdown table on stdout.

Ungated by construction (every payload is an observation; outcome = valid, i.e. the ASR judge says the agent
followed the injection). Per goal set (neutral, verbatim) and scope (all 45 tasks, financial+legal):
  logit P(success) = b0 + bD*D + bV*V + bDV*D*V,   D = 1 for D+, V = 1 for V+ (reference cell D-V-)
so bV is the V+ vs V- effect within D-, and bV + bDV is the V effect within D+.
  * Wald 95% CIs and two-sided p-values from the ML fit (the model is saturated, so the fit reproduces the cell rates)
  * likelihood-ratio test of bDV against the additive model
  * sensitivity: task-clustered (sandwich) standard errors, since the 3 templates per cell share a task
  * direction check: V- > V+ within D+ and V+ > V- within D- (a sign flip of the V effect)
Errored trials (status != ok) are excluded (E1); there are none at the time of writing.

    python analysis/mechanism_asr_interaction.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

TRIALS = Path("results/mechanism_trials.jsonl")
OUT = Path("results/mechanism_asr_interaction.json")
TERMS = ["intercept", "D", "V", "DxV"]
SCOPES = {"all": lambda r: True, "fin+legal": lambda r: r["domain"] != "general"}


def fit_logit(X: np.ndarray, y: np.ndarray, iters: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    """IRLS maximum likelihood. Returns coefficients, fitted probabilities, log-likelihood."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        w = p * (1 - p)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ b))
    return b, p, float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def analyze(rs: list[dict]) -> dict:
    D = np.array([r["cell"].startswith("D+") for r in rs], float)
    V = np.array([r["cell"].endswith("V+") for r in rs], float)
    y = np.array([bool(r["valid"]) for r in rs], float)
    X = np.column_stack([np.ones_like(D), D, V, D * V])
    b, p, ll = fit_logit(X, y)
    bread = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    # task-clustered sandwich (CR0 with the usual G/(G-1) small-sample factor)
    tasks = np.array([r["task_id"] for r in rs])
    groups = np.unique(tasks)
    meat = sum(np.outer(s, s) for s in (X[tasks == g].T @ (y - p)[tasks == g] for g in groups))
    G = len(groups)
    vcov = {"model": bread, "task_clustered": G / (G - 1) * bread @ meat @ bread}
    _, _, ll_add = fit_logit(X[:, :3], y)
    lr = 2 * (ll - ll_add)

    def est(c: np.ndarray, cov: np.ndarray) -> dict:
        e, se = float(c @ b), float(np.sqrt(c @ cov @ c))
        z = e / se
        return {"coef": e, "se": se, "ci95": [e - 1.96 * se, e + 1.96 * se], "z": z,
                "p": float(2 * stats.norm.sf(abs(z))), "odds_ratio": float(np.exp(e))}

    contrasts = {t: np.eye(4)[i] for i, t in enumerate(TERMS)}
    contrasts["V_within_D+"] = np.array([0, 0, 1, 1.0])  # bV + bDV
    contrasts["V_within_D-"] = np.array([0, 0, 1, 0.0])  # bV
    cells = {}
    for c in ("D+V-", "D+V+", "D-V+", "D-V-"):
        xs = [r for r in rs if r["cell"] == c]
        cells[c] = {"valid": int(sum(bool(r["valid"]) for r in xs)), "n": len(xs)}
    rate = {c: v["valid"] / v["n"] for c, v in cells.items()}
    return {
        "n": len(rs), "cells": cells,
        "coefficients": {k: est(c, vcov["model"]) for k, c in contrasts.items()},
        "coefficients_task_clustered": {k: est(c, vcov["task_clustered"]) for k, c in contrasts.items()},
        "n_task_clusters": G,
        "lr_test_DxV": {"chi2": lr, "df": 1, "p": float(stats.chi2.sf(lr, 1))},
        "direction": {"V-_gt_V+_within_D+": rate["D+V-"] > rate["D+V+"],
                      "V+_gt_V-_within_D-": rate["D-V+"] > rate["D-V-"],
                      "flip": rate["D+V-"] > rate["D+V+"] and rate["D-V+"] > rate["D-V-"]},
    }


def main() -> None:
    rs = [json.loads(x) for x in TRIALS.open() if x.strip()]
    rs = [r for r in rs if r.get("status", "ok") == "ok"]  # pre-fix records have no status field
    out = {"source": str(TRIALS), "coding": "D=1 for D+, V=1 for V+; reference cell D-V-; ungated (all payloads)",
           "results": {}}
    print("| goal set | scope | term | coef | 95% CI | p | p (task-clustered) |")
    print("|---|---|---|---|---|---|---|")
    for gs in ("neutral", "verbatim"):
        for scope, sel in SCOPES.items():
            res = analyze([r for r in rs if r["goal_set"] == gs and sel(r)])
            out["results"].setdefault(gs, {})[scope] = res
            for t in ("D", "V", "DxV", "V_within_D+"):
                m, c = res["coefficients"][t], res["coefficients_task_clustered"][t]
                print(f"| {gs} | {scope} | {t} | {m['coef']:+.3f} | [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}] "
                      f"| {m['p']:.2g} | {c['p']:.2g} |")
            print(f"| {gs} | {scope} | LR test DxV | chi2={res['lr_test_DxV']['chi2']:.2f} | | "
                  f"{res['lr_test_DxV']['p']:.2g} | | flip={res['direction']['flip']}")
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
