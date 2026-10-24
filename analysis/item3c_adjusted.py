"""
Item 3c covariate-adjusted delta (A-log 2026-10-21, user decision after the matching checks failed; before any 3c
detection). Per arm and detector (embedded, primary 43 tasks):

  Firth-penalised logistic regression  detected ~ 1 + camouflage + z(ACS) + z(spec/100) + z(vocabulary overlap)
  (features from results/item3c_caliper_features.json, z-scored over the arm's twins and camouflage), and the
  standardised (g-computation) adjusted delta = mean over all items of p(camouflage = 1, x) - p(camouflage = 0, x),
  with task-cluster BCa 95% CI (analysis/bootstrap_ci.cluster_bca, as the unadjusted delta). Firth keeps the fit
  finite under separation; a detector with no detections (or all detections) is reported as not estimable.
Bias direction to report: twins are more document-like than camouflage, which favours separation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.bootstrap_ci import cluster_bca

COVS = ("acs", "spec100", "vocab")


def firth(X: np.ndarray, y: np.ndarray, iters: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Firth-penalised logistic regression coefficients (Newton with step halving on the penalised likelihood)."""
    b = np.zeros(X.shape[1])

    def pll(b):
        p = 1 / (1 + np.exp(-X @ b))
        info = X.T @ (X * (p * (1 - p))[:, None])
        sign, logdet = np.linalg.slogdet(info)
        return np.sum(y * np.log(p + 1e-300) + (1 - y) * np.log(1 - p + 1e-300)) + 0.5 * logdet, p, info

    ll, p, info = pll(b)
    for _ in range(iters):
        w = p * (1 - p)
        inv = np.linalg.pinv(info)
        h = np.einsum("ij,jk,ik->i", X * w[:, None] ** 0.5, inv, X * w[:, None] ** 0.5)
        step = inv @ (X.T @ (y - p + h * (0.5 - p)))
        for _ in range(20):
            nb = b + step
            nll, np_, ninfo = pll(nb)
            if nll >= ll - 1e-12:
                break
            step /= 2
        done = np.max(np.abs(step)) < tol or abs(nll - ll) < tol
        b, ll, p, info = nb, nll, np_, ninfo
        if done:
            break
    return b


def design(rs: list[dict], feats: dict, mu: dict, sd: dict) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[1.0, float(r["kind"] == "camouflage")]
                  + [(feats[f"{r['kind']}|{r['item_id']}"][c] - mu[c]) / sd[c] for c in COVS] for r in rs])
    return X, np.array([float(r["detected"]) for r in rs])


def adj_delta(rs: list[dict], feats: dict, mu: dict, sd: dict) -> float:
    y = [r["detected"] for r in rs]
    if not rs or sum(y) in (0, len(y)) or len({r["kind"] for r in rs}) < 2:
        return float("nan")
    X, y = design(rs, feats, mu, sd)
    b = firth(X, y)
    X1, X0 = X.copy(), X.copy()
    X1[:, 1], X0[:, 1] = 1.0, 0.0
    return float(np.mean(1 / (1 + np.exp(-X1 @ b)) - 1 / (1 + np.exp(-X0 @ b))))


def adjusted(rs: list[dict], arm: str, detectors) -> dict:
    feats = json.loads(Path("results/item3c_caliper_features.json").read_text())[arm]["features"]
    rs = [r for r in rs if f"{r['kind']}|{r['item_id']}" in feats]
    items = {f"{r['kind']}|{r['item_id']}" for r in rs}
    mu = {c: float(np.mean([feats[i][c] for i in items])) for c in COVS}
    sd = {c: float(np.std([feats[i][c] for i in items], ddof=1)) for c in COVS}
    out = {}
    for d in detectors:
        sub = [r for r in rs if r["detector"] == d and r["mode"] == "embedded"]
        est = adj_delta(sub, feats, mu, sd)
        res = {"n": len(sub), "adjusted_delta": est, "estimable": not np.isnan(est)}
        if res["estimable"]:
            X, y = design(sub, feats, mu, sd)
            res["log_or_camouflage"] = float(firth(X, y)[1])
            res["ci"] = (cluster_bca(sub, lambda x: adj_delta(x, feats, mu, sd)) or {}).get("ci")
        out[f"{d}|embedded"] = res
    return out
