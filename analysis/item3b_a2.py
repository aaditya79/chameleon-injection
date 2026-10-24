"""
Item 3b EXPLORATORY follow-up (A-log 2026-10-20, decided after seeing the 3b results): arm (a'') declarative goal.

Contrasts, paired by source payload on the 96 primary sources, same methods as 3b (item3b_models.paired):
  (a) vs (a'')     directive form, goal held
  (a'') vs (a')    restating the goal without directive form
Per detector and mode; Holm across the 7 detectors (embedded) per contrast; ASR for both contrasts.
(a) and (a') come from the 3b logs, (a'') from results/item3b_a2_*.jsonl. All results are exploratory.

    python3 analysis/item3b_a2.py   -> results/item3b_a2_report.json, .md
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import item3b_data as D  # noqa: E402
import item3b_models as M  # noqa: E402

CONTRASTS = [("a", "a_dprime"), ("a_dprime", "a_prime")]
NAME = {"a": "(a)", "a_prime": "(a′)", "a_dprime": "(a″)"}


def main() -> None:
    meta = D.source_meta()
    pids = D.analysis_sets(meta)["primary"]
    dets, err3b = D.detections("results/item3b_detections.jsonl")
    dets2, err2 = D.detections("results/item3b_a2_detections.jsonl")
    for k, y in dets2.items():
        dets.setdefault(k, {}).update(y)
    v, n_err = D.asr("results/item3b_trials.jsonl")
    v2, n_err2 = D.asr("results/item3b_a2_trials.jsonl")
    v.update(v2)
    keys = [(d, "embedded") for d in D.DETECTORS] + [(d, "alone") for d in D.CLASSIFIERS]
    rate = lambda y, arm: [sum(y[(p, arm)] for p in pids if (p, arm) in y), sum((p, arm) in y for p in pids)]

    res = {"label": "EXPLORATORY (decided after the 3b results)", "n_sources": len(pids),
           "errors_excluded": {"detections": err2, "asr": n_err2},
           "rates": {f"{d}|{m}": {arm: rate(dets[(d, m)], arm) for arm in NAME} for d, m in keys},
           "asr_rates": {arm: rate(v, arm) for arm in NAME}, "contrasts": {}}
    for x, y in CONTRASTS:
        c = {f"{d}|{m}": M.paired(dets[(d, m)], x, y, pids, meta) for d, m in keys}
        emb = [f"{d}|embedded" for d in D.DETECTORS]
        for k, ph in zip(emb, M.holm([c[k].get("p_perm", 1.0) for k in emb])):
            c[k]["p_holm"] = ph
        res["contrasts"][f"{x}_vs_{y}"] = {"detectors": c, "asr": M.paired(v, x, y, pids, meta)}
    Path("results/item3b_a2_report.json").write_text(json.dumps(res, indent=1))

    L = ["# Item 3b exploratory follow-up: (a″) declarative goal", "",
         "EXPLORATORY: decided after seeing the 3b results. (a″) = original + one declarative sentence stating the "
         "neutral goal, no second person, modals or imperatives (3 words longer than (a)'s sentence, user decision). "
         f"Primary sources: {len(pids)}. RD = X − Y, task-cluster BCa 95% CI, sign-flip permutation p, Holm across "
         "the 7 detectors (embedded) per contrast, exact McNemar (X-only/Y-only).", ""]
    for x, y in CONTRASTS:
        C = res["contrasts"][f"{x}_vs_{y}"]
        L += [f"## {NAME[x]} vs {NAME[y]}", "", "| detector | X | Y | RD [BCa] | p perm | Holm | McNemar (exact p) |",
              "|---|---|---|---|---|---|---|"]
        for k, r in list(C["detectors"].items()) + [("ASR (agent followed)", C["asr"])]:
            if not r.get("n"):
                L.append(f"| {k} | – | – | – | – | – | – |")
                continue
            ci = r["ci"] or [float("nan")] * 2
            L.append(f"| {k} | {r['k_x']}/{r['n']} | {r['k_y']}/{r['n']} | {r['rd']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                     f"{r['p_perm']:.4f} | {r['p_holm']:.4f} | {r['mcnemar_b']}/{r['mcnemar_c']} ({r['p_mcnemar_exact']:.4f}) |"
                     if "p_holm" in r else
                     f"| {k} | {r['k_x']}/{r['n']} | {r['k_y']}/{r['n']} | {r['rd']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                     f"{r['p_perm']:.4f} | – | {r['mcnemar_b']}/{r['mcnemar_c']} ({r['p_mcnemar_exact']:.4f}) |")
        L.append("")
    Path("results/item3b_a2_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
