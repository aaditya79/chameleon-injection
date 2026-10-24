"""
Item 3b EXPLORATORY follow-up (A-log 2026-10-21, decided after seeing the (a″) results): arm (a‴), the goal's
outcome asserted as a finding or evaluation (data/item3b_a3_sentences.json, user-approved, tagged assertion /
normative-action).

Contrasts, paired by source payload on the 96 primary sources, same methods as 3b (item3b_models.paired):
  (a‴) vs (a′)     asserting the outcome vs a neutral sentence
  (a″) vs (a‴)     the goal as a declarative action ("The appropriate conclusion is to ...") vs the outcome asserted
Per detector and mode; Holm across the 7 detectors (embedded) per contrast; ASR for both contrasts.
Exploratory split by sentence tag (assertion / normative-action): same paired statistics, no Holm.
Caveat: each (a‴) sentence is 1-3 words shorter than the task's (a″) sentence.

    python3 analysis/item3b_a3.py   -> results/item3b_a3_report.json, .md
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import item3b_data as D  # noqa: E402
import item3b_models as M  # noqa: E402

CONTRASTS = [("a_tprime", "a_prime"), ("a_dprime", "a_tprime")]
NAME = {"a_prime": "(a′)", "a_dprime": "(a″)", "a_tprime": "(a‴)"}


def load():
    dets, _ = D.detections("results/item3b_detections.jsonl")
    v, _ = D.asr("results/item3b_trials.jsonl")
    errs = {}
    for tag in ("a2", "a3"):
        d2, e2 = D.detections(f"results/item3b_{tag}_detections.jsonl")
        v2, n2 = D.asr(f"results/item3b_{tag}_trials.jsonl")
        for k, y in d2.items():
            dets.setdefault(k, {}).update(y)
        v.update(v2)
        errs[tag] = {"detections": e2, "asr": n2}
    return dets, v, errs


def main() -> None:
    meta = D.source_meta()
    pids = D.analysis_sets(meta)["primary"]
    tags = {r["task_id"]: r["tag"] for r in json.loads(Path("data/item3b_a3_sentences.json").read_text())["sentences"]}
    dets, v, errs = load()
    keys = [(d, "embedded") for d in D.DETECTORS] + [(d, "alone") for d in D.CLASSIFIERS]
    res = {"label": "EXPLORATORY (decided after the (a″) results)", "n_sources": len(pids), "errors_excluded": errs,
           "caveat": "each (a‴) sentence is 1-3 words shorter than the task's (a″) sentence", "contrasts": {}}
    for x, y in CONTRASTS:
        c = {f"{d}|{m}": M.paired(dets[(d, m)], x, y, pids, meta) for d, m in keys}
        emb = [f"{d}|embedded" for d in D.DETECTORS]
        for k, ph in zip(emb, M.holm([c[k].get("p_perm", 1.0) for k in emb])):
            c[k]["p_holm"] = ph
        split = {t: {f"{d}|embedded": M.paired(dets[(d, "embedded")], x, y, pids, meta,
                                              keep=lambda p, t=t: tags[meta[p]["task_id"]] == t) for d in D.DETECTORS}
                 for t in ("assertion", "normative-action")}
        for t in split:
            split[t]["ASR"] = M.paired(v, x, y, pids, meta, keep=lambda p, t=t: tags[meta[p]["task_id"]] == t)
        res["contrasts"][f"{x}_vs_{y}"] = {"detectors": c, "asr": M.paired(v, x, y, pids, meta), "by_tag": split}
    Path("results/item3b_a3_report.json").write_text(json.dumps(res, indent=1))

    def row(k, r, holm=True):
        if not r.get("n"):
            return f"| {k} | – | – | – | – | – | – |"
        ci = r["ci"] or [float("nan")] * 2
        return (f"| {k} | {r['k_x']}/{r['n']} | {r['k_y']}/{r['n']} | {r['rd']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                f"{r['p_perm']:.4f} | {r['p_holm']:.4f} | " if holm and "p_holm" in r else
                f"| {k} | {r['k_x']}/{r['n']} | {r['k_y']}/{r['n']} | {r['rd']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                f"{r['p_perm']:.4f} | – | ") + f"{r['mcnemar_b']}/{r['mcnemar_c']} ({r['p_mcnemar_exact']:.4f}) |"

    L = ["# Item 3b exploratory follow-up: (a‴) asserted outcome", "",
         "EXPLORATORY: decided after seeing the (a″) results. (a‴) = original + one user-approved sentence asserting the "
         "goal's outcome as a finding or evaluation (no second person, modals, imperatives or 'to {verb}'). "
         f"Primary sources: {len(pids)}. RD = X − Y, task-cluster BCa 95% CI, sign-flip permutation p, Holm across the "
         "7 detectors (embedded) per contrast, exact McNemar (X-only/Y-only). Caveat: each (a‴) sentence is 1-3 words "
         "shorter than the task's (a″) sentence.", ""]
    hdr = ["| detector | X | Y | RD [BCa] | p perm | Holm | McNemar (exact p) |", "|---|---|---|---|---|---|---|"]
    for x, y in CONTRASTS:
        C = res["contrasts"][f"{x}_vs_{y}"]
        L += [f"## {NAME[x]} vs {NAME[y]}", "", *hdr]
        L += [row(k, r) for k, r in C["detectors"].items()] + [row("ASR (agent followed)", C["asr"], False), ""]
        for t, S in C["by_tag"].items():
            L += [f"### by tag: {t} (exploratory, no Holm)", "", *hdr]
            L += [row(k, r, False) for k, r in S.items()] + [""]
    Path("results/item3b_a3_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
