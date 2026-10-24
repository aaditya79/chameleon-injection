"""
Item 3c caliper-matched subset (ITEM3C_BENIGN_TWINS_DESIGN.md, "Manipulation checks"; A-log 2026-10-21), built from
text features only, before any 3c detection.

A pair is a twin whose matching target was its own slot's camouflage payload, with that payload. It is kept when
|twin - payload| <= 0.25 SD on every matching dimension that failed for the arm (results/item3c_checks.json), the
SD being that of the arm's re-screened camouflage (43 tasks), as in the matching distance. Twins with a task-mean or
arm-mean target have no paired payload and are not in the subset.
Also written (A-log 2026-10-21, user decision after the checks): descriptive subsets at 0.5 and 1 SD ("wider"), and
per-item features ("features", raw) for the covariate-adjusted analysis.

    python3 analysis/item3c_caliper.py   -> results/item3c_caliper_features.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.item3c_checks import items, vocab_overlap
from experiments.item3c_generate import features

CALIPER = 0.25
WIDER = (0.5, 1.0)  # descriptive only


def main() -> None:
    checks = json.loads(Path("results/item3c_checks.json").read_text())["arms"]
    tasks = {t["task_id"]: t for t in json.loads(Path("data/tasks.json").read_text())}
    slots = json.loads(Path("data/item3c_twins.json").read_text())["arms"]
    feat = {}
    for x in items():
        ctx = tasks[x["task_id"]]["clean_context"]
        f = features(x["text"], ctx)
        feat[(x["arm"], x["kind"], x["id"])] = {"acs": f["acs"], "words": f["words"], "spec100": f["spec100"],
                                                "vocab": vocab_overlap(x["text"], ctx)}
    out = {}
    for arm in ("llama", "gpt4o"):
        failed = [k for k, d in checks[arm]["matching"].items() if not d["passed"]]
        cam = {i: v for (a, k, i), v in feat.items() if a == arm and k == "camouflage"}
        sd = {k: float(np.std([v[k] for v in cam.values()], ddof=1)) for k in failed}
        pairs = [(s["twin_id"], s["target_payload_id"]) for s in slots[arm]["slots"].values()
                 if s["status"] == "selected" and s["target"] == "slot payload" and s["target_payload_id"] in cam]
        within = lambda c: [(t, p) for t, p in pairs
                            if all(abs(feat[(arm, "twin", t)][k] - cam[p][k]) <= c * sd[k] for k in failed)]
        keep = within(CALIPER)
        out[arm] = {"failed_dimensions": failed, "sd_camouflage": sd, "n_candidate_pairs": len(pairs),
                    "n_pairs": len(keep), "twins": [t for t, _ in keep], "camouflage": sorted({p for _, p in keep}),
                    "wider": {str(c): {"n_pairs": len(w), "twins": [t for t, _ in w], "camouflage": sorted({p for _, p in w})}
                              for c in WIDER for w in [within(c)]},
                    "features": {f"{k}|{i}": v for (a, k, i), v in feat.items() if a == arm}}
        print(f"{arm}: failed {failed}; {len(keep)} of {len(pairs)} pairs within {CALIPER} SD")
    Path("results/item3c_caliper_features.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
