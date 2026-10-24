"""
Item 3b selection report (A-log amendment 7, item 4): for each arm, the detection rate of the ORIGINAL payloads
(existing v2 detection logs) in the accepted vs the excluded subset, per detector. Excluded = status "failed";
"not_applicable" payloads are left out of both subsets. Descriptive: payload-level counts with a two-sided Fisher
exact p (no task clustering, no multiplicity correction).

    python3 analysis/item3b_selection.py   -> results/item3b_selection_report.md, results/item3b_selection.json
"""

from __future__ import annotations

import json
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.refusal_policy import excluded, out  # integrity issue #4

from scipy.stats import fisher_exact

ARMS = ("a", "a_prime", "f", "b", "c", "d", "e")


def load(name: str) -> list[dict]:
    return [json.loads(x) for x in Path(f"results/{name}.jsonl").open() if x.strip()]


def detector_maps(ids: set) -> dict:
    """{detector label: {payload_id: detected bool}} for the 121 Llama v2 full-context payloads."""
    out = {}
    for r in load("detections_v2"):
        if r["payload_id"] in ids:
            out.setdefault(f"{r['detector_type']} (Llama 3.1 8B)", {})[r["payload_id"]] = r["verdict"] == "INJECTED"
    # detections_gpt4o is NOT a source here: it holds GPT-4o's detections of GPT-4o-written payloads, whose ids
    # (cam_fin_001_v1, ...) collide with the Llama ids. Bug found 2026-10-20 (integrity issue #4 work); the GPT-4o
    # detector saw the Llama originals only in cross_detect cell D.
    for r in load("detections_baselines"):
        if r["payload_id"] in ids and r["corpus"] in ("llama_full", "llama_full_alone") \
                and r["detector"] in ("llamaguard3_injonly", "protectai_v2", "promptguard2_86m", "promptguard2_22m"):
            mode = "alone" if r["corpus"].endswith("_alone") else "embedded"
            out.setdefault(f"{r['detector']} ({mode})", {})[r["payload_id"]] = r["verdict"] == "INJECTED"
    for r in load("cross_detect"):
        if r["payload_id"] in ids and r["cell"] == "D":
            out.setdefault("gpt4o static (cross_detect cell D)", {})[r["payload_id"]] = r["verdict"] == "INJECTED"
    for k, v in out.items():
        assert len(v) == len(ids), (k, len(v))
    return out


def main() -> None:
    d = json.loads(Path("data/item3b_payloads.json").read_text())
    ids = set(d["payloads"]) - excluded("v2_llama")
    dets = detector_maps(ids)
    res = {}
    for arm in ARMS:
        acc = [p for p, r in d["payloads"].items() if p in ids and r["arms"][arm]["status"] == "accepted"]
        exc = [p for p, r in d["payloads"].items() if p in ids and r["arms"][arm]["status"] == "failed"]
        res[arm] = {"n_accepted": len(acc), "n_excluded": len(exc), "detectors": {}}
        for name, m in dets.items():
            ka, ke = sum(m[p] for p in acc), sum(m[p] for p in exc)
            p = fisher_exact([[ka, len(acc) - ka], [ke, len(exc) - ke]])[1] if acc and exc else None
            res[arm]["detectors"][name] = {"accepted": [ka, len(acc)], "excluded": [ke, len(exc)], "fisher_p": p}
    out(Path("results/item3b_selection.json")).write_text(json.dumps(res, indent=1))
    pct = lambda k, n: f"{k}/{n} ({k / n:.0%})" if n else "–"
    lines = ["# Item 3b selection report (amendment 7, item 4)", "",
             "Detection rate of the ORIGINAL payloads (v2 logs) among payloads accepted vs excluded in each arm. "
             "Payload-level, two-sided Fisher exact p, descriptive only.", ""]
    for arm in ARMS:
        r = res[arm]
        lines += [f"## ({arm.replace('_prime', '′')}): {r['n_accepted']} accepted, {r['n_excluded']} excluded", "",
                  "| detector | accepted | excluded | Fisher p |", "|---|---|---|---|"]
        for name, x in r["detectors"].items():
            p = "–" if x["fisher_p"] is None else f"{x['fisher_p']:.3f}"
            lines.append(f"| {name} | {pct(*x['accepted'])} | {pct(*x['excluded'])} | {p} |")
        lines.append("")
    out(Path("results/item3b_selection_report.md")).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
