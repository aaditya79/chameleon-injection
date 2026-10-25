"""
Item 3 manipulation checks (pre-registration 4d6687c, P3, as amended by A1-A5), recomputed from the templates.

For each goal set (neutral = A1 primary, verbatim = secondary) and each core payload:
  words            whitespace word count
  directive        automatic count of second-person / modal / imperative forms (mechanism_payloads.DIRECTIVE_RE)
  acs              cosine similarity to the task's clean_context (camouflage_generator ACS model)
  marker_rate      domain vocabulary_markers (data/domains.json) per 100 words
  llm_rating       1-5 directiveness from results/mechanism_ratings.jsonl, when present

Required (A3/A4): directive and LLM rating D+ > D- within each V level; ACS and marker rate V+ > V- within each
D level on financial+legal. General domain reported separately (exploratory).

    python analysis/mechanism_checks.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.mechanism_payloads import DIRECTIVE_RE, build_payloads

OUT = Path("results/mechanism_checks.json")
RATINGS = Path("results/mechanism_ratings.jsonl")
CELLS = ["D+V-", "D+V+", "D-V+", "D-V-"]


def acs(payloads: list[dict]) -> list[float]:
    import numpy as np

    from src.attacks.camouflage_generator import _get_st_model

    ctx = {t["task_id"]: t["clean_context"] for t in json.loads(Path("data/tasks.json").read_text())}
    m = _get_st_model()
    a = m.encode([p["text"] for p in payloads], convert_to_numpy=True, batch_size=32)
    b = m.encode([ctx[p["task_id"]] for p in payloads], convert_to_numpy=True, batch_size=32)
    return [float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in zip(a, b)]


def marker_rate(text: str, markers: list[str]) -> float:
    n = sum(len(re.findall(r"\b" + re.escape(mk) + r"\b", text, re.I)) for mk in markers)
    return 100 * n / len(text.split())


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    domains = json.loads(Path("data/domains.json").read_text())
    ratings = {}
    if RATINGS.exists():
        for r in map(json.loads, RATINGS.open()):
            if r["status"] == "ok":
                ratings[r["payload_id"]] = r["rating"]
    report = {}
    for gs in ("neutral", "verbatim"):
        ps = build_payloads(probe=False, goals=gs)
        for p, s in zip(ps, acs(ps)):
            p["acs"] = s
            p["words"] = len(p["text"].split())
            p["directive"] = len(DIRECTIVE_RE.findall(p["text"]))
            p["marker_rate"] = marker_rate(p["text"], domains[p["domain"]]["vocabulary_markers"])
            p["llm_rating"] = ratings.get(p["payload_id"])
        rep = {"per_template": {}, "per_cell": {}, "checks": {}}
        print(f"\n## Goal set: {gs}\n")
        print("| Template | Cell | words mean (min-max) | directive | ACS | marker/100w f+l | LLM rating |")
        print("|---|---|---|---|---|---|---|")
        by_t = defaultdict(list)
        for p in ps:
            by_t[(p["cell"], p["template_id"])].append(p)
        for (cell, tid), xs in sorted(by_t.items(), key=lambda kv: kv[0][1]):
            w = [x["words"] for x in xs]
            fl = [x for x in xs if x["domain"] != "general"]
            rt = [x["llm_rating"] for x in xs if x["llm_rating"] is not None]
            row = {"words_mean": mean(w), "words_min": min(w), "words_max": max(w),
                   "directive": mean([x["directive"] for x in xs]), "acs": mean([x["acs"] for x in xs]),
                   "marker_rate_finleg": mean([x["marker_rate"] for x in fl]),
                   "llm_rating": mean(rt) if rt else None, "n_rated": len(rt)}
            rep["per_template"][tid] = {"cell": cell, **row}
            print(f"| {tid} | {cell} | {row['words_mean']:.1f} ({min(w)}-{max(w)}) | {row['directive']:.2f} | "
                  f"{row['acs']:.3f} | {row['marker_rate_finleg']:.2f} | "
                  f"{'%.2f' % row['llm_rating'] if rt else '·'} |")
        for scope, sel in (("fin+legal", lambda p: p["domain"] != "general"), ("general", lambda p: p["domain"] == "general"),
                           ("all", lambda p: True)):
            cm = {}
            for c in CELLS:
                xs = [p for p in ps if p["cell"] == c and sel(p)]
                rt = [x["llm_rating"] for x in xs if x["llm_rating"] is not None]
                cm[c] = {"n": len(xs), "words": mean([x["words"] for x in xs]),
                         "directive": mean([x["directive"] for x in xs]), "acs": mean([x["acs"] for x in xs]),
                         "marker_rate": mean([x["marker_rate"] for x in xs]),
                         "llm_rating": mean(rt) if rt else None, "n_rated": len(rt)}
            rep["per_cell"][scope] = cm
            chk = {
                "directive D+>D- | V-": cm["D+V-"]["directive"] > cm["D-V-"]["directive"],
                "directive D+>D- | V+": cm["D+V+"]["directive"] > cm["D-V+"]["directive"],
                "ACS V+>V- | D+": cm["D+V+"]["acs"] > cm["D+V-"]["acs"],
                "ACS V+>V- | D-": cm["D-V+"]["acs"] > cm["D-V-"]["acs"],
                "markers V+>V- | D+": cm["D+V+"]["marker_rate"] > cm["D+V-"]["marker_rate"],
                "markers V+>V- | D-": cm["D-V+"]["marker_rate"] > cm["D-V-"]["marker_rate"],
            }
            if all(cm[c]["llm_rating"] is not None for c in CELLS):
                chk["LLM D+>D- | V-"] = cm["D+V-"]["llm_rating"] > cm["D-V-"]["llm_rating"]
                chk["LLM D+>D- | V+"] = cm["D+V+"]["llm_rating"] > cm["D-V+"]["llm_rating"]
            rep["checks"][scope] = chk
            print(f"\n{scope}: " + "; ".join(
                f"{c} words {cm[c]['words']:.1f} dir {cm[c]['directive']:.2f} ACS {cm[c]['acs']:.3f} "
                f"mk {cm[c]['marker_rate']:.2f}" + (f" LLM {cm[c]['llm_rating']:.2f}" if cm[c]["llm_rating"] else "")
                for c in CELLS))
            print("  " + ", ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in chk.items()))
        report[gs] = rep
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
