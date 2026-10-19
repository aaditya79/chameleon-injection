"""
Detector-baseline statistics (item 1), recomputed from raw outputs only.

Input:  results/detections_baselines.jsonl  (experiments/baselines_v2.py)
Output: results/baselines_stats.json, plus a markdown table on stdout.

Per detector. Llama Guard 3's injection verdict is the injection category only (S15, or S1 in the
injection-only taxonomy); "any unsafe category" is reported only as a content-flagging diagnostic.
  * IDR_static     static template payloads, k/n with Wilson 95% CI
  * FPR            the 45 hand-written task contexts, k/n with Wilson 95% CI
  * FPR_gen        the 373 llama3.1-generated clean docs (data/clean_corpus.jsonl), reported separately
  * per camouflage corpus: IDR_camouflage k/n, CDG = IDR_static - IDR_camouflage with the paper's
    task-cluster BCa interval (analysis/bootstrap_ci.cluster_bca, B=10,000, stratified by domain).
llama_full is refusal-screened (n=121), matching the paper's headline v2 numbers; the unscreened value
(n=135) is stored in the JSON as well. For the classifiers the same quantities are also computed on each
payload with no host document (corpus suffix _alone), separating misses from dilution.

Llama Guard 3 category breakdown: for each variant and condition, the number of contexts judged unsafe
and how often each category code appears in the verdict (a verdict can list several), plus clean
documents split by domain, to show which categories drive "any unsafe".

    python analysis/baselines_stats.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.bootstrap_ci import cdg_stat, cluster_bca

IN = Path("results/detections_baselines.jsonl")
OUT = Path("results/baselines_stats.json")
CORPORA = ["llama_full", "llama_noctx", "gpt4o_full", "gpt4o_noctx"]
DETECTORS = [  # (row label, detector id, verdict function, table)
    ("ProtectAI DeBERTa-v3 v2", "protectai_v2", lambda r: r["verdict"], "main"),
    ("Prompt Guard 2 86M", "promptguard2_86m", lambda r: r["verdict"], "main"),
    ("Prompt Guard 2 22M", "promptguard2_22m", lambda r: r["verdict"], "main"),
    ("Llama Guard 3 8B, S1-S14 + S15 (S15 flagged)", "llamaguard3_s15", lambda r: r["verdict"], "main"),
    ("Llama Guard 3 8B, injection-only taxonomy", "llamaguard3_injonly", lambda r: r["verdict"], "main"),
    # Diagnostic only: any S1-S14 category is content flagging, not injection detection.
    ("Llama Guard 3 8B, S1-S14 + S15: any unsafe category (content flagging)", "llamaguard3_s15",
     lambda r: "INJECTED" if r["any_unsafe"] else "CLEAN", "diagnostic"),
]
CLASSIFIERS = {"protectai_v2", "promptguard2_86m", "promptguard2_22m"}
EXPECTED = {"static": 180, "clean": 45, "clean_gen": 373, "llama_full": 135, "llama_noctx": 135,
            "gpt4o_full": 135, "gpt4o_noctx": 135}
EXPECTED_ALONE = {"static_alone": 180, **{f"{c}_alone": 135 for c in CORPORA}}
CONDITIONS = ["static", "llama_full", "llama_noctx", "gpt4o_full", "gpt4o_noctx", "clean", "clean_gen"]
HEADER = ("| Detector | IDR static | Llama full (screened) | Llama no-ctx | GPT-4o full | GPT-4o no-ctx "
          "| FPR clean (45 hand-written) | FPR clean (373 generated) |\n|---|---|---|---|---|---|---|---|")


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def load() -> list[dict]:
    rows = [json.loads(line) for line in IN.open() if line.strip()]
    rows = [r for r in rows if not r.get("dry_run")]
    keys = Counter((r["detector"], r["corpus"], r["task_id"], r["payload_id"]) for r in rows)
    dup = [k for k, v in keys.items() if v > 1]
    assert not dup, f"duplicate records: {dup[:3]}"
    return rows


def hits(sel: list[dict]) -> tuple[int, int]:
    return sum(r["verdict"] == "INJECTED" for r in sel), len(sel)


def score_block(rs: list[dict], suffix: str) -> tuple[dict, list[str]]:
    """IDR_static, and IDR_camouflage + CDG (BCa) per corpus, for embedded (suffix "") or alone payloads."""
    static = [r for r in rs if r["corpus"] == "static" + suffix]
    s_k, s_n = hits(static)
    entry = {"idr_static": [s_k, s_n], "idr_static_wilson": wilson(s_k, s_n), "corpora": {}}
    cells = [f"{s_k}/{s_n} ({s_k / s_n:.1%})" if s_n else "–"]
    for corpus in CORPORA:
        cam_all = [r for r in rs if r["corpus"] == corpus + suffix]
        cam = [r for r in cam_all if not r["is_refusal"]] if corpus == "llama_full" else cam_all
        k, n = hits(cam)
        if not n or not s_n:
            cells.append("–")
            continue
        ci = cluster_bca(static + cam, cdg_stat)
        cdg = s_k / s_n - k / n
        e = {"idr_camouflage": [k, n], "idr_camouflage_wilson": wilson(k, n), "cdg": round(cdg, 4),
             "cdg_bca_ci": ci["ci"] if ci else None}
        if corpus == "llama_full":
            ku, nu = hits(cam_all)
            e["unscreened"] = {"idr_camouflage": [ku, nu], "cdg": round(s_k / s_n - ku / nu, 4)}
        entry["corpora"][corpus] = e
        lo, hi = ci["ci"] if ci and ci.get("ci") else (float("nan"),) * 2
        cells.append(f"{k}/{n} · {cdg:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return entry, cells


def main() -> None:
    rows = load()
    report, tables = {}, {"main": [], "diagnostic": [], "alone": []}
    for label, det, verdict, which in DETECTORS:
        rs = [{**r, "verdict": verdict(r)} for r in rows if r["detector"] == det]
        if not rs:
            continue
        counts = Counter(r["corpus"] for r in rs)
        complete = all(counts.get(c, 0) == n for c, n in EXPECTED.items())
        f_k, f_n = hits([r for r in rs if r["corpus"] == "clean"])
        g_k, g_n = hits([r for r in rs if r["corpus"] == "clean_gen"])
        entry, cells = score_block(rs, "")
        entry.update({"detector": det, "table": which, "complete": complete, "counts": dict(counts),
                      "fpr": [f_k, f_n], "fpr_wilson": wilson(f_k, f_n),
                      "fpr_gen": [g_k, g_n], "fpr_gen_wilson": wilson(g_k, g_n)})
        gen = f"{g_k}/{g_n} ({g_k / g_n:.1%})" if g_n else "–"
        tables[which].append(f"| {label}{'' if complete else ' (INCOMPLETE)'} | " + " | ".join(cells)
                             + f" | {f_k}/{f_n} ({f_k / f_n:.1%}) | {gen} |")
        if det in CLASSIFIERS:
            alone, a_cells = score_block(rs, "_alone")
            a_complete = all(counts.get(c, 0) == n for c, n in EXPECTED_ALONE.items())
            entry["payload_alone"] = alone
            tables["alone"].append(f"| {label}, payload alone{'' if a_complete else ' (INCOMPLETE)'} | "
                                   + " | ".join(a_cells) + " | – | – |")
            tables["alone"].append(f"| {label}, embedded in document | " + " | ".join(cells) + " | – | – |")
        report[label] = entry

    report["llamaguard_categories"] = category_breakdown(rows)
    OUT.write_text(json.dumps(report, indent=2))
    print("\nMAIN: injection detection (Llama Guard verdict = injection category only)\n" + HEADER)
    print("\n".join(tables["main"]))
    print("\nPAYLOAD ALONE vs EMBEDDED (classifiers)\n" + HEADER)
    print("\n".join(tables["alone"]))
    print("\nDIAGNOSTIC: Llama Guard 3 content flagging (any S1-S15 category; not an injection verdict)\n" + HEADER)
    print("\n".join(tables["diagnostic"]))
    print("\nCamouflage cells: detected/n · CDG [task-cluster BCa 95% CI]. Wrote", OUT)


def _cat_counts(sel: list[dict]) -> dict:
    cats = Counter(c for r in sel for c in r["categories"])
    return {"n": len(sel), "unsafe": sum(r["any_unsafe"] for r in sel),
            "categories": dict(sorted(cats.items(), key=lambda kv: int(kv[0][1:]) if kv[0][1:].isdigit() else 99)),
            "unsafe_without_categories": sum(r["any_unsafe"] and not r["categories"] for r in sel)}


def category_breakdown(rows: list[dict]) -> dict:
    out = {}
    for det in ("llamaguard3_s15", "llamaguard3_injonly"):
        rs = [r for r in rows if r["detector"] == det]
        if not rs:
            continue
        d = {}
        for cond in CONDITIONS:
            sel = [r for r in rs if r["corpus"] == cond and not (cond == "llama_full" and r["is_refusal"])]
            if sel:
                d[cond] = _cat_counts(sel)
        for cond in ("clean", "clean_gen"):
            for dom in ("financial", "legal", "general"):
                sel = [r for r in rs if r["corpus"] == cond and r["domain"] == dom]
                if sel:
                    d[f"{cond}:{dom}"] = _cat_counts(sel)
        out[det] = d
        print(f"\n{det}: unsafe/n and category mentions per condition")
        print("| Condition | unsafe/n | categories |")
        print("|---|---|---|")
        for cond, c in d.items():
            cats = ", ".join(f"{k} {v}" for k, v in c["categories"].items()) or "–"
            extra = f" (+{c['unsafe_without_categories']} unsafe, no category)" if c["unsafe_without_categories"] else ""
            print(f"| {cond} | {c['unsafe']}/{c['n']} | {cats}{extra} |")
    return out


if __name__ == "__main__":
    main()
