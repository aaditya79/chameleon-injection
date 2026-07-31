"""
Recompute every paper number from results/trials.jsonl with dry-run mock trials
excluded.

Run analysis/mark_dry_run.py --write first; this script requires the `dry_run`
field and refuses to run without it.

Emits results/paper_numbers_corrected.json plus a console report laid out to
match Table 1 and the prose figures in acl_latex.tex.

    python analysis/recompute_paper_numbers.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

TRIALS_PATH = Path("results/trials.jsonl")
OUT_PATH = Path("results/paper_numbers_corrected.json")

LLAMA = "llama3.1"
GEMINI = "google/gemini-2.0-flash-001"
MODELS = [(LLAMA, "Llama 3.1 8B"), (GEMINI, "Gemini 2.0 Flash")]
DOMAINS = ["financial", "legal", "general"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(include_dry_run: bool = False) -> list[dict]:
    rows = [json.loads(line) for line in TRIALS_PATH.open()]
    missing = [r for r in rows if "dry_run" not in r]
    if missing:
        raise SystemExit(
            f"{len(missing)} records lack the `dry_run` field. "
            "Run: python analysis/mark_dry_run.py --write"
        )
    return rows if include_dry_run else [r for r in rows if not r["dry_run"]]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def detections(rows, model, experiment, payload_type, detector, arch=None):
    return [
        r for r in rows
        if r["model"] == model
        and r["experiment"] == experiment
        and r["payload_type"] == payload_type
        and (arch is None or r["architecture"] == arch)
        and (r.get("detection_result") or {}).get("detector_type") == detector
    ]


def idr(rows) -> tuple[int, int, float]:
    if not rows:
        return 0, 0, float("nan")
    n = sum(1 for r in rows if r["detection_result"]["verdict"] == "INJECTED")
    return n, len(rows), n / len(rows)


def asr(rows) -> tuple[int, int, float]:
    judged = [r for r in rows if r.get("asr_judgment")]
    if not judged:
        return 0, 0, float("nan")
    n = sum(1 for r in judged if r["asr_judgment"].get("followed_injection"))
    return n, len(judged), n / len(judged)


def trials(rows, model, experiment, payload_type, arch=None):
    return [
        r for r in rows
        if r["model"] == model
        and r["experiment"] == experiment
        and r["payload_type"] == payload_type
        and (arch is None or r["architecture"] == arch)
    ]


def fmt(t) -> str:
    n, d, v = t
    return f"{n}/{d} = {v:.4f}" if d else "n/a"


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def cdg_block(rows, model, experiment, detector, arch=None) -> dict:
    s = idr(detections(rows, model, experiment, "static", detector, arch))
    c = idr(detections(rows, model, experiment, "camouflage", detector, arch))
    out = {
        "idr_static": s, "idr_camouflage": c,
        "cdg": s[2] - c[2] if s[1] and c[1] else float("nan"),
        "by_domain": {},
    }
    for d in DOMAINS:
        ds = idr([r for r in detections(rows, model, experiment, "static", detector, arch) if r["domain"] == d])
        dc = idr([r for r in detections(rows, model, experiment, "camouflage", detector, arch) if r["domain"] == d])
        out["by_domain"][d] = {
            "idr_static": ds, "idr_camouflage": dc,
            "cdg": ds[2] - dc[2] if ds[1] and dc[1] else float("nan"),
        }
    return out


def confidence_block(rows, model) -> dict:
    cam = detections(rows, model, "exp1", "camouflage", "static", "single_agent")
    missed = [r for r in cam if r["detection_result"]["verdict"] == "CLEAN"]
    caught = [r for r in cam if r["detection_result"]["verdict"] == "INJECTED"]
    mc = Counter(r["detection_result"]["confidence"] for r in missed)
    cc = Counter(r["detection_result"]["confidence"] for r in caught)
    return {
        "n_camouflage": len(cam),
        "n_missed": len(missed), "missed_confidence": dict(mc),
        "missed_high_share": (mc.get("HIGH", 0) / len(missed)) if missed else float("nan"),
        "n_caught": len(caught), "caught_confidence": dict(cc),
        "caught_by_domain": {
            d: (sum(1 for r in cam if r["domain"] == d and r["detection_result"]["verdict"] == "INJECTED"),
                sum(1 for r in cam if r["domain"] == d))
            for d in DOMAINS
        },
        "caught_by_variant": {
            v: (sum(1 for r in cam if (r["payload_id"] or "").endswith(f"_v{v}")
                    and r["detection_result"]["verdict"] == "INJECTED"),
                sum(1 for r in cam if (r["payload_id"] or "").endswith(f"_v{v}")))
            for v in ("1", "2", "3")
        },
    }


def daf_block(rows, model) -> dict:
    """DAF under each candidate single-agent baseline, with task-set overlap shown."""
    out = {}
    for pt in ("static", "camouflage"):
        debate = trials(rows, model, "exp2", pt, "debate")
        base_e2 = trials(rows, model, "exp2", pt, "single_agent")
        base_e1 = trials(rows, model, "exp1", pt, "single_agent")

        d_asr, e2_asr, e1_asr = asr(debate), asr(base_e2), asr(base_e1)
        e2_tasks = sorted({r["task_id"] for r in base_e2})
        matched = asr([r for r in debate if r["task_id"] in set(e2_tasks)])

        out[pt] = {
            "asr_debate_all_tasks": d_asr,
            "asr_single_exp2_baseline": e2_asr,
            "asr_single_exp1_baseline": e1_asr,
            "exp2_baseline_tasks": e2_tasks,
            "asr_debate_matched_tasks": matched,
            "daf_as_published": (d_asr[2] / e2_asr[2]) if e2_asr[1] and e2_asr[2] else float("nan"),
            "daf_vs_exp1_baseline": (d_asr[2] / e1_asr[2]) if e1_asr[1] and e1_asr[2] else float("nan"),
            "daf_matched_tasks": (matched[2] / e2_asr[2]) if e2_asr[1] and e2_asr[2] else float("nan"),
        }
    return out


def cdg_improvement(before: dict, after: dict) -> float:
    gap_b, gap_a = before["cdg"], after["cdg"]
    return 100.0 * (gap_b - gap_a) / gap_b if gap_b else float("nan")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main() -> None:
    kept = load(include_dry_run=False)
    allrows = load(include_dry_run=True)
    n_dry = len(allrows) - len(kept)

    print("=" * 72)
    print("CORRECTED PAPER NUMBERS  (dry-run mock trials excluded)")
    print("=" * 72)
    print(f"  trials total {len(allrows)} | excluded as dry_run {n_dry} | analysed {len(kept)}")

    report: dict = {
        "provenance": {
            "source": str(TRIALS_PATH),
            "total_trials": len(allrows),
            "excluded_dry_run": n_dry,
            "analysed_trials": len(kept),
            "exclusion_rule": "dry_run == True (see analysis/mark_dry_run.py)",
        },
        "models": {},
    }

    for model, label in MODELS:
        print(f"\n{'='*72}\n{label}  [{model}]\n{'='*72}")
        m: dict = {}

        # ---- Exp 1 ----
        e1 = cdg_block(kept, model, "exp1", "static", "single_agent")
        m["exp1_static_detector"] = e1
        print("  Exp 1 (single_agent, static detector)")
        print(f"    IDR static     : {fmt(e1['idr_static'])}")
        print(f"    IDR camouflage : {fmt(e1['idr_camouflage'])}")
        print(f"    CDG            : {e1['cdg']:.4f}")
        for d in DOMAINS:
            b = e1["by_domain"][d]
            print(f"      {d:<10} static {fmt(b['idr_static']):>18}  cam {fmt(b['idr_camouflage']):>16}  CDG {b['cdg']:.4f}")

        # ---- ASR ----
        m["asr"] = {}
        print("  ASR (exp1 single_agent)")
        for pt in ("static", "camouflage"):
            a = asr(trials(kept, model, "exp1", pt, "single_agent"))
            m["asr"][pt] = a
            print(f"    {pt:<11}: {fmt(a)}")

        # ---- Exp 3 ----
        e3 = cdg_block(kept, model, "exp3", "static", "detection_only")
        m["exp3_static_detector"] = e3
        print("  Exp 3 (detection_only, static detector)")
        print(f"    IDR static {fmt(e3['idr_static'])} | cam {fmt(e3['idr_camouflage'])} | CDG {e3['cdg']:.4f}")
        for d in DOMAINS:
            b = e3["by_domain"][d]
            print(f"      {d:<10} static {fmt(b['idr_static']):>18}  cam {fmt(b['idr_camouflage']):>16}  CDG {b['cdg']:.4f}")

        # ---- Exp 4 ----
        e4s = cdg_block(kept, model, "exp4", "static", "detection_only")
        e4a = cdg_block(kept, model, "exp4", "augmented", "detection_only")
        imp = cdg_improvement(e4s, e4a)
        m["exp4_static_detector"] = e4s
        m["exp4_augmented_detector"] = e4a
        m["cdg_improvement_pct"] = imp
        print("  Exp 4 (detection_only)")
        print(f"    static det.   : IDR static {fmt(e4s['idr_static'])} | cam {fmt(e4s['idr_camouflage'])} | CDG {e4s['cdg']:.4f}")
        print(f"    augmented det.: IDR static {fmt(e4a['idr_static'])} | cam {fmt(e4a['idr_camouflage'])} | CDG {e4a['cdg']:.4f}")
        print(f"    CDG improvement: {imp:+.1f}%")

        # ---- Confidence ----
        cb = confidence_block(kept, model)
        m["confidence"] = cb
        print("  Confidence breakdown (exp1 camouflage)")
        print(f"    n={cb['n_camouflage']}  missed={cb['n_missed']} {cb['missed_confidence']}"
              f"  (HIGH share {cb['missed_high_share']:.4f})")
        print(f"    caught={cb['n_caught']} {cb['caught_confidence']}")
        print(f"    caught by domain : {cb['caught_by_domain']}")
        print(f"    caught by variant: {cb['caught_by_variant']}")

        # ---- DAF ----
        db = daf_block(kept, model)
        m["daf"] = db
        print("  DAF (exp2)")
        for pt, v in db.items():
            print(f"    {pt}:")
            print(f"      ASR debate (all tasks)      : {fmt(v['asr_debate_all_tasks'])}")
            print(f"      ASR single (exp2 baseline)  : {fmt(v['asr_single_exp2_baseline'])}"
                  f"   tasks={len(v['exp2_baseline_tasks'])}")
            print(f"      ASR single (exp1 baseline)  : {fmt(v['asr_single_exp1_baseline'])}")
            print(f"      ASR debate (matched tasks)  : {fmt(v['asr_debate_matched_tasks'])}")
            print(f"      DAF as published            : {v['daf_as_published']:.3f}")
            print(f"      DAF vs exp1 baseline        : {v['daf_vs_exp1_baseline']:.3f}")
            print(f"      DAF matched tasks           : {v['daf_matched_tasks']:.3f}")

        report["models"][model] = m

    # ---- cost / volume ----
    report["cost_usd"] = sum(r.get("total_cost_usd") or 0 for r in kept)
    report["cost_usd_including_dry_run"] = sum(r.get("total_cost_usd") or 0 for r in allrows)
    print(f"\n  Total API cost (analysed trials): ${report['cost_usd']:.4f}")

    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"  Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
