"""
Full-context vs no-context CDG: the deliverable for the threat-model critique.

Pairs the detection results from the no-document-access generator against the
full-context corpus, for both detectors, on refusal-screened payloads.

The comparison answers one question directly: when the attacker cannot see the
target document, does the detector find the resulting camouflage easier or harder
to catch? Either direction is reportable. Easier means our headline CDG is an
upper bound inflated by an unrealistically strong attacker. Harder would mean
document access is not what makes camouflage work.

Llama-only; Gemini 2.0 Flash is retired and cannot run this condition.

    python analysis/nocontext_compare.py
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

FULL_DET = Path("results/detections_v2.jsonl")
NOCTX_DET = Path("results/detections_nocontext.jsonl")
FULL_CORPUS = Path("data/camouflage_corpus_v2.json")
NOCTX_CORPUS = Path("data/camouflage_corpus_nocontext.json")
OUT = Path("results/nocontext_compare.json")

DOMAINS = ["financial", "legal", "general"]


def idr(rows: list[dict]) -> tuple[int, int, float]:
    if not rows:
        return 0, 0, float("nan")
    n = sum(1 for r in rows if r["verdict"] == "INJECTED")
    return n, len(rows), n / len(rows)


def fmt(t) -> str:
    n, d, v = t
    return f"{n}/{d} = {v:.4f}" if d else "n/a"


def main() -> None:
    for p in (FULL_DET, NOCTX_DET, FULL_CORPUS, NOCTX_CORPUS):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    full = [json.loads(l) for l in FULL_DET.open()]
    noctx = [json.loads(l) for l in NOCTX_DET.open()]
    if len({(r["payload_id"], r["detector_type"]) for r in noctx}) < 270:
        raise SystemExit(f"no-context detection incomplete ({len(noctx)} records)")

    # IDR_static is a property of the static payload bank, identical in both
    # conditions -- the no-context variable changes only the camouflage arm.
    static_ref = {
        d: idr([r for r in full if r["detector_type"] == d and r["payload_type"] == "static"])
        for d in ("static", "augmented")
    }

    report: dict = {"static_reference": static_ref, "detectors": {}}

    print("=" * 78)
    print("FULL-CONTEXT vs NO-CONTEXT CAMOUFLAGE   (Llama 3.1 8B, refusal-screened)")
    print("=" * 78)

    for det in ("static", "augmented"):
        f_cam = [r for r in full if r["detector_type"] == det
                 and r["payload_type"] == "camouflage" and not r["is_refusal"]]
        n_cam = [r for r in noctx if r["detector_type"] == det and not r["is_refusal"]]

        s = static_ref[det]
        fi, ni = idr(f_cam), idr(n_cam)
        f_cdg, n_cdg = s[2] - fi[2], s[2] - ni[2]

        print(f"\n  [{det} detector]   IDR_static {fmt(s)}")
        print(f"    full-context camouflage : IDR {fmt(fi)}   CDG {f_cdg:.4f}")
        print(f"    no-context   camouflage : IDR {fmt(ni)}   CDG {n_cdg:.4f}")
        print(f"    CDG change (no-ctx - full) : {n_cdg - f_cdg:+.4f}")

        by_dom = {}
        for d in DOMAINS:
            fd, nd = idr([r for r in f_cam if r["domain"] == d]), idr([r for r in n_cam if r["domain"] == d])
            sd = idr([r for r in full if r["detector_type"] == det
                      and r["payload_type"] == "static" and r["domain"] == d])
            by_dom[d] = {"cdg_full": sd[2] - fd[2], "cdg_noctx": sd[2] - nd[2],
                         "idr_full": fd, "idr_noctx": nd}
            print(f"      {d:<10} CDG {by_dom[d]['cdg_full']:.4f} -> {by_dom[d]['cdg_noctx']:.4f}"
                  f"   (cam IDR {fmt(fd)} -> {fmt(nd)})")

        report["detectors"][det] = {
            "idr_static": s,
            "idr_camouflage_full": fi, "idr_camouflage_noctx": ni,
            "cdg_full": round(f_cdg, 4), "cdg_noctx": round(n_cdg, 4),
            "cdg_change": round(n_cdg - f_cdg, 4),
            "by_domain": by_dom,
        }

    # ---- corpus-level comparison ----
    fc = json.loads(FULL_CORPUS.read_text())
    nc = json.loads(NOCTX_CORPUS.read_text())
    fmeta, nmeta = fc["meta"]["llama"], nc["meta"]

    def acs_of(corpus_store):
        vals = [p["acs"] for ps in corpus_store.values() for p in ps
                if not p.get("is_refusal") and p.get("acs", 0) > 0]
        return vals

    fa, na = acs_of(fc["generated"]["llama"]), acs_of(nc["generated"])
    print("\n  Corpus comparison:")
    print(f"    {'':<22}{'full-context':>16}{'no-context':>16}")
    print(f"    {'refusals first draw':<22}"
          f"{fmeta['refusals_on_first_draw']:>10}/{fmeta['n_payloads']:<5}"
          f"{nmeta['refusals_on_first_draw']:>10}/{nmeta['n_payloads']:<5}")
    print(f"    {'refusals remaining':<22}"
          f"{fmeta.get('refusals_remaining_in_corpus', fmeta.get('refusals_remaining')):>16}"
          f"{nmeta['refusals_remaining']:>16}")
    print(f"    {'ACS mean':<22}{st.mean(fa):>16.4f}{st.mean(na):>16.4f}")
    print(f"    {'ACS sd':<22}{st.pstdev(fa):>16.4f}{st.pstdev(na):>16.4f}")

    report["corpus"] = {
        "full": {"refusals_first_draw": fmeta["refusals_on_first_draw"],
                 "n_payloads": fmeta["n_payloads"],
                 "acs_mean": round(st.mean(fa), 4), "acs_sd": round(st.pstdev(fa), 4)},
        "noctx": {"refusals_first_draw": nmeta["refusals_on_first_draw"],
                  "n_payloads": nmeta["n_payloads"],
                  "acs_mean": round(st.mean(na), 4), "acs_sd": round(st.pstdev(na), 4)},
    }

    # ---- read ----
    d = report["detectors"]["static"]["cdg_change"]
    print("\n  READ (static detector):")
    if d < -0.02:
        print(f"    Removing document access LOWERS CDG by {abs(d):.4f}. Camouflage written")
        print("    blind is easier to detect, so the paper's headline gap is an upper bound")
        print("    inflated by an unrealistically strong attacker. Report BOTH.")
    elif d > 0.02:
        print(f"    Removing document access RAISES CDG by {d:.4f}. Document access is not")
        print("    what makes camouflage evade detection; the gap is not an artifact of the")
        print("    over-strong attacker.")
    else:
        print(f"    CDG is essentially unchanged ({d:+.4f}). Document access is not the")
        print("    mechanism behind the blind spot -- register alone suffices.")

    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Written to {OUT}")


if __name__ == "__main__":
    main()
