"""
Compute CDG on the regenerated corpus, with and without refusal screening.

This is the number that decides whether the paper's headline finding survives
both integrity fixes applied together:

  * dry-run mock exclusion  -- pushed CDG UP   (0.840 -> 0.896)
  * refusal screening       -- pushes CDG DOWN (refusals are scored as missed
                               injections, so they inflate the gap)

The screened figure is the honest one. The unscreened figure is reported beside
it so the size of the refusal bug is visible rather than asserted, and because
the unscreened set is the closest available reconstruction of what the submitted
pipeline actually measured.

    python analysis/cdg_v2.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DET = Path("results/detections_v2.jsonl")
OUT = Path("results/cdg_v2.json")

DOMAINS = ["financial", "legal", "general"]

# Published and mock-excluded reference points, for the comparison table.
PUBLISHED = {"idr_static": 0.938, "idr_cam": 0.097, "cdg": 0.840}
MOCK_EXCLUDED = {"idr_static": 1.000, "idr_cam": 0.1037, "cdg": 0.8963}


def idr(rows: list[dict]) -> tuple[int, int, float]:
    if not rows:
        return 0, 0, float("nan")
    n = sum(1 for r in rows if r["verdict"] == "INJECTED")
    return n, len(rows), n / len(rows)


def fmt(t) -> str:
    n, d, v = t
    return f"{n}/{d} = {v:.4f}" if d else "n/a"


def main() -> None:
    if not DET.exists():
        raise SystemExit("results/detections_v2.jsonl not found -- run redetect_v2.py")
    rows = [json.loads(l) for l in DET.open()]

    static = [r for r in rows if r["payload_type"] == "static"]
    cam_all = [r for r in rows if r["payload_type"] == "camouflage"]
    cam_scr = [r for r in cam_all if not r["is_refusal"]]
    cam_ref = [r for r in cam_all if r["is_refusal"]]

    s, c_all, c_scr, c_ref = idr(static), idr(cam_all), idr(cam_scr), idr(cam_ref)
    cdg_all = s[2] - c_all[2]
    cdg_scr = s[2] - c_scr[2]

    print("=" * 74)
    print("CDG ON THE REGENERATED CORPUS  (Llama 3.1 8B, StaticDetector)")
    print("=" * 74)
    print(f"  IDR static                    : {fmt(s)}")
    print(f"  IDR camouflage, ALL           : {fmt(c_all)}")
    print(f"  IDR camouflage, SCREENED      : {fmt(c_scr)}   <-- honest")
    print(f"  IDR on refusals only          : {fmt(c_ref)}   (these are not payloads)")
    print()
    print(f"  CDG unscreened : {cdg_all:.4f}")
    print(f"  CDG screened   : {cdg_scr:.4f}   <-- headline")
    print(f"  refusal-bug inflation of CDG  : {cdg_all - cdg_scr:+.4f}")

    print("\n  Trajectory of the headline number:")
    print(f"    as submitted                        CDG {PUBLISHED['cdg']:.4f}")
    print(f"    + dry-run mock exclusion            CDG {MOCK_EXCLUDED['cdg']:.4f}")
    print(f"    + refusal screening (regen corpus)  CDG {cdg_scr:.4f}")

    # ---- by domain ----
    print("\n  By domain (screened):")
    by_domain = {}
    for d in DOMAINS:
        ds = idr([r for r in static if r["domain"] == d])
        dc = idr([r for r in cam_scr if r["domain"] == d])
        dcg = ds[2] - dc[2] if ds[1] and dc[1] else float("nan")
        by_domain[d] = {"idr_static": ds, "idr_camouflage": dc, "cdg": dcg,
                        "n_refusals_dropped": sum(1 for r in cam_ref if r["domain"] == d)}
        print(f"    {d:<10} static {fmt(ds):>16}  cam {fmt(dc):>14}  CDG {dcg:.4f}"
              f"   (refusals dropped: {by_domain[d]['n_refusals_dropped']})")

    # ---- confidence ----
    missed = [r for r in cam_scr if r["verdict"] == "CLEAN"]
    caught = [r for r in cam_scr if r["verdict"] == "INJECTED"]
    mc, cc = Counter(r["confidence"] for r in missed), Counter(r["confidence"] for r in caught)
    high_share = mc.get("HIGH", 0) / len(missed) if missed else float("nan")
    print(f"\n  Confidence (screened camouflage):")
    print(f"    missed {len(missed)}: {dict(mc)}  -> HIGH share {high_share:.4f}")
    print(f"    caught {len(caught)}: {dict(cc)}")

    # ---- variant ----
    def variant(pid: str) -> str:
        return pid.rsplit("_v", 1)[-1] if "_v" in pid else "?"
    print("  Caught by variant (screened): ", {
        v: (sum(1 for r in cam_scr if variant(r["payload_id"]) == v and r["verdict"] == "INJECTED"),
            sum(1 for r in cam_scr if variant(r["payload_id"]) == v))
        for v in ("1", "2", "3")})

    report = {
        "idr_static": s, "idr_camouflage_all": c_all,
        "idr_camouflage_screened": c_scr, "idr_refusals_only": c_ref,
        "cdg_unscreened": round(cdg_all, 4), "cdg_screened": round(cdg_scr, 4),
        "refusal_inflation": round(cdg_all - cdg_scr, 4),
        "trajectory": {"as_submitted": PUBLISHED["cdg"],
                       "mock_excluded": MOCK_EXCLUDED["cdg"],
                       "mock_excluded_plus_refusal_screened": round(cdg_scr, 4)},
        "by_domain": by_domain,
        "confidence_missed": dict(mc), "confidence_caught": dict(cc),
        "missed_high_share": round(high_share, 4) if missed else None,
        "n_refusals_dropped": len(cam_ref),
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Written to {OUT}")


if __name__ == "__main__":
    main()
