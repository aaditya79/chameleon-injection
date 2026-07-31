"""
CDG improvement from the augmented detector, on the screened v2 corpus.

Exp 4 asks whether adding one camouflaged exemplar per domain to the few-shot
pool closes the gap. Every other Table 1 figure now rests on the corrected,
refusal-screened corpus, so the augmentation delta has to be recomputed there
too or the table mixes corpora.

    CDG_delta = 100 * (CDG_static_detector - CDG_augmented_detector) / CDG_static_detector

Reported on the screened set (the honest figure) and the full set (comparable to
what the submitted pipeline measured).

    python analysis/cdg_delta_v2.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DET = Path("results/detections_v2.jsonl")
OUT = Path("results/cdg_delta_v2.json")
DOMAINS = ["financial", "legal", "general"]

# Submitted Exp 4 figures, on the original unscreened corpus with mock records.
SUBMITTED = {"idr_static": 0.938, "idr_cam": 0.111,
             "aug_idr_static": 0.936, "aug_idr_cam": 0.194, "delta_pct": 10.2}
MOCK_EXCLUDED = {"idr_static": 1.000, "idr_cam": 0.1185,
                 "aug_idr_static": 0.9989, "aug_idr_cam": 0.2074, "delta_pct": 10.2}


def idr(rows: list[dict]) -> tuple[int, int, float]:
    if not rows:
        return 0, 0, float("nan")
    n = sum(1 for r in rows if r["verdict"] == "INJECTED")
    return n, len(rows), n / len(rows)


def fmt(t) -> str:
    n, d, v = t
    return f"{n}/{d} = {v:.4f}" if d else "n/a"


def block(rows: list[dict], detector: str, screened: bool) -> dict:
    sel = [r for r in rows if r["detector_type"] == detector]
    static = [r for r in sel if r["payload_type"] == "static"]
    cam = [r for r in sel if r["payload_type"] == "camouflage"]
    if screened:
        cam = [r for r in cam if not r["is_refusal"]]
    s, c = idr(static), idr(cam)
    return {"idr_static": s, "idr_camouflage": c, "cdg": s[2] - c[2],
            "by_domain": {
                d: {"idr_static": idr([r for r in static if r["domain"] == d]),
                    "idr_camouflage": idr([r for r in cam if r["domain"] == d])}
                for d in DOMAINS}}


def main() -> None:
    if not DET.exists():
        raise SystemExit("results/detections_v2.jsonl not found")
    rows = [json.loads(l) for l in DET.open()]

    n_aug = sum(1 for r in rows if r["detector_type"] == "augmented")
    if n_aug < 1035:
        raise SystemExit(f"augmented pass incomplete: {n_aug}/1035 -- wait for the run")

    report = {}
    print("=" * 76)
    print("EXP 4 -- AUGMENTATION DELTA ON THE SCREENED v2 CORPUS (Llama 3.1 8B)")
    print("=" * 76)

    for screened in (True, False):
        tag = "SCREENED (121 camouflage)" if screened else "UNSCREENED (135 camouflage)"
        st = block(rows, "static", screened)
        au = block(rows, "augmented", screened)
        delta = 100.0 * (st["cdg"] - au["cdg"]) / st["cdg"] if st["cdg"] else float("nan")

        print(f"\n  {tag}")
        print(f"    static detector    : IDR_static {fmt(st['idr_static'])} | "
              f"IDR_cam {fmt(st['idr_camouflage'])} | CDG {st['cdg']:.4f}")
        print(f"    augmented detector : IDR_static {fmt(au['idr_static'])} | "
              f"IDR_cam {fmt(au['idr_camouflage'])} | CDG {au['cdg']:.4f}")
        print(f"    CDG delta          : {delta:+.1f}%")
        print(f"    camouflage IDR moves {st['idr_camouflage'][2]:.4f} -> "
              f"{au['idr_camouflage'][2]:.4f} "
              f"({au['idr_camouflage'][2]-st['idr_camouflage'][2]:+.4f}); "
              f"relative gain {100*(au['idr_camouflage'][2]/st['idr_camouflage'][2]-1):+.1f}%"
              if st["idr_camouflage"][2] else "")
        report["screened" if screened else "unscreened"] = {
            "static_detector": st, "augmented_detector": au, "cdg_delta_pct": round(delta, 2)}

        if screened:
            print("    by domain (screened):")
            for d in DOMAINS:
                a, b = st["by_domain"][d], au["by_domain"][d]
                cs = a["idr_static"][2] - a["idr_camouflage"][2]
                ca = b["idr_static"][2] - b["idr_camouflage"][2]
                dd = 100.0 * (cs - ca) / cs if cs else float("nan")
                print(f"      {d:<10} CDG {cs:.4f} -> {ca:.4f}   delta {dd:+.1f}%   "
                      f"(cam {fmt(a['idr_camouflage'])} -> {fmt(b['idr_camouflage'])})")

    # Static-detection integrity: augmentation must not degrade it.
    aug_static = [r for r in rows if r["detector_type"] == "augmented" and r["payload_type"] == "static"]
    st_static = [r for r in rows if r["detector_type"] == "static" and r["payload_type"] == "static"]
    print(f"\n  Static-payload detection preserved? "
          f"{idr(st_static)[2]:.4f} -> {idr(aug_static)[2]:.4f}")

    conf = Counter(r["confidence"] for r in rows
                   if r["detector_type"] == "augmented" and r["payload_type"] == "camouflage"
                   and not r["is_refusal"] and r["verdict"] == "INJECTED")
    print(f"  Augmented caught-camouflage confidence (screened): {dict(conf)}")

    print("\n  Comparison of the Exp 4 delta across corpora:")
    print(f"    as submitted (unscreened + mock) : {SUBMITTED['delta_pct']:+.1f}%")
    print(f"    mock-excluded, original corpus   : {MOCK_EXCLUDED['delta_pct']:+.1f}%")
    print(f"    screened v2 corpus               : {report['screened']['cdg_delta_pct']:+.1f}%")

    report["comparison"] = {"submitted": SUBMITTED["delta_pct"],
                            "mock_excluded": MOCK_EXCLUDED["delta_pct"],
                            "screened_v2": report["screened"]["cdg_delta_pct"]}
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Written to {OUT}")


if __name__ == "__main__":
    main()
