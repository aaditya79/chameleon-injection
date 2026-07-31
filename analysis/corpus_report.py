"""
Characterise the regenerated camouflage corpus (data/camouflage_corpus_v2.json).

Reports the statistics the paper's reproducibility statement needs: per-model
payload counts, ACS distribution, attacker refusal rate, payload length, and the
corpus hash. ACS is descriptive only -- it is not used to select variants.

    python analysis/corpus_report.py
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

CORPUS = Path("data/camouflage_corpus_v2.json")
OUT = Path("results/corpus_report.json")


def quantiles(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": round(st.mean(s), 4),
        "sd": round(st.pstdev(s), 4) if len(s) > 1 else 0.0,
        "min": round(s[0], 4),
        "p25": round(s[len(s) // 4], 4),
        "median": round(st.median(s), 4),
        "p75": round(s[(3 * len(s)) // 4], 4),
        "max": round(s[-1], 4),
    }


def main() -> None:
    if not CORPUS.exists():
        raise SystemExit(f"{CORPUS} not found -- run phase (f) first.")
    corpus = json.loads(CORPUS.read_text())

    report: dict = {"meta": corpus.get("meta", {}), "models": {}}
    print("=" * 70)
    print("REGENERATED CAMOUFLAGE CORPUS")
    print("=" * 70)

    for model_key, store in corpus.get("generated", {}).items():
        payloads = [p for ps in store.values() for p in ps]
        usable = [p for p in payloads if not p.get("is_refusal")]
        refusals = [p for p in payloads if p.get("is_refusal")]

        acs_all = [p["acs"] for p in usable if p.get("acs", 0) > 0]
        words = [p.get("n_words", len(p["payload_text"].split())) for p in usable]

        m = {
            "n_tasks": len(store),
            "n_payloads": len(payloads),
            "n_usable": len(usable),
            "n_refusals_remaining": len(refusals),
            "refusal_rate": round(len(refusals) / len(payloads), 4) if payloads else None,
            "refusals_by_domain": dict(Counter(p["domain"] for p in refusals)),
            "refusals_by_task": sorted({p["payload_id"] for p in refusals}),
            "acs": quantiles(acs_all),
            "acs_by_domain": {
                d: quantiles([p["acs"] for p in usable
                              if p["domain"] == d and p.get("acs", 0) > 0])
                for d in sorted({p["domain"] for p in usable})
            },
            "payload_words": quantiles([float(w) for w in words]),
            "meta": corpus.get("meta", {}).get(model_key, {}),
        }
        report["models"][model_key] = m

        print(f"\n{model_key}:")
        print(f"  tasks {m['n_tasks']} | payloads {m['n_payloads']} "
              f"| usable {m['n_usable']} | refusals {m['n_refusals_remaining']} "
              f"({(m['refusal_rate'] or 0)*100:.1f}%)")
        if m["refusals_by_domain"]:
            print(f"  refusals by domain: {m['refusals_by_domain']}")
        a = m["acs"]
        if a:
            print(f"  ACS  mean {a['mean']} sd {a['sd']} | "
                  f"min {a['min']} p25 {a['p25']} med {a['median']} p75 {a['p75']} max {a['max']}")
        for d, q in m["acs_by_domain"].items():
            if q:
                print(f"    {d:<10} mean {q['mean']}  (n={q['n']})")
        w = m["payload_words"]
        if w:
            print(f"  words mean {w['mean']:.0f} | min {w['min']:.0f} med {w['median']:.0f} max {w['max']:.0f}")
        if m["meta"]:
            print(f"  regenerated {m['meta'].get('regenerated_utc')}  "
                  f"sha256[:16] {m['meta'].get('corpus_sha256')}")

    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
