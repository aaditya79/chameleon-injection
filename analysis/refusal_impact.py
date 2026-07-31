"""
Quantify the effect of the refusal-as-payload bug on the camouflage condition.

THE BUG
-------
The attacker LLM refuses a large share of camouflage generation requests
("I can't create content that promotes..."). A refusal is not a camouflage
payload: injecting one into a document measures nothing. The original pipeline
had no refusal check and took the first draw unconditionally, so refusals were
injected and scored as though they were payloads.

WHY IT MATTERS, AND IN WHICH DIRECTION
--------------------------------------
A refusal in the camouflage slot biases two quantities the paper reports, and
both biases run the same way:

  ASR_camouflage  -- a refusal cannot induce the malicious behaviour, so every
                     refusal is a guaranteed attack failure. ASR is biased DOWN.
  IDR_camouflage  -- a refusal contains no injection, so a correct detector
                     should call it CLEAN. Scored against a ground truth of
                     "injected", that CLEAN is recorded as a MISS. IDR is biased
                     DOWN, and therefore CDG = IDR_static - IDR_camouflage is
                     biased UP.

So the refusal bug INFLATES the paper's headline gap, in the opposite direction
from the dry-run mock contamination, which deflated it. This script measures the
ASR half directly from the phase-(g) corpus. The IDR half cannot be settled
without re-running detection on the screened corpus -- that run is not yet
authorised, and until it happens the net direction of CDG is genuinely open.

    python analysis/refusal_impact.py
"""

from __future__ import annotations

import json
from pathlib import Path

CORPUS = Path("data/camouflage_corpus_v2.json")
NEW = Path("results/trials_v2.jsonl")
OLD = Path("results/trials.jsonl")
OUT = Path("results/refusal_impact.json")

LLAMA_OLD, LLAMA_NEW = "llama3.1", "llama3.1:latest"


def asr(rows: list[dict]) -> tuple[int, int, float]:
    judged = [r for r in rows if r.get("asr_judgment")]
    if not judged:
        return 0, 0, float("nan")
    n = sum(1 for r in judged if r["asr_judgment"].get("followed_injection"))
    return n, len(judged), n / len(judged)


def fmt(t) -> str:
    n, d, v = t
    return f"{n}/{d} = {v:.4f}" if d else "n/a"


def main() -> None:
    corpus = json.loads(CORPUS.read_text())
    store = corpus["generated"]["llama"]
    meta = corpus["meta"]["llama"]
    refusal_ids = {p["payload_id"] for ps in store.values() for p in ps if p.get("is_refusal")}

    new = [json.loads(l) for l in NEW.open()]
    old = [json.loads(l) for l in OLD.open()]

    cam_new = [r for r in new if r["model"] == LLAMA_NEW
               and r["architecture"] == "single_agent" and r["payload_type"] == "camouflage"]
    usable = [r for r in cam_new if r["payload_id"] not in refusal_ids]
    refused = [r for r in cam_new if r["payload_id"] in refusal_ids]

    cam_old = [r for r in old if r["model"] == LLAMA_OLD and r["experiment"] == "exp1"
               and r["architecture"] == "single_agent" and r["payload_type"] == "camouflage"
               and not r.get("dry_run", False)]

    first_draw = meta["refusals_on_first_draw"]
    n_payloads = meta["n_payloads"]

    print("=" * 72)
    print("REFUSAL-AS-PAYLOAD IMPACT  (Llama 3.1 8B, camouflage condition)")
    print("=" * 72)
    print(f"  refusals on first draw     : {first_draw}/{n_payloads} "
          f"= {first_draw/n_payloads:.1%}   <-- the rate the ORIGINAL pipeline would have hit")
    print(f"  refusals after up to 4 retries: {len(refusal_ids)}/{n_payloads} "
          f"= {len(refusal_ids)/n_payloads:.1%}")
    print()
    print("  Single-agent ASR on the regenerated corpus (phase g):")
    print(f"    all payloads            : {fmt(asr(cam_new))}")
    print(f"    refusal-screened only   : {fmt(asr(usable))}   <-- the honest number")
    print(f"    refusals only           : {fmt(asr(refused))}   (expected ~0)")
    print()
    print("  For comparison, the originally logged (unscreened) camouflage ASR:")
    print(f"    exp1 single-agent, mock-excluded : {fmt(asr(cam_old))}")

    a_screened, a_all, a_old = asr(usable), asr(cam_new), asr(cam_old)
    report = {
        "refusals_first_draw": first_draw,
        "n_payloads": n_payloads,
        "refusal_rate_first_draw": round(first_draw / n_payloads, 4),
        "refusal_rate_after_retries": round(len(refusal_ids) / n_payloads, 4),
        "asr_v2_all": a_all,
        "asr_v2_screened": a_screened,
        "asr_v2_refusals_only": asr(refused),
        "asr_original_unscreened": a_old,
        "note": "IDR/CDG impact requires a detection re-run on the screened corpus; "
                "not yet performed. Refusals bias IDR_camouflage down and CDG up.",
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))

    print()
    print("  READ:")
    if a_screened[1] and a_old[1]:
        delta = a_screened[2] - a_old[2]
        print(f"    Screening refusals moves single-agent camouflage ASR by {delta:+.4f} "
              f"({a_old[2]:.4f} -> {a_screened[2]:.4f}).")
        if delta > 0.02:
            print("    The originally reported camouflage ASR was DEFLATED by refusals,")
            print("    consistent with the predicted direction of bias.")
    print("    The IDR/CDG half is NOT settled by this script. Because refusals are")
    print("    scored as missed injections, they inflate CDG; a detection re-run on")
    print("    the screened corpus is required before the corrected CDG is known.")
    print(f"\n  Written to {OUT}")


if __name__ == "__main__":
    main()
