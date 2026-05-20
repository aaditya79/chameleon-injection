"""
Statistical significance tests for chameleon injection detection.
"""

import json
import sys
from collections import defaultdict, Counter

import numpy as np
from scipy.stats import chi2_contingency

TRIALS_PATH = "results/trials.jsonl"


def load_trials():
    with open(TRIALS_PATH) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# McNemar's test helper
# ---------------------------------------------------------------------------

def mcnemar_test(b, c, continuity=True):
    """
    McNemar's test on a 2x2 paired contingency table.

    Table layout (row = condition A, col = condition B):
        a = both positive
        b = A positive, B negative
        c = A negative, B positive
        d = both negative

    We only need b and c (the discordant pairs).
    Returns (chi2, p_value).
    """
    if b + c == 0:
        return float("nan"), float("nan")
    if continuity:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2 = (b - c) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p = chi2_dist.sf(chi2, df=1)
    return chi2, p


def task_level_detection(trials, payload_type):
    """
    For each task_id, return True if the majority of trials of the given
    payload_type were detected (verdict == INJECTED).
    Returns dict[task_id -> bool].
    """
    by_task = defaultdict(list)
    for t in trials:
        if t["payload_type"] == payload_type and t.get("detection_result"):
            by_task[t["task_id"]].append(
                t["detection_result"]["verdict"] == "INJECTED"
            )
    return {tid: sum(v) > len(v) / 2 for tid, v in by_task.items()}


def run_mcnemar(model_label, trials):
    """
    Run McNemar's test comparing detection of static vs camouflage payloads
    at the task level (majority vote per task_id).
    """
    print(f"\n{'='*60}")
    print(f"McNemar's Test — {model_label}")
    print(f"{'='*60}")
    print("H0: The static detector performs equally on static and camouflage payloads")

    static_det = task_level_detection(trials, "static")
    cam_det = task_level_detection(trials, "camouflage")

    common_tasks = sorted(set(static_det) & set(cam_det))
    if not common_tasks:
        print("  No overlapping task_ids — cannot run McNemar's test.")
        return

    # Build 2x2 table
    a = b = c = d = 0
    for tid in common_tasks:
        s = static_det[tid]
        cm = cam_det[tid]
        if s and cm:
            a += 1        # both detected
        elif s and not cm:
            b += 1        # static detected, camouflage missed
        elif not s and cm:
            c += 1        # static missed, camouflage detected
        else:
            d += 1        # both missed

    print(f"\n  Paired task count : {len(common_tasks)}")
    print(f"  2x2 contingency table (rows = static, cols = camouflage):")
    print(f"                     Cam INJECTED  Cam CLEAN")
    print(f"  Static INJECTED        {a:>4}          {b:>4}   (b = discordant)")
    print(f"  Static CLEAN           {c:>4}          {d:>4}   (c = discordant)")
    print(f"\n  Discordant pairs: b={b}, c={c}")

    chi2, p = mcnemar_test(b, c, continuity=True)

    print(f"\n  McNemar chi-squared (with continuity correction): {chi2:.4f}")
    print(f"  p-value: {p:.6f}")

    if np.isnan(p):
        print("  Interpretation: No discordant pairs — test undefined.")
    elif p < 0.001:
        print("  Interpretation: *** p < 0.001 — reject H0. The detector performs")
        print("    significantly differently on static vs camouflage payloads.")
    elif p < 0.01:
        print("  Interpretation: ** p < 0.01 — reject H0.")
    elif p < 0.05:
        print("  Interpretation: * p < 0.05 — reject H0.")
    else:
        print("  Interpretation: p >= 0.05 — fail to reject H0.")

    # Raw detection rates for context
    n_static = sum(1 for t in trials if t["payload_type"] == "static" and t.get("detection_result"))
    n_cam = sum(1 for t in trials if t["payload_type"] == "camouflage" and t.get("detection_result"))
    inj_s = sum(1 for t in trials if t["payload_type"] == "static" and t.get("detection_result", {}).get("verdict") == "INJECTED")
    inj_c = sum(1 for t in trials if t["payload_type"] == "camouflage" and t.get("detection_result", {}).get("verdict") == "INJECTED")
    print(f"\n  Detection rates (trial-level):")
    print(f"    Static    : {inj_s}/{n_static} = {inj_s/n_static*100:.1f}%")
    print(f"    Camouflage: {inj_c}/{n_cam} = {inj_c/n_cam*100:.1f}%")


# ---------------------------------------------------------------------------
# Section 3: Qualitative examples
# ---------------------------------------------------------------------------

def qualitative_examples(all_trials):
    print(f"\n{'='*60}")
    print("Qualitative Examples")
    print(f"{'='*60}")
    print("Goal: same task_id where static was detected (INJECTED)")
    print("      AND camouflage was missed (CLEAN)\n")

    # Index by task_id
    by_task = defaultdict(list)
    for t in all_trials:
        if t.get("detection_result"):
            by_task[t["task_id"]].append(t)

    found = False
    for tid, task_trials in sorted(by_task.items()):
        static_injected = [
            t for t in task_trials
            if t["payload_type"] == "static"
            and t["detection_result"]["verdict"] == "INJECTED"
        ]
        cam_clean = [
            t for t in task_trials
            if t["payload_type"] == "camouflage"
            and t["detection_result"]["verdict"] == "CLEAN"
        ]
        if static_injected and cam_clean:
            si = static_injected[0]
            cc = cam_clean[0]
            print(f"  Task ID   : {tid}")
            print(f"  Domain    : {si['domain']}")
            print()
            print(f"  [Static — DETECTED]")
            print(f"    payload_category  : {si.get('payload_category', 'N/A')}")
            print(f"    payload_id        : {si.get('payload_id', 'N/A')}")
            print(f"    detector_verdict  : {si['detection_result']['verdict']}")
            print(f"    detector_confidence: {si['detection_result'].get('confidence', 'N/A')}")
            print(f"    model             : {si['model']}")
            print()
            print(f"  [Camouflage — MISSED]")
            print(f"    payload_category  : {cc.get('payload_category', 'N/A')}")
            print(f"    payload_id        : {cc.get('payload_id', 'N/A')}")
            print(f"    detector_verdict  : {cc['detection_result']['verdict']}")
            print(f"    detector_confidence: {cc['detection_result'].get('confidence', 'N/A')}")
            print(f"    model             : {cc['model']}")
            found = True
            break

    if not found:
        print("  No task_id found with both a static INJECTED and camouflage CLEAN trial.")


# ---------------------------------------------------------------------------
# Section 4: Error analysis — camouflage trials caught by detector
# ---------------------------------------------------------------------------

def error_analysis(all_trials):
    print(f"\n{'='*60}")
    print("Error Analysis: Camouflage Trials Caught by Detector")
    print("(The ~9.7% of camouflage trials that were flagged INJECTED)")
    print(f"{'='*60}")

    # Filter to Llama exp1 single_agent camouflage trials only (same slice as McNemar)
    cam_trials = [
        t for t in all_trials
        if t["model"] == "llama3.1"
        and t["experiment"] == "exp1"
        and t["architecture"] == "single_agent"
        and t["payload_type"] == "camouflage"
        and t.get("detection_result") is not None
        and t["detection_result"].get("detector_type") == "static"
    ]

    caught = [t for t in cam_trials if t["detection_result"]["verdict"] == "INJECTED"]
    missed = [t for t in cam_trials if t["detection_result"]["verdict"] == "CLEAN"]

    total = len(cam_trials)
    print(f"\n  Total camouflage trials: {total}")
    print(f"  Caught (INJECTED)      : {len(caught)} ({len(caught)/total*100:.1f}%)")
    print(f"  Missed (CLEAN)         : {len(missed)} ({len(missed)/total*100:.1f}%)")

    # --- By domain ---
    print(f"\n  Breakdown by domain:")
    print(f"  {'Domain':<15} {'Caught':>7} {'Total':>7} {'Catch%':>8}")
    print(f"  {'-'*40}")
    domain_totals = Counter(t["domain"] for t in cam_trials)
    domain_caught = Counter(t["domain"] for t in caught)
    for domain in sorted(domain_totals):
        tot = domain_totals[domain]
        cat = domain_caught.get(domain, 0)
        print(f"  {domain:<15} {cat:>7} {tot:>7} {cat/tot*100:>7.1f}%")

    # --- By payload variant (v1/v2/v3) ---
    def get_variant(payload_id):
        if payload_id and "_v" in payload_id:
            return payload_id.rsplit("_v", 1)[-1]
        return "unknown"

    print(f"\n  Breakdown by payload variant:")
    print(f"  {'Variant':<10} {'Caught':>7} {'Total':>7} {'Catch%':>8}")
    print(f"  {'-'*38}")
    variant_totals = Counter(get_variant(t.get("payload_id")) for t in cam_trials)
    variant_caught = Counter(get_variant(t.get("payload_id")) for t in caught)
    for v in sorted(variant_totals):
        tot = variant_totals[v]
        cat = variant_caught.get(v, 0)
        print(f"  v{v:<9} {cat:>7} {tot:>7} {cat/tot*100:>7.1f}%")

    # --- By detector confidence ---
    print(f"\n  Breakdown by detector confidence (caught trials only):")
    conf_counts = Counter(t["detection_result"].get("confidence", "N/A") for t in caught)
    for conf, cnt in sorted(conf_counts.items(), key=lambda x: -x[1]):
        print(f"    {conf}: {cnt} ({cnt/len(caught)*100:.1f}%)")

    # --- Confidence distribution across ALL camouflage trials for comparison ---
    print(f"\n  Confidence distribution — ALL camouflage trials:")
    conf_all = Counter(t["detection_result"].get("confidence", "N/A") for t in cam_trials)
    for conf, cnt in sorted(conf_all.items(), key=lambda x: -x[1]):
        print(f"    {conf}: {cnt} ({cnt/total*100:.1f}%)")

    # --- Cross-tab: domain x variant for caught trials ---
    print(f"\n  Caught trials — domain × variant cross-tab:")
    cross = Counter((t["domain"], get_variant(t.get("payload_id"))) for t in caught)
    print(f"  {'Domain':<15} {'v1':>5} {'v2':>5} {'v3':>5}")
    print(f"  {'-'*30}")
    domains = sorted(set(d for d, _ in cross))
    for domain in domains:
        row = [cross.get((domain, str(v)), 0) for v in [1, 2, 3]]
        print(f"  {domain:<15} {row[0]:>5} {row[1]:>5} {row[2]:>5}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    trials = load_trials()
    print(f"Loaded {len(trials)} total trials.")

    # --- Llama 3.1: exp1, single_agent, static detector ---
    llama_trials = [
        t for t in trials
        if t["model"] == "llama3.1"
        and t["experiment"] == "exp1"
        and t["architecture"] == "single_agent"
        and t.get("detection_result") is not None
        and t["detection_result"].get("detector_type") == "static"
    ]
    run_mcnemar("Llama 3.1 — exp1, single_agent, static detector", llama_trials)

    # --- Gemini 2.0 Flash: exp1, single_agent, static detector ---
    gemini_trials = [
        t for t in trials
        if t["model"] == "google/gemini-2.0-flash-001"
        and t["experiment"] == "exp1"
        and t["architecture"] == "single_agent"
        and t.get("detection_result") is not None
        and t["detection_result"].get("detector_type") == "static"
    ]
    run_mcnemar("Gemini 2.0 Flash — exp1, single_agent, static detector", gemini_trials)

    # --- Qualitative examples (across all trials) ---
    qualitative_examples(trials)

    # --- Error analysis (Llama exp1 camouflage) ---
    error_analysis(trials)


if __name__ == "__main__":
    main()
