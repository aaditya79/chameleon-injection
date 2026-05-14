"""
Load trials.jsonl and compute all metrics, outputting CSV and JSON files.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import CONFIG
from src.utils.logger import TrialLogger
from src.evaluation.metrics import MetricsComputer


def main(results_dir: str = None):
    results_dir = results_dir or CONFIG.results_dir
    logger = TrialLogger(results_dir)
    trials = logger.load_trials()

    if not trials:
        print(f"No trials found in {results_dir}/trials.jsonl")
        return

    print(f"Loaded {len(trials)} trials")
    df = pd.DataFrame(trials)

    mc = MetricsComputer()

    # ---- Flatten nested fields ----
    def _extract_followed(row):
        j = row.get("asr_judgment")
        if j:
            return j.get("followed_injection", None)
        return None

    def _extract_verdict(row, detector_type=None):
        d = row.get("detection_result")
        if d:
            if detector_type and d.get("detector_type") != detector_type:
                return None
            return d.get("verdict")
        return None

    df["followed_injection"] = df.apply(lambda r: _extract_followed(r), axis=1)
    df["detection_verdict_static"] = df.apply(lambda r: _extract_verdict(r, "static"), axis=1)
    df["detection_verdict_augmented"] = df.apply(lambda r: _extract_verdict(r, "augmented"), axis=1)

    # ---- Metrics by payload type ----
    injected = df[df["payload_type"].isin(["static", "camouflage"])].copy()
    metrics_by_payload = (
        injected.groupby("payload_type")
        .apply(lambda g: pd.Series({
            "n_trials": len(g),
            "asr": g["followed_injection"].mean() if g["followed_injection"].notna().any() else 0,
            "idr_static_detector": (g["detection_verdict_static"] == "INJECTED").mean() if g["detection_verdict_static"].notna().any() else 0,
            "idr_augmented_detector": (g["detection_verdict_augmented"] == "INJECTED").mean() if g["detection_verdict_augmented"].notna().any() else 0,
        }))
        .reset_index()
    )

    # ---- Metrics by domain ----
    metrics_by_domain = (
        injected.groupby(["domain", "payload_type"])
        .apply(lambda g: pd.Series({
            "n_trials": len(g),
            "asr": g["followed_injection"].mean() if g["followed_injection"].notna().any() else 0,
            "idr_static_detector": (g["detection_verdict_static"] == "INJECTED").mean() if g["detection_verdict_static"].notna().any() else 0,
            "idr_augmented_detector": (g["detection_verdict_augmented"] == "INJECTED").mean() if g["detection_verdict_augmented"].notna().any() else 0,
        }))
        .reset_index()
    )

    # ---- Metrics by architecture ----
    metrics_by_arch = (
        injected.groupby(["architecture", "payload_type"])
        .apply(lambda g: pd.Series({
            "n_trials": len(g),
            "asr": g["followed_injection"].mean() if g["followed_injection"].notna().any() else 0,
        }))
        .reset_index()
    )

    # ---- CDG per domain ----
    cdg_rows = []
    for domain in ["financial", "legal", "general"]:
        s_trials = injected[(injected["domain"] == domain) & (injected["payload_type"] == "static")]
        c_trials = injected[(injected["domain"] == domain) & (injected["payload_type"] == "camouflage")]

        idr_s = (s_trials["detection_verdict_static"] == "INJECTED").mean() if len(s_trials) > 0 else 0
        idr_c = (c_trials["detection_verdict_static"] == "INJECTED").mean() if len(c_trials) > 0 else 0
        idr_s_aug = (s_trials["detection_verdict_augmented"] == "INJECTED").mean() if len(s_trials) > 0 else 0
        idr_c_aug = (c_trials["detection_verdict_augmented"] == "INJECTED").mean() if len(c_trials) > 0 else 0

        cdg_before = mc.cdg(idr_s, idr_c)
        cdg_after = mc.cdg(idr_s_aug, idr_c_aug)
        improvement = mc.cdg_improvement(idr_s, idr_c, idr_s_aug, idr_c_aug)

        cdg_rows.append({
            "domain": domain,
            "idr_static_detector_on_static": round(idr_s, 4),
            "idr_static_detector_on_camouflage": round(idr_c, 4),
            "cdg_before": round(cdg_before, 4),
            "idr_augmented_on_camouflage": round(idr_c_aug, 4),
            "cdg_after": round(cdg_after, 4),
            "cdg_improvement_pct": improvement["improvement_pct"],
        })

    cdg_df = pd.DataFrame(cdg_rows)

    # ---- Save CSVs ----
    out = Path(results_dir)
    metrics_by_payload.to_csv(out / "metrics_by_payload_type.csv", index=False)
    metrics_by_domain.to_csv(out / "metrics_by_domain.csv", index=False)
    metrics_by_arch.to_csv(out / "metrics_by_architecture.csv", index=False)
    cdg_df.to_csv(out / "metrics_cdg_by_domain.csv", index=False)

    # ---- Full JSON ----
    full_summary = mc.compute_from_trials(trials)
    with open(out / "metrics_full.json", "w", encoding="utf-8") as f:
        import dataclasses
        json.dump(dataclasses.asdict(full_summary), f, indent=2, default=str)

    print(f"\nOutputs written to {results_dir}/:")
    print("  metrics_by_payload_type.csv")
    print("  metrics_by_domain.csv")
    print("  metrics_by_architecture.csv")
    print("  metrics_cdg_by_domain.csv")
    print("  metrics_full.json")

    print("\n--- Quick Summary ---")
    print(metrics_by_payload.to_string(index=False))
    print("\nCDG by domain:")
    print(cdg_df.to_string(index=False))


if __name__ == "__main__":
    main()
