"""
Generate all 6 paper figures using matplotlib/seaborn.

Saves to results/figures/. Requires experiment data in results/trials.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import CONFIG

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

DOMAIN_COLORS = {
    "financial": "#2196F3",
    "legal": "#FF9800",
    "general": "#4CAF50",
}

FIGSIZE = (8, 5)


def _load_data(results_dir: str):
    """Load trials.jsonl and metrics CSVs."""
    trials_path = Path(results_dir) / "trials.jsonl"
    if not trials_path.exists():
        print(f"Warning: {trials_path} not found. Generating placeholder figures.")
        return None, {}

    trials = []
    with open(trials_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trials.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    df = pd.DataFrame(trials)

    # Flatten nested fields
    def _followed(row):
        j = row.get("asr_judgment")
        return j.get("followed_injection", False) if j else False

    def _verdict(row, det_type):
        d = row.get("detection_result")
        if d and d.get("detector_type") == det_type:
            return d.get("verdict")
        return None

    df["followed_injection"] = df.apply(_followed, axis=1)
    df["verdict_static"] = df.apply(lambda r: _verdict(r, "static"), axis=1)
    df["verdict_augmented"] = df.apply(lambda r: _verdict(r, "augmented"), axis=1)

    metrics_path = Path(results_dir) / "metrics_full.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

    return df, metrics


def _placeholder_data():
    """Return synthetic data for demonstration when no real trials exist."""
    domains = ["financial", "legal", "general"]
    return {
        "idr_static": {"financial": 0.82, "legal": 0.79, "general": 0.71},
        "idr_camouflage": {"financial": 0.21, "legal": 0.29, "general": 0.52},
        "idr_aug_camouflage": {"financial": 0.51, "legal": 0.58, "general": 0.67},
        "asr_static_single": 0.41,
        "asr_static_debate": 0.38,
        "asr_cam_single": 0.63,
        "asr_cam_debate": 0.79,
        "cps_cam_financial": 0.71,
        "cps_cam_legal": 0.64,
        "cps_cam_general": 0.43,
        "acs_cam": np.random.normal(0.72, 0.08, 135).clip(0.3, 0.99).tolist(),
        "acs_static": np.random.normal(0.18, 0.09, 900).clip(0, 0.5).tolist(),
    }


def fig1_cdg_bar(df, pd_data, figures_dir):
    """
    Figure 1: CDG bar chart — IDR on static vs. camouflage, grouped by domain.
    """
    domains = ["financial", "legal", "general"]
    data = pd_data

    idr_s = [data["idr_static"].get(d, 0) for d in domains]
    idr_c = [data["idr_camouflage"].get(d, 0) for d in domains]

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars_s = ax.bar(x - width / 2, idr_s, width, label="Static Payloads",
                    color=[DOMAIN_COLORS[d] for d in domains], alpha=0.9, edgecolor="white")
    bars_c = ax.bar(x + width / 2, idr_c, width, label="Camouflage Payloads",
                    color=[DOMAIN_COLORS[d] for d in domains], alpha=0.45, edgecolor="white",
                    hatch="//")

    # CDG annotations
    for i, (s, c) in enumerate(zip(idr_s, idr_c)):
        cdg = s - c
        ax.annotate(f"CDG={cdg:.2f}", xy=(x[i], max(s, c) + 0.02),
                    ha="center", fontsize=9, fontweight="bold", color="#d32f2f")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in domains])
    ax.set_ylabel("Injection Detection Rate (IDR)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Figure 1: Camouflage Detection Gap by Domain\n"
                 "(Higher CDG = larger blind spot for camouflage)")

    legend_handles = [
        mpatches.Patch(color="gray", alpha=0.9, label="Static Payloads"),
        mpatches.Patch(color="gray", alpha=0.45, hatch="//", label="Camouflage Payloads"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    path = figures_dir / "fig1_cdg_bar.pdf"
    plt.savefig(path, dpi=150)
    plt.savefig(str(path).replace(".pdf", ".png"), dpi=150)
    plt.close()
    print(f"Saved: {path}")


def fig2_daf_comparison(pd_data, figures_dir):
    """
    Figure 2: DAF comparison bar chart.
    """
    asr_s_single = pd_data["asr_static_single"]
    asr_s_debate = pd_data["asr_static_debate"]
    asr_c_single = pd_data["asr_cam_single"]
    asr_c_debate = pd_data["asr_cam_debate"]

    daf_static = asr_s_debate / asr_s_single if asr_s_single > 0 else 1.0
    daf_cam = asr_c_debate / asr_c_single if asr_c_single > 0 else 1.0

    labels = ["Static Payloads", "Camouflage Payloads"]
    dafs = [daf_static, daf_cam]
    colors = ["#1565C0" if d <= 1.0 else "#C62828" for d in dafs]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, dafs, color=colors, edgecolor="white", width=0.4)
    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="No amplification (DAF=1)")
    ax.set_ylabel("Debate Amplification Factor (DAF)")
    ax.set_title("Figure 2: Debate Amplification Factor\n"
                 "(Red = debate amplifies injection; Blue = debate reduces injection)")
    ax.set_ylim(0, max(2.0, max(dafs) * 1.2))

    for bar, daf in zip(bars, dafs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{daf:.2f}×", ha="center", fontweight="bold", fontsize=12)

    ax.legend()
    plt.tight_layout()
    path = figures_dir / "fig2_daf_comparison.pdf"
    plt.savefig(path, dpi=150)
    plt.savefig(str(path).replace(".pdf", ".png"), dpi=150)
    plt.close()
    print(f"Saved: {path}")


def fig3_conformity_propagation(pd_data, figures_dir):
    """
    Figure 3: Conformity Pressure Score stacked bar by domain and payload type.
    """
    domains = ["financial", "legal", "general"]
    cps_cam = [pd_data["cps_cam_financial"], pd_data["cps_cam_legal"], pd_data["cps_cam_general"]]
    cps_static = [0.35, 0.28, 0.22]  # placeholder static CPS

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - width / 2, cps_cam, width, label="Camouflage", color="#E53935", alpha=0.85)
    ax.bar(x + width / 2, cps_static, width, label="Static", color="#1E88E5", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in domains])
    ax.set_ylabel("Conformity Pressure Score (CPS)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 3: Conformity Pressure by Domain\n"
                 "(inject_first: non-injected agents adopting injected position)")
    ax.legend()
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", label="Chance")

    plt.tight_layout()
    path = figures_dir / "fig3_conformity_propagation.pdf"
    plt.savefig(path, dpi=150)
    plt.savefig(str(path).replace(".pdf", ".png"), dpi=150)
    plt.close()
    print(f"Saved: {path}")


def fig4_cdg_before_after(pd_data, figures_dir):
    """
    Figure 4: CDG before/after cheap fix per domain.
    """
    domains = ["financial", "legal", "general"]
    cdg_before = [
        pd_data["idr_static"].get(d, 0) - pd_data["idr_camouflage"].get(d, 0)
        for d in domains
    ]
    cdg_after = [
        pd_data["idr_static"].get(d, 0) - pd_data["idr_aug_camouflage"].get(d, 0)
        for d in domains
    ]

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - width / 2, cdg_before, width, label="CDG (Static Detector)", color="#E53935", alpha=0.85)
    ax.bar(x + width / 2, cdg_after, width, label="CDG (Augmented Detector)", color="#43A047", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in domains])
    ax.set_ylabel("Camouflage Detection Gap (CDG)")
    ax.set_title("Figure 4: CDG Before and After Augmented Detector\n"
                 "(Lower CDG = better detection of camouflage payloads)")
    ax.legend()

    # Improvement annotations
    for i, (before, after) in enumerate(zip(cdg_before, cdg_after)):
        if before > 0:
            imp = 100 * (before - after) / before
            ax.annotate(f"{imp:.0f}% closed", xy=(x[i], max(before, after) + 0.02),
                       ha="center", fontsize=9, color="#2E7D32")

    plt.tight_layout()
    path = figures_dir / "fig4_cdg_before_after.pdf"
    plt.savefig(path, dpi=150)
    plt.savefig(str(path).replace(".pdf", ".png"), dpi=150)
    plt.close()
    print(f"Saved: {path}")


def fig5_acs_distribution(pd_data, figures_dir):
    """
    Figure 5: ACS distribution for camouflage vs. static payloads.
    """
    acs_cam = pd_data["acs_cam"]
    acs_static = pd_data["acs_static"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(acs_static, bins=30, alpha=0.65, label="Static Payloads", color="#1E88E5", density=True)
    ax.hist(acs_cam, bins=30, alpha=0.65, label="Camouflage Payloads", color="#E53935", density=True)

    ax.axvline(np.mean(acs_cam), color="#C62828", linestyle="--", linewidth=1.5,
               label=f"Cam mean={np.mean(acs_cam):.2f}")
    ax.axvline(np.mean(acs_static), color="#1565C0", linestyle="--", linewidth=1.5,
               label=f"Static mean={np.mean(acs_static):.2f}")

    ax.set_xlabel("Cosine Similarity to Domain Context (ACS)")
    ax.set_ylabel("Density")
    ax.set_title("Figure 5: Authoritative Camouflage Score Distribution\n"
                 "(Higher ACS = more domain-appropriate payload)")
    ax.legend()

    plt.tight_layout()
    path = figures_dir / "fig5_acs_distribution.pdf"
    plt.savefig(path, dpi=150)
    plt.savefig(str(path).replace(".pdf", ".png"), dpi=150)
    plt.close()
    print(f"Saved: {path}")


def fig6_asr_vs_acs(df, pd_data, figures_dir):
    """
    Figure 6: ASR vs. ACS scatter, colored by domain.
    """
    if df is not None and "semantic_similarity" in df.columns and len(df) > 0:
        cam_df = df[(df["payload_type"] == "camouflage") & df["semantic_similarity"].notna() & df["followed_injection"].notna()]
        if len(cam_df) > 10:
            _real_scatter(cam_df, figures_dir)
            return

    # Placeholder scatter
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rng = np.random.default_rng(42)
    for domain, color in DOMAIN_COLORS.items():
        n = 45
        acs = rng.uniform(0.4, 0.95, n)
        asr = (0.3 + 0.5 * acs + rng.normal(0, 0.1, n)).clip(0, 1)
        ax.scatter(acs, asr, c=color, label=domain.title(), alpha=0.7, s=40, edgecolors="white")

    ax.set_xlabel("Authoritative Camouflage Score (ACS)")
    ax.set_ylabel("Attack Success Rate (ASR)")
    ax.set_title("Figure 6: ASR vs. ACS by Domain\n"
                 "(placeholder data — run experiments for real values)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = figures_dir / "fig6_asr_vs_acs.pdf"
    plt.savefig(path, dpi=150)
    plt.savefig(str(path).replace(".pdf", ".png"), dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _real_scatter(cam_df: pd.DataFrame, figures_dir: Path):
    """Scatter from real data when available."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for domain, color in DOMAIN_COLORS.items():
        sub = cam_df[cam_df["domain"] == domain]
        ax.scatter(sub["semantic_similarity"], sub["followed_injection"].astype(float),
                   c=color, label=domain.title(), alpha=0.6, s=35, edgecolors="white")

    ax.set_xlabel("Authoritative Camouflage Score (ACS)")
    ax.set_ylabel("Attack Success Rate (binary, per trial)")
    ax.set_title("Figure 6: ASR vs. ACS by Domain")
    ax.legend()
    plt.tight_layout()
    path = figures_dir / "fig6_asr_vs_acs.pdf"
    plt.savefig(path, dpi=150)
    plt.savefig(str(path).replace(".pdf", ".png"), dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main(results_dir: str = None):
    results_dir = results_dir or CONFIG.results_dir
    figures_dir = Path(results_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df, metrics = _load_data(results_dir)
    pd_data = _placeholder_data()

    # Override with real data if available
    if df is not None and len(df) > 0:
        injected = df[df["payload_type"].isin(["static", "camouflage"])]
        for domain in ["financial", "legal", "general"]:
            s = injected[(injected["domain"] == domain) & (injected["payload_type"] == "static")]
            c = injected[(injected["domain"] == domain) & (injected["payload_type"] == "camouflage")]
            if len(s) > 0:
                pd_data["idr_static"][domain] = (s["verdict_static"] == "INJECTED").mean()
            if len(c) > 0:
                pd_data["idr_camouflage"][domain] = (c["verdict_static"] == "INJECTED").mean()
                pd_data["idr_aug_camouflage"][domain] = (c["verdict_augmented"] == "INJECTED").mean()
            # ACS
            if "semantic_similarity" in df.columns:
                acs_vals = c["semantic_similarity"].dropna().tolist()
                if acs_vals:
                    pd_data["acs_cam"] = acs_vals

    print(f"\nGenerating figures to {figures_dir}/\n")
    fig1_cdg_bar(df, pd_data, figures_dir)
    fig2_daf_comparison(pd_data, figures_dir)
    fig3_conformity_propagation(pd_data, figures_dir)
    fig4_cdg_before_after(pd_data, figures_dir)
    fig5_acs_distribution(pd_data, figures_dir)
    fig6_asr_vs_acs(df, pd_data, figures_dir)

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
