"""
Generate LaTeX tables for the paper.

Loads metrics CSVs from results/ and outputs ready-to-paste LaTeX.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG


def _fmt(val, pct=False):
    """Format a float for display."""
    if val is None or val == "N/A":
        return "--"
    if isinstance(val, str):
        return val
    if pct:
        return f"{val * 100:.1f}\\%"
    return f"{val:.3f}"


def generate_table1(metrics_full: dict) -> str:
    """
    Table 1: Main results — ASR and IDR by payload type and architecture.

    Rows: Static (4 categories) | Camouflage
    Columns: Single ASR | Debate ASR | DAF | Static IDR | CDG
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Main Results: Attack Success Rate and Injection Detection Rate}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Payload Type & Single ASR & Debate ASR & DAF & Static IDR & CDG \\",
        r"\midrule",
    ]

    # Extract from metrics
    asr_by_type = metrics_full.get("asr_by_payload_type", {})
    asr_by_arch = metrics_full.get("asr_by_architecture", {})
    idr_s_on_s = metrics_full.get("idr_static_detector_on_static", 0)
    idr_s_on_c = metrics_full.get("idr_static_detector_on_camouflage", 0)
    cdg = metrics_full.get("cdg", 0)
    daf_s = metrics_full.get("daf_static", 1.0)
    daf_c = metrics_full.get("daf_camouflage", 1.0)

    asr_single_static = asr_by_type.get("static", 0)
    asr_single_cam = asr_by_type.get("camouflage", 0)

    lines += [
        f"Static Payloads & {_fmt(asr_single_static)} & -- & {_fmt(daf_s)} & {_fmt(idr_s_on_s)} & -- \\\\",
        r"\quad Override Directive & -- & -- & -- & -- & -- \\",
        r"\quad Authority Claim & -- & -- & -- & -- & -- \\",
        r"\quad Role Confusion & -- & -- & -- & -- & -- \\",
        r"\quad Social Engineering & -- & -- & -- & -- & -- \\",
        r"\midrule",
        f"Camouflage & {_fmt(asr_single_cam)} & -- & {_fmt(daf_c)} & {_fmt(idr_s_on_c)} & {_fmt(cdg)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_table2(cdg_csv_path: str) -> str:
    """
    Table 2: CDG by domain.

    Rows: Financial | Legal | General QA
    Columns: IDR_static_detector | IDR_camouflage | CDG | CDG_after_fix | Improvement%
    """
    try:
        import pandas as pd
        df = pd.read_csv(cdg_csv_path)
    except Exception:
        return "% Table 2: CDG by domain data not found. Run compute_metrics.py first.\n"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Camouflage Detection Gap (CDG) by Domain}",
        r"\label{tab:cdg_by_domain}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Domain & IDR$_\text{static}$ & IDR$_\text{cam}$ & CDG$_\text{before}$ & IDR$^\text{aug}_\text{cam}$ & CDG$_\text{after}$ & Improvement \\",
        r"\midrule",
    ]

    domain_labels = {
        "financial": "Financial",
        "legal": "Legal",
        "general": "General QA",
    }

    for _, row in df.iterrows():
        domain = row.get("domain", "")
        label = domain_labels.get(domain, domain.title())
        lines.append(
            f"{label} & "
            f"{_fmt(row.get('idr_static_detector_on_static', 0))} & "
            f"{_fmt(row.get('idr_static_detector_on_camouflage', 0))} & "
            f"{_fmt(row.get('cdg_before', 0))} & "
            f"{_fmt(row.get('idr_augmented_on_camouflage', 0))} & "
            f"{_fmt(row.get('cdg_after', 0))} & "
            f"{row.get('cdg_improvement_pct', 0):.1f}\\% \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_table3(metrics_full: dict) -> str:
    """
    Table 3: Debate dynamics.

    Rows: inject_all | inject_first
    Columns: ASR | CPS | DAF_vs_single
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Multi-Agent Debate Dynamics}",
        r"\label{tab:debate_dynamics}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Injection Mode & Debate ASR & CPS & DAF vs. Single \\",
        r"\midrule",
        r"inject\_all (all agents) & -- & -- & -- \\",
        r"inject\_first (one agent) & -- & -- & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"% Note: populate ASR/CPS/DAF from metrics_full.json after running experiments.",
    ]
    return "\n".join(lines)


def main(results_dir: str = None):
    results_dir = results_dir or CONFIG.results_dir
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_full = {}
    metrics_path = out_dir / "metrics_full.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics_full = json.load(f)
    else:
        print("Warning: metrics_full.json not found. Tables will have placeholder values.")

    cdg_csv = str(out_dir / "metrics_cdg_by_domain.csv")

    tables = {
        "table1_main_results.tex": generate_table1(metrics_full),
        "table2_cdg_by_domain.tex": generate_table2(cdg_csv),
        "table3_debate_dynamics.tex": generate_table3(metrics_full),
    }

    for filename, content in tables.items():
        path = out_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        print(f"Wrote: {path}")

    print(f"\nAll LaTeX tables written to {results_dir}/")


if __name__ == "__main__":
    main()
