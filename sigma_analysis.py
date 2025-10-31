#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sigma-Dynamics • Analysis Module
Reads sigma_dynamics.csv, computes core stats/correlations/trends,
detects anomaly windows, renders plots, and writes a Markdown report.

Outputs:
- reports/sigma_analysis_report.md
- artifacts/plots/*.png
- artifacts/metrics_summary.json

CLI:
    python sigma_analysis.py --csv data/sigma_dynamics.csv
    python sigma_analysis.py               # auto-discovery
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Config ----------
DEFAULT_CSV_CANDIDATES = [
    "sigma_dynamics.csv",
    "artifacts/sigma_dynamics.csv",
    "data/sigma_dynamics.csv",
]

PLOT_DIR = Path("artifacts/plots")
REPORT_DIR = Path("reports")
REPORT_MD = REPORT_DIR / "sigma_analysis_report.md"
SUMMARY_JSON = Path("artifacts/metrics_summary.json")

ROLL_WINDOW = 10        # points for rolling means / derivatives
DERIV_EPS = 1e-9        # numerical guard

# Expected columns (flexible: we’ll pick what exists)
SIGNAL_COLS = [
    "theta_non_harm", "theta_equity", "theta_stability", "theta_resilience",
    "coherence", "veto", "context"
]

# ---------- Utilities ----------
def find_csv(path_hint: str | None) -> Path:
    if path_hint:
        p = Path(path_hint)
        if p.exists():
            return p
    for cand in DEFAULT_CSV_CANDIDATES:
        p = Path(cand)
        if p.exists():
            return p
    raise FileNotFoundError(
        "sigma_dynamics.csv introuvable. "
        "Passez --csv <chemin> ou placez le fichier à la racine / artifacts/ / data/."
    )

def ensure_dirs():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)

def slope_trend(y: np.ndarray) -> float:
    """Linear trend slope vs. index t."""
    if len(y) < 2:
        return float("nan")
    x = np.arange(len(y))
    a, b = np.polyfit(x, y, 1)   # slope a, intercept b
    return float(a)

def rolling_derivative(y: pd.Series, window: int = ROLL_WINDOW) -> pd.Series:
    dy = y.diff().fillna(0.0)
    return dy.rolling(window=max(2, window)).mean().fillna(0.0)

def zscore(y: pd.Series) -> pd.Series:
    mu, sd = y.mean(), y.std(ddof=0)
    if sd < DERIV_EPS:
        return pd.Series(np.zeros(len(y)), index=y.index)
    return (y - mu) / sd

def detect_anomaly_windows(df: pd.DataFrame) -> List[Tuple[int,int,str]]:
    """
    Simple anomaly detector:
    - sharp negative derivative on coherence OR
    - veto toggled to 1 OR
    - multi-signal zscore magnitude > 2 at same time
    Returns list of (start_idx, end_idx, reason).
    """
    reasons = []
    if "coherence" in df.columns:
        dC = rolling_derivative(df["coherence"])
        idx = np.where(dC < -0.01)[0]  # threshold adjustable
        for i in idx:
            reasons.append((int(max(0, i-1)), int(min(len(df)-1, i+1)), "coherence_drop"))

    if "veto" in df.columns:
        v = df["veto"].fillna(0)
        idx = np.where(v.values > 0.5)[0]
        for i in idx:
            reasons.append((int(i), int(i), "veto_triggered"))

    sigs = [c for c in ["theta_non_harm","theta_equity","theta_stability","theta_resilience"] if c in df.columns]
    if sigs:
        Z = np.vstack([zscore(df[c]).values for c in sigs]).T
        idx = np.where(np.any(np.abs(Z) > 2.0, axis=1))[0]
        for i in idx:
            reasons.append((int(max(0, i-1)), int(min(len(df)-1, i+1)), "multi_zscore>2"))

    # merge overlapping windows
    reasons.sort(key=lambda x: (x[0], x[1]))
    merged = []
    for w in reasons:
        if not merged or w[0] > merged[-1][1] + 1:
            merged.append(list(w))
        else:
            merged[-1][1] = max(merged[-1][1], w[1])
            merged[-1][2] += "|" + w[2]
    return [(int(a), int(b), r) for a,b,r in merged]

def save_line_plot(x, y, title, ylabel, outpath: Path):
    plt.figure(figsize=(10,3))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def save_corr_heatmap(df: pd.DataFrame, outpath: Path):
    cols = [c for c in ["theta_non_harm","theta_equity","theta_stability","theta_resilience","coherence"] if c in df.columns]
    if len(cols) < 2:
        return
    C = df[cols].corr()
    plt.figure(figsize=(5,4))
    plt.imshow(C, interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ---------- Main analysis ----------
def analyze(csv_path: Path) -> Dict:
    df = pd.read_csv(csv_path)

    # keep known columns if present
    cols = [c for c in SIGNAL_COLS if c in df.columns]
    df = df[cols].copy()

    # infer time index if absent
    if "t" in df.columns:
        t = df["t"].values
    else:
        t = np.arange(len(df))

    # compute rolling derivatives for core signals
    derivs = {}
    for c in ["theta_non_harm","theta_equity","theta_stability","theta_resilience","coherence"]:
        if c in df.columns:
            derivs[c] = rolling_derivative(df[c])

    # basic stats & trends
    stats = {}
    for c in ["theta_non_harm","theta_equity","theta_stability","theta_resilience","coherence"]:
        if c in df.columns:
            y = df[c]
            stats[c] = {
                "mean": float(y.mean()),
                "std": float(y.std(ddof=0)),
                "min": float(y.min()),
                "max": float(y.max()),
                "trend_slope": slope_trend(y.values),
            }

    # anomaly windows
    windows = detect_anomaly_windows(df)

    # plots
    ensure_dirs()
    for c in ["theta_non_harm","theta_equity","theta_stability","theta_resilience","coherence","veto"]:
        if c in df.columns:
            save_line_plot(t, df[c].values, f"Sigma-Dynamics: {c} Over Time", c, PLOT_DIR / f"{c}.png")

    save_corr_heatmap(df, PLOT_DIR / "correlations.png")

    # coherence early-warning index (EWI): negative mean derivative + variance spike
    ewi = None
    if "coherence" in df.columns:
        dC = derivs["coherence"]
        var = df["coherence"].rolling(ROLL_WINDOW).var().fillna(0.0)
        ewi = float((-dC.clip(lower=0).mean()) + var.mean())

    # write summary json
    summary = {
        "source_csv": str(csv_path),
        "n_points": int(len(df)),
        "stats": stats,
        "anomaly_windows": [{"start":a, "end":b, "reason":r} for a,b,r in windows],
        "early_warning_index": ewi,
        "plots_dir": str(PLOT_DIR),
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # markdown report
    md = []
    md.append("# Sigma-Dynamics • Analysis Report\n")
    md.append(f"- **Source CSV**: `{csv_path}`  \n- **Points**: **{len(df)}**\n")
    if ewi is not None:
        md.append(f"- **Early-Warning Index (coherence)**: **{ewi:.4f}**  \n")
    md.append("\n## Summary Statistics\n")
    for c, s in stats.items():
        md.append(
            f"**{c}** — mean: `{s['mean']:.4f}`, std: `{s['std']:.4f}`, "
            f"min/max: `{s['min']:.4f}` / `{s['max']:.4f}`, trend slope: `{s['trend_slope']:.5f}`  \n"
        )

    md.append("\n## Anomaly Windows\n")
    if windows:
        md.append("| start | end | reason |\n|---:|---:|---|\n")
        for a,b,r in windows:
            md.append(f"| {a} | {b} | {r} |\n")
    else:
        md.append("_No anomalies detected with current thresholds._\n")

    md.append("\n## Plots\n")
    for name in ["theta_non_harm","theta_equity","theta_stability","theta_resilience","coherence","veto","correlations"]:
        p = PLOT_DIR / f"{name}.png"
        if p.exists():
            md.append(f"- {name}: `artifacts/plots/{name}.png`\n")

    md.append("\n---\n_Generated by `sigma_analysis.py`._\n")

    REPORT_MD.write_text("".join(md), encoding="utf-8")
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to sigma_dynamics.csv")
    args = parser.parse_args()

    ensure_dirs()
    csv_path = find_csv(args.csv)
    summary = analyze(csv_path)

    print("\n[OK] Analysis complete.")
    print(f"- Report: {REPORT_MD}")
    print(f"- Plots:  {PLOT_DIR}")
    print(f"- JSON:   {SUMMARY_JSON}\n")

if __name__ == "__main__":
    main()
