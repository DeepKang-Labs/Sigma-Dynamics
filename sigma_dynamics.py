#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sigma-Dynamics — Canonical Adaptive Moral Control (REAL DATA default)
DeepKang-Labs (2025) — Axiom-to-Code

Closed loop:
    θ_i(t) = f_i(E_t, M_{t-1})
    M_t    = Σ_k w_k · C_k    (EMA ≈ logistic recency)
    C̄_t   = (1/n) Σ_k C_k

Where C_k = (non_harm, equity, stability, resilience)

This script:
  1) Downloads REAL market data (default: BTC-USD, 200d) via yfinance
  2) Derives network-like proxies: latency / throughput / stability
  3) Maps them to the comprehension vector C_k
  4) Runs the closed-loop dynamics and guardrails
  5) Saves deterministic artifacts under: <out_dir>/...

Artifacts:
  - <out_dir>/raw_<SYMBOL>.csv            (prices/volumes downloaded)
  - <out_dir>/sigma_dynamics.csv          (all time-series of the loop)
  - <out_dir>/coherence.png, theta_*.png, veto.png
  - <out_dir>/skywire_metrics.csv         (optional bridge snapshots)

Usage (examples):
  python sigma_dynamics.py
  python sigma_dynamics.py --symbol ETH-USD --lookback-days 365
  python sigma_dynamics.py --bridge-every 50
"""

from __future__ import annotations
import os, math, csv, argparse
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- external real data (market) ------------------------------------------------
import yfinance as yf
import requests

RNG = np.random.default_rng(42)
plt.switch_backend("Agg")  # headless-safe for CI

# -------------------------- helpers ---------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def clip01(x):
    return float(np.clip(x, 0.0, 1.0))

def _safe_norm(x: pd.Series) -> pd.Series:
    """Normalize to [0,1] safely (constant-safe)."""
    if len(x) == 0:
        return x
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax - xmin == 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - xmin) / (xmax - xmin)

# ------------------- data acquisition (REAL by default) -------------------------

def load_market_data(symbol: str, lookback_days: int) -> pd.DataFrame:
    """Download OHLCV with yfinance (auto_adjusted)."""
    df = yf.download(
        symbol, period=f"{lookback_days}d", interval="1d",
        auto_adjust=True, progress=False
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol}.")
    df = df.reset_index()  # Date, Open, High, Low, Close, Adj Close?, Volume
    # Keep only essential cols
    keep = [c for c in ["Date","Open","High","Low","Close","Volume"] if c in df.columns]
    return df[keep].copy()

def proxies_from_market(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute network-like proxies from price/volume:
      - volatility (rolling std of returns) -> inverse => latency proxy
      - normalized volume -> throughput proxy
      - trend stability (1 - |zscore of rolling mean returns|) -> stability proxy
    All proxies are clipped into [0,1].
    """
    df = df.copy()
    df["ret"] = df["Close"].pct_change().fillna(0.0)
    # rolling windows
    win_vol = 14
    win_mean = 7

    vol = df["ret"].rolling(win_vol).std().fillna(df["ret"].rolling(win_vol, min_periods=1).std())
    vol_norm = _safe_norm(vol)
    latency = 1.0 - vol_norm  # low volatility => high latency-score (better)

    volu_norm = _safe_norm(df["Volume"].astype(float))
    throughput = volu_norm  # more activity => higher throughput

    mean_ret = df["ret"].rolling(win_mean).mean().fillna(0.0)
    mean_z = (mean_ret - mean_ret.mean()) / (mean_ret.std() + 1e-9)
    stability = 1.0 - np.minimum(1.0, np.abs(mean_z))  # flatter trend => more stable

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d"),
        "latency": np.clip(latency, 0.0, 1.0),
        "throughput": np.clip(throughput, 0.0, 1.0),
        "stability": np.clip(stability, 0.0, 1.0),
    })
    return out

# -------------------------- optional public bridge ------------------------------

def fetch_public_metrics() -> Dict[str, float] | None:
    """Try to pull a tiny public snapshot; returns None on any error."""
    try:
        btc = requests.get("https://api.blockchair.com/bitcoin/stats", timeout=10).json()
        glob = requests.get("https://api.coinlore.net/api/global/", timeout=10).json()

        blocks_24h = max(1.0, float(btc["data"].get("blocks_24h", 144)))
        latency_raw = min(2.0, 144.0 / blocks_24h)
        latency = clip01(1.0 / latency_raw)

        coins_count = float(glob[0].get("coins_count", 8000.0))
        throughput = clip01(coins_count / 20000.0)

        stability = clip01(1.0 - (1.0 / (1.0 + math.exp(-2.0 * throughput))))
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "latency": round(latency, 4),
            "throughput": round(throughput, 4),
            "stability": round(stability, 4),
        }
    except Exception:
        return None

# --------------------------- Sigma core -----------------------------------------

def comprehension_from_metrics(m: Dict[str, float], context: str) -> np.ndarray:
    """
    Map raw (latency, throughput, stability) -> C_k \in [0,1]^4
    """
    lat  = float(np.clip(m.get("latency", 0.7), 0.0, 1.0))
    thr  = float(np.clip(m.get("throughput", 0.5), 0.0, 1.0))
    stab = float(np.clip(m.get("stability", 0.6), 0.0, 1.0))

    non_harm = clip01(0.5 * stab + 0.5 * lat)
    equity   = clip01(1.0 - abs(thr - 0.5) * 2.0)  # max quand l’activité est “équilibrée”
    stability = stab
    resilience = clip01(0.6 * stab + 0.4 * lat)

    C = np.array([non_harm, equity, stability, resilience], dtype=float)

    if context == "crisis":
        C[0] = clip01(C[0] - 0.08)
        C[2] = clip01(C[2] - 0.06)
    elif context == "recovery":
        C[3] = clip01(C[3] + 0.08)
    return C

def f_i(E_t: str, M_prev: np.ndarray, lam: float = 0.15) -> np.ndarray:
    """θ(t) = clip( M_prev ⊙ γ(context) + lam*0.1, [0,1] )"""
    gamma = np.ones(4, dtype=float)
    if E_t == "crisis":
        gamma = np.array([1.10, 1.05, 1.15, 1.12])
    elif E_t == "recovery":
        gamma = np.array([1.00, 1.02, 1.00, 1.08])
    theta = np.clip(M_prev * gamma + lam * 0.1, 0.0, 1.0)
    return theta

def veto_guardrail(C: np.ndarray, theta: np.ndarray, eps: float = 1e-3) -> bool:
    return bool(np.any((theta - C) > eps))

# ----------------------------- main loop ----------------------------------------

def run_sigma_dynamics(
    symbol: str = "BTC-USD",
    lookback_days: int = 200,
    out_dir: str = "artifacts",
    bridge_every: int = 0,         # 0 => désactivé ; N => fetch toutes les N itérations
) -> Dict[str, str]:

    out = ensure_dir(out_dir)

    # 1) REAL DATA
    market = load_market_data(symbol, lookback_days)
    raw_path = os.path.join(out, f"raw_{symbol.replace('-','_')}.csv")
    market.to_csv(raw_path, index=False)

    proxies = proxies_from_market(market)
    # base metrics vector (start with last row as initial "last_metrics")
    last_metrics = {
        "latency":   float(proxies["latency"].iloc[0]),
        "throughput":float(proxies["throughput"].iloc[0]),
        "stability": float(proxies["stability"].iloc[0]),
    }

    # 2) optional public-bridge CSV
    bridge_csv = os.path.join(out, "skywire_metrics.csv")
    if bridge_every > 0 and not os.path.exists(bridge_csv):
        with open(bridge_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp","latency","throughput","stability"])
            writer.writeheader()

    # 3) Sigma states
    dims = ["non_harm", "equity", "stability", "resilience"]
    M = np.array([0.70, 0.68, 0.66, 0.59], dtype=float)  # moral memory init
    theta = np.array([0.60, 0.60, 0.60, 0.58], dtype=float)
    alpha = 0.2  # EMA

    records: List[Dict[str, float | str | int]] = []

    steps = len(proxies)  # un pas par échantillon réel
    for t in range(steps):
        # contexte (boucle 90 pas)
        if t % 90 in (0,1,2,3,4,5):       context = "crisis"
        elif t % 90 in range(6,15):       context = "recovery"
        else:                             context = "normal"

        # refresh from REAL proxies
        last_metrics = {
            "latency":   float(proxies["latency"].iloc[t]),
            "throughput":float(proxies["throughput"].iloc[t]),
            "stability": float(proxies["stability"].iloc[t]),
        }

        # optional external bridge (append snapshot)
        if bridge_every > 0 and (t % bridge_every == 0):
            snap = fetch_public_metrics()
            if snap:
                with open(bridge_csv, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["timestamp","latency","throughput","stability"])
                    writer.writerow(snap)

        # comprehension vector
        C = comprehension_from_metrics(last_metrics, context)

        # memory + thresholds
        M = (1 - alpha) * M + alpha * C
        theta = f_i(context, M, lam=0.15)

        veto = veto_guardrail(C, theta, eps=1e-3)

        rec = {"t": t, "context": context,
               **{f"C_{d}": float(C[i]) for i,d in enumerate(dims)},
               **{f"M_{d}": float(M[i]) for i,d in enumerate(dims)},
               **{f"theta_{d}": float(theta[i]) for i,d in enumerate(dims)},
               "coherence": float(np.mean(C >= theta)),
               "veto": int(veto)}
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(out, "sigma_dynamics.csv")
    df.to_csv(csv_path, index=False)

    # 4) plots
    def save_plot(series: pd.Series, title: str, fname: str):
        plt.figure(figsize=(8,3))
        plt.plot(df["t"], series)
        plt.title(title)
        plt.xlabel("t")
        plt.tight_layout()
        path = os.path.join(out, fname)
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    art_paths = []
    art_paths.append(save_plot(df["coherence"], "Sigma-Dynamics: Coherence Over Time", "coherence.png"))
    for d in dims:
        art_paths.append(save_plot(df[f"theta_{d}"], f"Sigma-Dynamics: theta_{d} Over Time", f"theta_{d}.png"))
    art_paths.append(save_plot(df["veto"], "Sigma-Dynamics: Veto Triggered (1=yes)", "veto.png"))

    return {
        "symbol": symbol,
        "raw_csv": raw_path,
        "csv": csv_path,
        "bridge_csv": bridge_csv if bridge_every > 0 else "",
        "plots": ", ".join(art_paths),
        "note": f"Artifacts saved in {out_dir}/ using REAL data",
    }

# ----------------------------------- CLI ----------------------------------------

def main():
    p = argparse.ArgumentParser(description="Run Sigma-Dynamics on REAL market data")
    p.add_argument("--symbol", default="BTC-USD", help="Yahoo Finance ticker (ex: BTC-USD, ETH-USD)")
    p.add_argument("--lookback-days", type=int, default=200, help="History window (days)")
    p.add_argument("--out-dir", default="artifacts", help="Output directory for artifacts")
    p.add_argument("--bridge-every", type=int, default=0, help="Fetch tiny public snapshots every N steps (0=off)")
    args = p.parse_args()

    res = run_sigma_dynamics(
        symbol=args.symbol,
        lookback_days=args.lookback_days,
        out_dir=args.out_dir,
        bridge_every=args.bridge_every,
    )
    print("\n=== Run complete (REAL data) ===")
    for k,v in res.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
