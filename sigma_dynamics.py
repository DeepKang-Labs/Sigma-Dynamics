#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sigma-Dynamics — Canonical Adaptive Moral Control (with integrated Bridge)
DeepKang-Labs (2025) — Axiom-to-Code

Implements the closed loop:
    θ_i(t) = f_i(E_t, M_{t-1})
    M_t    = Σ_k w_k · C_k
    C̄_t   = (1/n) Σ_k C_k

Where C_k = (non_harm, equity, stability, resilience)

This script fetches real-world public network-like metrics (bridge) and
falls back to safe simulation if unavailable. It saves:
  - outputs/sigma_dynamics.csv
  - outputs/skywire_metrics.csv (from bridge)
  - outputs/*.png (plots)
"""

from __future__ import annotations
import os, math, csv, argparse, time, random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Bridge (public metrics) ------------------------------------------------
# Robust to failures; keeps the run offline-capable.
import json
import requests

def fetch_public_metrics() -> Dict[str, float] | None:
    """
    Pulls public indicators to mimic decentralized-network ‘health’.
    Fails gracefully if internet disabled/unreachable.
    Returns dict with keys: latency, throughput, stability
    """
    try:
        # Two lightweight public endpoints (no key):
        btc = requests.get("https://api.blockchair.com/bitcoin/stats", timeout=10).json()
        glob = requests.get("https://api.coinlore.net/api/global/", timeout=10).json()

        # Heuristics to derive proxy signals (bounded [0,1] after scaling)
        # blocks_24h ~144 when normal => latency proxy = 144/blocks_24h (capped)
        blocks_24h = max(1.0, float(btc["data"].get("blocks_24h", 144)))
        latency_raw = min(2.0, 144.0 / blocks_24h)           # 1.0 ≈ nominal, >1 worse
        latency = float(np.clip(1.0 / latency_raw, 0.0, 1.0)) # higher is better (lower real latency)

        coins_count = float(glob[0].get("coins_count", 8000.0))
        throughput = float(np.clip(coins_count / 20000.0, 0.0, 1.0))  # crude density proxy
        stability = float(np.clip(1.0 - (1.0 / (1.0 + math.exp(-2.0 * throughput))), 0.0, 1.0))

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "latency": round(latency, 4),
            "throughput": round(throughput, 4),
            "stability": round(stability, 4),
        }
    except Exception:
        return None

# -------- Sigma-Dynamics core ----------------------------------------------------

RNG = np.random.default_rng(42)

def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))

def make_outputs_dir() -> str:
    out = "outputs"
    os.makedirs(out, exist_ok=True)
    return out

def comprehension_from_metrics(m: Dict[str, float], context: str) -> np.ndarray:
    """
    Map raw bridge metrics → C_k components in [0,1].
    non_harm    ↑ with stability, and with low latency
    equity      proxy from ‘throughput evenness’ (bounded transform)
    stability   directly from bridge
    resilience  from stability & latency smoothness (simple transform)
    """
    lat  = float(np.clip(m.get("latency", 0.7), 0.0, 1.0))
    thr  = float(np.clip(m.get("throughput", 0.5), 0.0, 1.0))
    stab = float(np.clip(m.get("stability", 0.6), 0.0, 1.0))

    non_harm = float(np.clip(0.5 * stab + 0.5 * lat, 0.0, 1.0))
    equity   = float(np.clip(1.0 - abs(thr - 0.5) * 2.0, 0.0, 1.0))   # best when ~0.5 (balanced)
    stability= stab
    resilience = float(np.clip(0.6 * stab + 0.4 * lat, 0.0, 1.0))

    C = np.array([non_harm, equity, stability, resilience], dtype=float)

    # Contextual nudges (crisis lowers non_harm/stability a bit; recovery boosts resilience)
    if context == "crisis":
        C[0] = float(np.clip(C[0] - 0.08, 0.0, 1.0))
        C[2] = float(np.clip(C[2] - 0.06, 0.0, 1.0))
    elif context == "recovery":
        C[3] = float(np.clip(C[3] + 0.08, 0.0, 1.0))

    return C

def f_i(E_t: str, M_prev: np.ndarray, lam: float = 0.15) -> np.ndarray:
    """
    Threshold update function.
    θ(t) = clip( M_prev ⊙ γ(context) + λ*1, [0,1] ) where γ reweights dims by context.
    """
    gamma = np.ones(4, dtype=float)
    if E_t == "crisis":
        gamma = np.array([1.10, 1.05, 1.15, 1.12])  # tighten non_harm/stability/resilience
    elif E_t == "recovery":
        gamma = np.array([1.00, 1.02, 1.00, 1.08])  # focus resilience slightly

    theta = np.clip(M_prev * gamma + lam * 0.1, 0.0, 1.0)
    return theta

def veto_guardrail(C: np.ndarray, theta: np.ndarray, eps: float = 1e-6) -> bool:
    """Veto triggers if any component falls below its threshold by a margin."""
    gaps = theta - C
    return bool(np.any(gaps > eps))

def run_sigma_dynamics(steps: int = 200, use_bridge: bool = True, fetch_every: int = 50) -> Dict[str, str]:
    outdir = make_outputs_dir()

    # State
    dims = ["non_harm", "equity", "stability", "resilience"]
    M = np.array([0.70, 0.68, 0.66, 0.59], dtype=float)  # moral memory init
    theta = np.array([0.60, 0.60, 0.60, 0.58], dtype=float)

    records: List[Dict[str, float | str | int]] = []
    last_metrics = {"latency": 0.7, "throughput": 0.5, "stability": 0.6}
    bridge_csv = os.path.join(outdir, "skywire_metrics.csv")

    # Prepare bridge CSV (appendable)
    if use_bridge and not os.path.exists(bridge_csv):
        with open(bridge_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "latency", "throughput", "stability"])
            writer.writeheader()

    for t in range(steps):
        # Context schedule
        if t % 90 in (0, 1, 2, 3, 4, 5):   context = "crisis"
        elif t % 90 in range(6, 15):       context = "recovery"
        else:                               context = "normal"

        # Fetch bridge metrics periodically
        if use_bridge and (t % fetch_every == 0):
            data = fetch_public_metrics()
            if data:
                last_metrics = {k: data[k] for k in ("latency","throughput","stability")}
                # append to bridge csv
                with open(bridge_csv, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["timestamp","latency","throughput","stability"])
                    writer.writerow(data)

        # Build comprehension vector from latest metrics (or fallback)
        C = comprehension_from_metrics(last_metrics, context)

        # Weighted moral memory update: logistic recency weights
        # recent samples matter more; here we approximate with EMA
        alpha = 0.2
        M = (1 - alpha) * M + alpha * C

        # Thresholds from f_i
        theta = f_i(context, M, lam=0.15)

        # Veto
        veto = veto_guardrail(C, theta, eps=1e-3)

        rec = {
            "t": t,
            "context": context,
            **{f"C_{d}": float(C[i]) for i, d in enumerate(dims)},
            **{f"M_{d}": float(M[i]) for i, d in enumerate(dims)},
            **{f"theta_{d}": float(theta[i]) for i, d in enumerate(dims)},
            "coherence": float(np.mean(C >= theta)),
            "veto": int(veto),
        }
        records.append(rec)

        # Small stochasticity so plots aren’t perfectly flat
        last_metrics["throughput"] = float(np.clip(last_metrics["throughput"] + RNG.normal(0, 0.01), 0.0, 1.0))
        last_metrics["latency"]    = float(np.clip(last_metrics["latency"]    + RNG.normal(0, 0.008), 0.0, 1.0))

    # Save CSV
    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(outdir, "sigma_dynamics.csv")
    df.to_csv(csv_path, index=False)

    # Plots
    def save_plot(series: pd.Series, title: str, fname: str):
        plt.figure(figsize=(8,3))
        plt.plot(df["t"], series)
        plt.title(title)
        plt.xlabel("t")
        plt.tight_layout()
        path = os.path.join(outdir, fname)
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    art_paths = []
    art_paths.append(save_plot(df["coherence"], "Sigma-Dynamics: Coherence Over Time", "coherence.png"))
    for d in dims:
        art_paths.append(save_plot(df[f"theta_{d}"], f"Sigma-Dynamics: theta_{d} Over Time", f"theta_{d}.png"))
    art_paths.append(save_plot(df["veto"], "Sigma-Dynamics: Veto Triggered (1=yes)", "veto.png"))

    return {
        "csv": csv_path,
        "bridge_csv": bridge_csv if use_bridge else "",
        "plots": ", ".join(art_paths),
        "note": "Artifacts saved in outputs/"
    }

# -------- CLI -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Run Sigma-Dynamics with integrated Bridge")
    p.add_argument("--steps", type=int, default=200, help="number of timesteps")
    p.add_argument("--no-bridge", action="store_true", help="disable public metrics bridge")
    p.add_argument("--fetch-every", type=int, default=50, help="bridge fetch period in steps")
    args = p.parse_args()

    result = run_sigma_dynamics(
        steps=args.steps,
        use_bridge=not args.no_bridge,
        fetch_every=args.fetch_every,
    )
    print("\n=== Run complete ===")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
