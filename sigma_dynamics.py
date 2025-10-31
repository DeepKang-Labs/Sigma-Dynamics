#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sigma-Dynamics (v0.1) — DeepKang-Labs
-------------------------------------
Experimental simulation engine for the Sigma-Lab Framework.

Implements the canonical ethical control loop:
    θ_i(t) = f_i(E_t, M_{t-1})
    M_t    = Σ_k w_k * C_k
    C̄_t   = (1/n) Σ_k C_k

Core Principles:
- Adaptive moral thresholds (θ_i)
- Weighted moral memory (M_t)
- Contextual feedback from environment (E_t)
- Veto Guardrail: prevents coherence degradation
- Contextual modes: normal / crisis / recovery

Usage:
    pip install numpy matplotlib
    python sigma_dynamics.py

Artifacts (CSV + PNG plots) will be created under:
    ./sigma_dynamics_artifacts_YYYYMMDD-HHMMSS/
"""

from __future__ import annotations
import math
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Configuration & Defaults
# -------------------------
DEFAULT_STEPS = 200
DEFAULT_LAMBDA = 0.08  # moral memory persistence
RNG_SEED = 42

AXIOMS = ["non_harm", "equity", "stability", "resilience"]
CONTEXT_MODES = ["normal", "crisis", "recovery"]

@dataclass
class SimConfig:
    steps: int = DEFAULT_STEPS
    lam: float = DEFAULT_LAMBDA
    alpha: Dict[str, float] = None
    beta: Dict[str, float] = None
    gamma: Dict[str, float] = None
    context_weights: Dict[str, Dict[str, float]] = None
    veto_tolerance: float = 1e-6
    context_switch_prob: float = 0.1
    noise_scale: float = 0.03

    def __post_init__(self):
        if self.alpha is None:
            self.alpha = {k: 0.50 for k in AXIOMS}
        if self.beta is None:
            self.beta = {k: 0.40 for k in AXIOMS}
        if self.gamma is None:
            self.gamma = {k: 0.15 for k in AXIOMS}
        if self.context_weights is None:
            self.context_weights = {
                "normal":   {"non_harm": 0.30, "equity": 0.30, "stability": 0.20, "resilience": 0.20},
                "crisis":   {"non_harm": 0.15, "equity": 0.15, "stability": 0.35, "resilience": 0.35},
                "recovery": {"non_harm": 0.25, "equity": 0.30, "stability": 0.25, "resilience": 0.20},
            }

@dataclass
class State:
    t: int
    context: str
    C_k: Dict[str, float]
    M_t: Dict[str, float]
    theta: Dict[str, float]
    coherence: float
    veto_triggered: bool

def bounded(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))

def normalize_weights(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values())
    if s <= 0:
        n = len(d)
        return {k: 1.0/n for k in d}
    return {k: v/s for k, v in d.items()}

def measure_context(mode: str, noise_scale: float = 0.02) -> Dict[str, float]:
    base = {
        "normal":   {"non_harm": 0.80, "equity": 0.75, "stability": 0.70, "resilience": 0.70},
        "crisis":   {"non_harm": 0.55, "equity": 0.55, "stability": 0.40, "resilience": 0.45},
        "recovery": {"non_harm": 0.70, "equity": 0.72, "stability": 0.60, "resilience": 0.75},
    }[mode]
    out = {}
    for k, v in base.items():
        noise = np.random.normal(0.0, noise_scale)
        out[k] = bounded(v + noise, 0.0, 1.0)
    return out

def compute_weights(history_len: int, lam: float) -> np.ndarray:
    if history_len == 0:
        return np.array([])
    t = history_len - 1
    w = []
    for k in range(history_len):
        dt = t - k
        w_k = 1.0 / (1.0 + math.exp(-lam * dt))
        w.append(w_k)
    w = np.array(w, dtype=float)
    w /= w.sum()
    return w

def aggregate_memory(C_history: List[Dict[str, float]], lam: float) -> Dict[str, float]:
    n = len(C_history)
    if n == 0:
        return {k: 0.0 for k in AXIOMS}
    w = compute_weights(n, lam)
    M = {k: 0.0 for k in AXIOMS}
    for idx, Ck in enumerate(C_history):
        for key in AXIOMS:
            M[key] += w[idx] * Ck[key]
    return M

def f_i(E_t: Dict[str, float], M_prev: Dict[str, float],
        alpha: Dict[str, float], beta: Dict[str, float], gamma: Dict[str, float]) -> Dict[str, float]:
    theta = {}
    for key in AXIOMS:
        e = E_t[key]; m = M_prev[key]
        val = alpha[key]*e + beta[key]*m + gamma[key]*(e*m)
        theta[key] = bounded(val, 0.0, 1.0)
    return theta

def coherence_metric(theta: Dict[str, float]) -> float:
    vals = np.array([theta[k] for k in AXIOMS], dtype=float)
    mean = float(vals.mean())
    var  = float(vals.var())
    balance = bounded(1.0 - var, 0.0, 1.0)
    cost = 0.9
    phi = (mean * balance) / cost
    return float(phi)

def simulate(config: SimConfig, seed: int = RNG_SEED) -> Dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)

    context = "normal"
    C_history: List[Dict[str, float]] = []
    M_prev = {k: 0.5 for k in AXIOMS}
    theta_prev = {k: 0.5 for k in AXIOMS}
    coherence_prev = coherence_metric(theta_prev)

    log: List[State] = []

    for t in range(config.steps):
        if random.random() < config.context_switch_prob:
            context = random.choice(CONTEXT_MODES)

        C_k = measure_context(context, noise_scale=config.noise_scale)
        cw = normalize_weights(config.context_weights[context])
        C_k_weighted = {k: bounded(C_k[k] * (0.8 + 0.4*cw[k])) for k in AXIOMS}

        C_history.append(C_k_weighted)
        M_t = aggregate_memory(C_history, config.lam)
        theta_t = f_i(C_k_weighted, M_prev, config.alpha, config.beta, config.gamma)

        coherence_now = coherence_metric(theta_t)
        d_coh = coherence_now - coherence_prev
        veto = (d_coh < -config.veto_tolerance)

        if veto:
            theta_eff = theta_prev.copy()
            coherence_eff = coherence_prev
        else:
            theta_eff = theta_t
            coherence_eff = coherence_now

        log.append(State(
            t=t, context=context, C_k=C_k_weighted, M_t=M_t,
            theta=theta_eff, coherence=coherence_eff, veto_triggered=bool(veto)
        ))

        M_prev = M_t
        theta_prev = theta_eff
        coherence_prev = coherence_eff

    return {"config": asdict(config), "log": log}

def export_csv(sim_result: Dict[str, object], out_csv: Path) -> None:
    import csv
    log: List[State] = sim_result["log"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["t","context",
                  *[f"C_{k}" for k in AXIOMS],
                  *[f"M_{k}" for k in AXIOMS],
                  *[f"theta_{k}" for k in AXIOMS],
                  "coherence","veto"]
        writer.writerow(header)
        for s in log:
            row = [s.t, s.context,
                   *[s.C_k[k] for k in AXIOMS],
                   *[s.M_t[k] for k in AXIOMS],
                   *[s.theta[k] for k in AXIOMS],
                   s.coherence, int(s.veto_triggered)]
            writer.writerow(row)

def plot_series(sim_result: Dict[str, object], out_png_dir: Path) -> None:
    log: List[State] = sim_result["log"]
    t = [s.t for s in log]

    plt.figure(figsize=(10,5))
    plt.plot(t, [s.coherence for s in log])
    plt.xlabel("t"); plt.ylabel("coherence"); plt.title("Sigma-Dynamics: Coherence Over Time")
    plt.tight_layout(); plt.savefig(out_png_dir / "coherence_over_time.png"); plt.close()

    for key in AXIOMS:
        plt.figure(figsize=(10,5))
        plt.plot(t, [s.theta[key] for s in log])
        plt.xlabel("t"); plt.ylabel(f"theta_{key}"); plt.title(f"Sigma-Dynamics: theta_{key} Over Time")
        plt.tight_layout(); plt.savefig(out_png_dir / f"theta_{key}_over_time.png"); plt.close()

    plt.figure(figsize=(10,3))
    veto_series = [1 if s.veto_triggered else 0 for s in log]
    plt.step(t, veto_series, where="post")
    plt.xlabel("t"); plt.ylabel("veto"); plt.title("Sigma-Dynamics: Veto Triggered (1=yes)")
    plt.tight_layout(); plt.savefig(out_png_dir / "veto_over_time.png"); plt.close()

def main():
    cfg = SimConfig()
    result = simulate(cfg, seed=RNG_SEED)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(f"./sigma_dynamics_artifacts_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps(result["config"], indent=2), encoding="utf-8")
    export_csv(result, out_dir / "log.csv")
    plot_series(result, out_dir)
    print("Artifacts written to:", str(out_dir.resolve()))

if __name__ == "__main__":
    main()
