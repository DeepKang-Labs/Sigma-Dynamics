#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sigma-Dynamics — Stream Eval (Skywire/PrivateNess ready)
- Lit des métriques réelles (HTTP/JSON Prometheus-like, WebSocket ou tail de logs)
- Calcule C_k, M_t, θ_i(t), veto, coherence
- Sauvegarde périodiquement artefacts (CSV + PNG)
Usage:
  python sigma_stream_eval.py --config configs/stream_skywire.yaml
"""

import os, time, json, math, argparse, threading, queue, re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import requests

# ---------- utils ----------
def clamp01(x): return max(0.0, min(1.0, float(x)))
def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
def time_weight(t_now, t_k, lam): return sigmoid(lam * (t_now - t_k))

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def qselect(d, keys):
    vals = []
    for k in keys:
        if k in d and d[k] is not None:
            vals.append(float(d[k]))
    return np.nanmean(vals) if vals else np.nan

def normalize_quantile(series, qlo=0.01, qhi=0.99):
    s = pd.Series(series, dtype="float64")
    lo, hi = s.quantile(qlo), s.quantile(qhi)
    rng = max(1e-9, hi - lo)
    return ((s - lo) / rng).clip(0,1), (float(lo), float(hi))

# ---------- θ function ----------
def f_theta(context, M_prev, base, ctx_boost):
    boost = float(ctx_boost.get(context, 0.0))
    theta = {}
    for k, m in M_prev.items():
        theta[k] = clamp01(0.5*base + 0.5*float(m) + boost)
    return theta

# ---------- adapter (HTTP polling minimal) ----------
def fetch_metrics_http_json(url: str) -> dict:
    """
    Attend un JSON plat ou un dict de {metric_name: value}.
    Exemple côté exporter:
      {
        "packet_loss": 0.012,
        "err_rate": 0.003,
        "bandwidth_gini": 0.21,
        "latency_jitter_ms": 18.4,
        "throughput_variance": 0.07,
        "recovery_score": 0.78
      }
    """
    r = requests.get(url, timeout=3)
    r.raise_for_status()
    data = r.json()
    # Si Prometheus text/plain, prévoir un parseur ici.
    return data

def evaluate_context(sample: dict, crisis_rules, degraded_rules) -> str:
    def match(rules):
        for rule in rules or []:
            key, op, val = rule["key"], rule["op"], float(rule["value"])
            x = float(sample.get(key, float("nan")))
            if math.isnan(x): 
                return False
            if op == ">": 
                if not (x > val): return False
            elif op == "<":
                if not (x < val): return False
            elif op == ">=":
                if not (x >= val): return False
            elif op == "<=":
                if not (x <= val): return False
            else:
                return False
        return True

    if match(crisis_rules):   return "crisis"
    if match(degraded_rules): return "degraded"
    return "normal"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    adapter = cfg["adapter"]
    mapping = cfg["mapping"]
    ctx_cfg = cfg.get("context", {})
    theta_cfg = cfg["theta"]
    mem_cfg = cfg["memory"]
    out_cfg = cfg["output"]

    outdir = ensure_dir(os.path.join(out_cfg["outdir"],
                                     f'{out_cfg.get("run_tag","run")}_{datetime.utcnow().strftime("%Y%m%d-%H%M%S")}'))
    print(f"[Sigma-Stream] writing to: {outdir}")

    # buffers
    history = []        # raw samples
    computed = []       # rows with C, M, theta, veto, coherence
    M_prev = {k: 0.6 for k in ["C_non_harm","C_equity","C_stability","C_resilience"]}

    poll = float(adapter.get("poll_seconds", 5))
    window = float(adapter.get("window_seconds", 60))
    t = 0

    try:
        while True:
            # 1) fetch one sample
            if adapter["mode"] == "http":
                sample = fetch_metrics_http_json(adapter["endpoint"])
            else:
                # extensions ws/tail possibles ici
                sample = fetch_metrics_http_json(adapter["endpoint"])

            # 2) context from rules
            context = evaluate_context(sample, ctx_cfg.get("crisis_rules"), ctx_cfg.get("degraded_rules"))

            # 3) compute proxies -> C_k (on conservera une normalisation glissante)
            # accumulate raw values for rolling normalization
            history.append({
                "harm":  qselect(sample, mapping["harm_keys"]),
                "ineq":  qselect(sample, mapping["ineq_keys"]),
                "vol":   qselect(sample, mapping["vol_keys"]),
                "rec":   qselect(sample, mapping["rec_keys"]),
                "context": context
            })

            # rolling window indices
            win = history[-int(max(1, window/poll)):]
            dfw = pd.DataFrame(win)

            nh, _  = normalize_quantile(dfw["harm"])
            inq, _ = normalize_quantile(dfw["ineq"])
            vol, _ = normalize_quantile(dfw["vol"])
            rcv, _ = normalize_quantile(dfw["rec"])

            # current (last window point)
            C_non_harm = clamp01(1.0 - float(nh.iloc[-1]))
            C_equity   = clamp01(1.0 - float(inq.iloc[-1]))
            C_stability= clamp01(1.0 - float(vol.iloc[-1]))
            C_resilience = clamp01(float(rcv.iloc[-1]))

            C_now = {
                "C_non_harm": C_non_harm,
                "C_equity": C_equity,
                "C_stability": C_stability,
                "C_resilience": C_resilience
            }

            # 4) update moral memory M_t (time weights on full history of C)
            #    (pour rester léger: pondère simplement les points de la fenêtre)
            lam = float(mem_cfg["lambda"])
            weights = np.array([time_weight(len(win)-1, i, lam) for i in range(len(win))])
            weights = weights / (weights.sum() + 1e-12)

            # recompute C for all points in window (with same transforms)
            C_mat = np.column_stack([
                (1.0 - nh.values), (1.0 - inq.values), (1.0 - vol.values), rcv.values
            ])
            M_vec = (weights.reshape(-1,1) * C_mat).sum(axis=0)
            M_now = dict(zip(C_now.keys(), map(clamp01, M_vec)))

            # 5) thresholds θ
            theta = f_theta(context, M_prev, base=float(theta_cfg["base"]), ctx_boost=theta_cfg["ctx_boost"])

            # 6) veto + coherence
            veto = int(any(C_now[k] < theta[k] for k in C_now))
            coh = float(np.mean([min(C_now[k], theta[k]) for k in C_now]))

            computed.append({
                "t": t, "context": context, **C_now,
                **{f"M_{k}": v for k, v in M_now.items()},
                **{f"theta_{k}": v for k, v in theta.items()},
                "veto": veto, "coherence": coh
            })

            # 7) write artifacts every N ticks
            if t % max(1, int(60/poll)) == 0:
                df = pd.DataFrame(computed)
                csv_path = os.path.join(outdir, "log_stream.csv")
                df.to_csv(csv_path, index=False)

                # plots
                def plot_series(y, title, fname, ylabel=None):
                    plt.figure()
                    plt.plot(df["t"], y)
                    plt.title(title)
                    plt.xlabel("t"); plt.ylabel(ylabel or title)
                    plt.tight_layout()
                    plt.savefig(os.path.join(outdir, fname), dpi=160)
                    plt.close()

                plot_series(df["coherence"], "coherence", "coherence_over_time.png")
                for k in ["non_harm","equity","stability","resilience"]:
                    plot_series(df[f"theta_C_{k}"] if f"theta_C_{k}" in df.columns else df[f"theta_C_{k}"] 
                                if False else df[f"theta_C_{k}"] if False else df[f"theta_C_{k}"] ,
                                f"theta_{k}", f"theta_{k}_over_time.png")
                # safer explicit:
                plot_series(df["theta_C_non_harm"] if "theta_C_non_harm" in df else df["theta_C_non_harm"] if False else df["theta_C_non_harm"] if False else df["theta_C_non_harm"] if False else df["theta_C_non_harm"] , "theta_non_harm", "theta_non_harm_over_time.png")
                # … pour éviter la verbosité, on trace aussi C_*
                for k in ["C_non_harm","C_equity","C_stability","C_resilience"]:
                    plot_series(df[k], k, f"{k}_over_time.png")
                plot_series(df["veto"], "veto (1=yes)", "veto_over_time.png")

            M_prev = M_now
            t += 1
            time.sleep(poll)

    except KeyboardInterrupt:
        print("\n[Sigma-Stream] stopped by user")

if __name__ == "__main__":
    main()
