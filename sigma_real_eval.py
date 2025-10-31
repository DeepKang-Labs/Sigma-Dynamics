#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sigma-Dynamics — Real Data Evaluation
Branche l'équation morale adaptative sur un CSV réel.
Usage:
  python sigma_real_eval.py --csv data/real_series.csv --outdir artifacts_real --context-col context
"""

import argparse, json, os, math, time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Helpers ----------
def clamp01(x): 
    return max(0.0, min(1.0, float(x)))

def sigmoid(x): 
    return 1.0 / (1.0 + math.exp(-x))

def time_weight(t_now, t_k, lam=0.08):
    # poids croissant avec la fraicheur (récence)
    return sigmoid(lam * (t_now - t_k))

def norm_series(s, lo=None, hi=None):
    s = pd.to_numeric(s, errors="coerce")
    if lo is None: lo = np.nanquantile(s, 0.01)
    if hi is None: hi = np.nanquantile(s, 0.99)
    rng = max(1e-9, hi - lo)
    return ((s - lo) / rng).clip(0, 1), (lo, hi)

# ---------- f_i : seuils adaptatifs ----------
def f_theta(context, M_prev, base=0.6, ctx_boost={"crisis": -0.05, "degraded": -0.02, "normal": 0.0}):
    # plus le contexte est “risqué”, plus on exige (ou on relâche) selon ta politique
    # ici: en crise on relâche légèrement l’exigence sur stabilité/résilience, mais on
    # peut durcir non_harm/équité en changeant ctx_boost par dimension si besoin.
    boost = ctx_boost.get(context, 0.0)
    theta = {}
    for key, mval in M_prev.items():
        # mix base + mémoire (fidélité) + contexte
        theta[key] = clamp01(0.5 * base + 0.5 * mval + boost)
    return theta

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Chemin du CSV réel")
    ap.add_argument("--outdir", default="sigma_real_artifacts", help="Répertoire de sortie")
    ap.add_argument("--context-col", default="context", help="Nom de la colonne contexte")
    # mapping des colonnes -> proxies (si noms différents dans ton CSV)
    ap.add_argument("--col-harm", default="harm_rate")
    ap.add_argument("--col-ineq", default="gini_or_gap")
    ap.add_argument("--col-vol",  default="volatility")
    ap.add_argument("--col-recov",default="recovery_score")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    assert args.context_col in df.columns, f"Colonne contexte absente: {args.context_col}"

    # Normalisation robuste (quantiles 1%-99%) pour chaque proxy
    nh, nh_rng = norm_series(df[args.col_harm])
    inq, inq_rng = norm_series(df[args.col_ineq])
    vol, vol_rng = norm_series(df[args.col_vol])
    rcv, rcv_rng = norm_series(df[args.col_recov])

    # Vecteur C_k par ligne (proxies -> valeurs morales)
    C = pd.DataFrame({
        "C_non_harm": 1.0 - nh,     # moins de nuisance => mieux
        "C_equity":   1.0 - inq,    # moins d’iniquité => mieux
        "C_stability":1.0 - vol,    # moins d’instabilité => mieux
        "C_resilience": rcv         # meilleure récupération => mieux
    })
    C = C.applymap(clamp01)

    n = len(df)
    M = []  # mémoires pondérées
    Theta = []  # seuils
    VETO = []  # 1/0
    coherence = []

    # états init
    M_prev = {k: 0.6 for k in C.columns}

    # boucle temporelle réelle
    for t in range(n):
        context = str(df.iloc[t][args.context_col]).lower()

        # mémoire pondérée jusqu’à t (récence >)
        weights = np.array([time_weight(t, k, lam=0.08) for k in range(t+1)])
        weights = weights / (weights.sum() + 1e-12)
        C_hist = C.iloc[:t+1].values
        M_vec = (weights.reshape(-1,1) * C_hist).sum(axis=0)
        M_now = dict(zip(C.columns, map(clamp01, M_vec)))

        # seuils
        theta = f_theta(context, M_prev)

        # veto si une dimension sous le seuil
        c_now = dict(C.iloc[t])
        veto_now = int(any(c_now[k] < theta[k] for k in C.columns))

        # cohérence = moyenne des min(C_i, theta_i) (autre choix possible)
        coh = float(np.mean([min(c_now[k], theta[k]) for k in C.columns]))

        M.append(M_now); Theta.append(theta); VETO.append(veto_now); coherence.append(coh)
        M_prev = M_now

    # sortie
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(args.outdir, f"run_{ts}")
    os.makedirs(outdir, exist_ok=True)

    # log détaillé
    out = df.copy()
    out[["C_non_harm","C_equity","C_stability","C_resilience"]] = C
    for k in C.columns:
        out[f"M_{k}"] = [m[k] for m in M]
        out[f"theta_{k}"] = [th[k] for th in Theta]
    out["veto"] = VETO
    out["coherence"] = coherence
    out.to_csv(os.path.join(outdir, "log_real.csv"), index=False)

    # config et ranges
    cfg = {
        "csv": args.csv,
        "context_col": args.context_col,
        "columns_mapping": {
            "harm": args.col_harm, "ineq": args.col_ineq,
            "volatility": args.col_vol, "recovery": args.col_recov
        },
        "normalization_ranges": {
            "harm": nh_rng, "ineq": inq_rng, "volatility": vol_rng, "recovery": rcv_rng
        }
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Graphiques
    def plot_series(y, title, fname):
        plt.figure()
        plt.plot(range(n), y)
        plt.title(title)
        plt.xlabel("t"); plt.ylabel(title.split(":")[0])
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=160)
        plt.close()

    plot_series(coherence, "coherence: real data", "coherence_over_time.png")
    for k in C.columns:
        plot_series([th[k] for th in Theta], f"theta_{k}: real data", f"theta_{k}_over_time.png")
    plot_series(VETO, "veto (1=yes): real data", "veto_over_time.png")

    print(f"[OK] Artifacts written to: {outdir}")

if __name__ == "__main__":
    main()
