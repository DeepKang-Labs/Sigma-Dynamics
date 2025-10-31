#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sigma-Dynamics — Canonical Adaptive Moral Control (REAL bridge + Human plots)
DeepKang-Labs (2025) — Axiom-to-Code

Boucle fermée (cadre canonique) :
    θ_i(t) = f_i(E_t, M_{t-1})
    M_t    = Σ_k w_k · C_k
    C̄_t   = (1/n) Σ_k C_k
avec C_k = (non_harm, equity, stability, resilience)

Ce script :
  1) Essaie de récupérer des données réelles publiques (BTC OHLC + indicateurs).
  2) Si indisponible (CI hors-ligne, API down), bascule en simulation sûre.
  3) Calcule les features, applique Sigma-Dynamics et exporte :
     - outputs/sigma_dynamics.csv
     - outputs/bridge_metrics.csv (si pont réel)
     - outputs/*.png (graphes lisibles)

CLI :
    python sigma_dynamics.py --symbol BTC --lookback-days 365 --out-dir outputs --bridge-every 50
"""

from __future__ import annotations

import os
import io
import csv
import math
import time
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration graphique lisible (aucune couleur forcée, grille, labels)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.figsize": (10, 4),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 140,
})


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def today_str() -> str:
    return datetime.utcnow().date().isoformat()

def robust_norm(x: pd.Series) -> pd.Series:
    """
    ZFIX: Normalisation robuste 0..1 via quantiles 1%–99%.
    Correction d’ambiguïté : forcer lo/hi en float pour éviter l’erreur
    "The truth value of a Series is ambiguous".
    """
    lo = float(x.quantile(0.01))
    hi = float(x.quantile(0.99))
    if (hi - lo) <= 0:  # aucune variance → renvoie des zéros
        return pd.Series(np.zeros(len(x)), index=x.index)
    normed = (x - lo) / (hi - lo)
    return normed.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Bridge — Données réelles (sans clé), fallback en simulation
# ---------------------------------------------------------------------------

def fetch_btc_ohlc(days: int = 365) -> Optional[pd.DataFrame]:
    """
    Récupère des OHLC Bitcoin via une source publique simple (Stooq).
    - Stooq propose BTCUSD en CSV quotidiennes (sans clé) ; si ça échoue, on None.
    """
    try:
        url = "https://stooq.com/q/d/l/?s=btc_usd&i=d"
        df = pd.read_csv(url)
        # Colonnes attendues: Date, Open, High, Low, Close, Volume
        # Harmonise et tronque la fenêtre
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        # Couper aux N derniers jours
        cutoff = datetime.utcnow().date() - timedelta(days=days)
        df = df[df.index.date >= cutoff]
        # garder uniquement ce dont on a besoin
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        if len(cols) < 4:
            return None
        return df[cols].astype(float)
    except Exception:
        return None


def fetch_public_indicators() -> Optional[Dict[str, float]]:
    """
    Indicateurs publics légers (pas d'API key) pour mimer des signaux réseau.
    - Blockchair bitcoin stats
    - Coinlore global
    """
    try:
        import requests
        btc = requests.get("https://api.blockchair.com/bitcoin/stats", timeout=10).json()
        glob = requests.get("https://api.coinlore.net/api/global/", timeout=10).json()

        blocks_24h = max(1.0, float(btc["data"].get("blocks_24h", 144)))
        latency_raw = min(2.0, 144.0 / blocks_24h)            # 1.0 ≈ nominal, >1 pire
        latency = float(np.clip(1.0 / latency_raw, 0.0, 1.0)) # plus grand = meilleur

        coins_count = float(glob[0].get("coins_count", 8000.0))
        throughput = float(np.clip(coins_count / 20000.0, 0.0, 1.0))
        stability = float(np.clip(1.0 - (1.0 / (1.0 + math.exp(-2.0 * throughput))), 0.0, 1.0))

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "latency": round(latency, 4),
            "throughput": round(throughput, 4),
            "stability": round(stability, 4),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Construction des features & mapping vers C_k
# ---------------------------------------------------------------------------

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    À partir d'OHLC (quotidien), dérive quelques features objectives.
      - ret: log-returns
      - vol30: vol rolling 30
      - dd: drawdown
      - mom: momentum 30j
      - liq: volume normalisé
    Puis mappe vers (non_harm, equity, stability, resilience) dans [0,1].
    """
    df = raw.copy()

    # log-returns
    df["ret"] = np.log(df["close"]).diff()
    # vol 30 (std des ret) → normalisation robuste
    df["vol30"] = df["ret"].rolling(30, min_periods=5).std()

    # drawdown depuis les plus hauts
    roll_max = df["close"].cummax()
    df["dd"] = df["close"] / roll_max - 1.0  # <= 0

    # momentum simple 30j
    df["mom30"] = df["close"].pct_change(30)

    # normalisation robuste
    for col in ["ret", "vol30", "dd", "mom30"]:
        df[col] = df[col].fillna(0.0)

    # Échelle 0..1 robuste
    ret_n  = robust_norm(df["ret"])
    vol_n  = robust_norm(df["vol30"])
    dd_n   = robust_norm(-df["dd"])      # drawdown plus faible = meilleur ⇒ inverse
    mom_n  = robust_norm(df["mom30"])

    # Volume si dispo
    if "volume" in df.columns:
        liq_n = robust_norm(df["volume"].fillna(method="ffill").fillna(0.0))
    else:
        liq_n = pd.Series(np.full(len(df), 0.5), index=df.index)

    # Mapping vers C_k (tous en [0,1])
    # - non_harm : faible vol & drawdown bas ⇒ moyenne de (1-vol, dd)
    non_harm = (1.0 - vol_n) * 0.6 + dd_n * 0.4

    # - equity : dispersion “juste” (proche de médiane); ici proxy via symétrie des ret
    #   On favorise |ret| modéré ⇒ 1 - |ret_n - 0.5|*2
    equity = 1.0 - (ret_n - 0.5).abs() * 2.0
    equity = equity.clip(0.0, 1.0)

    # - stability : volatilité basse et momentum pas extrême
    stability = (1.0 - vol_n) * 0.7 + (1.0 - (mom_n - 0.5).abs() * 2.0) * 0.3
    stability = stability.clip(0.0, 1.0)

    # - resilience : recovery potentiel (mom positif modéré) + drawdown faible + liquidité
    resilience = (mom_n * 0.45 + dd_n * 0.35 + liq_n * 0.20).clip(0.0, 1.0)

    out = pd.DataFrame({
        "non_harm": non_harm,
        "equity": equity,
        "stability": stability,
        "resilience": resilience,
    }, index=df.index)

    return out.dropna()


# ---------------------------------------------------------------------------
# Sigma-Dynamics core
# ---------------------------------------------------------------------------

def f_theta(context: str, M_prev: np.ndarray, lam: float = 0.15) -> np.ndarray:
    """θ(t) = clip( M_prev ⊙ γ(context) + lam*0.1 , 0..1 )"""
    gamma = np.ones(4, dtype=float)
    if context == "crisis":
        gamma = np.array([1.10, 1.05, 1.15, 1.12])
    elif context == "recovery":
        gamma = np.array([1.00, 1.02, 1.00, 1.08])
    theta = np.clip(M_prev * gamma + lam * 0.1, 0.0, 1.0)
    return theta

def veto_guardrail(C: np.ndarray, theta: np.ndarray, eps: float = 1e-3) -> bool:
    return bool(np.any(theta - C > eps))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(symbol: str = "BTC", lookback_days: int = 365, out_dir: str = "outputs",
        bridge_every: int = 50) -> Dict[str, str]:
    """
    1) Essaie les données réelles (BTC OHLC + indicateurs publics).
    2) Si KO, génère une série simulée stable/modérée.
    3) Applique la boucle Sigma-Dynamics et exporte CSV + PNG.
    """
    # Dossier de sortie
    out_dir = ensure_dir(out_dir)

    # --- Bridge données réelles
    used_real = False
    bridge_rows: List[Dict[str, str | float]] = []

    ohlc = fetch_btc_ohlc(days=lookback_days)
    if ohlc is not None and len(ohlc) > 50:
        used_real = True

    # Si pas de données : simulation sûre
    if not used_real:
        idx = pd.date_range(end=datetime.utcnow().date(), periods=lookback_days, freq="D")
        price = 30000.0 + np.cumsum(RNG.normal(0, 120, len(idx)))
        price = np.maximum(1000.0, price)
        ohlc = pd.DataFrame({
            "open": price,
            "high": price * (1 + np.abs(RNG.normal(0, 0.01, len(idx)))),
            "low":  price * (1 - np.abs(RNG.normal(0, 0.01, len(idx)))),
            "close": price + RNG.normal(0, 60, len(idx)),
            "volume": RNG.uniform(1e3, 1e5, len(idx)),
        }, index=idx)

    # Features → C_k
    C_df = build_features(ohlc)

    # Mémoire morale init + états
    dims = ["non_harm", "equity", "stability", "resilience"]
    M = np.array([0.70, 0.68, 0.66, 0.59], dtype=float)
    theta = np.array([0.60, 0.60, 0.60, 0.58], dtype=float)

    # Timeline discrète (t = 0..N-1) alignée sur C_df
    dates = C_df.index
    N = len(C_df)

    records: List[Dict[str, float | str | int]] = []

    # Bridge CSV (si réel)
    bridge_csv_path = os.path.join(out_dir, "bridge_metrics.csv")
    if used_real:
        with open(bridge_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "latency", "throughput", "stability"])
            w.writeheader()

    # Boucle temporelle
    for i, (ts, row) in enumerate(C_df.iterrows()):
        # Contexte
        if i % 90 in (0, 1, 2, 3, 4, 5):
            context = "crisis"
        elif i % 90 in range(6, 15):
            context = "recovery"
        else:
            context = "normal"

        # Bridge indicateurs (faible fréquence)
        if used_real and (i % bridge_every == 0):
            ind = fetch_public_indicators()
            if ind:
                bridge_rows.append(ind)

        # Vecteur C à partir des features
        C = np.array([float(row[d]) for d in dims], dtype=float)

        # Mise à jour EMA de la mémoire morale
        alpha = 0.2
        M = (1 - alpha) * M + alpha * C

        # Seuils adaptatifs
        theta = f_theta(context, M, lam=0.15)

        # Veto
        veto = veto_guardrail(C, theta)

        # Cohérence (part des dimensions respectant θ)
        coherence = float(np.mean(C >= theta))

        rec = {
            "t": i,
            "date": ts.strftime("%Y-%m-%d"),
            "context": context,
            **{f"C_{d}": float(C[j]) for j, d in enumerate(dims)},
            **{f"M_{d}": float(M[j]) for j, d in enumerate(dims)},
            **{f"theta_{d}": float(theta[j]) for j, d in enumerate(dims)},
            "coherence": coherence,
            "veto": int(veto),
        }
        records.append(rec)

    # Écritures CSV
    dyn_df = pd.DataFrame.from_records(records).set_index("t")
    dyn_csv = os.path.join(out_dir, "sigma_dynamics.csv")
    dyn_df.to_csv(dyn_csv, index=True)

    if used_real and bridge_rows:
        with open(bridge_csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "latency", "throughput", "stability"])
            for r in bridge_rows:
                w.writerow(r)
    elif not used_real:
        bridge_csv_path = ""  # rien écrit

    # -----------------------------------------------------------------------
    # Graphes lisibles (dates en abscisse + grilles + titres explicites)
    # -----------------------------------------------------------------------

    def plot_series(x_dates: pd.Index, y: pd.Series, title: str, ylabel: str, fname: str):
        plt.figure()
        plt.plot(x_dates, y.values)
        plt.title(title)
        plt.xlabel("Date (UTC)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        path = os.path.join(out_dir, fname)
        plt.savefig(path)
        plt.close()
        return path

    art_paths: List[str] = []
    x_dates = pd.to_datetime(dyn_df["date"])

    art_paths.append(plot_series(x_dates, dyn_df["coherence"], 
        "Sigma-Dynamics : Cohérence dans le temps", "Cohérence (0..1)", "coherence.png"))

    for d in dims:
        art_paths.append(plot_series(x_dates, dyn_df[f"theta_{d}"], 
            f"Sigma-Dynamics : θ_{d} (seuil adaptatif)", f"θ_{d} (0..1)", f"theta_{d}.png"))

    art_paths.append(plot_series(x_dates, dyn_df["veto"], 
        "Sigma-Dynamics : Veto déclenché (1 = oui)", "veto (0/1)", "veto.png"))

    # Graphes supplémentaires lisibles : C_k et M_k
    for d in dims:
        art_paths.append(plot_series(x_dates, dyn_df[f"C_{d}"], 
            f"Comprehension C_{d} (observé)", f"C_{d} (0..1)", f"C_{d}.png"))
        art_paths.append(plot_series(x_dates, dyn_df[f"M_{d}"], 
            f"Mémoire morale M_{d} (EMA)", f"M_{d} (0..1)", f"M_{d}.png"))

    note = "Pont réel utilisé" if used_real else "Simulation sûre (offline)"
    return {
        "csv": dyn_csv,
        "bridge_csv": bridge_csv_path,
        "plots": ", ".join(art_paths),
        "note": note,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Sigma-Dynamics avec pont données réelles et graphes lisibles")
    p.add_argument("--symbol", type=str, default="BTC", help="Symbole (pour titre/trace)")
    p.add_argument("--lookback-days", type=int, default=365, help="Fenêtre en jours (historique)")
    p.add_argument("--out-dir", type=str, default="outputs", help="Dossier de sortie")
    p.add_argument("--bridge-every", type=int, default=50, help="Période d’échantillonnage des indicateurs publics")
    args = p.parse_args()

    res = run(
        symbol=args.symbol,
        lookback_days=args.lookback_days,
        out_dir=args.out_dir,
        bridge_every=args.bridge_every,
    )

    print("\n=== Sigma-Dynamics — Exécution terminée ===")
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
