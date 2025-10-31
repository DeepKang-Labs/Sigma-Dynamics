#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sigma-Dynamics — Canonical Adaptive Moral Control (REAL data + Human-readable plots)
DeepKang-Labs (2025) — Axiom-to-Code

Boucle canonique (par dimension i ∈ {non_harm, equity, stability, resilience}):
    θ_i(t) = f_i(E_t, M_{t-1})
    M_t    = EMA(C_t)                      # mémoire morale pondérée (EMA)
    C̄_t   = (1/n) Σ_k C_k                 # ici C_t est dérivé des métriques réelles (BTC)

Entrées réelles par défaut : BTC-USD (yfinance, 1D, auto_adjust=True)
Sorties :
  - artifacts/YYYY-MM-DD/sigma_dynamics.csv          (toutes les séries)
  - artifacts/YYYY-MM-DD/*.png                       (graphiques lisibles)
Options CLI : --symbol, --lookback-days, --out-dir, --bridge-every, --human-plots
"""

from __future__ import annotations
import os, math, argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# dépendances data
import yfinance as yf
import requests


# ========= Utilitaires I/O =========

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def utc_today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ========= Acquisition des données réelles (BTC par défaut) =========

def fetch_ohlcv(symbol: str = "BTC-USD", lookback_days: int = 5000) -> pd.DataFrame:
    """
    Télécharge OHLCV journalières via yfinance.
    Renvoie un DataFrame avec colonnes: [date, close, volume] (UTC).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days))
    df = yf.download(
        symbol, start=start.date().isoformat(), end=end.date().isoformat(),
        interval="1d", auto_adjust=True, progress=False
    )
    if df.empty:
        raise RuntimeError(f"Aucune donnée téléchargée pour {symbol}.")
    df = df.rename(columns={"Close": "close", "Volume": "volume"})
    df = df.reset_index()
    # yfinance renvoie un DatetimeIndex -> on force 'date' en timezone-naive UTC pour l'affichage
    df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df[["date", "close", "volume"]].sort_values("date").dropna()
    return df


# ========= Proxies -> Comprehension Vector C_t =========

@dataclass
class SigmaConfig:
    ema_alpha: float = 0.20     # poids récence mémoire morale
    lam: float = 0.15           # intensité d'ajustement des seuils θ
    crisis_vol_z: float = 1.25  # seuil crise (volatilité)
    crisis_ret_20d: float = -0.10  # seuil crise (rendement 20j)
    recovery_win: int = 10      # fenêtre "recovery"


def rolling_dd(close: pd.Series) -> pd.Series:
    """Drawdown (0..1) sur série de prix:  dd = 1 - close/rolling_peak."""
    peak = close.cummax()
    dd = 1.0 - (close / peak)
    return dd.clip(lower=0.0, upper=1.0)


def drawdown_duration(close: pd.Series) -> pd.Series:
    """Approx durée (en jours) passée sous le dernier sommet."""
    peak = close.cummax()
    under = (close < peak).astype(int)
    # compteur de jours consécutifs sous le sommet
    dur = []
    run = 0
    for x in under:
        if x == 1:
            run += 1
        else:
            run = 0
        dur.append(run)
    return pd.Series(dur, index=close.index, dtype=float)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit des proxies normalisés ∈ [0,1] pour C_t = (non_harm, equity, stability, resilience)
    à partir de prix/volume BTC.
    - non_harm: 1 - vol_30 normalisée (faible vol = peu de "harm")
    - equity:   équilibre des rendements -> 1 - |moyenne_30| normalisée
    - stability: 1 - drawdown (proche du sommet = stable)
    - resilience: fonction décroissante de la durée sous sommet (récupération)
    """
    df = df.copy()

    # Retours et volatilité
    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["vol30"] = df["ret"].rolling(30, min_periods=5).std().fillna(method="bfill")

    # normalisation robuste (quantiles 1%)
    def robust_norm(x: pd.Series) -> pd.Series:
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        if hi - lo <= 0:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return ((x - lo) / (hi - lo)).clip(0.0, 1.0)

    vol_n = robust_norm(df["vol30"])
    df["non_harm"] = (1.0 - vol_n).clip(0.0, 1.0)

    # "Equity" ~ faible biais directionnel moyen à 30j
    df["mu30"] = df["ret"].rolling(30, min_periods=5).mean().fillna(0.0).abs()
    df["equity"] = (1.0 - robust_norm(df["mu30"])).clip(0.0, 1.0)

    # Stabilité ~ 1 - drawdown
    dd = rolling_dd(df["close"])
    df["stability"] = (1.0 - robust_norm(dd)).clip(0.0, 1.0)

    # Résilience ~ e^{-durée_dd/const} combinée avec la pente récente
    dur = drawdown_duration(df["close"])
    df["resilience"] = np.exp(-dur / 90.0)
    # bonus si pente 10j > 0
    slope10 = (df["close"].pct_change(10)).fillna(0.0)
    df["resilience"] = (0.7 * df["resilience"] + 0.3 * (robust_norm(slope10))).clip(0.0, 1.0)

    return df


# ========= Contexte & mise à jour des seuils θ =========

def detect_context(df: pd.DataFrame, cfg: SigmaConfig) -> pd.Series:
    """Retourne 'crisis' / 'recovery' / 'normal' par jour."""
    vol_z = (df["vol30"] - df["vol30"].rolling(120, min_periods=30).mean()) / \
            (df["vol30"].rolling(120, min_periods=30).std() + 1e-9)
    ret20 = df["close"].pct_change(20).fillna(0.0)

    ctx = np.full(len(df), "normal", dtype=object)
    crisis_idx = (vol_z > cfg.crisis_vol_z) | (ret20 < cfg.crisis_ret_20d)
    ctx[crisis_idx.values] = "crisis"

    # recovery si on sort d'une crise et que 10j de ret cumulés > 0
    rec = (ctx == "crisis")
    rec_shift = pd.Series(rec).shift(1, fill_value=False).values
    pos10 = (df["close"].pct_change(cfg.recovery_win).fillna(0.0) > 0).values
    recovery_idx = (~rec) & rec_shift & pos10
    ctx[recovery_idx] = "recovery"
    return pd.Series(ctx, index=df.index)


def theta_update(context: str, M_prev: np.ndarray, lam: float) -> np.ndarray:
    """θ(t) = clip(M_prev ⊙ γ(context) + lam*0.1, [0,1])"""
    gamma = np.ones(4, dtype=float)
    if context == "crisis":
        gamma = np.array([1.10, 1.05, 1.15, 1.12])
    elif context == "recovery":
        gamma = np.array([1.00, 1.02, 1.00, 1.08])
    theta = np.clip(M_prev * gamma + lam * 0.1, 0.0, 1.0)
    return theta


def veto_guardrail(C: np.ndarray, theta: np.ndarray, eps: float = 1e-6) -> bool:
    """Déclenchement si une composante < seuil."""
    return bool(np.any((theta - C) > eps))


# ========= Tracé lisible (axe temps en dates) =========

def save_timeplot(df: pd.DataFrame, ycol: str, title: str, outdir: str, fname: str):
    plt.figure(figsize=(10, 3.2))
    plt.plot(df["date"], df[ycol])
    plt.title(title)
    plt.xlabel("Date (UTC)")
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=170)
    plt.close()
    return path


# ========= Exécution principale =========

def run(symbol: str, lookback_days: int, out_dir: str, bridge_every: int, human_plots: bool) -> Dict[str, str]:
    cfg = SigmaConfig()
    outdir = ensure_dir(out_dir)

    # 1) Données réelles
    raw = fetch_ohlcv(symbol, lookback_days)
    feat = build_features(raw)
    feat["context"] = detect_context(feat, cfg)

    # 2) Boucle dynamique (C, M, θ, veto, cohérence)
    dims = ["non_harm", "equity", "stability", "resilience"]
    M = np.array([0.70, 0.68, 0.66, 0.59], dtype=float)
    theta = np.array([0.60, 0.60, 0.60, 0.58], dtype=float)

    rows: List[Dict] = []
    for idx, row in feat.iterrows():
        C = np.array([row[d] for d in dims], dtype=float)

        # mémoire morale (EMA)
        alpha = cfg.ema_alpha
        M = (1 - alpha) * M + alpha * C

        # seuils
        theta = theta_update(str(row["context"]), M, lam=cfg.lam)

        # veto + cohérence
        veto = int(veto_guardrail(C, theta, eps=1e-3))
        coherence = float(np.mean(C >= theta))

        rows.append({
            "date": row["date"],
            "context": row["context"],
            **{f"C_{d}": float(C[i]) for i, d in enumerate(dims)},
            **{f"M_{d}": float(M[i]) for i, d in enumerate(dims)},
            **{f"theta_{d}": float(theta[i]) for i, d in enumerate(dims)},
            "coherence": coherence,
            "veto": veto,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "sigma_dynamics.csv")
    df.to_csv(csv_path, index=False)

    # 3) Graphiques lisibles
    plot_paths = []
    plot_paths.append(save_timeplot(df, "coherence",
                                    "Sigma-Dynamics : Cohérence dans le temps",
                                    outdir, "coherence.png"))
    for d in dims:
        plot_paths.append(save_timeplot(df, f"theta_{d}",
                                        f"Sigma-Dynamics : θ_{d} (seuil) dans le temps",
                                        outdir, f"theta_{d}.png"))
    plot_paths.append(save_timeplot(df, "veto",
                                    "Sigma-Dynamics : Veto déclenché (1 = oui)",
                                    outdir, "veto.png"))

    # 4) Petit rapport lisible (facultatif mais utile)
    rep_lines = [
        f"# Rapport Sigma-Dynamics — {symbol}",
        f"- Période : {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}",
        f"- Observations : {len(df)} jours",
        f"- Cohérence moyenne : {df['coherence'].mean():.3f}",
        f"- Veto : {int(df['veto'].sum())} jours déclenchés",
        "",
        "## Seuils moyens (θ) :",
        *[f"- {d} : {df[f'theta_{d}'].mean():.3f}" for d in dims],
    ]
    with open(os.path.join(outdir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep_lines))

    return {
        "csv": csv_path,
        "plots": ", ".join(plot_paths),
        "report": os.path.join(outdir, "REPORT.md"),
        "note": f"Résultats enregistrés dans {outdir}"
    }


# ========= CLI =========

def parse_args():
    p = argparse.ArgumentParser(description="Sigma-Dynamics (réel BTC + graphiques lisibles)")
    p.add_argument("--symbol", default="BTC-USD", help="Symbole yfinance (ex: BTC-USD)")
    p.add_argument("--lookback-days", type=int, default=5000, help="Fenêtre en jours (historique)")
    p.add_argument("--out-dir", default=f"artifacts/{utc_today_str()}", help="Dossier de sortie")
    p.add_argument("--bridge-every", type=int, default=0, help="(Réservé) période de fetch pour un pont externe")
    p.add_argument("--human-plots", action="store_true", help="Activer styles lisibles (déjà le cas par défaut)")
    return p.parse_args()


def main():
    args = parse_args()
    res = run(
        symbol=args.symbol,
        lookback_days=args.lookback_days,
        out_dir=args.out_dir,
        bridge_every=args.bridge_every,
        human_plots=args.human_plots,
    )
    print("\n=== Exécution terminée ===")
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
