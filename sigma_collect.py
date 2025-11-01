#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sigma-Collect
- Récupère quotidiennement des données marché (Top 20 CoinGecko),
  des métriques chaîne (BTC), du gas ETH (snapshot), des taux FX et DXY.
- Écrit des CSV/JSON bruts, arborescence datée, prête pour Sigma-Dynamics & Analysis.

Sources gratuites:
- CoinGecko: /coins/{id}/market_chart?vs_currency=usd&days=max
- blockchain.info charts: /charts/{metric}?timespan=all&format=json
- exchangerate.host timeseries
- Stooq DXY CSV

NOTE: CoinGecko ne fournit pas OHLC complet sur tout l'historique via cette route.
On exporte donc `price.csv`, `market_cap.csv`, `volume.csv` par actif.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List

import requests
import pandas as pd

CG_BASE = "https://api.coingecko.com/api/v3"
BTC_CHAIN_BASE = "https://api.blockchain.info/charts"
FX_BASE = "https://api.exchangerate.host"
DXY_STOOQ = "https://stooq.com/q/d/l/?s=dxy&i=d"

# Top 20 (modifiable)
COINS: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "USDT": "tether",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "TRX": "tron",
    "TON": "the-open-network",
    "BCH": "bitcoin-cash",
    "LINK": "chainlink",
    "LTC": "litecoin",
    "XLM": "stellar",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "MATIC": "polygon-pos",
    "ATOM": "cosmos",
    "ETC": "ethereum-classic",
    "XMR": "monero",
}

BTC_METRICS = [
    "hash-rate",
    "difficulty",
    "mempool-size",
    "transaction-fees-usd",
]

FX_BASE_CCY = "EUR"
FX_SYMBOLS = ["USD", "GBP", "CHF", "JPY", "CNY", "RUB"]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ts_to_date(ts_ms: int) -> str:
    # CoinGecko timestamps in ms
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()


def write_series_csv(path: str, series: List[List[float | int]] | List[Dict]):
    """
    Accept either:
      - CoinGecko style [[ts_ms, value], ...]
      - blockchain.info style [{"x": ts, "y": value}, ...]
    """
    if not series:
        return
    if isinstance(series[0], list) or isinstance(series[0], tuple):
        rows = [{"date": ts_to_date(int(ts)), "value": float(val)} for ts, val in series]
    else:
        rows = [{
            "date": datetime.fromtimestamp(int(pt["x"]), tz=timezone.utc).date().isoformat(),
            "value": float(pt["y"])
        } for pt in series]
    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(path, index=False)


def coingecko_market_chart(coin_id: str, vs="usd"):
    url = f"{CG_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs, "days": "max"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()  # keys: prices, market_caps, total_volumes


def fetch_btc_chain_metric(metric: str):
    url = f"{BTC_CHAIN_BASE}/{metric}"
    params = {"timespan": "all", "format": "json"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("values", [])


def fetch_eth_gas_oracle(api_key: str | None):
    if not api_key:
        return None
    url = "https://api.etherscan.io/api"
    params = {"module": "gastracker", "action": "gasoracle", "apikey": api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_fx_timeseries(base=FX_BASE_CCY, symbols: List[str] = FX_SYMBOLS):
    # full history since 2010
    url = f"{FX_BASE}/timeseries"
    params = {
        "start_date": "2010-01-01",
        "end_date": datetime.utcnow().date().isoformat(),
        "base": base,
        "symbols": ",".join(symbols),
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()  # { 'rates': { 'YYYY-MM-DD': {'USD': 1.xx, ...}, ... } }


def fetch_dxy_stooq_csv():
    r = requests.get(DXY_STOOQ, timeout=60)
    r.raise_for_status()
    return r.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Répertoire de sortie (ex: data/2025-11-01)")
    args = ap.parse_args()

    out_root = args.out
    ensure_dir(out_root)

    # ---------- A. CoinGecko (marché) ----------
    for sym, cg_id in COINS.items():
        print(f"[CG] {sym} ← {cg_id}")
        try:
            data = coingecko_market_chart(cg_id, "usd")
        except Exception as e:
            print(f"  ! coin {sym} erreur CoinGecko: {e}")
            continue

        base_dir = os.path.join(out_root, "crypto", sym)
        ensure_dir(base_dir)

        try:
            write_series_csv(os.path.join(base_dir, "price.csv"), data.get("prices", []))
            write_series_csv(os.path.join(base_dir, "market_cap.csv"), data.get("market_caps", []))
            write_series_csv(os.path.join(base_dir, "volume.csv"), data.get("total_volumes", []))
        except Exception as e:
            print(f"  ! coin {sym} écriture CSV: {e}")

        # Respecter un léger délai pour éviter un rate-limit
        time.sleep(1.2)

    # ---------- B. BTC chain metrics ----------
    btc_dir = os.path.join(out_root, "chain", "BTC")
    ensure_dir(btc_dir)
    for m in BTC_METRICS:
        print(f"[BTC-chain] {m}")
        try:
            vals = fetch_btc_chain_metric(m)
            write_series_csv(os.path.join(btc_dir, f"{m}.csv"), vals)
        except Exception as e:
            print(f"  ! BTC metric {m} erreur: {e}")

    # ---------- C. ETH gas oracle (snapshot) ----------
    eth_api_key = os.getenv("ETHERSCAN_API_KEY")
    eth_dir = os.path.join(out_root, "chain", "ETH")
    ensure_dir(eth_dir)
    if eth_api_key:
        try:
            print("[ETH] gas oracle snapshot")
            gas = fetch_eth_gas_oracle(eth_api_key) or {}
            pd.json_normalize(gas).to_json(os.path.join(eth_dir, "gas_oracle.json"), orient="records")
        except Exception as e:
            print(f"  ! ETH gas oracle erreur: {e}")
    else:
        # créer un témoin pour montrer que c'est volontairement ignoré
        with open(os.path.join(eth_dir, "gas_oracle.SKIPPED.txt"), "w", encoding="utf-8") as f:
            f.write("ETHERSCAN_API_KEY non fourni – snapshot gas non collecté.\n")

    # ---------- D. FX & DXY ----------
    fiat_dir = os.path.join(out_root, "fiat")
    ensure_dir(fiat_dir)

    # FX timeseries
    try:
        print("[FX] timeseries EUR→{...}")
        fx = fetch_fx_timeseries(FX_BASE_CCY, FX_SYMBOLS)
        rates = fx.get("rates", {})
        if rates:
            # construire un DF long (date, ccy, value)
            rows = []
            for d, ccys in rates.items():
                for ccy, val in ccys.items():
                    rows.append({"date": d, "ccy": ccy, "value": float(val)})
            df = pd.DataFrame(rows).sort_values(["ccy", "date"])
            for ccy in FX_SYMBOLS:
                dff = df[df["ccy"] == ccy][["date", "value"]]
                dff.to_csv(os.path.join(fiat_dir, f"FX_EUR_{ccy}.csv"), index=False)
    except Exception as e:
        print(f"  ! FX erreur: {e}")

    # DXY (csv direct)
    try:
        print("[DXY] stooq")
        csv_txt = fetch_dxy_stooq_csv()
        # sauvegarde brut
        with open(os.path.join(fiat_dir, "DXY.csv"), "w", encoding="utf-8") as f:
            f.write(csv_txt)
    except Exception as e:
        print(f"  ! DXY erreur: {e}")

    print(f"\n✔ Done. Files in: {out_root}")


if __name__ == "__main__":
    sys.exit(main())
