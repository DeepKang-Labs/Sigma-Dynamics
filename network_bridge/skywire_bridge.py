"""
Skywire Bridge - Public Metrics Connector
DeepKang-Labs (2025)

Ce module simule un pont de données vers le réseau Skywire
en récupérant des métriques publiques et en les convertissant
dans le format Sigma-Dynamics (latency, throughput, stability...).
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
BRIDGE_NAME = "Skywire_Public_Bridge"
DATA_URLS = {
    "latency": "https://api.blockchair.com/bitcoin/stats",  # exemple public
    "network": "https://api.coinlore.net/api/global/",       # indicateurs réseau globaux
}

# --- CHARGEMENT ---
def fetch_public_metrics():
    """Récupère des métriques simulant le comportement d’un réseau décentralisé."""
    try:
        btc_data = requests.get(DATA_URLS["latency"], timeout=10).json()
        global_data = requests.get(DATA_URLS["network"], timeout=10).json()

        latency = btc_data["data"]["blocks_24h"] / 144  # simulation
        throughput = global_data[0]["coins_count"] / 10000
        stability = np.clip(1 - (1 / (1 + np.exp(-throughput * 2))), 0, 1)

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "latency": round(latency, 3),
            "throughput": round(throughput, 3),
            "stability": round(stability, 3),
        }

        print(f"[{BRIDGE_NAME}] Metrics fetched:", metrics)
        return metrics

    except Exception as e:
        print(f"[{BRIDGE_NAME}] Error fetching metrics:", e)
        return None


def export_to_sigma_format(metrics):
    """Convertit les données en format CSV compatible Sigma-Dynamics."""
    df = pd.DataFrame([metrics])
    df.to_csv("network_bridge/skywire_metrics.csv", index=False)
    print("[Sigma-Bridge] Metrics exported → skywire_metrics.csv")


if __name__ == "__main__":
    data = fetch_public_metrics()
    if data:
        export_to_sigma_format(data)
