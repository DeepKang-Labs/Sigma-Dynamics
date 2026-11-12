# Sigma-Dynamics — experimental simulation (v0.1)

**DeepKang-Labs** · Research software associated with [Sigma-Lab Framework](https://github.com/DeepKang-Labs/Sigma-Lab-Framework)

> **Scope:** an experimental, inspectable simulation of adaptive diagnostic scores. It is **not** an empirically validated model of human morality, a production ethical decision-maker, or a verified live bridge to another network.

## Research model

The historical research label is *Theory of Algorithmic Moral Relativity*. Here it names a **proposed model**, not a demonstrated physical or moral law.

The simulation describes the illustrative loop `θ → C → M → θ`:

```text
theta_i(t) = f_i(E_t, M_(t-1))
M_t        = sum_k w_k * C_k
C_mean(t)  = (1/n) * sum_k C_k
```

These expressions specify a modeling approach; they do not establish predictive accuracy or ethical validity.

## Run locally

Requirements: Python 3, NumPy, pandas, matplotlib and requests (see [requirements.txt](requirements.txt)).

```bash
python -m pip install -r requirements.txt
python sigma_dynamics.py --symbol BTC --lookback-days 365 --out-dir outputs --bridge-every 50
```

Inspect generated CSVs and plots in `outputs/` (or the path selected with `--out-dir`). The script tries to retrieve public market input; when unavailable it may **fall back to synthetic data**. Do not describe a successful run as proof of live data ingestion without checking the actual input provenance in its artifacts and logs.

## Reproducibility and evidence limits

- [Executable source](sigma_dynamics.py) is the implementation to inspect; this README is not a scientific proof.
- [Scheduled workflow](.github/workflows/run-sigma-dynamics.yml) runs analysis and uploads artifacts for its configured retention period. A green job means the configured steps succeeded, **not** that a real-data bridge or external scientific validation occurred.
- Outputs are exploratory diagnostic traces. Compare against held-out real-world data and independent baselines before making generalization claims.
- Previous versions referenced an image under `docs/img/` and an `export-visual.yml` workflow; those files were not present on the default branch during the 12 November 2026 documentation review. No generated visualization is claimed here.

## Provenance

Research direction: Yuri Kang / DeepKang-Labs. AI assistants contributed to historical research and development workflows as tools. Their participation does not substitute for independent review.

See the [Sigma-Lab Framework](https://github.com/DeepKang-Labs/Sigma-Lab-Framework) for surrounding research context.
