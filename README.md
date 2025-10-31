# Sigma-Dynamics (v0.1) — DeepKang-Labs

**Experimental simulation engine for the Sigma-Lab Framework.**

Sigma-Dynamics implements the canonical loop of ethical coherence:

\[
\theta_i(t) = f_i(E_t, M_{t-1}), \quad M_t = \sum_k w_k C_k, \quad \overline{C}_t = \frac{1}{n}\sum_k C_k
\]

### 🔬 Core Principles
- Adaptive moral control through dynamic thresholds (`theta_i`)
- Time-weighted moral memory (`M_t`)
- Semantic environment reactivity (`E_t`)
- Veto Guardrail: prevents coherence loss
- Contextual modes: `normal`, `crisis`, `recovery`

### ⚙️ Run the simulation
```bash
pip install -r requirements.txt
python sigma_dynamics.py

Results → /sigma_dynamics_artifacts_YYYYMMDD-HHMMSS/
(contains plots + CSV logs for each simulation run)

🧠 Vision

Sigma-Dynamics is part of the broader Sigma-Lab Framework,
bridging conceptual ethics and computational implementation.

> DeepKang-Labs (2025) — Axiom to Code.
