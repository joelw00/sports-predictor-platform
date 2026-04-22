# Modelling approach

The platform is explicitly **multi-model**: different families answer different market questions, results are calibrated, and an ensemble combines them.

## 1. Targets

### Football

| Market              | Target type                          |
|---------------------|--------------------------------------|
| 1X2                 | categorical (home / draw / away)     |
| Double chance       | derived from 1X2                     |
| Over/Under 0.5…5.5  | binary per line                      |
| BTTS                | binary                               |
| Correct score       | categorical over scoreline grid      |
| Asian handicap      | derived from scoreline distribution  |
| HT/FT               | categorical 3×3                      |
| Corners / cards     | count (Poisson / NB)                 |
| Goalscorers         | binary per player                    |
| Clean sheet         | binary                               |

### Table tennis (Phase 4)

| Market              | Target type                |
|---------------------|----------------------------|
| Match winner        | binary                     |
| Set winner          | binary per set             |
| Total points        | count                      |
| Handicap set/points | derived                    |
| Exact set score     | categorical over grid      |

## 2. Model families

### 2.1 Statistical baselines

- **Independent Poisson** — λ_home, λ_away from team attack/defence strengths and home advantage.
- **Dixon–Coles** — adds a low-score dependency correction (φ) plus time-weighted fitting.
- **Elo** — incremental team rating with K-factor and home advantage; feeds both as a feature and as a standalone logistic model.

### 2.2 Gradient-boosted models

- **LightGBM** (default) / **XGBoost** (backup).
- One booster per `(sport, market)` target.
- Isotonic calibration on out-of-fold predictions.
- Monotonic constraints where meaningful (e.g. Elo diff ↑ → home prob ↑).

### 2.3 Ensembles

- Weighted average of calibrated probabilities from statistical + GBM models.
- Weights fit by minimising log-loss on a held-out recent window (`scipy.optimize.minimize`, simplex constraints).
- A final Platt / isotonic calibrator runs on the ensemble output.

## 3. Training protocol

- **Split**: walk-forward by match date.
- **Folds**: expanding window, step = 1 month (football) / 1 week (table tennis).
- **Hyperparams**: Optuna, time-budgeted per market.
- **Leakage guards**: features never use post-match info; H2H / form windows lag by ≥1 day.
- **Class imbalance**: log-loss objective (no resampling); monitor Brier and calibration.
- **Artifacts**: persisted to `models/artifacts/<sport>/<market>/<version>/` and registered in `model_registry`.

## 4. Evaluation metrics

| Metric              | Why it matters                           |
|---------------------|------------------------------------------|
| Log loss            | proper scoring rule                      |
| Brier score         | proper scoring rule, decomposable        |
| Calibration curve   | probabilities must match empirics        |
| AUC                 | ranking quality (where applicable)       |
| Accuracy            | sanity check only                        |
| ROI / yield         | betting utility                          |
| Profit factor       | robustness to variance                   |
| Hit rate            | complements ROI                          |
| Max drawdown        | risk                                     |
| CLV                 | predictive power vs. closing line        |

## 5. Calibration

- **Isotonic regression** by default (non-parametric, monotonic).
- **Platt scaling** as a fallback on small datasets.
- Calibrator fit on the last training fold; refreshed every retrain cycle.

## 6. Explainability

- **Feature importance** (gain-based) summarised per model.
- **SHAP** values cached per prediction (Phase 6) so the UI can show "why".
- **Confidence intervals** via bootstrap aggregation over GBM models trained on bagged folds.

## 7. Live models

- **Football in-play**: decay the pre-match scoreline distribution with an in-play goal-rate λ(t) conditioned on current score, minute, red cards, and xG accumulated. Re-derive 1X2 / OU from the updated scoreline distribution.
- **Table tennis in-play**: Markov chain over the current set/match state parameterised by each player's point-win probability (estimated from recent form and H2H).

## 8. Drift monitoring

- **PSI** on feature distributions vs. training window; alert at > 0.25.
- Rolling **Brier score** on predictions with realised outcomes; alert on sustained degradation.
- **Odds drift**: systematic gap between `p_model` and `p_closing` flagged.

## 9. Non-goals

- No claim of beating the closing line consistently on every market.
- No bankroll automation — Kelly is reported for reference, staking is user-driven.
- No proprietary data sources bundled — the quality ceiling is set by the sources you enable.
