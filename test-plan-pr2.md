# PR #2 Test Plan — Walk-forward backtester kills training-set leakage

## What changed (user-visible)
Backtest page now exposes a **Mode** selector with two options:
- **`walk_forward`** (new default, honest): retrains the predictor at each fold boundary, only scores matches strictly after the training cut-off, reports Brier / log-loss / holdout size.
- **`pretrained`** (legacy, leaky): the old path — kept purely so we can show the delta.

Expected impact on the demo universe: ROI drops from the old inflated `+22–23%` down to roughly `−7 to −10%` when switching to walk-forward, confirming the leakage is closed. Calibration metrics (Brier ≈ 0.6, log-loss ≈ 1.0, holdout ≈ 1200) appear only for walk-forward runs.

## UI path (from code)
`frontend/src/pages/Backtest.tsx`:
- Mode dropdown rendered by `<Select value={mode} onValueChange=...>` around lines 75–90 inside the same grid as Sport/Market/Strategy/Stake/Min edge.
- Clicking **Run backtest** calls `runBacktest({ mode, n_folds: 6, min_train_folds: 2, ... })` via `POST /backtests/run`.
- Returned `BacktestResult` is pushed to state; `Stat` cards render ROI / Yield / Hit rate / Drawdown / Profit factor / Bets from top-level fields, and **Mode / Brier (1X2) / Log-loss / Holdout matches** from `result.breakdown` and `result.breakdown.calibration`.

Backend routing: `backend/app/api/backtests.py::run_backtest` dispatches to `WalkForwardBacktester` when `mode == "walk_forward"`, else legacy `Backtester`.

## Primary end-to-end flow
Run both modes back-to-back against the same demo DB from the Backtest page and compare. Recording captures dashboard → Backtest navigation → both runs → both result cards.

### Pinned values observed on this machine's DB (via curl sanity)
| Mode | n_bets | ROI | Brier | log_loss | n_holdout |
|------|-------:|-----:|------:|---------:|----------:|
| pretrained (leaky) | 1226 | **+22.86%** | (n/a) | (n/a) | (n/a) |
| walk_forward (honest) | 859 | **−9.63%** | 0.609 | 1.010 | 1256 |

Used as the source of truth for the assertions below.

## Test cases

### Test 1 — Pretrained mode reproduces the leaky inflated ROI (regression baseline)
**Steps**
1. Go to `http://localhost:5173`, click **Backtest** in the nav.
2. Leave Sport=`football`, Market=`1x2`, Strategy=`flat`, Stake=`1`, Min edge=`0.03`.
3. Open the **Mode** dropdown, select **`pretrained`**.
4. Click **Run backtest**; wait for result card.

**Pass criteria (exact)**
- ROI stat renders between **+18.00% and +26.00%** (expected ≈ `+22.9%`).
- Bets stat renders between **1100 and 1350** (expected 1226).
- Mode stat renders the literal string **`pretrained`**.
- Brier / Log-loss / Holdout stats render **`—`** (no calibration metrics for legacy mode).

**Would a broken fix look identical?** No — if the mode selector weren't wired, the default would still be walk_forward and ROI would be ≈ `−9%`.

### Test 2 — Walk-forward mode drops ROI by ≥ 25 points and exposes calibration metrics (THE fix)
**Steps**
1. On the same Backtest page, change **Mode** dropdown to **`walk_forward`**.
2. Leave other controls unchanged.
3. Click **Run backtest**; wait for result card (retrains 4× — can take ≈ 2 min on demo).

**Pass criteria (exact)**
- ROI stat renders between **−12.00% and −3.00%** (expected ≈ `−9.6%`).
- |ROI(pretrained) − ROI(walk_forward)| ≥ **0.25** (i.e. at least 25 pp drop).
- Bets stat renders between **700 and 1000** (expected 859).
- Mode stat renders the literal string **`walk_forward`**.
- Brier (1X2) stat renders a number between **0.55 and 0.70** (expected 0.609).
- Log-loss stat renders a number between **0.90 and 1.15** (expected 1.010).
- Holdout matches stat renders an integer between **1100 and 1400** (expected 1256).

**Would a broken fix look identical?** No — if walk-forward silently fell back to leaky pretrained, ROI would still be ≈ +22% and Brier/Log-loss/Holdout would stay `—`.

## Out of scope for this run
- Multiple-strategy / multi-market variations.
- Regression on Dashboard/ValueBets/MatchDetail (unchanged by this PR).
- Performance — a single walk-forward run takes ~2 min with 4 retrains; acceptable for now.

## Evidence to collect
- Screenshot of each result card (pretrained vs walk_forward) side-by-side.
- Network trace of `POST /backtests/run` showing `mode=pretrained` then `mode=walk_forward` in request bodies.
- Short screen recording covering both runs.
