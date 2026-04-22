# Test report — PR #2 (walk-forward kills training-set leakage)

Ran the Backtest page against the same demo DB twice: once in the legacy `pretrained` mode (to reproduce the +22.9% inflated ROI), once in the new `walk_forward` mode (the honest evaluation). Both runs used identical controls: `football / 1x2 / flat / stake 1 / min_edge 0.03`.

## Headline

- **Pretrained (leaky)** — ROI **+22.9%** on 1226 bets; calibration metrics blank (as designed for the legacy path).
- **Walk-forward (honest)** — ROI **−9.6%** on 859 bets; Brier 0.609, log-loss 1.010, holdout 1256 matches.
- **Delta** — |+22.9% − (−9.6%)| = **32.5 pp**. Well above the 25 pp gate: the leakage bug is closed on the demo universe.

## Assertions

| # | Test | Expected | Observed | Result |
|---|------|----------|----------|--------|
| 1 | Pretrained ROI is within +18% to +26% | +18%–+26% | +22.86% | pass |
| 1 | Pretrained bets within 1100–1350 | 1100–1350 | 1226 | pass |
| 1 | Pretrained Mode stat = `pretrained` | `pretrained` | `pretrained` | pass |
| 1 | Pretrained Brier/Log-loss/Holdout = `—` | `—` | `—` | pass |
| 2 | Walk-forward ROI within −12% to −3% | −12%–−3% | −9.63% | pass |
| 2 | ROI delta ≥ 25 pp | ≥25 | 32.49 pp | pass |
| 2 | Walk-forward bets within 700–1000 | 700–1000 | 859 | pass |
| 2 | Walk-forward Mode stat = `walk_forward` | `walk_forward` | `walk_forward` | pass |
| 2 | Brier (1X2) within 0.55–0.70 | 0.55–0.70 | 0.6089 | pass |
| 2 | Log-loss (holdout) within 0.90–1.15 | 0.90–1.15 | 1.0102 | pass |
| 2 | Holdout matches within 1100–1400 | 1100–1400 | 1256 | pass |

## Evidence

### Pretrained (leaky) — reproduces the old +22.9% inflated ROI
![pretrained mode backtest result](https://app.devin.ai/attachments/5ab20064-b714-4ddd-838a-0b9d139dcb19/pr2-pretrained-leaky.png)

Key visible stats: Bets 1226, Hit 43.4%, ROI **+22.9%**, Yield +22.86%, Mode **pretrained**, Brier/Log-loss/Holdout **—**. Equity curve climbs to +285 (characteristic of leakage).

### Walk-forward (honest) — ROI drops 32 pp, calibration metrics populate
![walk-forward mode backtest result](https://app.devin.ai/attachments/b08ef277-ab7d-4a50-85ae-54cf3d10d88d/pr2-walkforward-honest.png)

Key visible stats: Bets 859, Hit 34.2%, ROI **−9.6%**, Yield −9.63%, Mode **walk_forward**, Brier **0.6089**, Log-loss **1.0102**, Holdout **1256**. Equity curve drifts down to around −100 — no free money.

### Full walkthrough
- Recording: https://app.devin.ai/attachments/84b6dcfa-2ed8-4fff-931c-68b6f9a6b131/rec-08978f8e-5396-4881-b90e-f87d9024941b-edited.mp4

## Out of scope / not tested here
- Different strategies (Kelly / proportional), multi-market (O/U, BTTS), multi-sport.
- Dashboard / ValueBets / MatchDetail pages — unchanged by this PR.
- Real-data evaluation — intentionally deferred to PR 3+ (Football-Data.org / Odds API adapters).

## Caveats
- The absolute −9.6% walk-forward ROI is still on the synthetic demo universe, so it has no predictive meaning in itself; it's only here to prove the gap vs the leaky path.
- The walk-forward run takes ~2 minutes on the demo because it retrains 4× at fold boundaries. Acceptable for now; will revisit with cached fold splits when real data is wired up.
