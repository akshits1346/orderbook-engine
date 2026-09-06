# stat-arb-lab

A statistical arbitrage research platform: cointegration testing
(Engle-Granger + Johansen), a from-scratch Kalman filter for
time-varying pairs hedge ratios, and a PCA-based multi-asset stat-arb
signal (Avellaneda & Lee 2010) with a genuine walk-forward backtest
engine. Built to answer three concrete questions, not to be a generic
backtesting library:

1. Does a Kalman filter's better hedge-ratio *tracking* actually
   translate into better trading P&L, or is tracking accuracy the
   wrong thing to optimize for?
2. How much does an honest walk-forward split cost (or gain) you
   relative to fitting your model on the whole sample and backtesting
   on the same data -- and does the answer depend on whether the true
   world is stationary?
3. Does a real cointegration/PCA stat-arb signal survive realistic
   transaction costs, or does it only work before fees?

## Architecture

```
data_gen.py         -- two synthetic generators, each matching a textbook
                        model exactly (checkable against ground truth):
                        a cointegrated pair (shared I(1) trend + stationary
                        OU spread, optional time-varying hedge ratio), and
                        a factor basket (K common factors + OU-mean-
                        reverting idiosyncratic residuals, optional
                        structural break in factor loadings)
        |
        v
cointegration.py     -- Engle-Granger (OLS + ADF-on-residual) and Johansen
        |               tests, validated against BOTH a positive control
        |               (real cointegration) and a negative control (two
        |               independent random walks -- the "spurious
        |               regression" case these tests exist to catch)
        v
kalman.py            -- from-scratch 2-state (alpha, beta) Kalman filter
        |               for a time-varying pairs hedge ratio
        v
pca_signal.py         -- Avellaneda-Lee PCA factor extraction, per-asset
        |                factor-residual regression, OU fit, s-score
        v
backtest.py            -- walk-forward engine (refits everything on a
                          TRAILING window only, verified by direct test)
                          + an in-sample lookahead-biased control for
                          measuring how much that discipline costs/gains
        |
        v
metrics.py               -- Sharpe, Sortino, max drawdown, turnover
```

## Build & test

```bash
pip install -r requirements.txt
python3 tests/test_data_gen.py       # 12 checks: OU spread stationary, trend isn't, factor recovery
python3 tests/test_cointegration.py  # 5 checks: EG/Johansen positive AND negative controls
python3 tests/test_kalman.py         # 6 checks: static recovery, time-varying tracking beats OLS
python3 tests/test_pca_signal.py     # 8 checks: factor recovery, OU parameter recovery
python3 tests/test_backtest.py       # 9 checks: NO-LOOKAHEAD verified directly, cost drag, edge cases
```

```bash
python3 scripts/run_pairs_experiment.py       # Kalman vs static OLS, end-to-end P&L (~1s)
python3 scripts/run_multi_seed_robustness.py  # walk-forward vs in-sample, 20 seeds x 2 regimes (~15s)
```

## Finding 1: better hedge-ratio TRACKING does not mean better trading
P&L -- the two are genuinely different objectives

`test_kalman.py` establishes cleanly that the Kalman filter tracks a
time-varying hedge ratio much better than static OLS: mean absolute
tracking error 0.052-0.093 vs. OLS's 0.257 (single-fit) across the
tests here. The natural next question -- does that translate into
better P&L when you trade the resulting spread -- turned out to have
a much less obvious answer, and getting there required actually
diagnosing a real problem in the naive approach, not just running the
comparison once.

**First attempt (textbook defaults: `delta=1e-4`, adaptive R) silently
never traded at all.** The Kalman innovation z-score's std sat at
0.44-0.52 -- an entry threshold of 2.0 was essentially unreachable.
Why: a fast-adapting state (large `delta`) absorbs a genuine spread
deviation INTO its next beta/alpha estimate rather than leaving it as
a surprising, persistent innovation -- and adaptive R (an EWMA of the
squared innovation) compounds this by inflating right when a real
deviation appears, which *shrinks* the z-score computed as
`innovation / sqrt(innovation_var)` even further. A direct sweep
confirms the mechanism precisely:

| delta | adapt_R | hedge-ratio MAE | z-score std | frac(\|z\|>2) |
|---|---|---|---|---|
| 1e-3 | True | 0.2890 | 0.161 | 0.000 |
| 1e-4 | True | 0.2909 | 0.456 | 0.000 |
| 1e-5 | True | 0.3060 | 0.921 | 0.009 |
| **1e-5** | **False** | **0.2895** | **1.558** | **0.211** |
| 1e-6 | False | 0.2935 | 4.129 | 0.666 |

`delta=1e-5, adapt_R=False` was the empirically best point: it beats
static OLS on tracking accuracy (0.2895 vs 0.316) **and** produces a
usable, non-degenerate z-score -- not a tradeoff between the two, a
genuine improvement on both axes.

**And it still lost, badly, on every single seed.** Rerunning the
full pairs backtest (10 seeds, identical entry/exit rule and cost
assumptions for both strategies):

| | mean Sharpe | Kalman beats static (Sharpe) |
|---|---|---|
| static OLS | -1.029 | -- |
| Kalman (tuned) | **-7.093** | 0/10 seeds |

The cause, once counted directly: static OLS made 18-29 entries/exits
over the 1500-step backtest; the tuned Kalman signal made
**147-329** -- roughly 10-13x more. A z-score distribution with
genuine, frequent crossings of the entry threshold (which is exactly
what fixing the self-normalization problem produced) means frequent
whipsaw in and out of positions, and transaction costs on that
turnover overwhelmed the tracking-accuracy edge on every seed tested.
**The honest conclusion is not "use Kalman filters for pairs trading"
or "don't" -- it's that turning a good state estimator into a good
*trading signal* is a separate design problem** (a static entry
threshold calibrated for one z-score distribution doesn't transfer to
a differently-shaped one), and this project's naive threshold rule
doesn't solve it. See "What I'd build next."

## Finding 2: honest walk-forward beats a full-sample "cheat" on
STATIONARY data too -- for a reason that has nothing to do with
lookahead

The intuition "fitting on the whole sample (including the future)
should look artificially good" turned out to be wrong in the simplest
case tested, and wrong in a way worth understanding rather than
dismissing. 20-seed sweep, stationary factor basket (true dynamics
never change):

| | mean Sharpe | mean total PnL | mean daily turnover |
|---|---|---|---|
| walk-forward (252-day trailing window, refit every 5 days) | **3.372** | **1.810** | 0.619 |
| in-sample (fit once on all 750 days) | 1.396 | 1.463 | 1.636 |

Walk-forward wins on Sharpe in **20/20** seeds and on total PnL in
19/20. The mechanism is turnover, not signal quality: a single
full-sample OU fit scores every day's cumulative residual against ONE
global mean/std, which is less locally adaptive than a rolling
252-day window -- so the in-sample version enters and exits positions
roughly **2.6x more often** (1.636 vs 0.619 mean daily turnover), and
that extra churn's transaction-cost drag outweighs any benefit from
fitting on more data. On genuinely stationary data, there's no future
information to leak, so there's no "unfair advantage" to find --
only a less-adaptive model, which loses on its own merits.

## Finding 3: the "unfair advantage" appears exactly where it should
-- once the data stops being stationary

Same 20-seed sweep, but with every asset's factor loadings
independently redrawn at day 500 (a structural break -- sector
rotation, a business-mix change -- not a slow drift):

| | mean Sharpe | mean total PnL | mean daily turnover |
|---|---|---|---|
| walk-forward | 1.235 | 1.537 | 0.688 |
| in-sample (lookahead) | 1.027 | **2.936** | 0.785 |

Walk-forward still edges out in-sample on Sharpe in 14/20 seeds (the
turnover-cost effect from Finding 2 is still present) -- but on total
PnL, **in-sample wins on every single seed (0/20 for walk-forward)**,
nearly doubling mean PnL (2.936 vs 1.537). This is the real lookahead
effect, isolated: the full-sample fit's factor loadings and OU
parameters are estimated using the POST-break regime's own data, so
it starts the post-break period already calibrated to it, while
walk-forward has to detect and adapt to the new regime using only
data that arrives after the break, exactly as a live system would.
**The two metrics tell different stories, and that's the actual
finding**: someone judging this by Sharpe alone would still lean
walk-forward; someone judging by raw PnL would see an unambiguous,
20-for-20 case for the "cheat" -- which is exactly why a real backtest
report needs to show both, and exactly why "my backtest has a good
Sharpe" is not by itself proof against lookahead bias.

## Honest limitations

- **All data here is synthetic** -- no real market data was used (see
  the sibling projects' READMEs for why: this sandbox's network access
  is a package-registry allowlist, not general internet). The
  cointegration/Kalman/PCA machinery itself doesn't know or care
  whether its input is synthetic or real; swapping in real return
  series requires no code changes to `cointegration.py`, `kalman.py`,
  or `pca_signal.py`.
- **The factor basket's idiosyncratic residuals are a simple OU
  process, and factor loadings are constant except at one deliberate,
  instantaneous break** -- real markets have continuously drifting
  loadings and residual dynamics that are far less clean than a single
  OU parameter per asset. The regime-change experiment is a
  deliberately simple, sharp version of a much messier real
  phenomenon.
- **The PCA backtest refits and rescoring only happen at rebalance
  dates (every 5 trading days), not daily** -- a documented
  simplification for tractability, not a claim that daily refitting
  wouldn't change the results.
- **The pairs-trading z-score threshold rule (entry=2.0, exit=0.5) was
  never itself tuned for profitability** -- both static and Kalman
  variants lose money net of the assumed 5bps transaction cost in
  Finding 1; the comparison there is about the RELATIVE difference
  between the two signal sources under an identical (untuned) trading
  rule, not a claim that either is a profitable strategy as built.
- **`min_kappa` (the mean-reversion-speed filter in `compute_signals`)
  and the entry/exit/stop thresholds throughout are fixed constants**,
  not calibrated per-asset or cross-validated.

## What I'd build next

- A trading rule for the Kalman-filtered spread that's actually
  designed for its z-score distribution (e.g. an adaptive threshold
  scaled by the innovation variance's own recent history, or a
  hysteresis band wide enough to filter the ~10x turnover blowup found
  in Finding 1) rather than reusing a threshold tuned for a
  differently-shaped static-OLS signal.
- A slower, continuous factor-loading drift (rather than one sharp
  break) to test whether Finding 3's PnL gap shrinks, grows, or
  behaves differently under a more realistic non-stationarity profile.
- Daily (not every-5-days) refitting for the PCA backtest, to check
  whether Finding 2's turnover-driven result is sensitive to rebalance
  frequency.
- Real return data, the moment it's reachable from wherever this runs
  next -- no code changes needed in the core modules, per the note
  above.
