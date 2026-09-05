"""
End-to-end pairs-trading comparison: does the Kalman filter's better
hedge-ratio TRACKING (already shown directly in test_kalman.py) actually
translate into better trading P&L, under a genuinely time-varying true
hedge ratio? Two strategies, same synthetic pair, same entry/exit rule,
same transaction costs -- the only difference is where the spread
z-score comes from:
  - STATIC: one OLS hedge ratio fit on a trailing window, held fixed
    until the next (infrequent) refit.
  - KALMAN: hedge ratio re-estimated every single step via the Kalman
    filter (src/kalman.py).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import statsmodels.api as sm

from data_gen import generate_cointegrated_pair
from kalman import kalman_hedge_ratio
from metrics import summary


def static_ols_zscore(log_p1, log_p2, refit_every=60, lookback=120):
    """Refits OLS beta every `refit_every` steps on the trailing
    `lookback` window, computes the spread and a rolling z-score of it."""
    n = len(log_p1)
    beta = np.zeros(n)
    alpha = np.zeros(n)
    b, a = 1.0, 0.0
    for t in range(n):
        if t >= lookback and t % refit_every == 0:
            window = slice(t - lookback, t)
            X = sm.add_constant(log_p1[window])
            model = sm.OLS(log_p2[window], X).fit()
            a, b = model.params
        alpha[t], beta[t] = a, b
    spread = log_p2 - (alpha + beta * log_p1)
    z = np.zeros(n)
    for t in range(lookback, n):
        window = spread[max(0, t - lookback):t]
        mu, sigma = window.mean(), window.std(ddof=1)
        z[t] = (spread[t] - mu) / sigma if sigma > 1e-10 else 0.0
    return z, beta


def backtest_pair(log_p1, log_p2, z, beta, entry=2.0, exit=0.5, cost_bps=5.0, burn_in=120):
    """Trades the spread y - beta*x: long the spread (long p2, short
    beta*p1) when z < -entry, short when z > entry, flat when |z| < exit.
    Position held constant between signal updates; PnL marked against
    the ACTUAL next-step log-price changes (proportional-return
    approximation, consistent with using log-prices throughout)."""
    n = len(z)
    state = 0
    daily_pnl = np.zeros(n - burn_in - 1)
    daily_turnover = np.zeros(n - burn_in - 1)
    n_state_changes = 0
    prev_state = 0
    prev_w = np.array([0.0, 0.0])  # [weight on p1, weight on p2]
    for idx, t in enumerate(range(burn_in, n - 1)):
        if state == 0:
            if z[t] < -entry:
                state = 1
            elif z[t] > entry:
                state = -1
        elif state == 1 and z[t] > -exit:
            state = 0
        elif state == -1 and z[t] < exit:
            state = 0
        if state != prev_state:
            n_state_changes += 1
        prev_state = state

        w = np.array([-state * beta[t], state * 1.0])
        turnover = np.abs(w - prev_w).sum()
        cost = cost_bps * 1e-4 * turnover
        d_log_p1 = log_p1[t + 1] - log_p1[t]
        d_log_p2 = log_p2[t + 1] - log_p2[t]
        daily_pnl[idx] = w[0] * d_log_p1 + w[1] * d_log_p2 - cost
        daily_turnover[idx] = turnover
        prev_w = w
    return daily_pnl, daily_turnover, n_state_changes


def main():
    print("## Time-varying hedge ratio: static OLS vs Kalman filter, end-to-end P&L")
    print()
    print("| seed | static Sharpe | static PnL | static #entries/exits | "
          "kalman Sharpe | kalman PnL | kalman #entries/exits |")
    print("|---|---|---|---|---|---|---|")

    # Rescaling log_p1/log_p2 by the SAME constant leaves the OLS/Kalman-
    # recovered hedge ratio exactly unchanged (slope is invariant to
    # scaling both x and y identically) and preserves the trend-vs-spread
    # signal-to-noise ratio exactly -- unlike changing trend_vol, which
    # would alter that ratio and make hedge-ratio ESTIMATION itself a
    # different, not-directly-comparable problem. trend_vol=1.0 (the
    # default, and what test_kalman.py's tracking-accuracy comparison
    # uses) gives per-step log-price changes with std=1.0, which is an
    # appropriately large signal for a scale-invariant unit-root/OLS
    # test but an unrealistic 100% daily move fed directly into a P&L
    # backtest -- RESCALE=0.01 brings that down to a realistic ~1%/day
    # without touching the estimation problem's actual difficulty.
    RESCALE = 0.01
    static_sharpes, kalman_sharpes = [], []
    for seed in range(10):
        d = generate_cointegrated_pair(n=1500, hedge_ratio=1.5, theta=0.08, spread_vol=0.3,
                                        time_varying_hedge=True, hedge_ratio_amplitude=0.5,
                                        hedge_ratio_period=200, seed=seed)
        log_p1 = d["log_p1"] * RESCALE
        log_p2 = d["log_p2"] * RESCALE

        z_static, beta_static = static_ols_zscore(log_p1, log_p2)
        pnl_static, turnover_static, n_static = backtest_pair(log_p1, log_p2, z_static, beta_static)

        # delta=1e-4 (a common textbook default) and adaptive R turned out
        # to be a bad combination here -- see README's "self-normalizing
        # z-score" finding: with fast state adaptation, the filter absorbs
        # genuine spread deviations INTO its beta/alpha estimate rather
        # than leaving them as a surprising (tradeable) innovation, and
        # an EWMA-adaptive R compounds this by inflating right when a
        # real deviation appears, further shrinking z. delta=1e-5 with a
        # FIXED R was the empirically best point found by direct sweep
        # (see the table in README): better hedge-ratio tracking AND a
        # usable, non-degenerate z-score distribution, not a tradeoff
        # between the two.
        kf = kalman_hedge_ratio(log_p1, log_p2, delta=1e-5, adapt_R=False, R_init=1e-6)
        z_kalman = kf["z_score"]
        pnl_kalman, turnover_kalman, n_kalman = backtest_pair(log_p1, log_p2, z_kalman, kf["beta"])

        s_static = summary(pnl_static)
        s_kalman = summary(pnl_kalman)
        static_sharpes.append(s_static["sharpe"])
        kalman_sharpes.append(s_kalman["sharpe"])
        print(f"| {seed} | {s_static['sharpe']:.3f} | {s_static['total_pnl']:.4f} | {n_static} | "
              f"{s_kalman['sharpe']:.3f} | {s_kalman['total_pnl']:.4f} | {n_kalman} |")

    print()
    print(f"Mean Sharpe -- static: {np.mean(static_sharpes):.3f}, kalman: {np.mean(kalman_sharpes):.3f}")
    print(f"Kalman beats static on {sum(k > s for k, s in zip(kalman_sharpes, static_sharpes))}/10 seeds")


if __name__ == "__main__":
    main()
