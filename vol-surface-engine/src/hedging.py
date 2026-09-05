"""
Discrete delta-hedging simulation under deliberate model mismatch: the
"true" world evolves under Heston (stochastic vol), but the hedger only
ever computes deltas from Black-Scholes at one fixed vol (the ATM
implied vol at trade inception) -- exactly what a desk without a
real-time vol-surface-aware hedger does. The gap between "BS delta
hedged frequently enough" and "actually replicated the payoff" is the
object being measured.

Short one European call, delta-hedge with the underlying, at a
grid of rehedge frequencies and per-trade transaction costs.
"""
import math
import numpy as np

from black_scholes import bs_price, bs_delta
from heston import simulate_paths, HestonParams


def simulate_delta_hedge(S0, K, T, r, heston_params: HestonParams, hedge_vol,
                          n_steps, rehedge_every, n_paths, cost_bps=0.0, q=0.0, seed=None):
    """
    Sells 1 call at its Black-Scholes price (computed at hedge_vol, T),
    delta-hedges every `rehedge_every` steps using BS delta at hedge_vol
    and the ACTUAL remaining time to maturity, under paths simulated
    from the true Heston dynamics.

    Returns hedging_pnl: array of length n_paths, the final P&L of the
    (short option + hedge portfolio) position at T. A perfect hedge
    under a correctly-specified model and continuous rehedging would
    have hedging_pnl == 0 on every path; here it's the sum of
    (a) discretization error from finite rehedge_every,
    (b) volatility-model-mismatch error from using a fixed BS vol
        against true stochastic-vol dynamics, and
    (c) transaction costs.
    """
    if n_steps % rehedge_every != 0:
        raise ValueError(f"n_steps={n_steps} must be a multiple of rehedge_every={rehedge_every}")

    x_paths, v_paths = simulate_paths(S0, r, heston_params, T, n_steps, n_paths, q=q, seed=seed)
    S_paths = np.exp(x_paths)
    dt = T / n_steps

    premium = bs_price(S0, K, T, r, hedge_vol, "call", q=q)
    cash = np.full(n_paths, premium, dtype=np.float64)  # received premium for selling the call
    position = np.zeros(n_paths, dtype=np.float64)      # shares of underlying held

    # Self-financing accounting: cash left over between trades is neither
    # created nor destroyed, it earns/pays the risk-free rate -- omitting
    # this (i.e. treating cash as static between rehedge dates) was an
    # actual bug caught by test_hedging.py: it showed up as a mean hedging
    # PnL stuck around +0.5 even at near-continuous rehedging in the
    # exactly-matched-model case, where it should go to ~0. The size was
    # the tell: roughly (average delta notional) * r * T, i.e. the
    # financing cost/benefit of the stock position, not discretization
    # noise (noise shrinks with rehedge frequency; this didn't).
    t_prev = 0.0
    rehedge_steps = list(range(0, n_steps, rehedge_every))
    for step in rehedge_steps:
        t = step * dt
        if t > t_prev:
            cash *= math.exp(r * (t - t_prev))
        tau = T - t
        S = S_paths[:, step]
        if tau > 1e-8:
            target_delta = bs_delta(S, K, tau, r, hedge_vol, "call", q=q)
        else:
            target_delta = (S > K).astype(np.float64)  # at expiry: 1 if ITM else 0
        trade = target_delta - position
        cost = cost_bps * 1e-4 * np.abs(trade) * S
        cash -= trade * S + cost
        position = target_delta
        t_prev = t
    if T > t_prev:
        cash *= math.exp(r * (T - t_prev))  # accrue interest from the last trade to expiry

    S_T = S_paths[:, -1]
    payoff = np.maximum(S_T - K, 0.0)
    final_portfolio_value = cash + position * S_T - payoff
    # discount to time-0 using the same r used throughout
    hedging_pnl = final_portfolio_value * math.exp(-r * T)
    return hedging_pnl


def hedging_error_summary(pnl):
    return {
        "mean": float(np.mean(pnl)),
        "std": float(np.std(pnl, ddof=1)),
        "stderr": float(np.std(pnl, ddof=1) / math.sqrt(len(pnl))),
        "p05": float(np.percentile(pnl, 5)),
        "p95": float(np.percentile(pnl, 95)),
    }
