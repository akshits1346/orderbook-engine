"""
Synthetic data generators for two different stat-arb setups, each
matching a classic model exactly so correctness is checkable against
ground truth (see tests/test_data_gen.py):

  1. generate_cointegrated_pair: a single pair built from a shared I(1)
     stochastic trend plus a stationary OU spread -- the textbook
     Engle-Granger construction, with an optional slowly time-varying
     hedge ratio (the reason a Kalman filter earns its keep over static
     OLS: see src/kalman.py).
  2. generate_factor_basket: a basket of N assets driven by K common
     factors plus OU-mean-reverting idiosyncratic residuals -- the
     assumption Avellaneda & Lee's PCA statistical arbitrage (2010) is
     built on, so a PCA fit to this data recovering the true factor
     loadings and residual OU dynamics is a real correctness check on
     src/pca_signal.py, not just "the code runs."
"""
import numpy as np


def generate_cointegrated_pair(n=1000, hedge_ratio=1.5, theta=0.05, spread_vol=0.3,
                                trend_vol=1.0, time_varying_hedge=False,
                                hedge_ratio_amplitude=0.4, hedge_ratio_period=250,
                                seed=None):
    """
    log(P1)_t = X_t                         (X_t: random walk, the shared I(1) trend)
    log(P2)_t = beta_t * X_t + spread_t      (spread_t: OU, mean-reverting -> stationary)

    theta: OU mean-reversion speed (per step) for the spread.
    spread_vol: OU diffusion vol.
    trend_vol: per-step std of the random-walk trend's increments.
    time_varying_hedge: if True, beta_t = hedge_ratio + amplitude*sin(2*pi*t/period)
        instead of a constant -- static OLS hedge-ratio estimation is
        provably suboptimal here since it estimates one number for a
        target that's actually moving; a Kalman filter that re-estimates
        beta_t each step is not (see run_pairs_experiment.py).

    Returns dict with keys: log_p1, log_p2, p1, p2, true_beta (array,
    same length, constant or time-varying), true_spread.
    """
    rng = np.random.default_rng(seed)
    dX = rng.normal(0, trend_vol, n)
    X = np.cumsum(dX)

    if time_varying_hedge:
        t = np.arange(n)
        true_beta = hedge_ratio + hedge_ratio_amplitude * np.sin(2 * np.pi * t / hedge_ratio_period)
    else:
        true_beta = np.full(n, hedge_ratio)

    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] + theta * (0.0 - spread[i - 1]) + spread_vol * rng.normal()

    log_p1 = X
    log_p2 = true_beta * X + spread

    return {
        "log_p1": log_p1, "log_p2": log_p2,
        "p1": np.exp(log_p1 / 20 + 4.0),  # rescaled to a plausible $50ish price level
        "p2": np.exp(log_p2 / 20 + 4.0),
        "true_beta": true_beta, "true_spread": spread,
    }


def generate_factor_basket(n_assets=20, n_days=750, n_factors=3, factor_vol=0.01,
                            idio_theta=0.03, idio_vol=0.008, loading_scale=1.0,
                            regime_change_at=None, seed=None):
    """
    Daily returns: R_{i,t} = sum_k beta_{i,k} * F_{k,t} + d(residual_i)_t
    where residual_i is an OU process in LOG-PRICE space (so its
    INCREMENT is what contributes to the return): mean-reverting,
    stationary by construction. Factor returns F_{k,t} are iid N(0,
    factor_vol^2) (independent factors, the simplifying assumption PCA
    on a correlation matrix is designed to recover). beta_{i,k} ~
    N(0, loading_scale^2), fixed per asset for the whole sample UNLESS
    regime_change_at is set.

    regime_change_at: if given (a day index), every asset's factor
    loadings are independently redrawn from the SAME N(0, loading_scale^2)
    distribution starting at that day -- a structural break (sector
    rotation, a business mix change) rather than a slow drift. This is
    the scenario that makes an in-sample/full-history model fit
    meaningfully different from a walk-forward one: a full-sample PCA +
    beta fit averages across both regimes (blending stale and current
    loadings) AND gets to "see" the post-break OU parameters in advance,
    while a walk-forward fit using only trailing data is still hedging
    with the OLD loadings right after the break and has to relearn the
    new regime's OU parameters from scratch, same as a real desk would.
    See backtest.py's run_in_sample_backtest / run_walk_forward_backtest
    and README's lookahead-bias finding.

    Returns dict: returns (n_days x n_assets), true_betas (n_assets x
    n_factors, POST-break loadings if regime_change_at is set -- pre-break
    loadings aren't separately returned, they're a generation-only detail),
    factor_returns (n_days x n_factors), residuals (n_days x n_assets, the
    OU increments actually used -- ground truth for checking PCA recovery).
    """
    rng = np.random.default_rng(seed)
    true_betas = rng.normal(0, loading_scale, (n_assets, n_factors))
    factor_returns = rng.normal(0, factor_vol, (n_days, n_factors))

    residual_level = np.zeros(n_assets)
    residuals = np.zeros((n_days, n_assets))
    for t in range(n_days):
        shock = idio_vol * rng.normal(size=n_assets)
        d_level = -idio_theta * residual_level + shock
        residuals[t] = d_level
        residual_level = residual_level + d_level

    if regime_change_at is not None:
        betas_pre = true_betas
        betas_post = rng.normal(0, loading_scale, (n_assets, n_factors))
        systematic = np.empty((n_days, n_assets))
        systematic[:regime_change_at] = factor_returns[:regime_change_at] @ betas_pre.T
        systematic[regime_change_at:] = factor_returns[regime_change_at:] @ betas_post.T
        true_betas = betas_post
    else:
        systematic = factor_returns @ true_betas.T

    returns = systematic + residuals

    return {
        "returns": returns, "true_betas": true_betas,
        "factor_returns": factor_returns, "residuals": residuals,
    }
