"""
PCA-based statistical arbitrage signal generation, following Avellaneda
& Lee (2010), "Statistical Arbitrage in the U.S. Equities Market":

  1. Extract systematic risk factors via PCA on the CORRELATION matrix
     of standardized returns -- eigenportfolio k's return is
     F_k,t = sum_i Q_{i,k} * R_{i,t}, with weights Q_{i,k} = v_{i,k}/sigma_i
     (eigenvector entry scaled by asset i's own return vol -- this is
     the Avellaneda-Lee convention, not an arbitrary normalization: it's
     what makes Q a valid dollar-neutral portfolio weight vector, since
     PCA itself only operates on standardized returns).
  2. Regress each asset's own return on the factor returns over a
     trailing window -> residual r_i,t and factor loadings beta_{i,k}.
  3. Cumulatively sum the residuals into an auxiliary process X_i,t and
     fit a discrete AR(1) to it -- X_n = a + b*X_{n-1} + xi_n is the
     discretization of an OU process with kappa = -ln(b)/dt (dt=1/252
     for daily data), long-run mean m = a/(1-b), and equilibrium std
     sigma_eq = sqrt(Var(xi)/(1-b^2)).
  4. s-score = (X_i,t - m) / sigma_eq -- a standardized mean-reversion
     signal: trade against a large |s-score|, exit as it reverts to 0.

All of this happens on a TRAILING window only (see backtest.py) so a
walk-forward run never uses information from beyond the current date.
"""
import numpy as np


def fit_pca_factors(returns_window, n_factors):
    """
    returns_window: (T, N) trailing-window returns.
    Returns (factor_returns (T, K), Q (N, K) portfolio weights, asset_vols (N,)).
    """
    asset_vols = returns_window.std(axis=0, ddof=1)
    asset_vols = np.where(asset_vols < 1e-10, 1e-10, asset_vols)
    standardized = returns_window / asset_vols

    corr = np.corrcoef(standardized, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(corr)  # ascending order
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    top_vecs = eigvecs[:, :n_factors]  # (N, K)
    Q = top_vecs / asset_vols[:, None]  # Avellaneda-Lee weighting
    factor_returns = returns_window @ Q  # (T, K)
    return factor_returns, Q, asset_vols


def fit_asset_residual(asset_returns_window, factor_returns_window):
    """OLS asset_returns ~ const + factor_returns over the trailing
    window. Returns (beta (K,), intercept, residuals (T,))."""
    T = len(asset_returns_window)
    X = np.column_stack([np.ones(T), factor_returns_window])
    coeffs, _, _, _ = np.linalg.lstsq(X, asset_returns_window, rcond=None)
    intercept, beta = coeffs[0], coeffs[1:]
    fitted = X @ coeffs
    residuals = asset_returns_window - fitted
    return beta, intercept, residuals


def fit_ou_params(residuals, periods_per_year=252):
    """
    residuals: (T,) trailing-window residuals for one asset. Cumulatively
    sums them into X_t and fits a discrete AR(1) to it. Returns the OU
    parameters only (kappa, m, sigma_eq) -- NOT a score, since callers
    need this split two ways: a walk-forward backtest only ever wants
    the score at the LAST point of a trailing window (see
    fit_ou_and_score below), while the in-sample lookahead control in
    backtest.py deliberately reuses ONE full-sample fit's (m, sigma_eq)
    to score EVERY day in the sample -- the realistic form of lookahead
    bias this project measures (see README): not "seeing future
    returns directly", but estimating a model's distributional
    parameters from data that includes the future, then scoring the
    past against them.

    is_mean_reverting is False when the fitted b is outside (0, 1), i.e.
    no valid mean-reversion speed; Avellaneda-Lee's own recommendation
    is to skip trading such names rather than force a kappa out of a
    nonsensical b.
    """
    X = np.cumsum(residuals)
    X_lag = X[:-1]
    X_now = X[1:]
    n = len(X_now)
    A = np.column_stack([np.ones(n), X_lag])
    coeffs, _, _, _ = np.linalg.lstsq(A, X_now, rcond=None)
    a, b = coeffs
    xi = X_now - A @ coeffs

    if not (0 < b < 1):
        return {"kappa": None, "m": None, "sigma_eq": None, "is_mean_reverting": False}

    kappa = -np.log(b) * periods_per_year
    m = a / (1 - b)
    var_xi = xi.var(ddof=2)
    sigma_eq = np.sqrt(var_xi / (1 - b ** 2))

    return {"kappa": float(kappa), "m": float(m), "sigma_eq": float(sigma_eq),
            "is_mean_reverting": True}


def fit_ou_and_score(residuals, periods_per_year=252):
    """Convenience wrapper: fit_ou_params plus the s-score AT THE LAST
    POINT of the window (what a walk-forward backtest needs: "what's
    the signal today", using only data up to and including today)."""
    ou = fit_ou_params(residuals, periods_per_year)
    if not ou["is_mean_reverting"]:
        return {**ou, "s_score": None}
    X_last = np.cumsum(residuals)[-1]
    s_score = (X_last - ou["m"]) / ou["sigma_eq"] if ou["sigma_eq"] > 1e-12 else 0.0
    return {**ou, "s_score": float(s_score)}


def compute_signals(returns_window, n_factors, min_kappa=252 / 60):
    """
    Full pipeline on one trailing window: PCA -> per-asset residual ->
    OU fit -> s-score, for every asset. min_kappa (default: mean-reversion
    half-life <= ~60 trading days, i.e. kappa >= 252/60) filters out
    names whose fitted mean reversion is too slow to trade on this
    horizon -- exactly the Avellaneda-Lee recommendation to only trade
    "fast enough" residuals.

    Returns: s_scores (N,), betas (N, K), Q (N, K), tradeable (N,) bool mask.
    """
    N = returns_window.shape[1]
    factor_returns, Q, asset_vols = fit_pca_factors(returns_window, n_factors)

    s_scores = np.zeros(N)
    betas = np.zeros((N, n_factors))
    tradeable = np.zeros(N, dtype=bool)

    for i in range(N):
        beta, intercept, residuals = fit_asset_residual(returns_window[:, i], factor_returns)
        betas[i] = beta
        ou = fit_ou_and_score(residuals)
        if ou["is_mean_reverting"] and ou["kappa"] >= min_kappa:
            s_scores[i] = ou["s_score"]
            tradeable[i] = True

    return s_scores, betas, Q, tradeable
