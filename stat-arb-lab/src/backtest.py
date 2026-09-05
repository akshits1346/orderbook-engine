"""
Walk-forward backtest for the PCA stat-arb signal. "Walk-forward" here
means literally that: at each rebalance date t, PCA/factor-loadings/OU
parameters are refit ONLY on the trailing window `returns[t-window:t]`
(a Python slice that excludes t itself) -- the decision made for day t
never sees day t's own return or anything after it. Day t's REALIZED
return is only used afterward, to mark that decision's P&L, exactly as
a live trading system would experience it one day at a time.

Each held name's residual trade is expressed as a fully-hedged N-asset
portfolio weight vector: long $1 of asset i, short beta_{i,k} dollars
of each eigenportfolio k (which is itself a weighted combination of
ALL N assets via Q). Aggregating these across every currently-held name
gives one target weight vector per day -- this is what makes the
strategy genuinely dollar-neutral against the fitted factors, not just
"long this stock, short some proxy index."
"""
import numpy as np
from pca_signal import compute_signals, fit_pca_factors, fit_asset_residual, fit_ou_params


def run_walk_forward_backtest(returns, n_factors=3, train_window=252, rebalance_every=5,
                               s_open=1.25, s_close=0.5, s_stop=3.5, position_size=1.0,
                               cost_bps=5.0, min_kappa=252 / 60):
    """
    returns: (T, N) daily returns.
    s_open: |s-score| threshold to OPEN a position (Avellaneda-Lee use ~1.25).
    s_close: |s-score| threshold to CLOSE a position back toward 0.
    s_stop: |s-score| threshold for a hard stop-loss (the residual diverged
        further than the mean-reversion model expected -- exit rather than
        keep believing the model).
    position_size: dollar size per held name's $1-long leg (gross exposure
        scales with how many names are simultaneously held, not fixed).
    cost_bps: per-dollar-traded transaction cost, charged on portfolio
        weight CHANGES (turnover), not on gross exposure.

    Returns dict: daily_pnl, daily_turnover, daily_gross_exposure, state_history
    (T x N, the long/flat/short state of every name each day -- for
    diagnosing IN-sample vs OUT-of-sample behavior), s_score_history.
    """
    T, N = returns.shape
    if train_window >= T:
        raise ValueError(f"train_window={train_window} must be < T={T}")

    state = np.zeros(N, dtype=int)
    W_prev = np.zeros(N)

    daily_pnl = np.zeros(T - train_window)
    daily_turnover = np.zeros(T - train_window)
    daily_gross = np.zeros(T - train_window)
    state_history = np.zeros((T - train_window, N), dtype=int)
    s_score_history = np.full((T - train_window, N), np.nan)

    last_s_scores = np.zeros(N)
    last_betas = np.zeros((N, n_factors))
    last_Q = np.zeros((N, n_factors))
    last_tradeable = np.zeros(N, dtype=bool)
    W_t = W_prev

    for idx, t in enumerate(range(train_window, T)):
        if (t - train_window) % rebalance_every == 0:
            window = returns[t - train_window:t]  # strictly data before t: no lookahead
            last_s_scores, last_betas, last_Q, last_tradeable = compute_signals(
                window, n_factors, min_kappa)

            for i in range(N):
                if not last_tradeable[i]:
                    state[i] = 0
                    continue
                s = last_s_scores[i]
                if state[i] == 0:
                    if s < -s_open:
                        state[i] = 1
                    elif s > s_open:
                        state[i] = -1
                elif state[i] == 1:
                    if s > -s_close or abs(s) > s_stop:
                        state[i] = 0
                elif state[i] == -1:
                    if s < s_close or abs(s) > s_stop:
                        state[i] = 0

            W_t = np.zeros(N)
            for i in range(N):
                if state[i] != 0:
                    w_i = np.zeros(N)
                    w_i[i] = 1.0
                    w_i -= last_Q @ last_betas[i]
                    W_t = W_t + state[i] * position_size * w_i

        turnover = np.abs(W_t - W_prev).sum()
        cost = cost_bps * 1e-4 * turnover
        pnl_t = W_t @ returns[t] - cost

        daily_pnl[idx] = pnl_t
        daily_turnover[idx] = turnover
        daily_gross[idx] = np.abs(W_t).sum()
        state_history[idx] = state
        s_score_history[idx] = last_s_scores

        W_prev = W_t

    return {
        "daily_pnl": daily_pnl, "daily_turnover": daily_turnover,
        "daily_gross_exposure": daily_gross, "state_history": state_history,
        "s_score_history": s_score_history,
    }


def run_in_sample_backtest(returns, n_factors=3, s_open=1.25, s_close=0.5, s_stop=3.5,
                            position_size=1.0, cost_bps=5.0, min_kappa=252 / 60,
                            burn_in=20):
    """
    THE LOOKAHEAD-BIASED CONTROL. Fits PCA loadings (Q), per-asset factor
    betas, AND each asset's OU parameters (m, sigma_eq, kappa) ONCE on
    the ENTIRE return series -- including, for every trading day in the
    "backtest" below, data from that day's own future. It then walks
    forward day by day using the SAME entry/exit state machine as
    run_walk_forward_backtest, scoring each day's cumulative residual
    against those full-sample-fitted (m, sigma_eq), and marking P&L
    against that day's ACTUAL return only (so it isn't literally
    peeking at tomorrow's return -- the lookahead here is specifically
    in the model's fitted PARAMETERS, which is the realistic, easy-to-
    miss version of this mistake: "I estimated my mean/vol/hedge-ratio
    from my whole sample, then backtested on that same sample," not the
    much more obvious "my backtest uses tomorrow's price today").

    Returns dict: daily_pnl, daily_turnover, state_history, s_score_history
    -- same shape/meaning as run_walk_forward_backtest's outputs, so the
    two are directly comparable metric-for-metric.
    """
    T, N = returns.shape
    factor_returns, Q, asset_vols = fit_pca_factors(returns, n_factors)

    betas = np.zeros((N, n_factors))
    residuals_all = np.zeros((T, N))
    ou_params = []
    for i in range(N):
        beta, intercept, residuals = fit_asset_residual(returns[:, i], factor_returns)
        betas[i] = beta
        residuals_all[:, i] = residuals
        ou_params.append(fit_ou_params(residuals))

    cum_residuals = np.cumsum(residuals_all, axis=0)  # (T, N), X_i,t for every t at once

    state = np.zeros(N, dtype=int)
    W_prev = np.zeros(N)
    n_days = T - burn_in
    daily_pnl = np.zeros(n_days)
    daily_turnover = np.zeros(n_days)
    state_history = np.zeros((n_days, N), dtype=int)
    s_score_history = np.full((n_days, N), np.nan)

    for idx, t in enumerate(range(burn_in, T)):
        for i in range(N):
            ou = ou_params[i]
            if not ou["is_mean_reverting"] or ou["sigma_eq"] < 1e-12:
                state[i] = 0
                continue
            s = (cum_residuals[t, i] - ou["m"]) / ou["sigma_eq"]
            s_score_history[idx, i] = s
            if state[i] == 0:
                if s < -s_open:
                    state[i] = 1
                elif s > s_open:
                    state[i] = -1
            elif state[i] == 1:
                if s > -s_close or abs(s) > s_stop:
                    state[i] = 0
            elif state[i] == -1:
                if s < s_close or abs(s) > s_stop:
                    state[i] = 0

        W_t = np.zeros(N)
        for i in range(N):
            if state[i] != 0:
                w_i = np.zeros(N)
                w_i[i] = 1.0
                w_i -= Q @ betas[i]
                W_t = W_t + state[i] * position_size * w_i

        turnover = np.abs(W_t - W_prev).sum()
        cost = cost_bps * 1e-4 * turnover
        daily_pnl[idx] = W_t @ returns[t] - cost
        daily_turnover[idx] = turnover
        state_history[idx] = state
        W_prev = W_t

    return {
        "daily_pnl": daily_pnl, "daily_turnover": daily_turnover,
        "state_history": state_history, "s_score_history": s_score_history,
    }
