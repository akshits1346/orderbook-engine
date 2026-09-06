"""
A from-scratch Kalman filter for a time-varying pairs-trading hedge
ratio (the classic formulation in e.g. Chan, "Algorithmic Trading",
ch. 5) -- implemented directly rather than pulled from a library, since
the whole point of using it here is understanding exactly what it buys
over static OLS: a hedge ratio that ADAPTS each step, which matters
precisely when the true hedge ratio moves over time (see
data_gen.generate_cointegrated_pair(time_varying_hedge=True) and
run_pairs_experiment.py for the direct comparison).

State-space model:
    theta_t = [alpha_t, beta_t]^T           (regression intercept + slope)
    theta_t = theta_{t-1} + w_t,  w_t ~ N(0, Q)      (random-walk state)
    y_t     = H_t @ theta_t + v_t,  v_t ~ N(0, R)     (H_t = [1, x_t])

y_t = log(P2)_t, x_t = log(P1)_t. R is the observation-noise variance
(the "instantaneous" part of the spread); Q controls how fast
alpha/beta are allowed to drift -- both estimated from data via
kalman_hedge_ratio's delta/R parameters, not hand-tuned per pair.
"""
import numpy as np


def kalman_hedge_ratio(x, y, delta=1e-4, R_init=1e-3, adapt_R=True, R_halflife=20):
    """
    Runs the 2-state (alpha, beta) Kalman filter forward through the
    series, one step at a time (no lookahead: theta_t's estimate at
    time t only ever uses data up to and including t).

    delta: controls Q = delta/(1-delta) * I -- larger delta lets the
        hedge ratio drift faster (standard Chan-style parameterization,
        so this project's one free "how fast can beta move" knob has a
        commonly-used interpretable form instead of an ad hoc Q).
    R_init: initial observation-noise variance estimate.
    adapt_R: if True, R is re-estimated online as an EWMA of the
        squared innovation (a simple adaptive-noise variant) rather
        than held fixed for the whole series -- real spread variance is
        not constant, and a fixed R systematically over/under-weights
        new information once realized noise drifts away from the
        initial guess.

    Returns dict: alpha, beta, innovation (e_t = y_t - H_t@theta_pred,
    the ACTUAL trading signal's numerator), innovation_var (S_t, so
    z-score = innovation / sqrt(innovation_var)), theta_cov_trace (for
    diagnosing filter convergence).
    """
    n = len(x)
    Q = (delta / (1 - delta)) * np.eye(2)

    theta = np.array([0.0, 1.0])  # [alpha, beta] initial guess
    P = np.eye(2) * 1.0

    alphas = np.zeros(n)
    betas = np.zeros(n)
    innovations = np.zeros(n)
    innovation_vars = np.zeros(n)
    cov_trace = np.zeros(n)

    R = R_init
    ewma_lambda = np.log(2) / R_halflife  # EWMA decay matching the stated half-life

    for t in range(n):
        H = np.array([1.0, x[t]])

        # predict
        theta_pred = theta  # random walk: no deterministic drift term
        P_pred = P + Q

        # update
        y_pred = H @ theta_pred
        e = y[t] - y_pred
        S = H @ P_pred @ H + R
        K = P_pred @ H / S

        theta = theta_pred + K * e
        P = P_pred - np.outer(K, H) @ P_pred

        if adapt_R:
            R = (1 - ewma_lambda) * R + ewma_lambda * (e ** 2)

        alphas[t], betas[t] = theta
        innovations[t] = e
        innovation_vars[t] = S
        cov_trace[t] = np.trace(P)

    return {
        "alpha": alphas, "beta": betas,
        "innovation": innovations, "innovation_var": innovation_vars,
        "z_score": innovations / np.sqrt(innovation_vars),
        "theta_cov_trace": cov_trace,
    }
