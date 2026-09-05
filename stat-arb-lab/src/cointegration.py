"""
Engle-Granger two-step cointegration test and a Johansen-test wrapper.
Both operate on log-prices (the standard convention -- cointegration in
log-price space corresponds to a stationary log-return spread, i.e. a
stable long-run ratio between the two assets' price LEVELS, not just
their returns).
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def engle_granger_test(log_p1, log_p2, adf_regression="c"):
    """
    Step 1: OLS log_p2 ~ alpha + beta*log_p1 -> static hedge ratio beta.
    Step 2: ADF test on the OLS residual (the "spread"). Rejecting the
    unit-root null on the RESIDUAL (not on either series alone) is what
    cointegration actually means.

    Returns dict: beta, alpha, residual (the fitted spread), adf_stat,
    adf_pvalue, is_cointegrated (pvalue < 0.05, the conventional cutoff
    -- NOT proof, just the standard convention, called out explicitly
    because it matters later in the walk-forward robustness checks).
    """
    X = sm.add_constant(log_p1)
    model = sm.OLS(log_p2, X).fit()
    alpha, beta = model.params
    residual = log_p2 - (alpha + beta * log_p1)

    adf_result = adfuller(residual, regression=adf_regression)
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]

    return {
        "beta": float(beta), "alpha": float(alpha), "residual": residual,
        "adf_stat": float(adf_stat), "adf_pvalue": float(adf_pvalue),
        "is_cointegrated": bool(adf_pvalue < 0.05),
    }


def johansen_test(log_prices, det_order=0, k_ar_diff=1):
    """
    log_prices: (n_obs, n_series) array (n_series >= 2). Unlike
    Engle-Granger, Johansen doesn't require picking a "dependent"
    variable and can detect MULTIPLE cointegrating relationships among
    more than 2 series -- the natural test for the PCA/basket case,
    where Engle-Granger's one-regression-per-pair approach doesn't
    generalize cleanly.

    Returns dict: trace_stats, trace_crit_vals (90/95/99%), n_coint_95
    (number of cointegrating relationships for which the trace stat
    exceeds the 95% critical value, testing r=0,1,2,... in order and
    stopping at the first non-rejection -- the standard sequential
    procedure), eigenvectors (columns are candidate cointegrating
    vectors, ordered by eigenvalue).
    """
    result = coint_johansen(log_prices, det_order, k_ar_diff)
    trace_stats = result.lr1
    crit_vals_95 = result.cvt[:, 1]  # columns are 90%, 95%, 99%

    n_coint_95 = 0
    for stat, crit in zip(trace_stats, crit_vals_95):
        if stat > crit:
            n_coint_95 += 1
        else:
            break

    return {
        "trace_stats": trace_stats, "crit_vals_95": crit_vals_95,
        "n_coint_95": n_coint_95, "eigenvectors": result.evec,
        "eigenvalues": result.eig,
    }
