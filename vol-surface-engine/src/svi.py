"""
SVI (Gatheral 2004) and SSVI (Gatheral & Jacquier 2014) implied-vol
surface parameterizations, fit to total variance w(k, T) = sigma_impl^2 * T
as a function of log-moneyness k = ln(K/F).

Two fitting strategies are implemented and compared empirically in
scripts/run_surface_experiment.py:

  1. Raw SVI, one independent fit per expiry slice -- flexible, fits
     each slice's shape well, but nothing stops adjacent slices from
     crossing (calendar arbitrage): a butterfly spread expiring at T1
     could then be worth more than the same butterfly at T2 > T1, which
     is a static arbitrage.
  2. SSVI, one joint fit across ALL expiries at once, using a power-law
     ATM-total-variance curve theta(T) and a single (rho, eta, gamma)
     shared across the whole surface -- more constrained (worse
     per-slice RMSE), but calendar-arbitrage-free by construction under
     a checkable parameter condition (Gatheral-Jacquier 2014, Theorem
     4.1 / Corollary 4.2).

The point of building both is the tradeoff itself: flexibility (raw
SVI) vs. a global no-arbitrage guarantee (SSVI) is a real, practitioner-
relevant design decision, not a toy exercise.
"""
import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Raw SVI: w(k) = a + b*(rho*(k - m) + sqrt((k - m)^2 + sigma^2))
# ---------------------------------------------------------------------------

def raw_svi_total_variance(k, params):
    """params = (a, b, rho, m, sigma). Vectorized over k (array or scalar)."""
    a, b, rho, m, sigma = params
    k = np.asarray(k, dtype=np.float64)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def _svi_bounds():
    # a: level (can be negative, w just needs to stay >= 0 on the fitted range)
    # b >= 0: negative b would make wings decrease with |k|, unphysical
    # |rho| < 1: correlation-like skew parameter
    # m: any real (horizontal shift)
    # sigma > 0: controls ATM curvature
    lo = [-1.0, 1e-6, -0.999, -2.0, 1e-4]
    hi = [1.0, 5.0, 0.999, 2.0, 2.0]
    return lo, hi


def fit_svi_slice(k, w_market, x0=None, weights=None):
    """
    Least-squares fit of raw SVI to one expiry slice's observed total
    variance. Returns (params, rmse_in_vol_terms) -- rmse is reported in
    (implied-vol) units, not total-variance units, since that's what's
    actually interpretable ("the fit is accurate to 0.3 vol points").
    Caller supplies T separately to convert back to iv = sqrt(w/T).
    """
    k = np.asarray(k, dtype=np.float64)
    w_market = np.asarray(w_market, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(k)

    if x0 is None:
        a0 = max(w_market.min() * 0.9, 1e-6)
        b0 = 0.1
        x0 = [a0, b0, -0.3, 0.0, 0.3]

    lo, hi = _svi_bounds()

    def resid(params):
        model_w = raw_svi_total_variance(k, params)
        return weights * (model_w - w_market)

    res = least_squares(resid, x0=x0, bounds=(lo, hi), method="trf", max_nfev=5000)
    fitted_w = raw_svi_total_variance(k, res.x)
    # RMSE in vol terms needs T; caller (surface.py) divides by T there.
    return tuple(res.x), fitted_w


def durrleman_g(k, params, h=1e-4):
    """
    Durrleman (2004) function g(k): the SVI slice is free of butterfly
    (strike-space, i.e. calendar-fixed) arbitrage on a range of k iff
    g(k) >= 0 there. Computed via central finite differences of w(k) --
    exact closed-form derivatives exist for raw SVI, but the numerical
    version is what actually gets checked against an SSVI closed form
    it DOESN'T share, so it's a genuinely independent check rather than
    two paths through the same algebra.

        g(k) = (1 - k*w'/(2w))^2 - (w'/2)^2 * (1/w + 1/4) + w''/2
    """
    k = np.asarray(k, dtype=np.float64)
    w = raw_svi_total_variance(k, params)
    w_up = raw_svi_total_variance(k + h, params)
    w_dn = raw_svi_total_variance(k - h, params)
    wp = (w_up - w_dn) / (2 * h)
    wpp = (w_up - 2 * w + w_dn) / (h ** 2)

    term1 = (1 - k * wp / (2 * w)) ** 2
    term2 = (wp / 2) ** 2 * (1 / w + 0.25)
    return term1 - term2 + wpp / 2


def check_butterfly_arbitrage(params, k_grid):
    """Returns (is_arb_free, min_g). is_arb_free requires g(k) >= 0 (up to
    float tolerance) across the whole grid -- a single negative point is
    a real static arbitrage (a butterfly spread with negative cost)."""
    g = durrleman_g(k_grid, params)
    min_g = float(g.min())
    return min_g >= -1e-6, min_g


# ---------------------------------------------------------------------------
# SSVI: w(k, theta) = theta/2 * (1 + rho*phi(theta)*k + sqrt((phi(theta)*k+rho)^2 + (1-rho^2)))
# power-law phi(theta) = eta / theta^gamma
# ---------------------------------------------------------------------------

def ssvi_total_variance(k, theta, rho, eta, gamma_power):
    """theta = ATM total variance for this slice's expiry (theta(T) = sigma_atm^2 * T)."""
    k = np.asarray(k, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    phi = eta * theta ** (-gamma_power)
    inner = phi * k + rho
    return 0.5 * theta * (1 + rho * phi * k + np.sqrt(inner ** 2 + (1 - rho ** 2)))


def ssvi_calendar_arbitrage_free(rho, eta, gamma_power):
    """
    Sufficient condition for NO calendar arbitrage across the whole
    surface, for ALL theta > 0 simultaneously (Gatheral & Jacquier 2014,
    Corollary 4.2, power-law case): with phi(theta) = eta*theta^-gamma,
        0 < gamma_power < 1/2   and   eta * (1 + |rho|) <= 2.
    This is checked directly on the fitted (rho, eta, gamma) rather than
    on a finite k-grid -- it's a global guarantee, not a spot-check.
    """
    cond1 = 0 < gamma_power < 0.5
    cond2 = eta * (1 + abs(rho)) <= 2.0 + 1e-9
    return bool(cond1 and cond2)


def fit_ssvi_surface(theta_by_expiry, k_by_expiry, w_by_expiry):
    """
    Joint fit of (rho, eta, gamma_power) across ALL expiry slices at once
    (theta per-slice is taken as given -- fit directly from each slice's
    own ATM total variance beforehand, see surface.py), minimizing total
    squared error in total-variance space across every (k, w) point from
    every expiry pooled together.

    theta_by_expiry: array of per-expiry ATM total variances, len = n_expiries
    k_by_expiry, w_by_expiry: lists of arrays, one per expiry
    """
    theta_by_expiry = np.asarray(theta_by_expiry, dtype=np.float64)

    def resid(params):
        rho, eta, gamma_power = params
        out = []
        for theta, k, w in zip(theta_by_expiry, k_by_expiry, w_by_expiry):
            model_w = ssvi_total_variance(k, theta, rho, eta, gamma_power)
            out.append(model_w - w)
        return np.concatenate(out)

    x0 = [-0.5, 1.0, 0.3]
    lo = [-0.999, 1e-4, 1e-4]
    hi = [0.999, 2.0, 0.499]
    res = least_squares(resid, x0=x0, bounds=(lo, hi), method="trf", max_nfev=5000)
    rho, eta, gamma_power = res.x
    return rho, eta, gamma_power
