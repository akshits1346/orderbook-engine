"""
Black-Scholes-Merton pricing, Greeks, and an implied-vol solver.

This is the "quoting convention" layer: everything upstream (Heston
prices, real market quotes) gets translated into Black-Scholes implied
vol before it touches the SVI/SSVI surface code, because strikes and
vols are what practitioners actually look at, not raw prices. Getting
this layer exactly right matters -- every downstream arbitrage check is
only as correct as the implied vols it's checking.
"""
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

N = norm.cdf
n = norm.pdf


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """
    S, K, sigma may be scalars or numpy arrays (T, r, q are treated as
    scalars throughout this module -- the hedging engine's use case is
    "many paths, one fixed maturity/rate/vol per rehedge date", never a
    vectorized T). Uses numpy (not math) specifically so bs_delta/
    bs_price etc. work unmodified on an array of simulated spot paths,
    which is exactly how hedging.py calls them.
    """
    if np.any(np.asarray(T) <= 0) or np.any(np.asarray(sigma) <= 0):
        raise ValueError(f"T and sigma must be positive, got T={T}, sigma={sigma}")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """European option price under Black-Scholes-Merton with a continuous dividend yield q."""
    if T <= 0:
        payoff = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return payoff
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)
    elif option_type == "put":
        return K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_delta(S, K, T, r, sigma, option_type="call", q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return math.exp(-q * T) * N(d1)
    return math.exp(-q * T) * (N(d1) - 1.0)


def bs_gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return math.exp(-q * T) * n(d1) / (S * sigma * math.sqrt(T))


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Per unit of volatility (i.e. d(price)/d(sigma), not per 1% vol point)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * n(d1) * math.sqrt(T)


def bs_theta(S, K, T, r, sigma, option_type="call", q=0.0):
    """Per year (i.e. d(price)/d(T) with T counting down), not per calendar day."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    term1 = -S * math.exp(-q * T) * n(d1) * sigma / (2 * math.sqrt(T))
    if option_type == "call":
        term2 = -r * K * math.exp(-r * T) * N(d2)
        term3 = q * S * math.exp(-q * T) * N(d1)
    else:
        term2 = r * K * math.exp(-r * T) * N(-d2)
        term3 = -q * S * math.exp(-q * T) * N(-d1)
    return term1 + term2 + term3


def bs_rho(S, K, T, r, sigma, option_type="call", q=0.0):
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * math.exp(-r * T) * N(d2)
    return -K * T * math.exp(-r * T) * N(-d2)


def intrinsic_value(S, K, T, r, option_type="call", q=0.0):
    """Lower no-arbitrage bound: the value if sigma -> 0 (forward intrinsic, discounted)."""
    F = S * math.exp((r - q) * T)
    disc = math.exp(-r * T)
    if option_type == "call":
        return disc * max(F - K, 0.0)
    return disc * max(K - F, 0.0)


def implied_vol(price, S, K, T, r, option_type="call", q=0.0, lo=1e-6, hi=8.0):
    """
    Invert bs_price for sigma via Brent's method (bracketed, so it can't
    diverge the way Newton's method can near-atm-but-far-from-x0). Raises
    ValueError if `price` is outside the no-arbitrage band
    [intrinsic, S*exp(-qT)] (call) / [intrinsic, K*exp(-rT)] (put) --
    no sigma reproduces a price outside that band, so silently returning
    a garbage number would be worse than failing loudly.
    """
    intrinsic = intrinsic_value(S, K, T, r, option_type, q)
    upper_bound = S * math.exp(-q * T) if option_type == "call" else K * math.exp(-r * T)
    eps = 1e-10
    if price < intrinsic - eps or price > upper_bound + eps:
        raise ValueError(
            f"price {price} outside no-arbitrage band [{intrinsic}, {upper_bound}] "
            f"for {option_type} S={S} K={K} T={T}"
        )
    price = min(max(price, intrinsic), upper_bound)

    def obj(sigma):
        return bs_price(S, K, T, r, sigma, option_type, q) - price

    f_lo, f_hi = obj(lo), obj(hi)
    if f_lo > 0:
        return lo
    if f_hi < 0:
        return hi
    return brentq(obj, lo, hi, xtol=1e-10, rtol=1e-12, maxiter=200)
