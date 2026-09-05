"""
Generate a synthetic European option chain by pricing under Heston and
solving each price back out to Black-Scholes implied vol. This is the
project's stand-in for a real listed-options snapshot (see the
Honest limitations section of the README for why: this sandbox's
outbound network access is an allowlist of package registries, not
general internet, so a live options data vendor is unreachable here --
exactly the same constraint the sibling orderbook-engine project
documents for LOBSTER data).

Using Heston to GENERATE the "market" and then fitting SVI/SSVI back
to it is not circular: Heston and SVI are different, independently
motivated parameterizations of the same object (an implied vol
surface), and nothing here injects Heston structure into the SVI
fitting code -- it fits generic (k, w) points, and would fit real
market points identically. The generator gives every downstream
arbitrage check ground truth to check itself against too: a Heston
surface is known to be arbitrage-free (it comes from a single
consistent risk-neutral model), so if the calendar-arbitrage check
below fires on independently-fit slices, that's the FITTING procedure
introducing arbitrage, not real structure in the "market" data --
precisely the finding this project reports.
"""
import math
import numpy as np
from black_scholes import implied_vol
from heston import heston_price, HestonParams


def generate_synthetic_chain(heston_params: HestonParams, S0=100.0, r=0.02,
                              expiries=(0.083, 0.25, 0.5, 1.0, 2.0),
                              moneyness_grid=None, q=0.0):
    """
    Returns a list of dict rows: {T, K, F, k (log-moneyness vs forward),
    price, iv}. moneyness_grid is in log-forward-moneyness units
    (k = ln(K/F)); default spans a realistic +/-40% strike range.
    """
    if moneyness_grid is None:
        moneyness_grid = np.linspace(-0.6, 0.6, 13)

    rows = []
    for T in expiries:
        F = S0 * math.exp((r - q) * T)
        for k in moneyness_grid:
            K = F * math.exp(k)
            option_type = "call" if K >= F else "put"
            price = heston_price(S0, K, T, r, heston_params, option_type, q=q)
            try:
                iv = implied_vol(price, S0, K, T, r, option_type, q=q)
            except ValueError:
                continue  # numerically outside the no-arb band; skip this point
            rows.append({"T": T, "K": K, "F": F, "k": k, "price": price,
                          "option_type": option_type, "iv": iv})
    return rows


def rows_by_expiry(rows):
    """Group generate_synthetic_chain's flat row list by T, sorted by k."""
    by_T = {}
    for row in rows:
        by_T.setdefault(row["T"], []).append(row)
    for T in by_T:
        by_T[T].sort(key=lambda r: r["k"])
    return dict(sorted(by_T.items()))
