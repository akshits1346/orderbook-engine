"""
Sanity checks for the delta-hedging simulator: when the "true" dynamics
match the hedging model exactly (xi=0, so Heston collapses to constant-
volatility Black-Scholes) and the option is priced at that same
constant vol, delta-hedging should replicate the payoff essentially
perfectly, with hedging-error variance shrinking as rehedge frequency
increases -- the textbook Black-Scholes replication result. This is
the correctness baseline the model-mismatch experiment (Heston truth,
BS-vol hedge) in the README is measured against.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from heston import HestonParams
from hedging import simulate_delta_hedge, hedging_error_summary

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    S0, K, T, r = 100.0, 100.0, 0.5, 0.02
    hedge_vol = 0.2
    # xi=0 collapses Heston to constant-variance GBM at v0=hedge_vol^2:
    # matches the hedging model exactly, so this isolates discretization
    # error from model-mismatch error.
    p_matched = HestonParams(kappa=1.0, theta=hedge_vol ** 2, xi=1e-6, rho=0.0, v0=hedge_vol ** 2)

    prev_std = None
    stds = []
    for rehedge_every in [50, 10, 2, 1]:
        pnl = simulate_delta_hedge(S0, K, T, r, p_matched, hedge_vol,
                                    n_steps=100, rehedge_every=rehedge_every,
                                    n_paths=40_000, cost_bps=0.0, seed=1)
        summary = hedging_error_summary(pnl)
        stds.append(summary["std"])
        check(abs(summary["mean"]) < 0.15,
              f"matched-model hedge, rehedge_every={rehedge_every}: mean PnL ~0, got {summary['mean']:.4f}")

    # Hedging error std should shrink monotonically as rehedging gets more frequent
    check(all(stds[i] > stds[i + 1] for i in range(len(stds) - 1)),
          f"hedging error std decreases with rehedge frequency: {[f'{s:.4f}' for s in stds]}")

    # Continuous-ish rehedging (every step) should be very tight
    check(stds[-1] < 0.5, f"near-continuous rehedging gives small hedging error std, got {stds[-1]:.4f}")

    # Zero-cost, zero-trade sanity: cost_bps should strictly worsen mean PnL (costs are a pure drag)
    pnl_no_cost = simulate_delta_hedge(S0, K, T, r, p_matched, hedge_vol,
                                        n_steps=100, rehedge_every=1, n_paths=20_000,
                                        cost_bps=0.0, seed=2)
    pnl_with_cost = simulate_delta_hedge(S0, K, T, r, p_matched, hedge_vol,
                                          n_steps=100, rehedge_every=1, n_paths=20_000,
                                          cost_bps=20.0, seed=2)
    check(np.mean(pnl_with_cost) < np.mean(pnl_no_cost),
          f"transaction costs reduce mean hedging PnL: {np.mean(pnl_with_cost):.4f} < {np.mean(pnl_no_cost):.4f}")

    print()
    if failures == 0:
        print("All hedging checks passed.")
        return 0
    print(f"{failures} check(s) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
