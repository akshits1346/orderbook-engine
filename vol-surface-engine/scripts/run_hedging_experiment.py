"""
Delta-hedging under model mismatch: true dynamics are Heston
(stochastic vol, leverage effect), the hedger only ever computes BS
delta at one fixed vol (the ATM implied vol at inception). Sweeps
rehedge frequency and per-trade transaction cost, and reports the
mean/std of hedging P&L -- this is the tradeoff a real hedging desk
actually faces: more frequent rehedging shrinks variance but costs more.
"""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from heston import HestonParams
from black_scholes import bs_price
from hedging import simulate_delta_hedge, hedging_error_summary


def main():
    S0, K, T, r = 100.0, 100.0, 0.5, 0.02
    p = HestonParams(kappa=1.5, theta=0.045, xi=0.55, rho=-0.75, v0=0.045)
    hedge_vol = math.sqrt(p.v0)  # ATM implied vol proxy at inception
    premium = bs_price(S0, K, T, r, hedge_vol, "call")

    print(f"True dynamics: Heston kappa={p.kappa} theta={p.theta} xi={p.xi} rho={p.rho} v0={p.v0}")
    print(f"Hedge model: Black-Scholes delta at fixed vol={hedge_vol:.4f} "
          f"(ATM vol at inception, never updated)")
    print(f"Option: ATM call, K={K}, T={T}, premium received = {premium:.4f}")
    print()

    n_steps = 250  # ~daily steps over 6 months
    n_paths = 200_000

    print("## Rehedge-frequency sweep (zero transaction costs)")
    print()
    print("| rehedge every N steps | ~calendar days/rehedge | mean PnL | std PnL | stderr |")
    print("|---|---|---|---|---|")
    freq_results = []
    for rehedge_every in [125, 50, 25, 10, 5, 1]:
        pnl = simulate_delta_hedge(S0, K, T, r, p, hedge_vol, n_steps, rehedge_every,
                                    n_paths, cost_bps=0.0, seed=42)
        s = hedging_error_summary(pnl)
        days_per_rehedge = (T * 365) / (n_steps / rehedge_every)
        freq_results.append((rehedge_every, s))
        print(f"| {rehedge_every} | {days_per_rehedge:.1f} | {s['mean']:.4f} | {s['std']:.4f} | {s['stderr']:.4f} |")
    print()

    print("## Transaction-cost sweep (daily rehedging, rehedge_every=1)")
    print()
    print("| cost (bps of notional per trade) | mean PnL | std PnL |")
    print("|---|---|---|")
    for cost_bps in [0.0, 1.0, 5.0, 10.0, 25.0, 50.0]:
        pnl = simulate_delta_hedge(S0, K, T, r, p, hedge_vol, n_steps, rehedge_every=1,
                                    n_paths=n_paths, cost_bps=cost_bps, seed=42)
        s = hedging_error_summary(pnl)
        print(f"| {cost_bps:.1f} | {s['mean']:.4f} | {s['std']:.4f} |")
    print()

    print("## Cost-risk tradeoff: sweeping BOTH rehedge frequency and cost together")
    print("(realistic cost of 5 bps per trade -- does more frequent rehedging still help?)")
    print()
    print("| rehedge every N steps | mean PnL | std PnL |")
    print("|---|---|---|")
    for rehedge_every in [125, 50, 25, 10, 5, 1]:
        pnl = simulate_delta_hedge(S0, K, T, r, p, hedge_vol, n_steps, rehedge_every,
                                    n_paths, cost_bps=5.0, seed=42)
        s = hedging_error_summary(pnl)
        print(f"| {rehedge_every} | {s['mean']:.4f} | {s['std']:.4f} |")


if __name__ == "__main__":
    main()
