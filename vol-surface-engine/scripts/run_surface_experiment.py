"""
Generate a synthetic option chain from Heston, fit raw SVI per-slice,
check butterfly/calendar arbitrage, then fit SSVI jointly and compare.
Prints the tables that go straight into README.md.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from heston import HestonParams
from data_gen import generate_synthetic_chain
from surface import VolSurface


def main():
    # A realistic equity-index-like Heston parameterization: negative
    # correlation (leverage effect / downside skew), moderate vol-of-vol.
    p = HestonParams(kappa=1.5, theta=0.045, xi=0.55, rho=-0.75, v0=0.045)
    print(f"Heston params: kappa={p.kappa} theta={p.theta} xi={p.xi} rho={p.rho} v0={p.v0}")
    print(f"Feller ratio 2*kappa*theta/xi^2 = {p.feller_ratio():.3f} "
          f"({'satisfied' if p.feller_ratio() >= 1 else 'VIOLATED -- realistic for equity indices'})")
    print()

    # Includes a closely-spaced weekly/monthly pair (0.083, 0.095yr = ~30d,
    # ~35d) deliberately -- calendar arbitrage from independent per-slice
    # fitting under noise shows up most easily between adjacent expiries
    # whose true total variance is close together, exactly like real
    # weekly-options term structures.
    expiries = (0.083, 0.095, 0.25, 0.5, 1.0, 2.0)
    rows = generate_synthetic_chain(p, S0=100.0, r=0.02, expiries=expiries)

    # Real quotes are never noiseless -- inject a bid-ask-scale jitter
    # (+/- 0.5 vol points, a realistic single-name/index options market,
    # wider than the tightest ATM index markets) onto the clean
    # Heston-implied vols before fitting. This matters: a perfectly
    # noiseless arbitrage-free surface fits without introducing calendar
    # arbitrage almost by definition (each slice recovers the *same*
    # underlying arbitrage-free curve independently, so of course they
    # agree) -- noise breaks that coincidence and is exactly the practical
    # case where raw per-slice fits are known to introduce arbitrage on
    # real desks.
    rng = np.random.default_rng(0)
    for row in rows:
        row["iv"] = max(row["iv"] + rng.normal(0, 0.005), 1e-4)

    surface = VolSurface(rows)

    print(f"Generated {len(rows)} (K, T) quotes across {len(expiries)} expiries "
          f"(implied vols perturbed by +/-0.5 vol pt Gaussian noise to emulate bid-ask).")
    print()

    # --- raw SVI per-slice ---
    surface.fit_raw_svi()
    print("## Raw SVI, independent per-expiry fits")
    print()
    print("| T (yrs) | RMSE (vol pts) | butterfly arb-free | min g(k) |")
    print("|---|---|---|---|")
    bfly = surface.butterfly_report()
    for T in surface.expiries:
        r = bfly[T]
        print(f"| {T:.3f} | {r['rmse_vol']*100:.4f} | {r['arb_free']} | {r['min_g']:.4f} |")
    print()

    n_violations, worst_gap, violations = surface.calendar_report()
    print(f"Calendar-arbitrage check (raw SVI, independent slices): "
          f"{n_violations} violating (k, T_i, T_{{i+1}}) points found, "
          f"worst total-variance gap = {worst_gap:.6f}")
    if violations:
        v = max(violations, key=lambda v: v["gap"])
        print(f"Worst single violation: k={v['k']:.3f}, T1={v['T1']} (w={v['w1']:.6f}) "
              f"-> T2={v['T2']} (w={v['w2']:.6f}), i.e. total variance DECREASED with maturity.")
    print()

    # --- SSVI joint fit ---
    rho, eta, gamma_power = surface.fit_ssvi()[:3]
    print(f"## SSVI joint fit: rho={rho:.4f}, eta={eta:.4f}, gamma={gamma_power:.4f}")
    print(f"eta*(1+|rho|) = {eta*(1+abs(rho)):.4f}  (Gatheral-Jacquier sufficient condition requires <= 2)")
    print(f"Calendar-arbitrage-free by the Gatheral-Jacquier sufficient condition: "
          f"{surface.ssvi_calendar_arb_free()}")
    n_ssvi_violations, ssvi_worst_gap, _ = surface.ssvi_calendar_report()
    print(f"Direct grid check on the fitted SSVI surface: {n_ssvi_violations} violations, "
          f"worst gap = {ssvi_worst_gap:.6f} (this is the check that actually matters -- "
          f"the condition above is only sufficient, not necessary)")
    print()
    print("| T (yrs) | theta(T) (ATM total var) | SSVI RMSE (vol pts) | raw-SVI RMSE (vol pts) |")
    print("|---|---|---|---|")
    ssvi_rmse = surface.ssvi_fit_quality()
    for T in surface.expiries:
        print(f"| {T:.3f} | {surface.ssvi_theta[T]:.6f} | {ssvi_rmse[T]*100:.4f} | {bfly[T]['rmse_vol']*100:.4f} |")


if __name__ == "__main__":
    main()
