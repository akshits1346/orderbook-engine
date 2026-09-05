"""
SVI/SSVI correctness tests: noiseless parameter recovery, the
Durrleman butterfly-arbitrage check correctly separating an arb-free
slice from a deliberately pathological one, and the SSVI
calendar-arbitrage-free CONDITION (checked on parameters directly, not
just a k-grid spot check).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from svi import (
    raw_svi_total_variance, fit_svi_slice, durrleman_g, check_butterfly_arbitrage,
    ssvi_total_variance, ssvi_calendar_arbitrage_free, fit_ssvi_surface,
)

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    # --- noiseless raw SVI recovery ---
    true_params = (0.02, 0.3, -0.4, 0.05, 0.2)
    k = np.linspace(-1, 1, 25)
    w = raw_svi_total_variance(k, true_params)
    fitted, fitted_w = fit_svi_slice(k, w)
    max_err = np.max(np.abs(fitted_w - w))
    check(max_err < 1e-6, f"raw SVI recovers noiseless total variance exactly, max err={max_err:.2e}")

    # --- butterfly arbitrage check: sane params pass, pathological params fail ---
    k_grid = np.linspace(-2, 2, 201)
    ok, min_g = check_butterfly_arbitrage(true_params, k_grid)
    check(ok and min_g > 0, f"reasonable SVI params are butterfly-arb-free, min_g={min_g:.4f}")

    # A flat smile (b=0) is trivially arb-free: g(k) = 1 everywhere
    flat_params = (0.04, 1e-6, 0.0, 0.0, 0.3)
    g_flat = durrleman_g(k_grid, flat_params)
    check(np.allclose(g_flat, 1.0, atol=1e-3), f"flat smile has g(k)=1 everywhere, got range [{g_flat.min():.4f}, {g_flat.max():.4f}]")

    # Deliberately extreme curvature (large b, tiny sigma, high |rho|) should violate butterfly no-arb
    bad_params = (0.0, 4.0, 0.99, 0.0, 0.01)
    ok_bad, min_g_bad = check_butterfly_arbitrage(bad_params, k_grid)
    check(not ok_bad and min_g_bad < 0, f"pathological SVI params correctly flagged as arb-violating, min_g={min_g_bad:.4f}")

    # --- SSVI calendar-arbitrage-free condition ---
    # Gatheral-Jacquier: 0 < gamma < 1/2 and eta*(1+|rho|) <= 2
    check(ssvi_calendar_arbitrage_free(rho=-0.5, eta=1.0, gamma_power=0.3),
          "SSVI params satisfying the sufficient condition are flagged arb-free")
    check(not ssvi_calendar_arbitrage_free(rho=-0.5, eta=3.0, gamma_power=0.3),
          "SSVI params violating eta*(1+|rho|)<=2 are flagged NOT arb-free")
    check(not ssvi_calendar_arbitrage_free(rho=-0.5, eta=1.0, gamma_power=0.6),
          "SSVI params violating gamma<1/2 are flagged NOT arb-free")

    # --- SSVI is manifestly non-decreasing in theta at fixed k for a
    # single (rho, eta, gamma) evaluated across several theta values,
    # given the condition holds (this is the calendar no-arb property
    # itself, checked directly rather than trusting the sufficient
    # condition blindly) ---
    rho, eta, gamma_power = -0.4, 1.0, 0.3
    check(ssvi_calendar_arbitrage_free(rho, eta, gamma_power), "chosen SSVI test params satisfy the sufficient condition")
    thetas = np.array([0.01, 0.03, 0.08, 0.15, 0.30])  # increasing, as a real ATM term structure is
    k_test = np.linspace(-0.5, 0.5, 21)
    prev_w = None
    monotone_ok = True
    for theta in thetas:
        w_theta = ssvi_total_variance(k_test, theta, rho, eta, gamma_power)
        if prev_w is not None and np.any(w_theta < prev_w - 1e-9):
            monotone_ok = False
        prev_w = w_theta
    check(monotone_ok, "SSVI total variance is non-decreasing in theta at every fixed k (no calendar arbitrage)")

    # --- SSVI joint fit recovers known params from noiseless multi-expiry data ---
    true_rho, true_eta, true_gamma = -0.35, 0.8, 0.25
    thetas_true = np.array([0.02, 0.05, 0.10, 0.18])
    ks = [np.linspace(-0.6, 0.6, 15) for _ in thetas_true]
    ws = [ssvi_total_variance(k_arr, theta, true_rho, true_eta, true_gamma)
          for k_arr, theta in zip(ks, thetas_true)]
    fit_rho, fit_eta, fit_gamma = fit_ssvi_surface(thetas_true, ks, ws)
    check(abs(fit_rho - true_rho) < 1e-4, f"SSVI joint fit recovers rho, true={true_rho} fit={fit_rho:.6f}")
    check(abs(fit_eta - true_eta) < 1e-4, f"SSVI joint fit recovers eta, true={true_eta} fit={fit_eta:.6f}")
    check(abs(fit_gamma - true_gamma) < 1e-4, f"SSVI joint fit recovers gamma, true={true_gamma} fit={fit_gamma:.6f}")

    print()
    if failures == 0:
        print("All SVI/SSVI checks passed.")
        return 0
    print(f"{failures} check(s) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
