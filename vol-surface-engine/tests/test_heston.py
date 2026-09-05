"""
Cross-validation of the semi-analytic Heston pricer against an
independent Monte Carlo simulation -- the core correctness proof for
this module. Also checks put-call parity (algebraic, not numerical)
and that the characteristic function satisfies phi(0) = 1, a basic
identity every valid characteristic function must satisfy.
"""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from heston import HestonParams, heston_price, mc_price, _char_function

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    S0, r, q = 100.0, 0.02, 0.0

    # phi(0) = E[exp(0)] = 1 for any valid characteristic function
    for T in [0.1, 1.0, 3.0]:
        p = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
        phi0 = _char_function(0.0, S0, r, q, T, p)
        check(abs(phi0 - 1.0) < 1e-8, f"phi(0)=1 at T={T}, got {phi0}")

    # Semi-analytic price vs Monte Carlo, across strikes/maturities/param regimes.
    # This is the load-bearing check: two independently-derived methods for
    # "the price of this option under these dynamics" must agree within MC
    # standard error, or the semi-analytic formula has a bug.
    param_sets = [
        ("moderate vol-of-vol", HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)),
        ("high vol-of-vol, Feller violated", HestonParams(kappa=1.0, theta=0.04, xi=1.0, rho=-0.5, v0=0.09)),
        ("low correlation", HestonParams(kappa=3.0, theta=0.09, xi=0.3, rho=-0.1, v0=0.09)),
    ]
    n_within = 0
    n_total = 0
    for label, p in param_sets:
        print(f"  -- {label} (Feller ratio={p.feller_ratio():.3f}) --")
        for T in [0.5, 1.5]:
            for K in [80, 100, 120]:
                analytic = heston_price(S0, K, T, r, p, "call", q)
                mc, se = mc_price(S0, K, T, r, p, "call", q, n_steps=100, n_paths=150_000, seed=7)
                diff = analytic - mc
                within = abs(diff) < 4 * se
                n_total += 1
                n_within += int(within)
                check(within,
                      f"[{label}] K={K} T={T}: analytic={analytic:.4f} mc={mc:.4f}+/-{se:.4f} diff={diff:+.4f}")

    check(n_within >= 0.9 * n_total,
          f"at least 90% of (param, strike, maturity) combos agree within 4 MC std errors "
          f"({n_within}/{n_total})")

    # Put-call parity for Heston prices (call - put should equal forward - strike, discounted)
    p = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
    for T in [0.25, 1.0]:
        for K in [90, 100, 110]:
            call = heston_price(S0, K, T, r, p, "call", q)
            put = heston_price(S0, K, T, r, p, "put", q)
            lhs = call - put
            rhs = S0 * math.exp(-q * T) - K * math.exp(-r * T)
            check(abs(lhs - rhs) < 1e-6, f"Heston put-call parity K={K} T={T}: {lhs:.6f} vs {rhs:.6f}")

    print()
    if failures == 0:
        print("All Heston checks passed.")
        return 0
    print(f"{failures} check(s) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
