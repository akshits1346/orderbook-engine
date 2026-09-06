"""Hand-checkable Black-Scholes correctness tests: known values, put-call
parity, Greek bounds, and implied-vol round-trip."""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from black_scholes import (
    bs_price, bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho,
    implied_vol, intrinsic_value,
)

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0

    # Known reference value (Hull's textbook example, S=K=100,r=5%,sigma=20%,T=1):
    # call ~= 10.4506
    call = bs_price(S, K, T, r, sigma, "call", q)
    check(abs(call - 10.4506) < 1e-3, f"ATM call matches textbook reference, got {call:.4f}")

    # Put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
    put = bs_price(S, K, T, r, sigma, "put", q)
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    check(abs(lhs - rhs) < 1e-9, f"put-call parity holds, {lhs:.6f} vs {rhs:.6f}")

    # Deep ITM call converges to intrinsic-ish behavior: delta -> 1
    d_deep_itm = bs_delta(S, 1.0, T, r, sigma, "call", q)
    check(d_deep_itm > 0.999, f"deep ITM call delta ~1, got {d_deep_itm:.6f}")

    # Deep OTM call: delta -> 0
    d_deep_otm = bs_delta(S, 100000.0, T, r, sigma, "call", q)
    check(d_deep_otm < 1e-6, f"deep OTM call delta ~0, got {d_deep_otm:.8f}")

    # Delta bounds
    for K_test in [50, 80, 100, 120, 200]:
        dc = bs_delta(S, K_test, T, r, sigma, "call", q)
        dp = bs_delta(S, K_test, T, r, sigma, "put", q)
        check(0 <= dc <= 1, f"call delta in [0,1] at K={K_test}, got {dc:.4f}")
        check(-1 <= dp <= 0, f"put delta in [-1,0] at K={K_test}, got {dp:.4f}")
        check(abs(dc - dp - 1.0) < 1e-9, f"delta parity dc-dp=1 at K={K_test}")

    # Gamma and vega positive everywhere (true for vanilla options)
    for K_test in [50, 80, 100, 120, 200]:
        g = bs_gamma(S, K_test, T, r, sigma, q)
        v = bs_vega(S, K_test, T, r, sigma, q)
        check(g > 0, f"gamma positive at K={K_test}, got {g:.6f}")
        check(v > 0, f"vega positive at K={K_test}, got {v:.6f}")

    # rho: call rho positive, put rho negative
    rc = bs_rho(S, K, T, r, sigma, "call", q)
    rp = bs_rho(S, K, T, r, sigma, "put", q)
    check(rc > 0, f"call rho positive, got {rc:.4f}")
    check(rp < 0, f"put rho negative, got {rp:.4f}")

    # theta: for a non-dividend-paying ATM call, theta is typically negative (time decay)
    th = bs_theta(S, K, T, r, sigma, "call", q)
    check(th < 0, f"ATM call theta negative (time decay), got {th:.4f}")

    # Implied vol round-trip: price -> solve for sigma -> should recover input
    # sigma. Only tested for the OTM/ATM leg at each strike (put for K<S,
    # call for K>=S) -- the standard market convention, and not an
    # arbitrary restriction: see the deep-ITM ill-conditioning check below
    # for why the ITM leg is deliberately excluded here.
    for true_sigma in [0.05, 0.15, 0.3, 0.6, 1.2]:
        for K_test in [70, 100, 130]:
            opt = "put" if K_test < S else "call"
            price = bs_price(S, K_test, T, r, true_sigma, opt, q)
            iv = implied_vol(price, S, K_test, T, r, opt, q)
            check(abs(iv - true_sigma) < 1e-6,
                  f"IV round-trip {opt} K={K_test} sigma={true_sigma}: recovered {iv:.6f}")

    # Deep-ITM implied vol inversion is a KNOWN ill-conditioned regime, not
    # a bug: a deep ITM call's time value can fall below float64 precision
    # (vega underflows), so a 1-ULP price error implies a large sigma
    # error. Confirmed directly: vega should be tiny, and the price at the
    # (wrong-looking) recovered sigma should still match the input price
    # to float precision -- i.e. Brent's method solved the equation it was
    # given correctly, the equation itself just doesn't pin down sigma
    # here. This is exactly why data_gen.py always inverts from the OTM
    # leg (put for K<F, call for K>=F), never the ITM one.
    deep_itm_sigma = 0.05
    deep_itm_K = 70.0
    deep_itm_price = bs_price(S, deep_itm_K, T, r, deep_itm_sigma, "call", q)
    deep_itm_vega = bs_vega(S, deep_itm_K, T, r, deep_itm_sigma, q)
    deep_itm_iv = implied_vol(deep_itm_price, S, deep_itm_K, T, r, "call", q)
    price_at_recovered = bs_price(S, deep_itm_K, T, r, deep_itm_iv, "call", q)
    check(deep_itm_vega < 1e-8,
          f"deep ITM call vega is numerically negligible (ill-conditioned regime), got {deep_itm_vega:.2e}")
    check(abs(price_at_recovered - deep_itm_price) < 1e-9,
          "recovered sigma still reproduces the exact input price (solver isn't wrong, the inversion is just ill-posed here)")

    # implied_vol must reject a price outside the no-arbitrage band
    try:
        # a call can never be worth more than the (discounted) forward-adjusted spot
        implied_vol(S * 2, S, K, T, r, "call", q)
        check(False, "implied_vol should reject an out-of-band price")
    except ValueError:
        check(True, "implied_vol correctly rejects an out-of-band price")

    # intrinsic_value sanity: deep ITM call ~ discounted forward - discounted K
    iv_val = intrinsic_value(S, 1.0, T, r, "call", q)
    check(iv_val > 0, f"deep ITM intrinsic value positive, got {iv_val:.4f}")

    print()
    if failures == 0:
        print("All Black-Scholes checks passed.")
        return 0
    print(f"{failures} check(s) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
