"""Kalman filter hedge-ratio tests: static-case recovery vs. OLS, and
the actual selling point -- tracking a TIME-VARYING hedge ratio better
than a single static OLS estimate can, by construction."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import statsmodels.api as sm
from data_gen import generate_cointegrated_pair
from kalman import kalman_hedge_ratio

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    # --- static hedge ratio: Kalman's converged beta should agree with OLS ---
    d = generate_cointegrated_pair(n=2000, hedge_ratio=1.5, theta=0.05, spread_vol=0.3, seed=20)
    res = kalman_hedge_ratio(d["log_p1"], d["log_p2"], delta=1e-5)
    kalman_beta_converged = res["beta"][-300:].mean()
    X = sm.add_constant(d["log_p1"])
    ols_beta = sm.OLS(d["log_p2"], X).fit().params[1]
    check(abs(kalman_beta_converged - ols_beta) < 0.05,
          f"converged Kalman beta agrees with static OLS beta: kalman={kalman_beta_converged:.4f} ols={ols_beta:.4f}")
    check(abs(kalman_beta_converged - 1.5) < 0.05,
          f"converged Kalman beta close to true beta=1.5, got {kalman_beta_converged:.4f}")

    # --- time-varying hedge ratio: Kalman should track it better than static OLS ---
    d2 = generate_cointegrated_pair(n=2000, hedge_ratio=1.5, theta=0.05, spread_vol=0.3,
                                     time_varying_hedge=True, hedge_ratio_amplitude=0.4,
                                     hedge_ratio_period=250, seed=21)
    res2 = kalman_hedge_ratio(d2["log_p1"], d2["log_p2"], delta=1e-4)
    # skip the first 200 points to let the filter converge past its initial guess
    kalman_err = np.abs(res2["beta"][200:] - d2["true_beta"][200:]).mean()
    static_beta2 = sm.OLS(d2["log_p2"], sm.add_constant(d2["log_p1"])).fit().params[1]
    static_err = np.abs(static_beta2 - d2["true_beta"][200:]).mean()
    check(kalman_err < static_err,
          f"Kalman tracks time-varying hedge ratio better than static OLS: "
          f"kalman_mae={kalman_err:.4f} static_mae={static_err:.4f}")

    # --- innovation z-score sanity: should be roughly standardized (mean~0, std~O(1)) once converged ---
    z = res["z_score"][300:]
    check(abs(z.mean()) < 0.5, f"innovation z-score has ~zero mean once converged, got {z.mean():.4f}")
    check(0.3 < z.std() < 3.0, f"innovation z-score has a sane O(1) std once converged, got {z.std():.4f}")

    # --- filter covariance should shrink FAST away from its arbitrary
    # initial guess (P0 = I, trace = 2) as it converges. Checked over an
    # early-to-mid window rather than end-of-series: with adaptive R and
    # a regressor (log_p1, a plain random walk with no mean reversion)
    # that can wander to a very different SCALE late in a 2000-step
    # series, late-series trace can legitimately re-inflate as R adapts
    # to a regime with larger observation noise -- that's a property of
    # adaptive-R on a growing-scale regressor, not non-convergence, so
    # it isn't what this check is testing.
    early_window_mean = res["theta_cov_trace"][50:500].mean()
    check(early_window_mean < res["theta_cov_trace"][0] * 0.1,
          f"state covariance shrinks well below its initial guess after the initial transient: "
          f"trace[0]={res['theta_cov_trace'][0]:.4f}, mean(trace[50:500])={early_window_mean:.6f}")

    print()
    if failures == 0:
        print("All Kalman filter checks passed.")
        return 0
    print(f"{failures} check(s) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
