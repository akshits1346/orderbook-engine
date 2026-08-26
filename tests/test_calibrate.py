"""
Hand-traced tests for calibrate.py's pure-math fitting functions.
These use constructed data where the true answer is known exactly (no
noise), so a passing test means the fitting math itself is correct --
separately from whether real market data would actually look like this.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from backtest.calibrate import estimate_sigma_from_mid_series, fit_kappa_from_distance_counts

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    # --- sigma: perfectly alternating +100/-100 price changes, dt=1 each ---
    # diffs = [100, -100, 100, -100], mean 0, population std = exactly 100
    mids = [1000000, 1000100, 1000000, 1000100, 1000000]
    times = [0, 1, 2, 3, 4]
    sigma = estimate_sigma_from_mid_series(mids, times)
    check(abs(sigma - 100.0) < 1e-9, f"sigma recovered exactly as 100.0, got {sigma}")

    # --- kappa: noiseless exponential decay, intensity = 100 * exp(-0.5 * distance) ---
    kappa_true = 0.5
    A = 100
    distances = [0, 1, 2, 3, 4, 5]
    counts = [A * math.exp(-kappa_true * d) for d in distances]
    kappa_fit = fit_kappa_from_distance_counts(distances, counts)
    check(abs(kappa_fit - 0.5) < 1e-6, f"kappa recovered as 0.5 from noiseless data, got {kappa_fit}")

    # --- kappa: zero-count buckets should be dropped, not break the fit ---
    distances_with_gaps = [0, 1, 2, 3, 4, 5]
    counts_with_gaps = [100, 0, 36.79, 0, 13.53, 0]  # matches kappa=0.5 pattern at 0,2,4; zeros elsewhere
    kappa_fit_gaps = fit_kappa_from_distance_counts(distances_with_gaps, counts_with_gaps)
    check(abs(kappa_fit_gaps - 0.5) < 0.01,
          f"kappa fit ignores zero-count buckets correctly, got {kappa_fit_gaps}")

    # --- kappa: should raise if fewer than 2 nonzero buckets ---
    try:
        fit_kappa_from_distance_counts([0, 1, 2], [50, 0, 0])
        check(False, "fit_kappa_from_distance_counts should raise with <2 nonzero buckets")
    except ValueError:
        check(True, "fit_kappa_from_distance_counts correctly raises with <2 nonzero buckets")

    print()
    if failures == 0:
        print("All calibration checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
