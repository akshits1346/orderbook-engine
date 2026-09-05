"""
PCA stat-arb signal tests: factor recovery against the synthetic
factor basket's ground truth, and OU parameter recovery on a
constructed AR(1) series with a known answer.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from data_gen import generate_factor_basket
from pca_signal import fit_pca_factors, fit_asset_residual, fit_ou_params, compute_signals

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    # --- PCA recovers the true factors, up to sign/permutation (a
    # fundamental identifiability property of PCA, not a limitation of
    # this implementation) ---
    b = generate_factor_basket(n_assets=20, n_days=750, n_factors=3, idio_theta=0.04, seed=1)
    factor_returns, Q, asset_vols = fit_pca_factors(b["returns"], n_factors=3)
    corr = np.abs(np.corrcoef(np.column_stack([factor_returns, b["factor_returns"]]), rowvar=False)[:3, 3:])
    best_match_corr = corr.max(axis=1)  # best-matching true factor for each recovered one
    check(np.all(best_match_corr > 0.6),
          f"every recovered PCA factor correlates strongly (>0.6) with some true factor, "
          f"got {np.round(best_match_corr, 3)}")
    # every true factor should also be matched by SOME recovered one (not just the reverse)
    best_match_by_true = corr.max(axis=0)
    check(np.all(best_match_by_true > 0.6),
          f"every true factor is recovered by some PCA factor, got {np.round(best_match_by_true, 3)}")

    # --- OU parameter recovery on a constructed AR(1) with known answer ---
    rng = np.random.default_rng(2)
    n = 3000
    true_kappa_target_b = 0.97  # b = exp(-kappa*dt), dt=1/252
    true_m = 0.0
    xi = rng.normal(0, 0.05, n)
    X = np.zeros(n)
    for t in range(1, n):
        X[t] = true_m + true_kappa_target_b * X[t - 1] + xi[t]
    residuals = np.diff(np.concatenate([[0.0], X]))  # increments that reconstruct X via cumsum
    ou = fit_ou_params(residuals)
    check(ou["is_mean_reverting"], "OU fit correctly identifies a mean-reverting series")
    implied_b = np.exp(-ou["kappa"] / 252)
    check(abs(implied_b - true_kappa_target_b) < 0.02,
          f"recovered kappa implies b close to true b={true_kappa_target_b}, got {implied_b:.4f}")

    # --- a pure random walk (b=1, non mean-reverting) should be flagged as NOT tradeable ---
    rw_residuals = rng.normal(0, 0.05, 1000)  # residuals of a pure random walk: cumsum has b~1
    ou_rw = fit_ou_params(rw_residuals)
    check(not ou_rw["is_mean_reverting"] or ou_rw["kappa"] < 1.0,
          f"a non-mean-reverting (near-random-walk) series is not flagged as strongly mean-reverting")

    # --- compute_signals end-to-end sanity ---
    s_scores, betas, Q2, tradeable = compute_signals(b["returns"], n_factors=3)
    check(len(s_scores) == 20 and betas.shape == (20, 3), "compute_signals returns correctly shaped output")
    check(tradeable.sum() >= 15, f"most assets are flagged tradeable on stationary synthetic data, got {tradeable.sum()}/20")
    check(np.all(np.isfinite(s_scores[tradeable])), "s-scores for tradeable assets are all finite")

    print()
    if failures == 0:
        print("All PCA signal checks passed.")
        return 0
    print(f"{failures} check(s) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
