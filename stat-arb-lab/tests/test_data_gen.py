"""Checks the synthetic data generators actually produce what they claim:
a stationary spread riding a non-stationary trend (cointegration), and
a factor basket whose residuals are stationary and whose systematic
component is recoverable."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from statsmodels.tsa.stattools import adfuller
from data_gen import generate_cointegrated_pair, generate_factor_basket

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    d = generate_cointegrated_pair(n=1500, hedge_ratio=1.5, theta=0.05, spread_vol=0.3, seed=1)

    adf_spread = adfuller(d["true_spread"])
    check(adf_spread[1] < 0.01, f"true spread is stationary (ADF p={adf_spread[1]:.2e})")

    adf_trend = adfuller(d["log_p1"])
    check(adf_trend[1] > 0.10, f"trend log_p1 is non-stationary (ADF p={adf_trend[1]:.2f})")

    adf_p2 = adfuller(d["log_p2"])
    check(adf_p2[1] > 0.10, f"log_p2 is also non-stationary on its own (ADF p={adf_p2[1]:.2f})")

    check(np.all(d["p1"] > 0) and np.all(d["p2"] > 0), "generated prices are all positive")

    d_tv = generate_cointegrated_pair(n=500, time_varying_hedge=True, hedge_ratio_amplitude=0.4,
                                       hedge_ratio_period=250, seed=2)
    check(d_tv["true_beta"].std() > 0.1, "time-varying hedge ratio actually varies")
    check(abs(d_tv["true_beta"].mean() - 1.5) < 0.1, "time-varying hedge ratio oscillates around its center")

    # --- factor basket ---
    b = generate_factor_basket(n_assets=15, n_days=1500, n_factors=3, seed=5)
    check(b["returns"].shape == (1500, 15), f"returns shape correct, got {b['returns'].shape}")
    check(b["true_betas"].shape == (15, 3), f"true_betas shape correct, got {b['true_betas'].shape}")

    # cumulative residual (= OU LEVEL, not increment) should be stationary per asset
    cum_residual = np.cumsum(b["residuals"], axis=0)
    n_stationary = sum(adfuller(cum_residual[:, i])[1] < 0.05 for i in range(15))
    check(n_stationary >= 12, f"most cumulative idiosyncratic residuals are stationary ({n_stationary}/15)")

    # systematic component should correlate strongly with true factor exposure:
    # regressing returns on true factor_returns should recover true_betas closely
    import statsmodels.api as sm
    recovered_betas = np.zeros((15, 3))
    for i in range(15):
        model = sm.OLS(b["returns"][:, i], b["factor_returns"]).fit()
        recovered_betas[i] = model.params
    max_err = np.max(np.abs(recovered_betas - b["true_betas"]))
    check(max_err < 0.15, f"OLS on true factor returns recovers true betas closely, max err={max_err:.4f}")

    print()
    if failures == 0:
        print("All data_gen checks passed.")
        return 0
    print(f"{failures} check(s) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
