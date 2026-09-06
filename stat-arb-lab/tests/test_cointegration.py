"""Engle-Granger and Johansen tests, checked against constructed
cointegrated and NON-cointegrated data (two independent random walks --
the standard negative control every cointegration test needs, since
correlation between two random walks is common even when there is no
real long-run relationship, the "spurious regression" problem
Engle-Granger's ADF-on-residual step exists specifically to catch)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from data_gen import generate_cointegrated_pair
from cointegration import engle_granger_test, johansen_test

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    # --- positive control: genuinely cointegrated pair ---
    d = generate_cointegrated_pair(n=1500, hedge_ratio=1.5, theta=0.05, spread_vol=0.3, seed=10)
    eg = engle_granger_test(d["log_p1"], d["log_p2"])
    check(eg["is_cointegrated"], f"Engle-Granger detects real cointegration (p={eg['adf_pvalue']:.2e})")
    check(abs(eg["beta"] - 1.5) < 0.1, f"Engle-Granger recovers hedge ratio ~1.5, got {eg['beta']:.4f}")

    log_prices = np.column_stack([d["log_p1"], d["log_p2"]])
    joh = johansen_test(log_prices)
    check(joh["n_coint_95"] >= 1, f"Johansen detects >=1 cointegrating relationship, got {joh['n_coint_95']}")

    # --- negative control: two INDEPENDENT random walks, no true relationship ---
    rng = np.random.default_rng(11)
    n = 1500
    n_false_positives_eg = 0
    n_false_positives_joh = 0
    n_trials = 20
    for trial in range(n_trials):
        rw1 = np.cumsum(rng.normal(0, 1, n))
        rw2 = np.cumsum(rng.normal(0, 1, n))
        eg_neg = engle_granger_test(rw1, rw2)
        if eg_neg["is_cointegrated"]:
            n_false_positives_eg += 1
        joh_neg = johansen_test(np.column_stack([rw1, rw2]))
        if joh_neg["n_coint_95"] >= 1:
            n_false_positives_joh += 1

    # At a 5% test size, false positives should be rare, not the majority --
    # this is the actual claim a cointegration test makes: it should mostly
    # say "no" when there's genuinely nothing there.
    check(n_false_positives_eg <= n_trials * 0.35,
          f"Engle-Granger false-positive rate is bounded on independent random walks "
          f"({n_false_positives_eg}/{n_trials}, test size 5% so some false positives are expected)")
    check(n_false_positives_joh <= n_trials * 0.35,
          f"Johansen false-positive rate is bounded on independent random walks "
          f"({n_false_positives_joh}/{n_trials})")

    print()
    if failures == 0:
        print("All cointegration checks passed.")
        return 0
    print(f"{failures} check(s) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
