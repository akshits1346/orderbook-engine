"""
Statistical validation that generate_realistic_lobster_data() actually
reproduces two well-known "stylized facts" of real market microstructure
data that generate_synthetic_lobster_data()'s plain random walk does
NOT have by construction (every execution moves mid by exactly +-1
tick, i.i.d. -- no mechanism could produce either property):

  - FAT TAILS: real return distributions have positive excess kurtosis
    (more extreme moves than a Gaussian of the same variance predicts).
  - VOLATILITY CLUSTERING: real |returns| are positively autocorrelated
    (big moves tend to follow big moves, calm periods follow calm
    periods) -- volatility clusters, it isn't i.i.d. noise.

VALIDATION STRATEGY, in two different ways for the two properties:

  - Fat tails: run BOTH generators on the SAME seed and length, and
    check the realistic one has measurably higher excess kurtosis than
    the plain one. A relative, self-contained comparison ("realistic
    scores higher than plain, same data-generating conditions") is a
    stronger and less arbitrary check than picking one fixed absolute
    number to clear.

  - Volatility clustering: comparing directly against the PLAIN
    generator here would be misleading, not just weaker -- the plain
    generator moves mid by an exactly-fixed 1 tick on every single
    execution (only the SIGN varies by side), so its |returns| series
    is nearly constant and its autocorrelation comes out spuriously
    close to 1.0, which is a degenerate artifact of having no
    magnitude variation at all, not genuine clustering. The correct
    way to isolate "does this series exhibit clustering" is to compare
    a series against a RANDOM PERMUTATION OF ITSELF: shuffling
    preserves the exact marginal distribution of |returns| (same
    values, same kurtosis, same everything about the distribution)
    while destroying any temporal structure. If the realistic
    generator's autocorrelation is measurably higher in its original,
    time-ordered form than in a shuffled version of the exact same
    values, that difference can only come from genuine temporal
    clustering, not from the distribution's shape.
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from backtest.generate_synthetic_data import generate_synthetic_lobster_data, generate_realistic_lobster_data

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def log_returns(mid_series):
    prices = np.array([m for _, m in mid_series], dtype=float)
    return np.diff(np.log(prices))


def excess_kurtosis(x):
    x = x - x.mean()
    m2 = np.mean(x ** 2)
    m4 = np.mean(x ** 4)
    return m4 / (m2 ** 2) - 3.0  # 0 for a true Gaussian; > 0 means fatter tails


def autocorr_abs_returns(x, lag=1):
    a = np.abs(x)
    a = a - a.mean()
    denom = np.sum(a ** 2)
    if denom == 0:
        return 0.0
    return np.sum(a[:-lag] * a[lag:]) / denom


def main():
    n_events = 6000
    seed = 7

    with tempfile.NamedTemporaryFile(suffix=".csv") as plain_f, \
         tempfile.NamedTemporaryFile(suffix=".csv") as realistic_f:

        _, plain_series = generate_synthetic_lobster_data(
            n_events=n_events, seed=seed, output_path=plain_f.name, record_mid_series=True,
        )
        _, realistic_series = generate_realistic_lobster_data(
            n_events=n_events, seed=seed, output_path=realistic_f.name, record_mid_series=True,
        )

    check(len(plain_series) > 100, f"plain generator produced enough executions to analyze, got {len(plain_series)}")
    check(len(realistic_series) > 100, f"realistic generator produced enough executions to analyze, got {len(realistic_series)}")

    plain_returns = log_returns(plain_series)
    realistic_returns = log_returns(realistic_series)

    # --- fat tails ---
    plain_kurt = excess_kurtosis(plain_returns)
    realistic_kurt = excess_kurtosis(realistic_returns)
    check(realistic_kurt > plain_kurt + 1.0,
          f"realistic generator has measurably fatter tails: excess kurtosis {realistic_kurt:.2f} "
          f"vs plain generator's {plain_kurt:.2f}")

    # --- volatility clustering: original time-order vs a shuffled
    # control with the IDENTICAL marginal distribution (see module
    # docstring for why comparing to the plain generator directly would
    # be misleading here, not just weaker) ---
    realistic_autocorr = autocorr_abs_returns(realistic_returns)

    shuffle_rng = np.random.RandomState(123)
    shuffled_autocorrs = []
    for _ in range(30):
        shuffled = realistic_returns.copy()
        shuffle_rng.shuffle(shuffled)
        shuffled_autocorrs.append(autocorr_abs_returns(shuffled))
    mean_shuffled_autocorr = float(np.mean(shuffled_autocorrs))
    std_shuffled_autocorr = float(np.std(shuffled_autocorrs))

    check(abs(mean_shuffled_autocorr) < 0.05,
          f"shuffled control has ~zero autocorrelation, as expected (mean over 30 shuffles: "
          f"{mean_shuffled_autocorr:.3f}) -- confirms the marginal distribution alone doesn't "
          f"produce clustering, only temporal order can")
    check(realistic_autocorr > mean_shuffled_autocorr + 5 * std_shuffled_autocorr,
          f"realistic generator's TIME-ORDERED |return| autocorrelation ({realistic_autocorr:.3f}) is "
          f"far above its own shuffled-control distribution (mean {mean_shuffled_autocorr:.3f}, "
          f"std {std_shuffled_autocorr:.3f}) -- clustering comes from temporal order, not just the "
          f"distribution's shape")
    check(realistic_autocorr > 0.1,
          f"realistic generator's volatility clustering is positive in absolute terms too, got {realistic_autocorr:.3f}")

    # --- determinism: same seed must reproduce the exact same series (a
    # generator whose "randomness" isn't actually seeded would make any
    # reported finding from it irreproducible) ---
    with tempfile.NamedTemporaryFile(suffix=".csv") as again_f:
        _, realistic_series_again = generate_realistic_lobster_data(
            n_events=n_events, seed=seed, output_path=again_f.name, record_mid_series=True,
        )
    check(realistic_series == realistic_series_again,
          "same seed reproduces an identical mid series (deterministic, reproducible)")

    print()
    if failures == 0:
        print("All synthetic data realism checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
