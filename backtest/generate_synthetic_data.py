"""
Generates a synthetic LOBSTER-format message file large enough to
actually exercise calibration and the strategy comparison end to end.

THIS IS NOT REAL MARKET DATA. It's a random walk with plausible-looking
order flow, built only so the calibration and comparison pipeline has
enough volume to produce a non-degenerate result while you're setting
things up. Any sigma/kappa estimated from this, and any "naive beats AS"
or "AS beats naive" result from running the comparison against it, is a
statement about this synthetic generator's random seed -- not a market
finding. Swap in a real downloaded LOBSTER sample before drawing any
actual conclusion; this script exists to prove the plumbing works while
you do that.
"""
import math
import random


def generate_synthetic_lobster_data(
    n_events: int = 500,
    start_price: int = 1000000,   # LOBSTER units: $100.00
    tick_size: int = 100,          # $0.01
    seed: int = 42,
    output_path: str = "data/synthetic_large.csv",
    record_mid_series: bool = False,
):
    rng = random.Random(seed)
    lines = []
    time = 34200.0  # 9:30:00 AM in seconds-since-midnight, matching LOBSTER convention
    next_order_id = 1
    mid = start_price
    mid_series = []

    # order book state we simulate ourselves here, just enough to know
    # what ids/prices/sizes exist so cancels/executions reference real ones
    resting_orders = {}  # id -> (side, price, size)

    def add_new_order():
        nonlocal next_order_id, mid
        side = rng.choice([1, -1])
        # orders cluster near mid, occasionally further out
        offset_ticks = rng.choice([0, 0, 1, 1, 2, 3, 5, 8])
        price = mid - offset_ticks * tick_size if side == 1 else mid + offset_ticks * tick_size
        size = rng.choice([100, 100, 200, 300, 500])
        oid = next_order_id
        next_order_id += 1
        resting_orders[oid] = [side, price, size]
        lines.append(f"{time:.6f},1,{oid},{size},{price},{side}")

    def random_execution():
        nonlocal mid
        if not resting_orders:
            return
        oid = rng.choice(list(resting_orders.keys()))
        side, price, size = resting_orders[oid]
        exec_size = min(size, rng.choice([50, 100, 100, 200]))
        resting_orders[oid][2] -= exec_size
        if resting_orders[oid][2] <= 0:
            del resting_orders[oid]
        lines.append(f"{time:.6f},4,{oid},{exec_size},{price},{side}")
        # a trade nudges mid slightly in the direction of the taker
        mid += tick_size if side == -1 else -tick_size
        if record_mid_series:
            mid_series.append((time, mid))

    def random_cancel():
        if not resting_orders:
            return
        oid = rng.choice(list(resting_orders.keys()))
        side, price, size = resting_orders[oid]
        lines.append(f"{time:.6f},3,{oid},{size},{price},{side}")
        del resting_orders[oid]

    for _ in range(n_events):
        time += rng.uniform(0.01, 0.5)
        action = rng.choices(
            ["new", "execute", "cancel"],
            weights=[0.5, 0.3, 0.2],
        )[0]
        if action == "new" or not resting_orders:
            add_new_order()
        elif action == "execute":
            random_execution()
        else:
            random_cancel()

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    if record_mid_series:
        return output_path, mid_series
    return output_path


def generate_realistic_lobster_data(
    n_events: int = 3000,
    start_price: int = 1000000,
    tick_size: int = 100,
    seed: int = 42,
    output_path: str = "data/synthetic_realistic.csv",
    vol_baseline: float = 1.0,       # baseline jump-size scale, in ticks (== long-run variance level)
    garch_alpha: float = 0.15,       # weight on the LATEST standardized shock^2 (reaction speed)
    garch_beta: float = 0.80,        # weight on the previous variance state (persistence)
    fat_tail_df: int = 3,            # Student-t degrees of freedom for jump draws (low df = fat tails)
    record_mid_series: bool = False,
):
    """
    A more statistically realistic synthetic order flow generator than
    generate_synthetic_lobster_data() above: that one nudges mid by
    exactly +-1 tick on every execution, i.i.d. -- which, by
    construction, produces an approximately Gaussian, NOT fat-tailed,
    return series with NO volatility clustering (nothing links one
    step's size to the next). Two well-documented stylized facts of
    real market microstructure data are simply absent from it.

    This version reproduces both, via a standard mechanism for each:

      - FAT TAILS: each execution's price-move size (in ticks) is drawn
        from a Student-t distribution (fat_tail_df degrees of freedom --
        low df means heavy tails) instead of being fixed at 1. This
        produces measurable EXCESS KURTOSIS in the resulting return
        series -- more extreme moves than a Gaussian of the same
        variance would predict, matching real return distributions.

      - VOLATILITY CLUSTERING: the Student-t draw is scaled by a
        persistent "current variance" state following a textbook
        GARCH(1,1) recursion, variance_t = omega + alpha*z_{t-1}^2 +
        beta*variance_{t-1} (z = the standardized shock, omega chosen
        so variance mean-reverts to vol_baseline). A big shock pushes
        variance up for the NEXT move, and that elevated variance
        persists (decaying geometrically at rate beta) rather than
        resetting instantly -- producing positively autocorrelated
        |returns| ("vol clusters"), instead of i.i.d. step sizes.
        Feeding back the raw standardized shock z (not the rounded,
        floor-clamped tick count actually written to the file) matters:
        ticks_moved is floored at 1 tick, which would otherwise mute
        the feedback signal on the frequent small moves and make the
        clustering effect much weaker than the underlying process
        actually has.

    Still not real market data -- see this module's honesty note above,
    which applies here too -- but a materially better stand-in for
    testing calibration/strategy code against something with the same
    basic statistical shape as real order flow, rather than a plain
    fixed-step random walk. See tests/test_synthetic_data_realism.py
    for the statistical checks (excess kurtosis, |return| autocorrelation)
    that validate both properties actually hold, measured directly
    against generate_synthetic_lobster_data()'s plain random walk on the
    same seed and length.
    """
    rng = random.Random(seed)
    lines = []
    time = 34200.0
    next_order_id = 1
    mid = start_price
    variance_state = vol_baseline  # GARCH(1,1) variance state, starts at its long-run mean
    omega = (1.0 - garch_alpha - garch_beta) * vol_baseline  # keeps E[variance_state] == vol_baseline
    mid_series = []

    resting_orders = {}

    def student_t(df):
        z = rng.gauss(0, 1)
        chi2 = sum(rng.gauss(0, 1) ** 2 for _ in range(df))
        return z / math.sqrt(chi2 / df)

    def add_new_order():
        nonlocal next_order_id
        side = rng.choice([1, -1])
        offset_ticks = rng.choice([0, 0, 1, 1, 2, 3, 5, 8])
        price = mid - offset_ticks * tick_size if side == 1 else mid + offset_ticks * tick_size
        size = rng.choice([100, 100, 200, 300, 500])
        oid = next_order_id
        next_order_id += 1
        resting_orders[oid] = [side, price, size]
        lines.append(f"{time:.6f},1,{oid},{size},{price},{side}")

    def random_execution():
        nonlocal mid, variance_state
        if not resting_orders:
            return
        oid = rng.choice(list(resting_orders.keys()))
        side, price, size = resting_orders[oid]
        exec_size = min(size, rng.choice([50, 100, 100, 200]))
        resting_orders[oid][2] -= exec_size
        if resting_orders[oid][2] <= 0:
            del resting_orders[oid]
        lines.append(f"{time:.6f},4,{oid},{exec_size},{price},{side}")

        z = student_t(fat_tail_df)  # standardized shock -- the raw, unclamped draw
        raw_ticks = abs(z) * math.sqrt(variance_state)
        ticks_moved = max(1, round(raw_ticks))
        mid += ticks_moved * tick_size if side == -1 else -ticks_moved * tick_size

        # Textbook GARCH(1,1): feed back z^2 (the FULL-PRECISION shock,
        # not ticks_moved, which is floored at 1 and would understate
        # small shocks) -- a big |z| pushes variance_state up, and that
        # elevated state decays geometrically at rate garch_beta rather
        # than resetting next step. That persistence is what creates
        # measurable autocorrelation in |returns| ("vol clustering").
        variance_state = omega + garch_alpha * (z ** 2) + garch_beta * variance_state

        if record_mid_series:
            mid_series.append((time, mid))

    def random_cancel():
        if not resting_orders:
            return
        oid = rng.choice(list(resting_orders.keys()))
        side, price, size = resting_orders[oid]
        lines.append(f"{time:.6f},3,{oid},{size},{price},{side}")
        del resting_orders[oid]

    for _ in range(n_events):
        time += rng.uniform(0.01, 0.5)
        action = rng.choices(
            ["new", "execute", "cancel"],
            weights=[0.5, 0.3, 0.2],
        )[0]
        if action == "new" or not resting_orders:
            add_new_order()
        elif action == "execute":
            random_execution()
        else:
            random_cancel()

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    if record_mid_series:
        return output_path, mid_series
    return output_path


if __name__ == "__main__":
    path = generate_synthetic_lobster_data()
    print(f"Wrote synthetic data to {path}")
