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
import random


def generate_synthetic_lobster_data(
    n_events: int = 500,
    start_price: int = 1000000,   # LOBSTER units: $100.00
    tick_size: int = 100,          # $0.01
    seed: int = 42,
    output_path: str = "data/synthetic_large.csv",
):
    rng = random.Random(seed)
    lines = []
    time = 34200.0  # 9:30:00 AM in seconds-since-midnight, matching LOBSTER convention
    next_order_id = 1
    mid = start_price

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

    return output_path


if __name__ == "__main__":
    path = generate_synthetic_lobster_data()
    print(f"Wrote synthetic data to {path}")
