"""
Direct test of the fix proposed in the README for AS's gamma-sensitivity
finding: does capping T-t with a rolling horizon restore fills at
gamma=0.01 (the model's own "reasonable default", which produced ZERO
fills against data/synthetic_large.csv under the fixed-session-end
version), without resorting to shrinking gamma to unrealistic values?

Sweeps horizon over a range of values (including horizon=None, i.e. the
original fixed-session-end behavior, as the control) at fixed gamma=0.01
and fixed calibrated sigma/kappa, and reports the resulting half-spread
at session start and the fill count -- the same two numbers the README's
gamma sweep reported, so the two tables are directly comparable.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lob_engine as lob
from strategies.avellaneda_stoikov import AvellanedaStoikov, AvellanedaStoikovConfig
from backtest.replay_backtest import ReplayBacktest
from backtest.calibrate import estimate_sigma_from_events, estimate_kappa_from_events


def main(data_path="data/synthetic_large.csv", gamma=0.01, tick_size=100,
         quote_size=100, max_inventory=500,
         horizons=(None, 30.0, 10.0, 5.0, 2.0, 1.0)):
    events = lob.read_message_file(data_path)
    print(f"Loaded {len(events)} events from {data_path}\n")

    sigma = estimate_sigma_from_events(events)
    kappa = estimate_kappa_from_events(events, tick_size=tick_size)
    print(f"Calibrated: sigma={sigma:.4f}, kappa={kappa:.4f}, gamma={gamma} (fixed, unchanged)\n")

    first_time = events[0].time
    last_time = events[-1].time
    session_length = last_time - first_time

    print(f"{'horizon (s)':>12} {'half-spread @ start (ticks)':>28} {'fills':>7}")
    print("-" * 51)

    results = []
    for horizon in horizons:
        as_cfg = AvellanedaStoikovConfig(
            gamma=gamma, kappa=kappa, sigma=sigma,
            session_end_time=last_time, horizon=horizon,
            tick_size=tick_size, quote_size=quote_size, max_inventory=max_inventory,
        )
        strategy = AvellanedaStoikov(as_cfg)

        bid, ask = strategy.desired_quotes(mid_price=0, time=first_time)
        half_spread_ticks = (ask - bid) / 2 / tick_size

        backtest = ReplayBacktest(events, strategy)
        backtest.run()

        label = "None (fixed)" if horizon is None else f"{horizon:g}"
        print(f"{label:>12} {half_spread_ticks:>28.2f} {len(strategy.fills):>7}")
        results.append({"horizon": horizon, "half_spread_ticks": half_spread_ticks,
                         "fills": len(strategy.fills)})

    print(f"\n(session length in this data: {session_length:.1f}s)")
    return results


if __name__ == "__main__":
    main()
