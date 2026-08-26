"""
Runs naive market maker and Avellaneda-Stoikov (with sigma/kappa
calibrated from the SAME data being replayed) through identical
backtests and prints a side-by-side comparison.

HONESTY NOTE, again, because it matters: whatever data path is passed
in here determines whether this comparison means anything. Point this
at data/synthetic_large.csv and you get a demonstration that the
pipeline works end-to-end. Point it at real downloaded LOBSTER data and
you get an actual finding. Don't mix these up in a README or a write-up.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lob_engine as lob
from strategies.naive_mm import NaiveMarketMaker, NaiveMarketMakerConfig
from strategies.avellaneda_stoikov import AvellanedaStoikov, AvellanedaStoikovConfig
from backtest.replay_backtest import ReplayBacktest
from backtest.calibrate import estimate_sigma_from_events, estimate_kappa_from_events


def run_one(strategy_name, strategy, events):
    backtest = ReplayBacktest(events, strategy)
    history = backtest.run()
    final_mid = history[-1][1] if history else None  # last MTM value, not price -- see note below
    return {
        "name": strategy_name,
        "fills": len(strategy.fills),
        "final_inventory": strategy.inventory,
        "final_cash": strategy.cash,
        "mtm_points": len(history),
        "final_mtm": history[-1][1] if history else 0.0,
    }


def main(data_path="data/synthetic_large.csv", gamma=0.01, tick_size=100, quote_size=100, max_inventory=500):
    events = lob.read_message_file(data_path)
    print(f"Loaded {len(events)} events from {data_path}\n")

    print("Calibrating sigma and kappa from this data...")
    sigma = estimate_sigma_from_events(events)
    kappa = estimate_kappa_from_events(events, tick_size=tick_size)
    print(f"  sigma = {sigma:.4f}")
    print(f"  kappa = {kappa:.4f}\n")

    last_time = events[-1].time
    first_time = events[0].time

    naive_cfg = NaiveMarketMakerConfig(
        half_spread_ticks=1, tick_size=tick_size, quote_size=quote_size, max_inventory=max_inventory
    )
    naive = NaiveMarketMaker(naive_cfg)

    as_cfg = AvellanedaStoikovConfig(
        gamma=gamma, kappa=kappa, sigma=sigma,
        session_end_time=last_time, tick_size=tick_size,
        quote_size=quote_size, max_inventory=max_inventory,
    )
    avellaneda = AvellanedaStoikov(as_cfg)

    results = [
        run_one("Naive symmetric", naive, events),
        run_one("Avellaneda-Stoikov (calibrated)", avellaneda, events),
    ]

    print(f"{'Strategy':<32} {'Fills':>7} {'Inventory':>11} {'Cash':>12} {'Final MTM':>12}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<32} {r['fills']:>7} {r['final_inventory']:>11} "
              f"{r['final_cash']:>12} {r['final_mtm']:>12.2f}")

    return results


if __name__ == "__main__":
    main()
