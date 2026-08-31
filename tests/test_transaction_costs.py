"""
Hand-traced test of ReplayBacktest's fee_bps transaction cost, reusing
the exact fixture and fill trace from test_naive_mm.py (see that file
for the full walk-through of why fills land on 999900 buy / 1000100
sell, both size 100).

Trace with fee_bps=10 (0.10%, a representative maker-fee-sized cost):
  buy  fill: price=999900,  size=100 -> notional=99,990,000
                                       -> fee = 99,990,000 * 10/10000 = 99,990.0
  sell fill: price=1000100, size=100 -> notional=100,010,000
                                       -> fee = 100,010,000 * 10/10000 = 100,010.0
  total fees = 200,000.0
  cash WITHOUT fees (test_naive_mm.py) = 20,000
  cash WITH fees = 20,000 - 200,000 = -180,000.0
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lob_engine as lob
from strategies.naive_mm import NaiveMarketMaker, NaiveMarketMakerConfig
from backtest.replay_backtest import ReplayBacktest

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    events = lob.read_message_file("data/fill_test_scenario.csv")

    # --- regression: fee_bps=0.0 (the default) must reproduce
    # test_naive_mm.py's fee-free result exactly ---
    cfg = NaiveMarketMakerConfig(half_spread_ticks=1, tick_size=100, quote_size=100)
    strategy_no_fee = NaiveMarketMaker(cfg)
    backtest_no_fee = ReplayBacktest(events, strategy_no_fee)  # fee_bps defaults to 0.0
    backtest_no_fee.run()
    check(strategy_no_fee.cash == 20000, f"fee_bps=0.0 (default) reproduces the fee-free cash of 20000, got {strategy_no_fee.cash}")
    check(backtest_no_fee.total_fees_paid == 0.0, f"no fees charged when fee_bps=0.0, got {backtest_no_fee.total_fees_paid}")

    # --- fee_bps=10 (0.10%) applied to both fills ---
    strategy_fee = NaiveMarketMaker(NaiveMarketMakerConfig(half_spread_ticks=1, tick_size=100, quote_size=100))
    backtest_fee = ReplayBacktest(events, strategy_fee, fee_bps=10.0)
    backtest_fee.run()

    check(abs(backtest_fee.total_fees_paid - 200000.0) < 1e-6,
          f"total fees paid == 200000.0, got {backtest_fee.total_fees_paid}")
    check(abs(strategy_fee.cash - (-180000.0)) < 1e-6,
          f"cash with fee_bps=10 == -180000.0, got {strategy_fee.cash}")

    # --- inventory and fills themselves are UNCHANGED by fees -- fees
    # affect cash only, not the strategy's view of what it traded ---
    check(strategy_fee.inventory == strategy_no_fee.inventory == 0,
          "inventory unaffected by fees (still nets to 0)")
    check(len(strategy_fee.fills) == len(strategy_no_fee.fills) == 2,
          "fill count/prices/sizes unaffected by fees")

    # --- a negative fee_bps models a maker REBATE: cash should go UP
    # relative to the no-fee case, not down ---
    strategy_rebate = NaiveMarketMaker(NaiveMarketMakerConfig(half_spread_ticks=1, tick_size=100, quote_size=100))
    backtest_rebate = ReplayBacktest(events, strategy_rebate, fee_bps=-10.0)
    backtest_rebate.run()
    check(abs(strategy_rebate.cash - 220000.0) < 1e-6,
          f"negative fee_bps (a rebate) INCREASES cash vs the fee-free case: got {strategy_rebate.cash}, want 220000.0")

    print()
    if failures == 0:
        print("All transaction cost checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
