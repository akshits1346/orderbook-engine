"""
Hand-traced test of NaiveMarketMaker + ReplayBacktest together, using
data/fill_test_scenario.csv. Every expected number below was computed
by hand before running the code -- see the trace comment.

Config used: tick_size=100, half_spread_ticks=1, quote_size=100.
That makes our quotes land EXACTLY on the existing historical price
levels (999900 bid, 1000100 ask) -- deliberately, so the test doesn't
need to fabricate executions at prices no historical order ever sat at.

Trace of data/fill_test_scenario.csv:
  1. new id=1 buy  300 @  999900          book: bid[999900]={1:300}
                                          (mid still None: no ask yet)
  2. new id=2 sell 300 @ 1000100          book: ask[1000100]={2:300}
                                          mid = (999900+1000100)//2 = 1000000
                                          -> strategy requotes:
                                             bid@999900 (behind id=1, size_ahead=300)
                                             ask@1000100 (behind id=2, size_ahead=300)
  3. exec id=1 for 300 @ 999900           bid_volume_since_quote: 0->300
                                          300 >= size_ahead_at_quote(300) -> FILL
                                          fill: buy 100 shares @ 999900
                                          (our bid order removed; id=1 then
                                           executed to 0 by apply_message)
  4. exec id=2 for 300 @ 1000100          ask_volume_since_quote: 0->300
                                          300 >= size_ahead_at_quote(300) -> FILL
                                          fill: sell 100 shares @ 1000100

Expected final state:
  fills = [ (buy, 999900, 100), (sell, 1000100, 100) ]
  inventory = 100 - 100 = 0
  cash = -999900*100 + 1000100*100 = -99,990,000 + 100,010,000 = 20,000
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
    check(len(events) == 4, "parsed 4 rows from fill_test_scenario.csv")

    cfg = NaiveMarketMakerConfig(half_spread_ticks=1, tick_size=100, quote_size=100)
    strategy = NaiveMarketMaker(cfg)
    backtest = ReplayBacktest(events, strategy)
    backtest.run()

    check(len(strategy.fills) == 2, f"exactly 2 fills recorded, got {len(strategy.fills)}")

    if len(strategy.fills) == 2:
        buy_fill, sell_fill = strategy.fills
        check(buy_fill["side"] == "buy" and buy_fill["price"] == 999900 and buy_fill["size"] == 100,
              "first fill is a buy of 100 @ 999900")
        check(sell_fill["side"] == "sell" and sell_fill["price"] == 1000100 and sell_fill["size"] == 100,
              "second fill is a sell of 100 @ 1000100")

    check(strategy.inventory == 0, f"final inventory is 0, got {strategy.inventory}")
    check(strategy.cash == 20000, f"final cash is 20000, got {strategy.cash}")

    print()
    if failures == 0:
        print("All naive market maker fill checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
