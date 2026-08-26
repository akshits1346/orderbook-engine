"""
Proves the pybind11 bindings actually work end-to-end from Python --
same hand-traced scenario as tests/test_reconstruction.cpp, so if
this passes, the C++ engine and the Python bridge agree on behavior.

Run with: python3 tests/test_bindings.py
(requires lob_engine*.so to be built and importable -- see build_bindings.sh)
"""
import sys
import os

# so `import lob_engine` finds the compiled .so sitting in the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lob_engine as lob

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    book = lob.OrderBook()

    book.add_limit_order(1, lob.Side.Buy, 999900, 100, 1.0)
    book.add_limit_order(2, lob.Side.Buy, 999800, 200, 2.0)
    book.add_limit_order(3, lob.Side.Sell, 1000100, 150, 3.0)

    check(book.best_bid() == 999900, "best bid is 999900 after two buy orders")
    check(book.best_ask() == 1000100, "best ask is 1000100 after one sell order")

    book.cancel_partial(1, 40)  # 100 -> 60, keeps queue position
    check(book.size_ahead_of(1) == 0, "id=1 still at front after partial cancel")

    book.add_limit_order(4, lob.Side.Buy, 999900, 50, 5.0)  # joins behind id=1
    check(book.size_ahead_of(4) == 60, "id=4 has 60 shares ahead of it (id=1's remainder)")

    book.execute_order(1, 60)  # fully fills id=1
    check(book.size_ahead_of(4) == 0, "id=4 moves to front once id=1 is fully executed")

    book.delete_order(2)

    bids = book.bid_levels(5)
    check(len(bids) == 1, "only one bid level remains")
    check(bids[0].price == 999900, "remaining bid level is 999900")
    check(bids[0].size == 50, "remaining bid level size is 50")

    asks = book.ask_levels(5)
    check(len(asks) == 1, "ask side untouched: still one level")
    check(asks[0].price == 1000100 and asks[0].size == 150, "ask level unchanged at 150")

    check(book.order_count() == 2, "exactly 2 live orders remain")

    # --- also exercise the LOBSTER replay path from Python ---
    events = lob.read_message_file("data/sample_message.csv")
    check(len(events) == 10, "parsed exactly 10 rows from sample_message.csv via Python")

    replay_book = lob.OrderBook()
    lob.replay_all(replay_book, events)
    check(replay_book.best_bid() == 1000000, "replay: best bid is 1000000")
    check(replay_book.best_ask() == 1000200, "replay: best ask is 1000200")
    check(replay_book.order_count() == 3, "replay: 3 live orders remain")

    print()
    if failures == 0:
        print("All Python binding checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
