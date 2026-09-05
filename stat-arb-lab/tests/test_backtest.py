"""
Backtest-engine correctness tests: no-lookahead guarantee (walk-forward
decisions must be reproducible from ONLY the data available at decision
time), transaction costs actually drag on PnL, and a flat (never-traded)
book has exactly zero PnL and turnover.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from data_gen import generate_factor_basket
from backtest import run_walk_forward_backtest, run_in_sample_backtest
from metrics import summary, max_drawdown, sharpe_ratio

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    b = generate_factor_basket(n_assets=15, n_days=600, n_factors=2, idio_theta=0.04, seed=30)

    # --- no-lookahead: truncating the return series AFTER a rebalance
    # date should not change any decision made ON OR BEFORE that date.
    # Concretely: running the backtest on returns[:600] and on
    # returns[:400] should produce IDENTICAL daily_pnl/state_history for
    # every day up to 400 - train_window, since a walk-forward fit at any
    # earlier date never used data past it. ---
    train_window = 200
    full = run_walk_forward_backtest(b["returns"][:600], n_factors=2, train_window=train_window,
                                      rebalance_every=5, cost_bps=5.0)
    truncated = run_walk_forward_backtest(b["returns"][:400], n_factors=2, train_window=train_window,
                                           rebalance_every=5, cost_bps=5.0)
    n_common = len(truncated["daily_pnl"])
    check(np.allclose(full["daily_pnl"][:n_common], truncated["daily_pnl"], atol=1e-10),
          f"truncating the future leaves every earlier day's P&L unchanged (no lookahead), "
          f"checked {n_common} common days")
    check(np.array_equal(full["state_history"][:n_common], truncated["state_history"]),
          "truncating the future leaves every earlier day's position state unchanged")

    # --- transaction costs strictly reduce PnL relative to zero-cost, all else equal ---
    zero_cost = run_walk_forward_backtest(b["returns"], n_factors=2, train_window=train_window,
                                           rebalance_every=5, cost_bps=0.0)
    with_cost = run_walk_forward_backtest(b["returns"], n_factors=2, train_window=train_window,
                                           rebalance_every=5, cost_bps=20.0)
    check(with_cost["daily_pnl"].sum() < zero_cost["daily_pnl"].sum(),
          f"transaction costs reduce total P&L: {with_cost['daily_pnl'].sum():.4f} < {zero_cost['daily_pnl'].sum():.4f}")
    check(np.allclose(with_cost["daily_turnover"], zero_cost["daily_turnover"]),
          "turnover itself (share volume) is unaffected by the cost RATE, only PnL is")

    # --- a threshold so wide nothing ever opens a position gives exactly zero PnL/turnover ---
    never_trade = run_walk_forward_backtest(b["returns"], n_factors=2, train_window=train_window,
                                             rebalance_every=5, s_open=1e6, cost_bps=5.0)
    check(np.all(never_trade["daily_pnl"] == 0.0), "an impossible entry threshold trades never, PnL is exactly 0")
    check(np.all(never_trade["daily_turnover"] == 0.0), "an impossible entry threshold has exactly 0 turnover")

    # --- metrics sanity: a strictly positive constant PnL series has Sharpe > 0 and zero drawdown ---
    constant_pnl = np.full(100, 0.001)
    check(sharpe_ratio(constant_pnl) == 0.0, "a truly constant (zero-variance) PnL series has Sharpe defined as 0 (not inf/nan)")
    check(max_drawdown(constant_pnl) >= -1e-12, "a monotonically non-decreasing PnL curve has ~zero max drawdown")

    increasing_then_drop = np.array([0.01] * 10 + [-0.5] + [0.01] * 10)
    dd = max_drawdown(increasing_then_drop)
    check(dd < -0.4, f"a sharp drop shows up as a large negative max drawdown, got {dd:.4f}")

    print()
    if failures == 0:
        print("All backtest engine checks passed.")
        return 0
    print(f"{failures} check(s) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
