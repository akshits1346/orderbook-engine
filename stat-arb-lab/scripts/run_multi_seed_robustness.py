"""
Multi-seed robustness check for the PCA stat-arb strategy: honest
walk-forward vs. the in-sample lookahead-biased control, under two
regimes -- a stationary factor basket (true dynamics never change) and
one with a single structural break in factor loadings partway through.
Mirrors the sibling orderbook-engine project's multi-seed methodology:
one seed is one random draw, and the point is whether a finding
generalizes across many, not whether it happened to look a certain way
once.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from data_gen import generate_factor_basket
from backtest import run_walk_forward_backtest, run_in_sample_backtest
from metrics import summary


def run_scenario(label, regime_change_at, n_seeds=20):
    wf_sharpes, is_sharpes = [], []
    wf_pnls, is_pnls = [], []
    wf_turnover, is_turnover = [], []

    for seed in range(n_seeds):
        b = generate_factor_basket(n_assets=20, n_days=750, n_factors=3, idio_theta=0.04,
                                    idio_vol=0.008, regime_change_at=regime_change_at, seed=seed)
        wf = run_walk_forward_backtest(b["returns"], n_factors=3, train_window=252,
                                        rebalance_every=5, cost_bps=5.0)
        is_ = run_in_sample_backtest(b["returns"], n_factors=3, cost_bps=5.0)

        sw = summary(wf["daily_pnl"], wf["daily_turnover"])
        si = summary(is_["daily_pnl"], is_["daily_turnover"])
        wf_sharpes.append(sw["sharpe"]); is_sharpes.append(si["sharpe"])
        wf_pnls.append(sw["total_pnl"]); is_pnls.append(si["total_pnl"])
        wf_turnover.append(sw["mean_daily_turnover"]); is_turnover.append(si["mean_daily_turnover"])

    print(f"## {label} ({n_seeds} seeds)")
    print()
    print("| | mean Sharpe | median Sharpe | mean total PnL | mean daily turnover |")
    print("|---|---|---|---|---|")
    print(f"| walk-forward | {np.mean(wf_sharpes):.3f} | {np.median(wf_sharpes):.3f} | "
          f"{np.mean(wf_pnls):.3f} | {np.mean(wf_turnover):.3f} |")
    print(f"| in-sample (lookahead) | {np.mean(is_sharpes):.3f} | {np.median(is_sharpes):.3f} | "
          f"{np.mean(is_pnls):.3f} | {np.mean(is_turnover):.3f} |")
    n_wf_wins = sum(w > i for w, i in zip(wf_sharpes, is_sharpes))
    n_wf_pnl_wins = sum(w > i for w, i in zip(wf_pnls, is_pnls))
    print()
    print(f"Walk-forward has higher Sharpe in {n_wf_wins}/{n_seeds} seeds, "
          f"higher total PnL in {n_wf_pnl_wins}/{n_seeds} seeds.")
    print()
    return {
        "wf_sharpes": wf_sharpes, "is_sharpes": is_sharpes,
        "wf_pnls": wf_pnls, "is_pnls": is_pnls,
    }


def main():
    run_scenario("Stationary factor basket (no regime change)", regime_change_at=None)
    run_scenario("Factor loadings undergo a structural break at day 500", regime_change_at=500)


if __name__ == "__main__":
    main()
