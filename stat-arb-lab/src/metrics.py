"""Standard performance/risk metrics for a daily PnL series."""
import numpy as np


def sharpe_ratio(daily_pnl, periods_per_year=252):
    std = daily_pnl.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(daily_pnl.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(daily_pnl, periods_per_year=252):
    downside = daily_pnl[daily_pnl < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if downside_std < 1e-12:
        return 0.0
    return float(daily_pnl.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(daily_pnl):
    cum = np.cumsum(daily_pnl)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    return float(drawdown.min())


def annualized_return(daily_pnl, periods_per_year=252):
    return float(daily_pnl.mean() * periods_per_year)


def summary(daily_pnl, daily_turnover=None, periods_per_year=252):
    out = {
        "total_pnl": float(np.sum(daily_pnl)),
        "annualized_pnl": annualized_return(daily_pnl, periods_per_year),
        "sharpe": sharpe_ratio(daily_pnl, periods_per_year),
        "sortino": sortino_ratio(daily_pnl, periods_per_year),
        "max_drawdown": max_drawdown(daily_pnl),
        "win_rate": float(np.mean(daily_pnl > 0)),
        "n_days": len(daily_pnl),
    }
    if daily_turnover is not None:
        out["mean_daily_turnover"] = float(np.mean(daily_turnover))
        out["total_turnover"] = float(np.sum(daily_turnover))
    return out
