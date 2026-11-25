"""Performance metrics computed from a series of daily returns.

Everything works on daily simple returns (0.01 = +1%). "Annualized" numbers use
252 trading days per year, the usual convention for US equities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(returns: pd.Series) -> float:
    """Overall growth over the whole period: 0.5 means +50%."""
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate: the constant yearly return giving the same end value."""
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    return float((1 + total_return(returns)) ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    """Standard deviation of daily returns scaled to a year: how bumpy the ride is."""
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized excess return per unit of volatility: return earned per unit of risk.

    Args:
        returns: Daily returns.
        risk_free_rate: Annual risk-free rate, e.g. 0.02 for 2%.
    """
    excess = returns - risk_free_rate / TRADING_DAYS
    std = excess.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Percentage below the previous peak on each day (0 at a new high, -0.3 = 30% down)."""
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss over the period: the most pain you would have sat through."""
    return float(drawdown_series(returns).min())


def number_of_trades(positions: pd.Series) -> int:
    """Count of position changes, including the first entry from cash."""
    return int(positions.diff().abs().fillna(positions.abs()).sum())


def time_in_market(positions: pd.Series) -> float:
    """Fraction of days holding the asset (position != 0)."""
    return float((positions != 0).mean())


def compute_metrics(
    returns: pd.Series, positions: pd.Series, risk_free_rate: float = 0.0
) -> dict[str, float]:
    """Compute every metric for one return stream."""
    return {
        "Total return": total_return(returns),
        "CAGR": cagr(returns),
        "Annualized volatility": annualized_volatility(returns),
        "Sharpe ratio": sharpe_ratio(returns, risk_free_rate),
        "Max drawdown": max_drawdown(returns),
        "Number of trades": number_of_trades(positions),
        "Time in market": time_in_market(positions),
    }


def summary_table(backtest: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """Side-by-side metrics for the strategy and buy-and-hold.

    Args:
        backtest: Output of `run_backtest()`.
        risk_free_rate: Annual risk-free rate used in the Sharpe ratio.
    """
    always_long = pd.Series(1, index=backtest.index)
    return pd.DataFrame(
        {
            "Strategy": compute_metrics(
                backtest["strategy_return"], backtest["position"], risk_free_rate
            ),
            "Buy & Hold": compute_metrics(
                backtest["buyhold_return"], always_long, risk_free_rate
            ),
        }
    )


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw numbers into readable strings (percentages, 2-decimal Sharpe)."""
    out = table.astype(object)
    for metric in table.index:
        if metric == "Sharpe ratio":
            out.loc[metric] = table.loc[metric].map("{:.2f}".format)
        elif metric == "Number of trades":
            out.loc[metric] = table.loc[metric].map("{:.0f}".format)
        else:
            out.loc[metric] = table.loc[metric].map("{:.1%}".format)
    return out
