"""Tests for the performance metrics on series small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    annualized_volatility,
    cagr,
    max_drawdown,
    number_of_trades,
    sharpe_ratio,
    time_in_market,
    total_return,
)
from tests.helpers import make_prices


def returns_from_equity(values: list[float]) -> pd.Series:
    """Daily returns that reproduce the given equity path."""
    return make_prices(values).pct_change().fillna(0.0)


def test_total_return():
    assert total_return(returns_from_equity([100, 110, 99, 150])) == pytest.approx(0.5)


def test_cagr_of_doubling_twice_in_two_years_is_100_percent():
    dates = pd.to_datetime(["2020-01-01", "2022-01-01"])
    returns = pd.Series([0.0, 3.0], index=dates)  # 1 -> 4 over 731 calendar days
    assert cagr(returns) == pytest.approx(1.0, abs=0.002)


def test_max_drawdown_is_worst_fall_from_a_peak():
    # Peak 120, then trough 60: -50%. The later fall from 130 to 117 is only -10%.
    returns = returns_from_equity([100, 120, 60, 90, 130, 117])
    assert max_drawdown(returns) == pytest.approx(-0.5)


def test_sharpe_matches_formula():
    returns = make_prices([0.02, 0.0, 0.02, 0.0])
    expected = 0.01 / np.std([0.02, 0.0, 0.02, 0.0], ddof=1) * np.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_risk_free_rate_lowers_sharpe():
    returns = make_prices([0.02, 0.0, 0.02, 0.0])
    assert sharpe_ratio(returns, risk_free_rate=0.05) < sharpe_ratio(returns)


def test_sharpe_is_nan_when_returns_never_vary():
    assert np.isnan(sharpe_ratio(make_prices([0.0, 0.0, 0.0])))


def test_volatility_is_zero_for_constant_returns():
    assert annualized_volatility(make_prices([0.01, 0.01, 0.01])) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "positions, trades, in_market",
    [
        ([0, 1, 1, 0, 1], 3, 0.6),  # enter, exit, enter
        ([1, 1, 0], 2, 2 / 3),  # starting long counts as an entry
        ([0, 0, 0], 0, 0.0),
    ],
)
def test_trades_and_time_in_market(positions, trades, in_market):
    series = make_prices(positions)
    assert number_of_trades(series) == trades
    assert time_in_market(series) == pytest.approx(in_market)
