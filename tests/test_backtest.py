"""Tests for returns, costs, equity curves, and the absence of look-ahead bias."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.strategy import generate_signals
from tests.helpers import make_prices


def test_returns_and_equity_match_hand_calculation(crossover_prices):
    bt = run_backtest(generate_signals(crossover_prices, 2, 3), cost=0.0)

    # Warm-up (days 0-1) is dropped, so the backtest starts on day 2 at close 10.
    assert bt["close"].tolist() == [10, 13, 16, 7, 4, 4]
    np.testing.assert_allclose(bt["asset_return"], [0, 3 / 10, 3 / 13, -9 / 16, -3 / 7, 0])
    # Long only on days 4 and 5 (position 1): earn 16/13 then 7/16.
    np.testing.assert_allclose(bt["strategy_return"], [0, 0, 3 / 13, -9 / 16, 0, 0])
    assert bt["strategy_equity"].iloc[-1] == pytest.approx(10_000 * 16 / 13 * 7 / 16)  # 5,384.62
    assert bt["buyhold_equity"].iloc[-1] == pytest.approx(10_000 * 4 / 10)  # 4,000


def test_costs_are_charged_on_every_position_change(crossover_prices):
    cost = 0.01
    bt = run_backtest(generate_signals(crossover_prices, 2, 3), cost=cost)

    assert bt["trade"].tolist() == [0, 0, 1, 0, 1, 0]  # enter on day 4, exit on day 6
    expected = 10_000 * (1 + 3 / 13 - cost) * (1 - 9 / 16) * (1 - cost)
    assert bt["strategy_equity"].iloc[-1] == pytest.approx(expected)
    # Buy-and-hold pays one entry cost on the first day only.
    assert bt["buyhold_equity"].iloc[-1] == pytest.approx(10_000 * (1 - cost) * 4 / 10)


def test_strategy_does_not_earn_the_move_that_creates_the_signal():
    """No look-ahead, case 1: a price jump that triggers a buy must not be captured.

    The price doubles on day 4. With short=1 the signal turns on at that same close.
    A backtest without the one-day shift would book +100% on day 4. Ours must book 0%,
    because we only learn about the jump after it has happened.
    """
    prices = make_prices([10, 10, 10, 10, 20, 20, 20])
    bt = run_backtest(generate_signals(prices, short=1, long=3), cost=0.0)

    jump_day = prices.index[4]
    assert bt.loc[jump_day, "signal"] == 1
    assert bt.loc[jump_day, "position"] == 0
    assert bt.loc[jump_day, "strategy_return"] == 0
    assert bt["strategy_equity"].iloc[-1] == pytest.approx(10_000)  # not 20,000


def test_changing_future_prices_does_not_change_past_decisions():
    """No look-ahead, case 2: results up to day k can't depend on prices after day k.

    We run the backtest twice on a random walk. The second time, every price from
    day k onward is pushed hard in whichever direction flips day k's signal. The
    position held on day k was decided at the close of day k-1, so it must not move,
    and neither may any return before day k.
    """
    rng = np.random.default_rng(seed=42)
    original = make_prices(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
    k = 150
    signal_k = generate_signals(original, 5, 20)["signal"].iloc[k]
    scrambled = original.copy()
    scrambled.iloc[k:] *= 0.2 if signal_k == 1 else 3.0  # crash or spike from day k on

    bt_a = run_backtest(generate_signals(original, 5, 20), cost=0.001)
    bt_b = run_backtest(generate_signals(scrambled, 5, 20), cost=0.001)

    day_k = original.index[k]
    before_k = original.index[k - 1]
    # The new prices really do change the signal on day k...
    assert bt_a.loc[day_k, "signal"] != bt_b.loc[day_k, "signal"]
    # ...but nothing the strategy did up to day k is affected.
    pd.testing.assert_series_equal(bt_a.loc[:day_k, "position"], bt_b.loc[:day_k, "position"])
    pd.testing.assert_series_equal(bt_a.loc[:before_k, "strategy_return"],
                                   bt_b.loc[:before_k, "strategy_return"])
