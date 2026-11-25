"""Tests for moving averages, signals, and the one-day position shift."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import add_moving_averages, generate_signals
from tests.helpers import make_prices


def test_sma_matches_hand_calculation():
    df = add_moving_averages(make_prices([1, 2, 3, 4, 5]), short=2, long=3)
    # SMA3 on day 2 = (1+2+3)/3 = 2, day 3 = (2+3+4)/3 = 3, day 4 = 4.
    np.testing.assert_allclose(df["sma_long"], [np.nan, np.nan, 2, 3, 4])
    np.testing.assert_allclose(df["sma_short"], [np.nan, 1.5, 2.5, 3.5, 4.5])


def test_signal_and_position_match_hand_calculation(crossover_prices):
    df = generate_signals(crossover_prices, short=2, long=3)
    assert df["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    assert df["position"].tolist() == [0, 0, 0, 0, 1, 1, 0, 0]


def test_position_is_previous_days_signal(crossover_prices):
    df = generate_signals(crossover_prices, short=2, long=3)
    assert df["position"].iloc[0] == 0
    assert (df["position"].iloc[1:].to_numpy() == df["signal"].iloc[:-1].to_numpy()).all()


def test_warm_up_period_is_in_cash():
    df = generate_signals(make_prices([5, 6, 7, 8, 9, 10]), short=2, long=4)
    # Prices only rise, but SMA4 doesn't exist until day 3, so days 0-2 must be 0.
    assert df["signal"].iloc[:3].tolist() == [0, 0, 0]


@pytest.mark.parametrize("short, long", [(200, 50), (50, 50), (0, 10)])
def test_invalid_windows_raise(short, long):
    with pytest.raises(ValueError):
        generate_signals(pd.Series(dtype=float), short, long)
