"""Shared fixtures: tiny hand-made price series."""

import pandas as pd
import pytest

from tests.helpers import make_prices


@pytest.fixture
def crossover_prices() -> pd.Series:
    """Eight prices with one up-crossover and one down-crossover for SMA 2/3.

    Worked out by hand (day: close, sma2, sma3, signal, position):
        0: 10   nan   nan   0  0
        1: 10   10    nan   0  0
        2: 10   10    10    0  0   (10 > 10 is False)
        3: 13   11.5  11    1  0   <- crossover at the close of day 3
        4: 16   14.5  13    1  1   <- we hold from day 4
        5:  7   11.5  12    0  1   <- cross back down at the close of day 5
        6:  4   5.5   9     0  0   <- in cash from day 6
        7:  4   4     5     0  0
    """
    return make_prices([10, 10, 10, 13, 16, 7, 4, 4])
