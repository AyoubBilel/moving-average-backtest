"""Moving averages, crossover signals, and the positions we actually hold.

Two columns matter and they mean different things:

- `signal`   : what the rule says at today's close (1 = short SMA above long SMA).
- `position` : what we hold during today. It is yesterday's signal, because we can't
               act on a close price until that close has happened.
"""

from __future__ import annotations

import pandas as pd


def add_moving_averages(prices: pd.Series, short: int, long: int) -> pd.DataFrame:
    """Return a DataFrame with the close price and two simple moving averages.

    A simple moving average (SMA) of window n on day t is the mean of the last n
    closes, including day t. The first n-1 values are NaN because there isn't
    enough history yet (the "warm-up" period).
    """
    return pd.DataFrame(
        {
            "close": prices,
            "sma_short": prices.rolling(window=short).mean(),
            "sma_long": prices.rolling(window=long).mean(),
        }
    )


def generate_signals(prices: pd.Series, short: int = 50, long: int = 200) -> pd.DataFrame:
    """Compute SMAs, the crossover signal, and the next-day position.

    Args:
        prices: Daily adjusted close prices indexed by date.
        short: Window of the fast moving average, in trading days.
        long: Window of the slow moving average, in trading days.

    Returns:
        DataFrame with columns close, sma_short, sma_long, signal, position.
        `signal` is 1 when sma_short > sma_long, else 0 (also 0 during warm-up).
        `position` is the signal shifted forward one day.
    """
    if not 0 < short < long:
        raise ValueError(f"Need 0 < short < long, got short={short}, long={long}.")

    df = add_moving_averages(prices, short, long)

    # NaN > NaN is False, so the warm-up period becomes 0 (in cash). That's what we want.
    df["signal"] = (df["sma_short"] > df["sma_long"]).astype(int)

    # The key line of the project. The signal uses day t's close, so we can only
    # hold the position from day t+1 onward. Without this shift, the backtest would
    # earn day t's return using information that only exists at the end of day t.
    df["position"] = df["signal"].shift(1).fillna(0).astype(int)
    return df
