"""Download daily adjusted close prices from Yahoo Finance and cache them as CSV.

The rest of the project only ever calls `load_prices()`. It returns a pandas Series
of adjusted close prices indexed by date, reading from the local cache when it can
and downloading only when it has to.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def cache_path(ticker: str, start: str, end: str | None, data_dir: Path = DATA_DIR) -> Path:
    """Return the CSV path used to cache one (ticker, start, end) download."""
    end_label = end if end is not None else "latest"
    return data_dir / f"{ticker.upper()}_{start}_{end_label}.csv"


def download_prices(ticker: str, start: str, end: str | None = None) -> pd.Series:
    """Download daily adjusted close prices from Yahoo Finance.

    `auto_adjust=True` makes the "Close" column already adjusted for splits and
    dividends, so a percentage change in this series is the total return an
    investor actually got.

    Args:
        ticker: Yahoo Finance symbol, e.g. "SPY".
        start: First date to fetch, "YYYY-MM-DD".
        end: Last date to fetch (exclusive, per yfinance). None means up to today.

    Returns:
        Series named "close", indexed by a timezone-naive DatetimeIndex named "date".
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        # yfinance prints network errors instead of raising them, so an empty
        # result can mean a bad symbol OR a failed connection.
        raise ValueError(
            f"No data returned for {ticker!r}. Check the symbol, the dates, and your "
            "connection (yfinance prints the underlying error above this message)."
        )

    # Recent yfinance versions return MultiIndex columns like ("Close", "SPY")
    # even for one ticker. Flatten them to plain "Close", "Open", ...
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna().astype(float).rename("close")
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).rename("date")
    return close


def validate_prices(prices: pd.Series) -> None:
    """Raise ValueError if the price series is unusable for a backtest."""
    if prices.empty:
        raise ValueError("Price series is empty.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted in ascending order.")
    if prices.index.has_duplicates:
        raise ValueError("Price series contains duplicate dates.")
    if (prices <= 0).any():
        raise ValueError("Price series contains zero or negative prices.")


def load_prices(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str | None = None,
    data_dir: Path = DATA_DIR,
    refresh: bool = False,
) -> pd.Series:
    """Load adjusted close prices, using the local CSV cache when available.

    Args:
        ticker: Yahoo Finance symbol.
        start: First date, "YYYY-MM-DD".
        end: Last date (exclusive), or None for "up to today".
        data_dir: Folder that holds the cached CSV files.
        refresh: If True, ignore the cache and download again. Useful because a
            cache with end=None goes stale as new trading days happen.

    Returns:
        Series of adjusted close prices named "close", indexed by date.
    """
    path = cache_path(ticker, start, end, data_dir)

    if path.exists() and not refresh:
        prices = pd.read_csv(path, index_col="date", parse_dates=True)["close"]
    else:
        prices = download_prices(ticker, start, end)
        data_dir.mkdir(parents=True, exist_ok=True)
        prices.to_csv(path)

    validate_prices(prices)
    return prices
