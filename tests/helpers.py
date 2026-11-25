"""Small helpers shared by the tests."""

import pandas as pd


def make_prices(values: list[float], start: str = "2024-01-01") -> pd.Series:
    """Build a price Series on consecutive business days, like load_prices() returns."""
    index = pd.bdate_range(start, periods=len(values), name="date")
    return pd.Series(values, index=index, name="close", dtype=float)
