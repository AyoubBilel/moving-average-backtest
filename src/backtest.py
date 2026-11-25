"""Turn positions into daily returns, apply transaction costs, build equity curves.

Input is the DataFrame from `strategy.generate_signals()`. Its `position` column is
already shifted, so every row here only uses information available the day before.
Also holds the parameter sweep and the in-sample / out-of-sample test, which are
just the backtest run many times.
"""

from __future__ import annotations

import pandas as pd

from src.metrics import compute_metrics, sharpe_ratio
from src.strategy import generate_signals


def run_backtest(
    signals: pd.DataFrame,
    cost: float = 0.001,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """Backtest the strategy and a buy-and-hold benchmark over the same dates.

    Rows before the long SMA exists (the warm-up) are dropped, so both strategies
    start on the same day with the same capital. Otherwise buy-and-hold would get a
    head start of ~200 days that the strategy could never have used.

    Args:
        signals: Output of `generate_signals()`: needs close, sma_long, position.
        cost: Transaction cost as a fraction of the traded amount, charged every
            time the position changes (0.001 = 0.1%).
        initial_capital: Starting value of both equity curves.

    Returns:
        The input columns plus asset_return, trade, strategy_return,
        buyhold_return, strategy_equity, buyhold_equity.
    """
    df = signals.loc[signals["sma_long"].first_valid_index():].copy()

    # Daily simple return of the asset: today's close vs yesterday's close.
    df["asset_return"] = df["close"].pct_change().fillna(0.0)

    # 1 on days the position changes (cash -> long or long -> cash), else 0.
    df["trade"] = df["position"].diff().abs().fillna(df["position"]).astype(int)

    # Earn the asset's return only when long, and pay the cost on trade days.
    df["strategy_return"] = df["position"] * df["asset_return"] - df["trade"] * cost

    # Buy-and-hold pays one entry cost on day one, then just holds.
    df["buyhold_return"] = df["asset_return"]
    df.iloc[0, df.columns.get_loc("buyhold_return")] -= cost

    # Equity curve = starting capital compounded by each day's return.
    df["strategy_equity"] = initial_capital * (1 + df["strategy_return"]).cumprod()
    df["buyhold_equity"] = initial_capital * (1 + df["buyhold_return"]).cumprod()
    return df


def backtest_period(
    prices: pd.Series,
    short: int,
    long: int,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    cost: float = 0.001,
) -> pd.DataFrame:
    """Backtest one parameter pair, evaluated only between `start` and `end`.

    Signals are computed on the full price history first, so the moving averages
    on `start` are already warmed up with earlier prices. That is not look-ahead:
    rolling means only ever use past data. Then we keep just the evaluation window.
    """
    signals = generate_signals(prices, short, long).loc[start:end]
    return run_backtest(signals, cost=cost)


def parameter_sweep(
    prices: pd.Series,
    shorts: list[int],
    longs: list[int],
    cost: float = 0.001,
    risk_free_rate: float = 0.0,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, float]:
    """Backtest every (short, long) pair and return a grid of Sharpe ratios.

    All pairs are evaluated over the same dates, starting once the longest window
    has warmed up (or at `start`, whichever is later). Otherwise a 50-day pair would
    be scored on more years than a 300-day pair and the cells wouldn't be comparable.

    Args:
        prices: Daily adjusted close prices.
        shorts: Short-window lengths to test (rows of the grid).
        longs: Long-window lengths to test (columns of the grid).
        cost: Transaction cost per trade.
        risk_free_rate: Annual risk-free rate for the Sharpe ratio.
        start: Optional first evaluation date.
        end: Optional last evaluation date.

    Returns:
        (grid, buyhold_sharpe): grid has shorts as rows and longs as columns, with
        NaN where short >= long. buyhold_sharpe is the benchmark over the same dates.
    """
    eval_start = prices.index[max(longs) - 1]
    if start is not None:
        eval_start = max(eval_start, pd.Timestamp(start))

    grid = pd.DataFrame(index=pd.Index(shorts, name="short"),
                        columns=pd.Index(longs, name="long"), dtype=float)
    buyhold_sharpe = float("nan")

    for short in shorts:
        for long in longs:
            if short >= long:
                continue
            bt = backtest_period(prices, short, long, eval_start, end, cost)
            grid.loc[short, long] = sharpe_ratio(bt["strategy_return"], risk_free_rate)
            buyhold_sharpe = sharpe_ratio(bt["buyhold_return"], risk_free_rate)

    return grid, buyhold_sharpe


def out_of_sample_test(
    prices: pd.Series,
    split_date: str,
    shorts: list[int],
    longs: list[int],
    cost: float = 0.001,
    risk_free_rate: float = 0.0,
    reference: tuple[int, int] = (50, 200),
) -> dict:
    """Pick the best pair on data before `split_date`, then test it on data after.

    The test period is never looked at while choosing parameters, so its results
    are an honest estimate of how the chosen pair would have done "live".

    Returns:
        Dict with the chosen pair, a metrics table for each period (best pair,
        the reference pair, buy-and-hold), and where the chosen pair ranks among
        all pairs in the test period.
    """
    split = pd.Timestamp(split_date)
    train_end = split - pd.Timedelta(days=1)
    train_start = prices.index[max(longs) - 1]  # same start as the in-sample sweep

    train_grid, _ = parameter_sweep(prices, shorts, longs, cost, risk_free_rate, end=train_end)
    best = tuple(int(x) for x in train_grid.stack().idxmax())
    test_grid, _ = parameter_sweep(prices, shorts, longs, cost, risk_free_rate, start=split)

    tables = {}
    for period, start, end in [("in_sample", train_start, train_end), ("out_of_sample", split, None)]:
        columns = {}
        for name, (short, long) in [(f"Best in-sample {best[0]}/{best[1]}", best),
                                    (f"SMA {reference[0]}/{reference[1]}", reference)]:
            bt = backtest_period(prices, short, long, start, end, cost)
            columns[name] = compute_metrics(bt["strategy_return"], bt["position"], risk_free_rate)
        always_long = pd.Series(1, index=bt.index)
        columns["Buy & Hold"] = compute_metrics(bt["buyhold_return"], always_long, risk_free_rate)
        tables[period] = pd.DataFrame(columns)
        tables[f"{period}_dates"] = (bt.index[0].date(), bt.index[-1].date())

    test_ranks = test_grid.stack().rank(ascending=False)
    return {
        "best_pair": best,
        "in_sample": tables["in_sample"],
        "out_of_sample": tables["out_of_sample"],
        "in_sample_dates": tables["in_sample_dates"],
        "out_of_sample_dates": tables["out_of_sample_dates"],
        "test_rank": int(test_ranks[best]),
        "n_pairs": int(test_ranks.size),
        "test_best_pair": tuple(int(x) for x in test_grid.stack().idxmax()),
    }
