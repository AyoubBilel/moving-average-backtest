"""Command-line entry point for the moving average crossover backtest.

Examples:
    python main.py
    python main.py --ticker SPY --short 50 --long 200 --start 2005-01-01 --cost 0.001
    python main.py --sweep          # also run the parameter grid and save a heatmap
    python main.py --oos            # also run the in-sample / out-of-sample test
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # we only save PNG files, so never open plot windows

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest import out_of_sample_test, parameter_sweep, run_backtest  # noqa: E402
from src.data import load_prices  # noqa: E402
from src.metrics import format_table, summary_table  # noqa: E402
from src.plots import RESULTS_DIR, plot_sharpe_heatmap, save_all_plots  # noqa: E402
from src.strategy import generate_signals  # noqa: E402

SHORT_GRID = list(range(10, 101, 10))
LONG_GRID = list(range(50, 301, 25))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line options."""
    parser = argparse.ArgumentParser(description="Backtest a moving average crossover strategy.")
    parser.add_argument("--ticker", default="SPY", help="Yahoo Finance symbol (default: SPY)")
    parser.add_argument("--short", type=int, default=50, help="short SMA window in days (default: 50)")
    parser.add_argument("--long", type=int, default=200, help="long SMA window in days (default: 200)")
    parser.add_argument("--start", default="2005-01-01", help="first date, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="last date (exclusive), default: today")
    parser.add_argument("--cost", type=float, default=0.001, help="cost per trade (0.001 = 0.1%%)")
    parser.add_argument("--rf", type=float, default=0.0, help="annual risk-free rate for Sharpe")
    parser.add_argument("--sweep", action="store_true", help="run the parameter grid + heatmap")
    parser.add_argument("--oos", action="store_true", help="run the out-of-sample test")
    parser.add_argument("--split", default="2018-01-01", help="first out-of-sample date")
    parser.add_argument("--refresh", action="store_true", help="re-download instead of using cache")
    parser.add_argument("--no-plots", action="store_true", help="don't save charts")
    args = parser.parse_args(argv)

    if not 0 < args.short < args.long:
        parser.error(f"need 0 < --short < --long, got {args.short} and {args.long}")
    if args.cost < 0:
        parser.error("--cost can't be negative")
    args.ticker = args.ticker.upper()
    return args


def run_single(args: argparse.Namespace, prices: pd.Series) -> None:
    """Backtest one parameter pair: print metrics, save them as CSV, save charts."""
    bt = run_backtest(generate_signals(prices, args.short, args.long), cost=args.cost)
    table = summary_table(bt, args.rf)

    print(f"\n{args.ticker} SMA {args.short}/{args.long} | cost {args.cost:.2%} per trade | "
          f"rf {args.rf:.1%} | {bt.index[0].date()} -> {bt.index[-1].date()}\n")
    print(format_table(table).to_string())
    print(f"\nFinal value of 10,000: strategy {bt['strategy_equity'].iloc[-1]:,.0f}, "
          f"buy & hold {bt['buyhold_equity'].iloc[-1]:,.0f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / f"metrics_{args.ticker}_{args.short}_{args.long}.csv")
    if not args.no_plots:
        for path in save_all_plots(bt, args.ticker, args.short, args.long):
            print(f"Saved {path.relative_to(RESULTS_DIR.parent)}")


def run_sweep(args: argparse.Namespace, prices: pd.Series) -> None:
    """Sharpe ratio for every pair in the grid, plus the heatmap."""
    grid, buyhold = parameter_sweep(prices, SHORT_GRID, LONG_GRID, args.cost, args.rf)
    values = grid.stack()
    print(f"\nParameter sweep: {values.size} pairs, buy & hold Sharpe {buyhold:.2f}, "
          f"{(values > buyhold).sum()} pairs beat it")
    print("Top 5 pairs by Sharpe:\n" + values.nlargest(5).round(2).to_string())

    if not args.no_plots:
        fig = plot_sharpe_heatmap(grid, buyhold, args.ticker, (args.short, args.long))
        path = RESULTS_DIR / "sharpe_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path.relative_to(RESULTS_DIR.parent)}")


def run_oos(args: argparse.Namespace, prices: pd.Series) -> None:
    """Pick the best pair before --split, then report how it did after."""
    result = out_of_sample_test(prices, args.split, SHORT_GRID, LONG_GRID, args.cost, args.rf,
                                reference=(args.short, args.long))
    short, long = result["best_pair"]
    print(f"\nOut-of-sample test: best in-sample pair is {short}/{long}")
    for period in ("in_sample", "out_of_sample"):
        first, last = result[f"{period}_dates"]
        print(f"\n{period.replace('_', '-').capitalize()} ({first} -> {last})")
        print(format_table(result[period]).to_string())
    test_short, test_long = result["test_best_pair"]
    print(f"\n{short}/{long} ranks {result['test_rank']} of {result['n_pairs']} pairs out of sample "
          f"(the best pair out of sample was {test_short}/{test_long}).")


def main(argv: list[str] | None = None) -> None:
    """Load data once, then run the single backtest and any optional analyses."""
    args = parse_args(argv)
    try:
        prices = load_prices(args.ticker, args.start, args.end, refresh=args.refresh)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from None
    print(f"Loaded {len(prices):,} daily prices for {args.ticker} "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")

    run_single(args, prices)
    if args.sweep:
        run_sweep(args, prices)
    if args.oos:
        run_oos(args, prices)


if __name__ == "__main__":
    main()
