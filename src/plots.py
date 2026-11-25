"""Charts for the backtest, saved as PNGs to results/ (they are embedded in the README).

Each plot function takes the DataFrame from `run_backtest()` and returns a matplotlib
Figure, so the same functions work in the CLI (saved to disk) and in the notebook.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, PercentFormatter

from src.metrics import drawdown_series

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# A small fixed palette: one color per role, used the same way in every chart.
COLORS = {
    "strategy": "#2a78d6",  # blue: the strategy / short SMA
    "benchmark": "#eb6834",  # orange: buy-and-hold / long SMA
    "price": "#52514e",  # dark gray: raw price
    "buy": "#0ca30c",  # green triangle up
    "sell": "#d03b3b",  # red triangle down
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "muted": "#898781",
    "surface": "#fcfcfb",
}

plt.rcParams.update(
    {
        # A fallback list: matplotlib uses the first font installed on the machine.
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": COLORS["surface"],
        "axes.facecolor": COLORS["surface"],
        "savefig.facecolor": COLORS["surface"],
        "figure.dpi": 110,
    }
)


def _style_axes(ax: plt.Axes, title: str) -> None:
    """Apply the shared look: hairline grid, no top/right spines, muted ticks."""
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", color="#0b0b0b", pad=10)
    ax.grid(True, color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["axis"])
    ax.tick_params(colors=COLORS["muted"], labelcolor="#52514e")
    ax.margins(x=0.01)


def _log_y_axis(ax: plt.Axes, label: str) -> None:
    """Log-scale y-axis with plain ticks at 1-2-5 steps (e.g. 50, 100, 200, 500).

    On a log scale, equal vertical distances mean equal percentage changes, so a
    -20% drop in 2008 looks as big as a -20% drop in 2022 even though prices differ.
    """
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(label, color="#52514e")


def _label_line_end(ax: plt.Axes, series: pd.Series, text: str, color: str) -> None:
    """Write a label just right of a line's last point (selective direct labeling)."""
    ax.annotate(
        text,
        xy=(series.index[-1], series.iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color="#0b0b0b",
        bbox={"boxstyle": "round,pad=0.25", "fc": COLORS["surface"], "ec": color, "lw": 1},
    )


def plot_price_signals(bt: pd.DataFrame, ticker: str, short: int, long: int) -> Figure:
    """Price with both SMAs and markers on the days a trade was executed."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(bt.index, bt["close"], color=COLORS["price"], lw=1.0, label=f"{ticker} adjusted close")
    ax.plot(bt.index, bt["sma_short"], color=COLORS["strategy"], lw=1.6, label=f"{short}-day SMA")
    ax.plot(bt.index, bt["sma_long"], color=COLORS["benchmark"], lw=1.6, label=f"{long}-day SMA")

    # Markers go on execution days (position changed), one day after the crossover.
    change = bt["position"].diff()
    buys, sells = bt.index[change == 1], bt.index[change == -1]
    ax.scatter(buys, bt.loc[buys, "close"], marker="^", s=70, color=COLORS["buy"],
               edgecolors=COLORS["surface"], linewidths=1.5, zorder=5, label="Buy (enter long)")
    ax.scatter(sells, bt.loc[sells, "close"], marker="v", s=70, color=COLORS["sell"],
               edgecolors=COLORS["surface"], linewidths=1.5, zorder=5, label="Sell (go to cash)")

    _log_y_axis(ax, "Price (log scale)")
    _style_axes(ax, f"{ticker}: price, {short}/{long}-day moving averages and trades")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def plot_equity(bt: pd.DataFrame, ticker: str, short: int, long: int) -> Figure:
    """Equity curves of the strategy and buy-and-hold on one log-scale axis."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    strategy, benchmark = bt["strategy_equity"], bt["buyhold_equity"]
    ax.plot(bt.index, strategy, color=COLORS["strategy"], lw=1.8, label=f"SMA {short}/{long} strategy")
    ax.plot(bt.index, benchmark, color=COLORS["benchmark"], lw=1.8, label=f"Buy & hold {ticker}")
    _label_line_end(ax, strategy, f"{strategy.iloc[-1]:,.0f}", COLORS["strategy"])
    _label_line_end(ax, benchmark, f"{benchmark.iloc[-1]:,.0f}", COLORS["benchmark"])

    _log_y_axis(ax, "Portfolio value (log scale)")
    start = bt.index[0].strftime("%Y-%m-%d")
    _style_axes(ax, f"Growth of 10,000 invested on {start}")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def plot_drawdown(bt: pd.DataFrame, ticker: str, short: int, long: int) -> Figure:
    """Drawdown (distance below the running peak) for both return streams."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for col, key, label in [
        ("buyhold_return", "benchmark", f"Buy & hold {ticker}"),
        ("strategy_return", "strategy", f"SMA {short}/{long} strategy"),
    ]:
        dd = drawdown_series(bt[col])
        ax.fill_between(dd.index, dd, 0, color=COLORS[key], alpha=0.12, lw=0)
        ax.plot(dd.index, dd, color=COLORS[key], lw=1.3,
                label=f"{label} (max {dd.min():.1%})")

    ax.axhline(0, color=COLORS["axis"], lw=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Below previous peak", color="#52514e")
    _style_axes(ax, "Drawdown: how far each portfolio sat below its previous high")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_sharpe_heatmap(
    grid: pd.DataFrame, buyhold_sharpe: float, ticker: str, highlight: tuple[int, int] = (50, 200)
) -> Figure:
    """Heatmap of Sharpe ratio for every (short, long) pair.

    Colors are diverging around buy-and-hold's Sharpe: blue cells beat the
    benchmark, red cells lose to it, gray is roughly equal. Only two cells are
    labeled (the best one and the classic 50/200); the color bar carries the rest.
    """
    # Red (worse) -> neutral gray -> blue (better), centered on buy-and-hold.
    cmap = LinearSegmentedColormap.from_list(
        "vs_benchmark", ["#b23434", "#e66767", "#f0efec", "#5598e7", "#184f95"]
    )
    cmap.set_bad(COLORS["surface"])  # invalid cells (short >= long) stay blank
    values = grid.to_numpy(dtype=float)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    spread = max(vmax - buyhold_sharpe, buyhold_sharpe - vmin, 0.01)
    norm = TwoSlopeNorm(vcenter=buyhold_sharpe, vmin=buyhold_sharpe - spread,
                        vmax=buyhold_sharpe + spread)

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(grid.columns)), labels=grid.columns)
    ax.set_yticks(range(len(grid.index)), labels=grid.index)
    ax.set_xlabel("Long window (days)", color="#52514e")
    ax.set_ylabel("Short window (days)", color="#52514e")
    ax.tick_params(colors=COLORS["muted"], labelcolor="#52514e", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    best_short, best_long = grid.stack().idxmax()
    cells = {(best_short, best_long): "best"}
    if highlight[0] in grid.index and highlight[1] in grid.columns:
        cells.setdefault(highlight, f"{highlight[0]}/{highlight[1]}")
    for (short, long), tag in cells.items():
        row, col = grid.index.get_loc(short), grid.columns.get_loc(long)
        ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False,
                                   ec="#0b0b0b", lw=1.8))
        ax.text(col, row, f"{tag}\n{grid.loc[short, long]:.2f}", ha="center", va="center",
                fontsize=8, color="#0b0b0b",
                bbox={"boxstyle": "round,pad=0.2", "fc": COLORS["surface"], "ec": "none", "alpha": 0.85})

    bar = fig.colorbar(image, ax=ax, shrink=0.85)
    bar.set_label("Sharpe ratio", color="#52514e")
    bar.ax.axhline(buyhold_sharpe, color="#0b0b0b", lw=1.2)
    bar.ax.text(-0.3, buyhold_sharpe, f"buy & hold\n{buyhold_sharpe:.2f}",
                transform=bar.ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8, color="#0b0b0b")
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=COLORS["muted"], labelcolor="#52514e")

    ax.set_title(f"{ticker}: Sharpe ratio by moving-average windows (blue beats buy & hold)",
                 loc="left", fontsize=12, fontweight="semibold", color="#0b0b0b", pad=10)
    fig.tight_layout()
    return fig


def save_all_plots(
    bt: pd.DataFrame, ticker: str, short: int, long: int, results_dir: Path = RESULTS_DIR
) -> list[Path]:
    """Create all three charts, save them as PNG, and return the file paths."""
    results_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "price_signals.png": plot_price_signals,
        "equity_curve.png": plot_equity,
        "drawdown.png": plot_drawdown,
    }
    paths = []
    for filename, make_chart in charts.items():
        fig = make_chart(bt, ticker, short, long)
        path = results_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
