"""Tests for command-line parsing (no network, no files)."""

import pytest

from main import parse_args


def test_defaults_match_the_project_spec():
    args = parse_args([])
    assert (args.ticker, args.short, args.long, args.start, args.cost) == (
        "SPY", 50, 200, "2005-01-01", 0.001,
    )


def test_ticker_is_uppercased():
    assert parse_args(["--ticker", "qqq"]).ticker == "QQQ"


@pytest.mark.parametrize("argv", [["--short", "200", "--long", "50"], ["--cost", "-0.01"]])
def test_invalid_arguments_exit_with_an_error(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)
