"""Command-line entry point for scanners and backtests."""

from datetime import date, timedelta
import importlib

import pandas as pd

import inquirer
import runpy

from config.settings import HISTORICAL_LOOKBACK_DAYS
from src.data_loader import load_or_download_ohlcv
from src.data_store import list_downloads, to_date

BUY_STRATEGIES = {
    "sma_slope": "scanners.sma_slope_buy_signal",
    "stochastic": "scanners.stochastic",
}

SELL_STRATEGIES = {
    "x_day_hold": "scanners.x_day_hold_sell_signal",
}

NEW_RANGE_CHOICE = "Pick a specific date range"


def _validate_date(_answers, value: str) -> bool:
    try:
        to_date(value)
    except ValueError:
        raise inquirer.errors.ValidationError("", reason="Use YYYY-MM-DD")
    return True


def _validate_end_date(answers, value: str) -> bool:
    _validate_date(answers, value)
    if to_date(value) <= to_date(answers["start"]):
        raise inquirer.errors.ValidationError("", reason="End must be after start")
    return True


def require_all_signals(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine signal frames so a ticker signals only when every frame does.

    Only dates and tickers present in every frame are kept.
    """
    if not frames:
        return pd.DataFrame()
    combined = frames[0]
    for frame in frames[1:]:
        combined, frame = combined.align(frame, join="inner")
        combined = combined & frame
    return combined


def select_backtest_data(
    buy_strategies: list[str],
) -> tuple[pd.DataFrame, date | None, date | None]:
    """Ask for a saved download or a date range and return the OHLCV data.

    Also returns the chosen [start, end) range so signals can be trimmed to
    it, or (None, None) when a saved download is used as-is.
    """
    downloads = list_downloads()
    choices = [(d.label, d) for d in downloads] + [(NEW_RANGE_CHOICE, None)]
    source = inquirer.prompt(
        [
            inquirer.List(
                "source",
                message="Select backtest data",
                choices=choices,
            )
        ]
    )["source"]

    if source is not None:
        print(f"Loading saved download '{source.data_path.name}'...")
        return source.load(), None, None

    answers = inquirer.prompt(
        [
            inquirer.Text(
                "start",
                message="Backtest start date (YYYY-MM-DD)",
                validate=_validate_date,
            ),
            inquirer.Text(
                "end",
                message="Backtest end date, exclusive (YYYY-MM-DD)",
                default=date.today().isoformat(),
                validate=_validate_end_date,
            ),
        ]
    )
    start, end = to_date(answers["start"]), to_date(answers["end"])

    tickers = sorted(
        {
            ticker
            for strategy in buy_strategies
            for ticker in importlib.import_module(BUY_STRATEGIES[strategy]).TICKERS
        }
    )
    # Fetch extra history before the start so long indicators (SMA200) are warm
    download_start = start - timedelta(days=HISTORICAL_LOOKBACK_DAYS)
    print(f"Loading data for {len(tickers)} tickers from {download_start} to {end}...")
    return load_or_download_ohlcv(tickers, download_start, end), start, end


def main() -> None:
    questions = [
        inquirer.Checkbox(
            "buy_strategy",
            message="Select a buy strategy or workflow to run",
            choices=list(BUY_STRATEGIES.keys()),
        ),
        inquirer.Checkbox(
            "sell_strategy",
            message="Select a sell strategy or workflow to run",
            choices=list(SELL_STRATEGIES.keys()),
        ),
        inquirer.List(
            "live_or_backtest",
            message="Select whether to run live or backtest",
            choices=["backtest"],
        ),
    ]
    answers = inquirer.prompt(questions)

    buy_signals_frames = []
    sell_signals_frames = []

    if answers["live_or_backtest"] == "backtest":
        days_to_hold = 5  # Default value
        ohlcv_data, start, end = select_backtest_data(answers["buy_strategy"])
        for strategy in answers["buy_strategy"]:
            strat_globals = runpy.run_module(
                BUY_STRATEGIES[strategy],
                run_name="__main__",
                init_globals={"ohlcv_data": ohlcv_data},
            )
            buy_signals_frames.append(strat_globals.get("scan_signals"))

        # overlay buy signals
        combined_buy_signals = require_all_signals(buy_signals_frames)
        if start is not None and not combined_buy_signals.empty:
            # Drop the indicator warm-up period before the chosen start date
            combined_buy_signals = combined_buy_signals[
                (combined_buy_signals.index >= pd.Timestamp(start))
                & (combined_buy_signals.index < pd.Timestamp(end))
            ]

        for strategy in answers["sell_strategy"]:
            strat_globals = runpy.run_module(
                SELL_STRATEGIES[strategy],
                run_name="__main__",
                init_globals={
                    "buy_signals": combined_buy_signals,
                    "days_to_hold": days_to_hold,
                },
            )
            sell_signals_frames.append(strat_globals.get("sell_signals"))

        # overlay sell signals
        combined_sell_signals = require_all_signals(sell_signals_frames)

        print("Combined buy signals:")
        print(combined_buy_signals)
        print("Combined sell signals:")
        print(combined_sell_signals)


if __name__ == "__main__":
    main()
