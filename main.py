"""Command-line entry point for scanners and backtests."""

from datetime import date, timedelta
import importlib

import pandas as pd

import inquirer
import runpy

from config.settings import HISTORICAL_LOOKBACK_DAYS, RISK_REWARD_STOP_LOSS_PCT
from src.data_loader import load_or_download_ohlcv
from src.data_store import list_downloads, to_date

BUY_STRATEGIES = {
    "sma_slope": "scanners.sma_slope_buy_signal",
    "stochastic": "scanners.stochastic",
}

SELL_STRATEGIES = {
    "x_day_hold": "scanners.x_day_hold_sell_signal",
    "risk_reward": "scanners.risk_reward_sell_signal",
}

NEW_RANGE_CHOICE = "Pick a specific date range"

RISK_REWARD_PRESETS = {"1:1": 1.0, "1:1.5": 1.5, "1:2": 2.0}
STOP_LOSS_PRESETS = {"2%": 0.02, "3%": 0.03, "5%": 0.05, "8%": 0.08}
CUSTOM_CHOICE = "Custom"


def _parse_reward_ratio(value: str) -> float:
    """Parse '1:2.5' or '2.5' into the reward multiple 2.5."""
    value = value.strip()
    if ":" in value:
        risk, reward = (float(part) for part in value.split(":", 1))
    else:
        risk, reward = 1.0, float(value)
    if risk <= 0 or reward <= 0:
        raise ValueError("ratio parts must be positive")
    return reward / risk


def _validate_reward_ratio(_answers, value: str) -> bool:
    try:
        _parse_reward_ratio(value)
    except ValueError:
        raise inquirer.errors.ValidationError(
            "", reason="Enter a positive ratio like 1:2.5 or 2.5"
        )
    return True


def select_reward_ratio() -> float:
    """Ask for a preset or custom risk:reward ratio and return the reward multiple."""
    choice = inquirer.prompt(
        [
            inquirer.List(
                "ratio",
                message="Select a risk to reward ratio",
                choices=[*RISK_REWARD_PRESETS, CUSTOM_CHOICE],
            )
        ]
    )["ratio"]
    if choice != CUSTOM_CHOICE:
        return RISK_REWARD_PRESETS[choice]

    custom = inquirer.prompt(
        [
            inquirer.Text(
                "ratio",
                message="Custom risk:reward ratio (e.g. 1:3)",
                validate=_validate_reward_ratio,
            )
        ]
    )["ratio"]
    return _parse_reward_ratio(custom)


def _parse_stop_loss_pct(value: str) -> float:
    """Parse '4' or '4%' into the fraction 0.04."""
    pct = float(value.strip().rstrip("%"))
    if not 0 < pct < 100:
        raise ValueError("stop-loss must be between 0 and 100 percent")
    return pct / 100


def _validate_stop_loss_pct(_answers, value: str) -> bool:
    try:
        _parse_stop_loss_pct(value)
    except ValueError:
        raise inquirer.errors.ValidationError(
            "", reason="Enter a percent between 0 and 100, like 4 or 4%"
        )
    return True


def select_stop_loss_pct() -> float:
    """Ask for a preset or custom stop-loss % and return it as a fraction."""
    choice = inquirer.prompt(
        [
            inquirer.List(
                "stop_loss",
                message="Select a stop-loss distance below entry (defines the risk)",
                choices=[*STOP_LOSS_PRESETS, CUSTOM_CHOICE],
                default=f"{RISK_REWARD_STOP_LOSS_PCT:.0%}",
            )
        ]
    )["stop_loss"]
    if choice != CUSTOM_CHOICE:
        return STOP_LOSS_PRESETS[choice]

    custom = inquirer.prompt(
        [
            inquirer.Text(
                "stop_loss",
                message="Custom stop-loss % below entry (e.g. 4 or 4%)",
                validate=_validate_stop_loss_pct,
            )
        ]
    )["stop_loss"]
    return _parse_stop_loss_pct(custom)


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
        reward_ratio = stop_loss_pct = None
        if "risk_reward" in answers["sell_strategy"]:
            stop_loss_pct = select_stop_loss_pct()
            reward_ratio = select_reward_ratio()
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
                    "ohlcv_data": ohlcv_data,
                    "reward_ratio": reward_ratio,
                    "stop_loss_pct": stop_loss_pct,
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
