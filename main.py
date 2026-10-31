"""Command-line entry point for scanners and backtests."""

import pandas as pd

import inquirer
import runpy

BUY_STRATEGIES = {
    "sma_slope": "scanners.sma_slope_buy_signal",
    "stochastic": "scanners.stochastic",
    "backtest": "backtesting.main_backtest",
    "analyze": "backtesting.analysis",
}

SELL_STRATEGIES = {
    "x_day_hold": "scanners.x_day_hold_sell_signal",
}


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
        inquirer.Text(
            "days_to_hold",
            message="Days to hold before selling",
            default="5",
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
        days_to_hold = int(answers["days_to_hold"])
        for strategy in answers["buy_strategy"]:
            strat_globals = runpy.run_module(
                BUY_STRATEGIES[strategy], run_name="__main__"
            )
            buy_signals_frames.append(strat_globals.get('scan_signals'))

        # overlay buy signals
        combined_buy_signals = (
            pd.concat(buy_signals_frames, axis=1)
            if buy_signals_frames
            else pd.DataFrame()
        )

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
        combined_sell_signals = (
            pd.concat(sell_signals_frames, axis=1)
            if sell_signals_frames
            else pd.DataFrame()
        )

        print("Combined buy signals:")
        print(combined_buy_signals)
        print("Combined sell signals:")
        print(combined_sell_signals)


if __name__ == "__main__":
    main()
