"""Command-line entry point for scanners and backtests."""

import argparse
import runpy


STRATEGIES = {
    "sma_slope": "scanners.sma_slope",
    "stochastic": "scanners.stochastic",
    "backtest": "backtesting.main_backtest",
    "analyze": "backtesting.analysis",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a stock scanner, backtest, or backtest analysis."
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        required=True,
        help="Strategy or workflow to run.",
    )
    args = parser.parse_args()
    runpy.run_module(STRATEGIES[args.strategy], run_name="__main__")


if __name__ == "__main__":
    main()
