"""Compatibility entry point for the repository's canonical backtest."""

import runpy


def main() -> None:
    runpy.run_module("backtesting.main_backtest", run_name="__main__")


if __name__ == "__main__":
    main()
