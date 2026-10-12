"""Technical indicators shared by scanners and backtesting strategies."""

import pandas as pd


def sma_slope_percent(
    closes: pd.DataFrame | pd.Series, window: int
) -> pd.DataFrame | pd.Series:
    """Return the one-bar percentage change in a simple moving average."""
    average = closes.rolling(window=window).mean()
    return (average - average.shift(1)) / average.shift(1)
