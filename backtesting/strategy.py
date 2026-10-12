import pandas as pd
import numpy as np
from config.settings import (
    SMA_LONG_WINDOW,
    SMA_MACRO_WINDOW,
    SMA_SHORT_WINDOW,
    STRETCH_LIMIT,
    VOLATILITY_WINDOW,
)
from src.indicators import sma_slope_percent


def sma_derivative_signal(
    closes,
    volumes,
    rolling_window_cross20: int,
    min_volume: float,
    min_price: float,
) -> np.ndarray:
    closes = pd.Series(closes)
    volumes = pd.Series(volumes, index=closes.index)

    sma20 = closes.rolling(window=SMA_LONG_WINDOW).mean()
    sma9 = closes.rolling(window=SMA_SHORT_WINDOW).mean()
    sma200 = closes.rolling(window=SMA_MACRO_WINDOW).mean()

    d1_20_pct = sma_slope_percent(closes, SMA_LONG_WINDOW)
    d1_9_pct = sma_slope_percent(closes, SMA_SHORT_WINDOW)

    daily_returns = closes.pct_change()
    volatility_20d = daily_returns.rolling(window=VOLATILITY_WINDOW).std()
    # risk_adjusted_score = (d1_20_pct + d1_9_pct) / volatility_20d

    stretch_factor = closes / sma20
    not_overbought = stretch_factor <= STRETCH_LIMIT

    cross20 = (d1_20_pct > 0) & (d1_20_pct.shift(1) <= 0)
    # recent_cross20 = cross20.rolling(window=rolling_window_cross20).max() > 0
    recent_cross20 = True

    short_term_up = d1_9_pct > 0
    short_term_down_cross = (d1_9_pct < 0) & (d1_9_pct.shift(1) >= 0)
    macro_uptrend = closes > sma200
    liquidity = (volumes.shift(1) > min_volume) & (closes > min_price)

    scan_signals = (
        liquidity
        & macro_uptrend
        & short_term_up
        & recent_cross20
        & not_overbought
    )
    signals = np.zeros(len(closes), dtype=int)
    signals[scan_signals.fillna(False).to_numpy()] = 1
    signals[short_term_down_cross.fillna(False).to_numpy()] = -1
    return signals

def sma_crossover_signal(close_prices, short_window: int, long_window: int) -> np.ndarray:
    sma_short = pd.Series(close_prices).rolling(window=short_window).mean()
    sma_long = pd.Series(close_prices).rolling(window=long_window).mean()

    bullish_cross = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
    bearish_cross = (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))

    signals = np.zeros(len(sma_short), dtype=int)
    signals[bullish_cross.to_numpy()] = 1
    signals[bearish_cross.to_numpy()] = -1

    return signals