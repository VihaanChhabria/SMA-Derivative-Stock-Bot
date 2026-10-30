from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import talib

from config.settings import (
    HISTORICAL_LOOKBACK_DAYS,
    STOCHASTIC_FASTK_PERIOD,
    STOCHASTIC_MA_TYPE,
    STOCHASTIC_SLOWD_PERIOD,
    STOCHASTIC_SLOWK_PERIOD,
)
from config.settings import (
    STOCHASTIC_LOOKBACK_DAYS as LOOKBACK_DAYS,
)
from config.universes import STOCHASTIC_TICKERS as TICKERS
from src.data_loader import download_ohlcv
from src.scan_results import export_and_print_scan_results


def main() -> None:

    # ==========================================
    # 1. PARAMETERS & UNIVERSE SETUP
    # ==========================================
    # Lookback to calculate 200 SMA and historical indicators
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HISTORICAL_LOOKBACK_DAYS)

    print(
        f"Fetching market data for {len(TICKERS)} tickers to evaluate latest day..."
    )
    data = download_ohlcv(
        TICKERS,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    )

    closes = data["Close"]
    highs = data["High"]
    lows = data["Low"]
    volumes = data["Volume"]

    # ==========================================
    # 2. INDICATOR CALCULATIONS
    # ==========================================

    # C. STOCHASTIC (10, 10, 3, EMA) & CROSSOVER CONDITION
    slowk_df = pd.DataFrame(index=closes.index)
    slowd_df = pd.DataFrame(index=closes.index)

    for ticker in closes.columns:
        h = highs[ticker].dropna().values
        l = lows[ticker].dropna().values
        c = closes[ticker].dropna().values

        # Calculate Stochastic
        k, d = talib.STOCH(
            h, l, c,
            fastk_period=STOCHASTIC_FASTK_PERIOD,
            slowk_period=STOCHASTIC_SLOWK_PERIOD,
            slowk_matype=STOCHASTIC_MA_TYPE,
            slowd_period=STOCHASTIC_SLOWD_PERIOD,
            slowd_matype=STOCHASTIC_MA_TYPE,
        )
        slowk_df[ticker] = pd.Series(k, index=closes[ticker].dropna().index)
        slowd_df[ticker] = pd.Series(d, index=closes[ticker].dropna().index)

    # Boolean Crossover Signal: %K crossed above %D on the current bar
    k_cross_d_raw = (slowk_df > slowd_df) & (slowk_df.shift(1) <= slowd_df.shift(1))

    # ==========================================
    # 3. LATEST DAY SCAN & RANKING BY CROSS RECENCY
    # ==========================================
    latest_date = closes.dropna(how="all").index[-1]
    print(f"Latest Market Data Date: {latest_date.strftime('%Y-%m-%d')}\n")

    all_tickers = closes.columns.tolist()

    # Find days since last cross for each ticker over the last LOOKBACK_DAYS bars
    # 0 = crossed on latest_date, 1 = 1 day ago, 2 = 2 days ago, 3 = 3 days ago
    recent_window = k_cross_d_raw.iloc[-LOOKBACK_DAYS:]

    days_since_cross = {}
    passed_scan = {}

    for ticker in all_tickers:
        ticker_series = recent_window[ticker].values  # Boolean array of length LOOKBACK_DAYS
        # Find indices where True occurred in reverse order (most recent first)
        true_indices = np.where(ticker_series)[0]

        if len(true_indices) > 0:
            most_recent_idx = true_indices[-1]  # rightmost True index
            days_ago = (LOOKBACK_DAYS - 1) - most_recent_idx
            days_since_cross[ticker] = days_ago
            passed_scan[ticker] = True
        else:
            days_since_cross[ticker] = None
            passed_scan[ticker] = False

    # 1. Build DataFrame containing ALL stocks
    full_df = pd.DataFrame(
        {
            "Ticker": all_tickers,
            "Close_Price": closes.loc[latest_date, all_tickers].values,
            "SlowK": slowk_df.loc[latest_date, all_tickers].values,
            "SlowD": slowd_df.loc[latest_date, all_tickers].values,
            "Days_Since_Cross": [days_since_cross[t] for t in all_tickers],
            "Passed_Scan": [passed_scan[t] for t in all_tickers],
        }
    )

    export_and_print_scan_results(
        full_df,
        latest_date=latest_date,
        filename=f"all_stocks_scan_stochastic_{latest_date.strftime('%Y%m%d')}.csv",
        heading=(
            "PASSED CANDIDATES: %K CROSS OVER %D IN PAST "
            f"{LOOKBACK_DAYS} DAYS ({{date}}) - Total: {{count}}"
        ),
        no_results_message=(
            f"No tickers had a %K over %D crossover in the past "
            f"{LOOKBACK_DAYS} days."
        ),
        sort_by="Days_Since_Cross",
        hidden_columns=("Passed_Scan",),
    )

if __name__ == "__main__":
    main()
