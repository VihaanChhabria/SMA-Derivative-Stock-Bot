from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# ==========================================
# 1. PARAMETERS & UNIVERSE SETUP
# ==========================================
TOP_N_POSITIONS = 10
MIN_VOLUME = 1000000
MIN_PRICE = 5.0
ROLLING_WINDOW_CROSS20 = 7

# Lookback to calculate 200 SMA and historical indicators
end_date = datetime.now()
start_date = end_date - timedelta(days=350)

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "NFLX",
    "INTC",
    "JPM",
    "BAC",
    "V",
    "MA",
    "UNH",
    "JNJ",
    "PFE",
    "PG",
    "XOM",
    "CVX",
    "HD",
    "COST",
    "PEP",
    "KO",
    "DIS",
    "CAT",
    "DE",
    "BA",
    "GE",
    "LMT",
    "MCHP"
]

# Deduplicate tickers
TICKERS = list(set(TICKERS))

print(
    f"Fetching market data for {len(TICKERS)} tickers to evaluate latest day..."
)
data = yf.download(
    TICKERS,
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
)

closes = data["Close"]
highs = data["High"]
volumes = data["Volume"]

# ==========================================
# 2. INDICATOR CALCULATIONS
# ==========================================
sma20 = closes.rolling(window=20).mean()
sma9 = closes.rolling(window=9).mean()
sma200 = closes.rolling(window=200).mean()

# A. PERCENTAGE SLOPES (Fixes Price-Scale Bias)
# Measures % change in SMA per bar rather than dollar change
d1_20_pct = (sma20 - sma20.shift(1)) / sma20.shift(1)
d1_9_pct = (sma9 - sma9.shift(1)) / sma9.shift(1)

# B. VOLATILITY / ATR NORMALIZATION (Fixes Single-Day Spike Bias)
# Calculate 20-day percentage volatility (std dev of daily returns)
daily_returns = closes.pct_change()
volatility_20d = daily_returns.rolling(window=20).std()

# Risk-Adjusted Momentum Score (Slope normalized by volatility)
risk_adjusted_score = (d1_20_pct + d1_9_pct) / volatility_20d

# C. OVERBOUGHT / STRETCH FILTER (Prevents Buying Parabolic Tops)
# Disqualify stocks stretched > 4% above their 20 SMA
stretch_factor = closes / sma20
not_overbought = stretch_factor <= 1.04

# Crossover tracking: d1_20_pct crossed above 0 within last ROLLING_WINDOW_CROSS20 bars
cross20 = (d1_20_pct > 0) & (d1_20_pct.shift(1) <= 0)
recent_cross20 = cross20.rolling(window=ROLLING_WINDOW_CROSS20).max() > 0

# Scan filters
short_term_up = d1_9_pct > 0
macro_uptrend = closes > sma200
liquidity = (volumes.shift(1) > MIN_VOLUME) & (closes > MIN_PRICE)

# Final Scan Condition (Includes Overbought Guardrail)
scan_signals = liquidity & macro_uptrend & short_term_up & recent_cross20 & not_overbought

# 50-day High Resistance Target
resistance_50d = highs.shift(1).rolling(window=50).max()

# ==========================================
# 3. LATEST DAY SCAN, CSV EXPORT & RANKING
# ==========================================
latest_date = closes.dropna(how="all").index[-1]
print(f"Latest Market Data Date: {latest_date.strftime('%Y-%m-%d')}\n")

# Get all universe tickers for the latest date
all_tickers = closes.columns.tolist()

# 1. Build DataFrame containing ALL stocks (Passed + Failed)
full_df = pd.DataFrame(
    {
        "Ticker": all_tickers,
        "Close_Price": closes.loc[latest_date, all_tickers].values,
        "Risk_Adj_Score": risk_adjusted_score.loc[latest_date, all_tickers].values,
        "SMA20_Slope_%": (d1_20_pct.loc[latest_date, all_tickers] * 100).values,
        "SMA9_Slope_%": (d1_9_pct.loc[latest_date, all_tickers] * 100).values,
        "Stretch_vs_SMA20_%": ((stretch_factor.loc[latest_date, all_tickers] - 1) * 100).values,
        "50d_Resistance": resistance_50d.loc[latest_date, all_tickers].values,
        "Passed_Scan": scan_signals.loc[latest_date, all_tickers].values,
    }
)

# Sort the complete dataset by Risk_Adj_Score descending
full_df = full_df.sort_values(by="Risk_Adj_Score", ascending=False).reset_index(drop=True)

# 2. Export ALL data (including non-passing stocks) to CSV
export_filename = f"all_stocks_scan_{latest_date.strftime('%Y%m%d')}.csv"
full_df.to_csv(export_filename, index=False)
print(f"Full dataset ({len(full_df)} stocks) exported to '{export_filename}'.\n")

# 3. Filter and display ONLY the stocks that PASSED all scan criteria
passed_df = full_df[full_df["Passed_Scan"] == True].copy().reset_index(drop=True)

if passed_df.empty:
    print("No tickers met all scan criteria for today.")
else:
    passed_df.index += 1
    passed_df.index.name = "Rank"

    print("=" * 80)
    print(f"PASSED CANDIDATES FOR TODAY ({latest_date.strftime('%Y-%m-%d')}) - Total: {len(passed_df)}")
    print("=" * 80)
    # Exclude the 'Passed_Scan' column from terminal display since all are True
    print(passed_df.drop(columns=["Passed_Scan"]).to_string())