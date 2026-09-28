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

d1_20 = sma20 - sma20.shift(1)
d1_9 = sma9 - sma9.shift(1)

# Crossover tracking: d1_20 crossed above 0 within last 3 bars
cross20 = (d1_20 > 0) & (d1_20.shift(1) <= 0)
recent_cross20 = cross20.rolling(window=ROLLING_WINDOW_CROSS20).max() > 0

# Scan filters
short_term_up = d1_9 > 0
macro_uptrend = closes > sma200
liquidity = (volumes.shift(1) > MIN_VOLUME) & (closes > MIN_PRICE)

# Final Scan Condition
scan_signals = liquidity & macro_uptrend & short_term_up & recent_cross20

# 50-day High Resistance Target
resistance_50d = highs.shift(1).rolling(window=50).max()

# ==========================================
# 3. LATEST DAY SCAN & RANKING (ALL CANDIDATES)
# ==========================================
latest_date = closes.dropna(how="all").index[-1]
print(f"Latest Market Data Date: {latest_date.strftime('%Y-%m-%d')}\n")

# Get slice for latest date
today_signals = scan_signals.loc[latest_date]
valid_candidates = today_signals[today_signals == True].index.tolist()

if not valid_candidates:
    print("No tickers met the scan criteria for the most recent date.")
else:
    # Compute ranking metrics for candidates
    today_d1_20 = d1_20.loc[latest_date, valid_candidates]
    today_d1_9 = d1_9.loc[latest_date, valid_candidates]
    today_closes = closes.loc[latest_date, valid_candidates]
    today_res = resistance_50d.loc[latest_date, valid_candidates]

    combined_slope = (today_d1_20 + today_d1_9).dropna()

    # Create detailed DataFrame for ALL valid candidates
    ranking_df = pd.DataFrame(
        {
            "Ticker": combined_slope.index,
            "Close_Price": today_closes[combined_slope.index].values,
            "Slope_Score": combined_slope.values,
            "SMA20_Slope": today_d1_20[combined_slope.index].values,
            "SMA9_Slope": today_d1_9[combined_slope.index].values,
            "50d_Resistance": today_res[combined_slope.index].values,
        }
    )

    # Sort descending by momentum slope score across ALL candidates
    ranking_df = ranking_df.sort_values(
        by="Slope_Score", ascending=False
    ).reset_index(drop=True)
    ranking_df.index += 1  # 1-based rank indexing
    ranking_df.index.name = "Rank"

    # Display ALL ranked candidates
    print("=" * 65)
    print(f"ALL RANKED CANDIDATES FOR TODAY ({latest_date.strftime('%Y-%m-%d')}) - Total: {len(ranking_df)}")
    print("=" * 65)
    print(ranking_df.to_string())

    # Export ALL ranked positions to CSV
    export_all_filename = "all_ranked_positions_today.csv"
    ranking_df.to_csv(export_all_filename)
    print(f"\nAll ranked positions saved to '{export_all_filename}'.")