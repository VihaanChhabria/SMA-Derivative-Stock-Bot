from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import talib
import numpy as np

# ==========================================
# 1. PARAMETERS & UNIVERSE SETUP
# ==========================================
LOOKBACK_DAYS = 4

# Lookback to calculate 200 SMA and historical indicators
end_date = datetime.now()
start_date = end_date - timedelta(days=350)

TICKERS = [
    # Top 1–50
    "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "AVGO", "TSLA", "MU", "BRK-B",
    "LLY", "AMD", "JPM", "WMT", "V", "XOM", "JNJ", "INTC", "MA", "ABBV",
    "PLTR", "CSCO", "ORCL", "COST", "CVX", "BAC", "LRCX", "AMAT", "KO", "CAT",
    "MRK", "DELL", "PG", "GE", "UNH", "MS", "PANW", "PM", "NFLX", "HD",
    "PEP", "TMUS", "NOW", "CRM", "ABT", "LIN", "GS", "DIS", "T", "AXP",

    # Top 51–100
    "IBM", "RTX", "HON", "ISRG", "PFE", "UBER", "MCD", "QCOM", "C",
    "NKE", "TMO", "PGR", "SBUX", "BKNG", "LOW", "UPS", "LMT", "ACN", "TXN",
    "BA", "SCHW", "AMGN", "ANET", "SYK", "COP", "KLAC", "BLK", "UNP", "SPGI",
    "TJX", "ADI", "MDLZ", "ELV", "GILD", "VRTX", "ADP", "INTU", "CI", "DHR",
    "BSX", "CB", "CME", "PNC", "REGN", "BX", "MDT", "SHW", "WM",

    # Top 101–150
    "PLD", "SO", "DUK", "DE", "CL", "APH", "EOG", "AON", "ICE",
    "FDX", "NOC", "BDX", "FCX", "MCK", "MOD", "CMG", "MAR", "HCA", "ITW",
    "USB", "MO", "ORLY", "SNPS", "CDNS", "TT", "PH", "MCO", "PWR", "EPR",
    "HUM", "CSX", "PYPL", "NSC", "ROP", "ADM", "AEM", "ADSK", "TDG",
    "TFC", "HMC", "COR", "AIG", "AFL", "RSG", "CINF", "AZO", "PAYX", "PCAR",

    # Top 151–200
    "EMR", "EW", "D", "HAL", "SLB", "O", "WELL", "PSA", "EXC", "XEL",
    "ALL", "MET", "TRV", "PRU", "DLR", "VMC", "MLM", "KMB", "GIS",
    "AEP", "SRE", "WEC", "ES", "PEG", "ED", "FAST", "CTAS", "GWW", "CPRT",
    "ODFL", "ROK", "AME", "VRSK", "IDXX", "IQV", "DXCM", "MTD", "RMD",
    "ZTS", "MOH", "HLT", "YUM", "DRI", "ROST", "DAL", "UAL", "AAL",

    "MCHP", "CCL"
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
    
    # Calculate Stochastic using your exact parameters (10, 10, 3, EMA=1)
    k, d = talib.STOCH(
        h, l, c,
        fastk_period=10,
        slowk_period=3,
        slowk_matype=1,  # 1 = EMA
        slowd_period=10,
        slowd_matype=1   # 1 = EMA
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

# 2. Export full dataset to CSV
export_filename = f"all_stocks_scan_{latest_date.strftime('%Y%m%d')}.csv"
full_df.to_csv(export_filename, index=False)
print(f"Full dataset ({len(full_df)} stocks) exported to '{export_filename}'.\n")

# 3. Filter for PASSED stocks only & SORT BY Days_Since_Cross (Ascending: 0, 1, 2, 3)
passed_df = (
    full_df[full_df["Passed_Scan"] == True]
    .sort_values(by="Days_Since_Cross", ascending=True)
    .copy()
    .reset_index(drop=True)
)

if passed_df.empty:
    print(f"No tickers had a %K over %D crossover in the past {LOOKBACK_DAYS} days.")
else:
    passed_df.index += 1
    passed_df.index.name = "Rank"

    print("=" * 90)
    print(
        f"PASSED CANDIDATES: %K CROSS OVER %D IN PAST {LOOKBACK_DAYS} DAYS ({latest_date.strftime('%Y-%m-%d')}) - Total: {len(passed_df)}"
    )
    print("=" * 90)
    print(passed_df.drop(columns=["Passed_Scan"]).to_string())