from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# ==========================================
# 1. PARAMETERS & UNIVERSE SETUP
# ==========================================
TOP_N_POSITIONS = 10
MIN_VOLUME = 1000000
MIN_PRICE = 5.0
ROLLING_WINDOW_CROSS20 = 4

# Lookback to calculate 200 SMA and historical indicators
end_date = datetime.now()
start_date = end_date - timedelta(days=350)

# TICKERS = [
#     "AAPL",
#     "MSFT",
#     "NVDA",
#     "AMZN",
#     "GOOGL",
#     "META",
#     "TSLA",
#     "AMD",
#     "NFLX",
#     "INTC",
#     "JPM",
#     "BAC",
#     "V",
#     "MA",
#     "UNH",
#     "JNJ",
#     "PFE",
#     "PG",
#     "XOM",
#     "CVX",
#     "HD",
#     "COST",
#     "PEP",
#     "KO",
#     "DIS",
#     "CAT",
#     "DE",
#     "BA",
#     "GE",
#     "LMT",
#     "MCHP",
# ]

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
    "ZTS", "MOH", "HLT", "YUM", "DRI", "ROST", "DAL", "UAL", "AAL"
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
d1_20_pct = (sma20 - sma20.shift(1)) / sma20.shift(1)
d1_9_pct = (sma9 - sma9.shift(1)) / sma9.shift(1)

# B. VOLATILITY / ATR NORMALIZATION (Fixes Single-Day Spike Bias)
daily_returns = closes.pct_change()
volatility_20d = daily_returns.rolling(window=20).std()

# Risk-Adjusted Momentum Score
risk_adjusted_score = (d1_20_pct + d1_9_pct) / volatility_20d

# C. INDIVIDUAL FILTER CONDITIONS
stretch_factor = closes / sma20
not_overbought = stretch_factor <= 1.10

# Crossover tracking: d1_20_pct crossed above 0 within last ROLLING_WINDOW_CROSS20 bars
cross20 = (d1_20_pct > 0) & (d1_20_pct.shift(1) <= 0)
recent_cross20 = cross20.rolling(window=ROLLING_WINDOW_CROSS20).max() > 0

short_term_up = d1_9_pct > 0
macro_uptrend = closes > sma200
liquidity = (volumes.shift(1) > MIN_VOLUME) & (closes > MIN_PRICE)

# Combined Final Scan Condition
scan_signals = (
    liquidity
    & macro_uptrend
    & short_term_up
    & recent_cross20
    & not_overbought
)

# 50-day High Resistance Target
resistance_50d = highs.shift(1).rolling(window=50).max()

# ==========================================
# 3. LATEST DAY SCAN, CSV EXPORT & RANKING
# ==========================================
latest_date = closes.dropna(how="all").index[-1]
print(f"Latest Market Data Date: {latest_date.strftime('%Y-%m-%d')}\n")

# Extract series for the latest bar
all_tickers = closes.columns.tolist()

liq_latest = liquidity.loc[latest_date]
macro_latest = macro_uptrend.loc[latest_date]
short_latest = short_term_up.loc[latest_date]
cross_latest = recent_cross20.loc[latest_date]
stretch_latest = not_overbought.loc[latest_date]

# Build granular rejection reasons per ticker
rejection_reasons = []
for ticker in all_tickers:
    reasons = []
    if not liq_latest[ticker]:
        reasons.append("Low Volume/Price")
    if not macro_latest[ticker]:
        reasons.append("Below 200 SMA")
    if not short_latest[ticker]:
        reasons.append("SMA9 Slope Down")
    if not cross_latest[ticker]:
        reasons.append("No Recent SMA20 Crossover")
    if not stretch_latest[ticker]:
        reasons.append("Overbought (above SMA20 threshold)")

    rejection_reasons.append("; ".join(reasons) if reasons else "None (Passed)")

# 1. Build DataFrame containing ALL stocks (Passed + Rejected)
full_df = pd.DataFrame(
    {
        "Ticker": all_tickers,
        "Close_Price": closes.loc[latest_date, all_tickers].values,
        "Risk_Adj_Score": risk_adjusted_score.loc[
            latest_date, all_tickers
        ].values,
        "SMA20_Slope_%": (d1_20_pct.loc[latest_date, all_tickers] * 100).values,
        "SMA9_Slope_%": (d1_9_pct.loc[latest_date, all_tickers] * 100).values,
        "Stretch_vs_SMA20_%": (
            (stretch_factor.loc[latest_date, all_tickers] - 1) * 100
        ).values,
        "50d_Resistance": resistance_50d.loc[latest_date, all_tickers].values,
        "Passed_Scan": scan_signals.loc[latest_date, all_tickers].values,
        "Rejection_Reason": rejection_reasons,
    }
)

# 2. Sort by Risk_Adj_Score descending (Highest to Lowest)
full_df = full_df.sort_values(
    by="Risk_Adj_Score", ascending=False
).reset_index(drop=True)

# 3. Export full dataset to CSV
export_filename = f"all_stocks_scan_{latest_date.strftime('%Y%m%d')}.csv"
full_df.to_csv(export_filename, index=False)
print(f"Full dataset ({len(full_df)} stocks) exported to '{export_filename}'.\n")

# 4. Filter and display ONLY stocks that PASSED in the console terminal
passed_df = (
    full_df[full_df["Passed_Scan"] == True].copy().reset_index(drop=True)
)

if passed_df.empty:
    print("No tickers met all scan criteria for today.")
else:
    passed_df.index += 1
    passed_df.index.name = "Rank"

    print("=" * 90)
    print(
        f"PASSED CANDIDATES FOR TODAY ({latest_date.strftime('%Y-%m-%d')}) - Total: {len(passed_df)}"
    )
    print("=" * 90)
    print(
        passed_df.drop(
            columns=["Passed_Scan", "Rejection_Reason"]
        ).to_string()
    )