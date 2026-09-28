import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ==========================================
# 1. PARAMETERS & UNIVERSE SETUP
# ==========================================
months_run = 4
end_date = datetime.now()
# 350 calendar days lookback ensures ~250 trading days for the 200 SMA calculation
start_date = end_date - timedelta(days=months_run * 30 + 350)

# TICKERS = [
#     # Top 1–50
#     "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "AVGO", "TSLA", "MU", "BRK-B",
#     "LLY", "AMD", "JPM", "WMT", "V", "XOM", "JNJ", "INTC", "MA", "ABBV",
#     "PLTR", "CSCO", "ORCL", "COST", "CVX", "BAC", "LRCX", "AMAT", "KO", "CAT",
#     "MRK", "DELL", "PG", "GE", "UNH", "MS", "PANW", "PM", "NFLX", "HD",
#     "PEP", "TMUS", "NOW", "CRM", "ABT", "LIN", "GS", "DIS", "T", "AXP",

#     # Top 51–100
#     "IBM", "RTX", "FI", "HON", "ISRG", "PFE", "UBER", "MCD", "QCOM", "C",
#     "NKE", "TMO", "PGR", "SBUX", "BKNG", "LOW", "UPS", "LMT", "ACN", "TXN",
#     "BA", "SCHW", "AMGN", "ANET", "SYK", "COP", "KLAC", "BLK", "UNP", "SPGI",
#     "TJX", "ADI", "MDLZ", "ELV", "GILD", "VRTX", "ADP", "INTU", "CI", "DHR",
#     "BSX", "CB", "MMC", "CME", "PNC", "REGN", "BX", "MDT", "SHW", "WM",

#     # Top 101–150
#     "PLD", "SO", "DUK", "CNE", "DE", "CL", "APH", "EOG", "AON", "ICE",
#     "FDX", "NOC", "BDX", "FCX", "MCK", "MOD", "CMG", "MAR", "HCA", "ITW",
#     "USB", "MO", "ORLY", "SNPS", "CDNS", "TT", "PH", "MCO", "PWR", "EPR",
#     "HUM", "CSX", "PYPL", "PAN", "NSC", "ROP", "ADM", "AEM", "ADSK", "TDG",
#     "TFC", "HMC", "COR", "AIG", "AFL", "RSG", "CINF", "AZO", "PAYX", "PCAR",

#     # Top 151–200
#     "EMR", "EW", "D", "HAL", "SLB", "O", "WELL", "PSA", "EXC", "XEL",
#     "ALL", "MET", "TRV", "PRU", "BK", "DLR", "VMC", "MLM", "KMB", "GIS",
#     "AEP", "SRE", "WEC", "ES", "PEG", "ED", "FAST", "CTAS", "GWW", "CPRT",
#     "ODFL", "ROK", "AME", "VRSK", "IDXX", "IQV", "DXCM", "MTD", "RMD",
#     "ZTS", "MOH", "HLT", "YUM", "DRI", "ROST", "DAL", "UAL", "AAL"
# ]

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "INTC",
    "JPM", "BAC", "V", "MA", "UNH", "JNJ", "PFE", "PG", "XOM", "CVX",
    "HD", "COST", "PEP", "KO", "DIS", "CAT", "DE", "BA", "GE", "LMT"
]

# Remove duplicates while maintaining clean format
TICKERS = list(set(TICKERS))

MIN_VOLUME = 1000000
MIN_PRICE = 5.0
TOP_N_POSITIONS = 10
INITIAL_CAPITAL = 1000.0

print(f"Fetching historical data for {len(TICKERS)} tickers...")
data = yf.download(TICKERS, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

closes = data['Close']
highs = data['High']
volumes = data['Volume']

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
recent_cross20 = cross20.rolling(window=3).max() > 0

# Scan filters
short_term_up = d1_9 > 0
macro_uptrend = closes > sma200
liquidity = (volumes.shift(1) > MIN_VOLUME) & (closes > MIN_PRICE)

# Final Scan Condition
scan_signals = liquidity & macro_uptrend & short_term_up & recent_cross20

# 50-day High Resistance Target
resistance_50d = highs.shift(1).rolling(window=50).max()

# ==========================================
# 3. BACKTEST ENGINE
# ==========================================
backtest_start_date = (end_date - timedelta(days=months_run * 30)).strftime('%Y-%m-%d')
trading_days = closes.index[closes.index >= backtest_start_date]

portfolio = {} 
cash = INITIAL_CAPITAL
portfolio_history = []
trades_log = []

for date in trading_days:
    today_closes = closes.loc[date]
    today_highs = highs.loc[date]
    today_d1_20 = d1_20.loc[date]
    today_d1_9 = d1_9.loc[date]
    today_res = resistance_50d.loc[date]
    
    # Check Exits
    closed_tickers = []
    for ticker, pos in list(portfolio.items()):
        curr_price = today_closes[ticker]
        curr_high = today_highs[ticker]
        curr_d1_9 = today_d1_9[ticker]
        
        # Exit Rule 1: Partial profit target at 50-day Resistance
        if not pos['res_hit'] and curr_high >= pos['res_target']:
            sell_shares = pos['shares'] // 2
            if sell_shares > 0:
                cash += sell_shares * pos['res_target']
                pos['shares'] -= sell_shares
                pos['res_hit'] = True
                trades_log.append({
                    'Date': date, 'Ticker': ticker, 'Action': 'PARTIAL_EXIT (Resistance Hit)', 
                    'Price': pos['res_target'], 'Shares': sell_shares
                })

        # Exit Rule 2: 9-Day SMA Slope turns negative
        if curr_d1_9 < 0:
            cash += pos['shares'] * curr_price
            trades_log.append({
                'Date': date, 'Ticker': ticker, 'Action': 'FULL_EXIT (d1_9 < 0)', 
                'Price': curr_price, 'Shares': pos['shares']
            })
            closed_tickers.append(ticker)
            
    for ticker in closed_tickers:
        del portfolio[ticker]

    # Check Entries
    open_slots = TOP_N_POSITIONS - len(portfolio)
    if open_slots > 0:
        valid_today = scan_signals.loc[date]
        candidates = valid_today[valid_today == True].index.tolist()
        candidates = [t for t in candidates if t not in portfolio]
        
        if candidates:
            combined_slope = (today_d1_20[candidates] + today_d1_9[candidates]).dropna()
            top_candidates = combined_slope.sort_values(ascending=False).head(open_slots).index.tolist()
            
            allocation_per_slot = cash / len(top_candidates) if top_candidates else 0
            
            for ticker in top_candidates:
                entry_price = today_closes[ticker]
                target_res = today_res[ticker]
                shares_to_buy = int(allocation_per_slot // entry_price)
                
                if shares_to_buy > 0 and cash >= (shares_to_buy * entry_price):
                    cost = shares_to_buy * entry_price
                    cash -= cost
                    portfolio[ticker] = {
                        'shares': shares_to_buy,
                        'entry_price': entry_price,
                        'res_target': target_res,
                        'res_hit': False
                    }
                    trades_log.append({
                        'Date': date, 'Ticker': ticker, 'Action': 'BUY', 
                        'Price': entry_price, 'Shares': shares_to_buy
                    })

    # Track Portfolio Value
    positions_value = sum([pos['shares'] * today_closes[t] for t, pos in portfolio.items()])
    total_val = cash + positions_value
    portfolio_history.append({'Date': date, 'Total_Value': total_val})

# ==========================================
# 4. RESULTS & CSV EXPORT
# ==========================================
results_df = pd.DataFrame(portfolio_history).set_index('Date')
trades_df = pd.DataFrame(trades_log)

# Export daily account performance
daily_csv = "daily_portfolio_value.csv"
results_df.rename(columns={'Total_Value': 'Net_Account_Value'}).to_csv(daily_csv)

# Export trade execution log
trades_csv = "executed_trades_log.csv"
trades_df.to_csv(trades_csv, index=False)

final_value = results_df['Total_Value'].iloc[-1]
net_pct_return = ((final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

print("\n" + "="*50)
print(f"BACKTEST RESULTS (LAST {months_run} MONTHS)")
print("="*50)
print(f"Starting Capital:     ${INITIAL_CAPITAL:,.2f}")
print(f"Final Capital:        ${final_value:,.2f}")
print(f"Net Percentage Gain:  {net_pct_return:.2f}%")
print(f"Total Trades Executed:{len(trades_log)}")
print(f"Daily Portfolio CSV:  Exported to '{daily_csv}'")
print(f"Trades Execution CSV: Exported to '{trades_csv}'")
print("="*50)