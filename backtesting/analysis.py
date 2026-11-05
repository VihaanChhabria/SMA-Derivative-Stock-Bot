import json
from pathlib import Path

import inquirer
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import download_ohlcv
from src.paths import PROJECT_ROOT

BACKTESTS_DIR = PROJECT_ROOT / "outputs" / "backtests"


def select_run_dir() -> Path | None:
    """Ask which saved backtest run to analyze, newest first."""
    run_dirs = sorted(
        (d for d in BACKTESTS_DIR.iterdir() if (d / "daily_equity.csv").exists()),
        key=lambda d: (d / "daily_equity.csv").stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        print(f"No backtest runs found in '{BACKTESTS_DIR}'. Run a backtest from main.py first.")
        return None
    return inquirer.prompt(
        [
            inquirer.List(
                "run_dir",
                message="Select a backtest run to compare against the S&P 500",
                choices=[(d.name, d) for d in run_dirs],
            )
        ]
    )["run_dir"]


def main() -> None:

    # ==========================================
    # 1. READ BACKTEST CSV DATA
    # ==========================================
    run_dir = select_run_dir()
    if run_dir is None:
        return

    settings_file = run_dir / "settings.json"
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    strategy_name = " + ".join(
        settings.get("buy_strategies", []) + settings.get("sell_strategies", [])
    ) or run_dir.name

    # Load the run's daily equity CSV
    df_strategy = pd.read_csv(run_dir / "daily_equity.csv")
    df_strategy['Date'] = pd.to_datetime(df_strategy['Date'])
    df_strategy.sort_values('Date', inplace=True)

    start_date = df_strategy['Date'].min()
    end_date = df_strategy['Date'].max()
    initial_capital = settings.get("initial_capital", df_strategy['Total_Equity'].iloc[0])

    print(f"Loaded backtest data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # ==========================================
    # 2. FETCH MATCHING S&P 500 (SPY) BENCHMARK
    # ==========================================
    print("Fetching S&P 500 (SPY) benchmark data...")

    # Download S&P 500 ETF data for the exact backtest timeframe
    sp500 = download_ohlcv(
        'SPY',
        start=start_date.strftime('%Y-%m-%d'),
        end=(end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        progress=False,
        save=False,
    )

    # Extract close prices (handles yfinance multi-index formats)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500_closes = sp500['Close']['SPY']
    else:
        sp500_closes = sp500['Close']

    df_sp500 = pd.DataFrame({'Date': sp500_closes.index, 'SPY_Close': sp500_closes.values})
    df_sp500['Date'] = pd.to_datetime(df_sp500['Date']).dt.tz_localize(None)

    # Merge backtest data with SPY benchmark data
    df_merged = pd.merge(df_strategy, df_sp500, on='Date', how='inner')

    # Normalize S&P 500 starting value to match the strategy's starting capital
    df_merged['SP500_Value'] = (df_merged['SPY_Close'] / df_merged['SPY_Close'].iloc[0]) * initial_capital

    # ==========================================
    # 3. PERFORMANCE & DRAWDOWN CALCULATIONS
    # ==========================================
    strat_final = df_merged['Total_Equity'].iloc[-1]
    spy_final = df_merged['SP500_Value'].iloc[-1]

    strat_return = ((strat_final - initial_capital) / initial_capital) * 100
    spy_return = ((spy_final - initial_capital) / initial_capital) * 100

    def calc_max_drawdown(series):
        peak = series.cummax()
        drawdown = (series - peak) / peak
        return drawdown.min() * 100

    strat_mdd = calc_max_drawdown(df_merged['Total_Equity'])
    spy_mdd = calc_max_drawdown(df_merged['SP500_Value'])

    print("\n" + "="*55)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*55)
    print(f"Strategy Final Value:  ${strat_final:,.2f} ({strat_return:+.2f}%) | Max DD: {strat_mdd:.2f}%")
    print(f"S&P 500 Final Value:   ${spy_final:,.2f} ({spy_return:+.2f}%) | Max DD: {spy_mdd:.2f}%")
    print(f"Excess Return (Alpha): {strat_return - spy_return:+.2f}%")
    print("="*55)

    # ==========================================
    # 4. PLOT VISUAL COMPARISON
    # ==========================================
    plt.figure(figsize=(12, 6))

    plt.plot(
        df_merged['Date'],
        df_merged['Total_Equity'],
        label=f'{strategy_name} ({strat_return:+.1f}%)',
        color='#1f77b4',
        linewidth=2.5
    )

    plt.plot(
        df_merged['Date'],
        df_merged['SP500_Value'],
        label=f'S&P 500 / SPY Benchmark ({spy_return:+.1f}%)',
        color='#ff7f0e',
        linestyle='--',
        linewidth=2.0
    )

    plt.title(f'{strategy_name} vs. S&P 500 Benchmark', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Portfolio Net Value ($)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=11, frameon=True)

    # Format Y-axis as currency
    plt.gca().yaxis.set_major_formatter('${x:,.0f}')
    plt.tight_layout()

    # Save chart as high-res PNG image
    chart_file = run_dir / "strategy_vs_sp500.png"
    plt.savefig(chart_file, dpi=300)
    print(f"Chart saved to '{chart_file}'.")

    # Display chart window
    plt.show()

if __name__ == "__main__":
    main()
