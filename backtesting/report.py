"""Save and print backtest results, one output folder per run."""

import json
from datetime import datetime
from pathlib import Path

from backtesting.engine import BacktestResult
from src.paths import output_path


def save_backtest_run(result: BacktestResult, settings: dict) -> Path:
    """Write trades, daily equity, summary and settings to a new run folder.

    The folder is ``outputs/backtests/<timestamp>_<buys>_<sells>/``.
    """
    run_name = "_".join(
        [
            f"{datetime.now():%Y%m%d_%H%M%S}",
            "+".join(settings["buy_strategies"]) or "none",
            "+".join(settings["sell_strategies"]) or "none",
        ]
    )
    run_dir = output_path(f"backtests/{run_name}", "trades.csv").parent

    result.trades.to_csv(run_dir / "trades.csv", index=False)
    result.equity.to_csv(run_dir / "daily_equity.csv")
    (run_dir / "summary.json").write_text(json.dumps(result.summary, indent=2))
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2, default=str))
    return run_dir


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}%"


def print_summary(result: BacktestResult, run_dir: Path | None = None) -> None:
    s = result.summary
    win_rate = "n/a" if s["Win_Rate_Pct"] is None else f"{s['Win_Rate_Pct']:.1f}%"
    profit_factor = "n/a" if s["Profit_Factor"] is None else f"{s['Profit_Factor']:.2f}"
    avg_days = "n/a" if s["Avg_Days_Held"] is None else f"{s['Avg_Days_Held']:.1f}"

    print("\n" + "=" * 50)
    print(f"BACKTEST RESULTS ({s['Start_Date']} to {s['End_Date']})")
    print("=" * 50)
    print(f"Starting Capital:     ${s['Starting_Capital']:,.2f}")
    print(f"Final Equity:         ${s['Final_Equity']:,.2f}")
    print(f"Total Return:         {_fmt_pct(s['Total_Return_Pct'])}")
    print(f"Avg Return per Year:  {_fmt_pct(s['Annualized_Return_Pct'])} (annualized)")
    print(f"Max Drawdown:         {_fmt_pct(s['Max_Drawdown_Pct'])}")
    print(f"Closed Trades:        {s['Closed_Trades']}")
    print(f"Open Positions:       {s['Open_Positions']}")
    print(f"Win Rate:             {win_rate}")
    print(f"Avg Win / Avg Loss:   {_fmt_pct(s['Avg_Win_Pct'])} / {_fmt_pct(s['Avg_Loss_Pct'])}")
    print(f"Profit Factor:        {profit_factor}")
    print(f"Avg Days Held:        {avg_days}")
    for reason, count in s["Exit_Reasons"].items():
        print(f"  Exit - {reason}: {count}")
    if run_dir is not None:
        print(f"Results saved to:     '{run_dir}'")
    print("=" * 50)
