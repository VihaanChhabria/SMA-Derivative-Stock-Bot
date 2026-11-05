"""Portfolio backtest engine driven by buy signals and position-aware exit rules."""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from config.settings import ALLOW_FRACTIONAL_SHARES


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    bars_held: int = 0  # Trading days since entry


@dataclass(frozen=True)
class Bar:
    open: float
    high: float
    low: float
    close: float


# Returns (exit_price, reason) to close the position today, or None to hold
ExitRule = Callable[[Position, pd.Timestamp, Bar], tuple[float, str] | None]

OPEN_REASON = "OPEN (marked to market)"

TRADE_COLUMNS = [
    "Ticker", "Entry_Date", "Entry_Price", "Shares", "Cost", "Exit_Date",
    "Exit_Price", "Exit_Reason", "PnL", "Return_Pct", "Days_Held",
]


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    summary: dict = field(default_factory=dict)


def _trade_row(pos: Position, exit_date, exit_price: float, reason: str) -> dict:
    cost = pos.shares * pos.entry_price
    pnl = pos.shares * exit_price - cost
    return {
        "Ticker": pos.ticker,
        "Entry_Date": pos.entry_date,
        "Entry_Price": pos.entry_price,
        "Shares": pos.shares,
        "Cost": cost,
        "Exit_Date": exit_date,
        "Exit_Price": exit_price,
        "Exit_Reason": reason,
        "PnL": pnl,
        "Return_Pct": pnl / cost * 100,
        "Days_Held": pos.bars_held,
    }


def run_backtest(
    ohlcv_data: pd.DataFrame,
    buy_signals: pd.DataFrame,
    buy_scores: pd.DataFrame | None,
    exit_rules: list[ExitRule],
    initial_capital: float,
    max_positions: int,
) -> BacktestResult:
    """Simulate trading every day in ``buy_signals.index``.

    Each day: check exits on held positions (first rule to fire wins), then buy
    the top-ranked new signals at the close, splitting cash evenly across them,
    then mark the portfolio to market.
    """
    dates = buy_signals.index
    tickers = buy_signals.columns
    prices = {
        name: ohlcv_data[name].reindex(index=dates, columns=tickers)
        for name in ("Open", "High", "Low", "Close")
    }
    # Value holdings at the last known close if a ticker has no bar today
    valuation_closes = prices["Close"].ffill()
    if buy_scores is not None:
        buy_scores = buy_scores.reindex(index=dates, columns=tickers)

    cash = initial_capital
    portfolio: dict[str, Position] = {}
    trades = []
    equity_rows = []
    peak_equity = initial_capital

    for date in dates:
        # 1. Exits
        for ticker, pos in list(portfolio.items()):
            pos.bars_held += 1
            bar = Bar(*(prices[name].at[date, ticker] for name in ("Open", "High", "Low", "Close")))
            if pd.isna(bar.close):
                continue
            for rule in exit_rules:
                exit_hit = rule(pos, date, bar)
                if exit_hit is not None:
                    exit_price, reason = exit_hit
                    cash += pos.shares * exit_price
                    trades.append(_trade_row(pos, date, exit_price, reason))
                    del portfolio[ticker]
                    break

        # 2. Entries
        open_slots = max_positions - len(portfolio)
        if open_slots > 0:
            closes_today = prices["Close"].loc[date]
            signals_today = buy_signals.loc[date]
            candidates = [
                t for t in tickers
                if signals_today[t] and t not in portfolio and pd.notna(closes_today[t])
            ]
            if buy_scores is not None:
                scores_today = buy_scores.loc[date].fillna(-math.inf)
                # Highest score first (unscored last), ties broken by ticker
                candidates.sort(key=lambda t: (-scores_today[t], t))
            selected = candidates[:open_slots]

            if selected:
                allocation = cash / len(selected)
                for ticker in selected:
                    price = closes_today[ticker]
                    shares = allocation / price
                    if not ALLOW_FRACTIONAL_SHARES:
                        shares = math.floor(shares)
                    if shares <= 0:
                        continue
                    # Clamp float rounding so the last buy can't leave cash at -1e-13
                    cash = max(cash - shares * price, 0.0)
                    portfolio[ticker] = Position(ticker, date, price, shares)

        # 3. Mark to market
        positions_value = sum(
            pos.shares * valuation_closes.at[date, t] for t, pos in portfolio.items()
        )
        total_equity = cash + positions_value
        peak_equity = max(peak_equity, total_equity)
        equity_rows.append(
            {
                "Date": date,
                "Cash": cash,
                "Positions_Value": positions_value,
                "Total_Equity": total_equity,
                "Open_Positions": len(portfolio),
                "Drawdown_Pct": (total_equity / peak_equity - 1) * 100,
            }
        )

    # Log positions still held at the end, valued at the last known close
    for ticker, pos in portfolio.items():
        trades.append(
            _trade_row(pos, dates[-1], valuation_closes.at[dates[-1], ticker], OPEN_REASON)
        )

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    equity_df = pd.DataFrame(equity_rows).set_index("Date")
    return BacktestResult(
        trades=trades_df,
        equity=equity_df,
        summary=summarize(trades_df, equity_df, initial_capital),
    )


def summarize(trades: pd.DataFrame, equity: pd.DataFrame, initial_capital: float) -> dict:
    """Headline performance stats; open positions only count toward equity."""
    closed = trades[trades["Exit_Reason"] != OPEN_REASON]
    wins = closed[closed["PnL"] > 0]
    losses = closed[closed["PnL"] <= 0]
    gross_loss = -losses["PnL"].sum()
    final_equity = float(equity["Total_Equity"].iloc[-1]) if not equity.empty else initial_capital

    def mean_or_none(series: pd.Series) -> float | None:
        return float(series.mean()) if not series.empty else None

    # Compound annual growth rate over the calendar span of the backtest
    years = (equity.index[-1] - equity.index[0]).days / 365.25 if not equity.empty else 0
    annualized_return = (
        ((final_equity / initial_capital) ** (1 / years) - 1) * 100
        if years > 0 and final_equity > 0
        else None
    )

    return {
        "Start_Date": str(equity.index[0].date()) if not equity.empty else None,
        "End_Date": str(equity.index[-1].date()) if not equity.empty else None,
        "Starting_Capital": float(initial_capital),
        "Final_Equity": final_equity,
        "Total_Return_Pct": (final_equity / initial_capital - 1) * 100,
        "Annualized_Return_Pct": annualized_return,
        "Max_Drawdown_Pct": float(equity["Drawdown_Pct"].min()) if not equity.empty else 0.0,
        "Closed_Trades": len(closed),
        "Open_Positions": len(trades) - len(closed),
        "Win_Rate_Pct": len(wins) / len(closed) * 100 if len(closed) else None,
        "Avg_Win_Pct": mean_or_none(wins["Return_Pct"]),
        "Avg_Loss_Pct": mean_or_none(losses["Return_Pct"]),
        "Profit_Factor": float(wins["PnL"].sum() / gross_loss) if gross_loss > 0 else None,
        "Realized_PnL": float(closed["PnL"].sum()),
        "Avg_Days_Held": mean_or_none(closed["Days_Held"]),
        "Exit_Reasons": dict(Counter(closed["Exit_Reason"])),
    }
