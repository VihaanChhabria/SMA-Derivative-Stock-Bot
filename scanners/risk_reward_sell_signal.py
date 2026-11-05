import pandas as pd

from config.settings import RISK_REWARD_STOP_LOSS_PCT

buy_signals = globals().get("buy_signals")
ohlcv_data = globals().get("ohlcv_data")
reward_ratio = globals().get("reward_ratio", 2.0)
stop_loss_pct = globals().get("stop_loss_pct", RISK_REWARD_STOP_LOSS_PCT)
sell_signals = None


def generate_sell_signals(
    buy_signals: pd.DataFrame,
    ohlcv_data: pd.DataFrame,
    reward_ratio: float,
    stop_loss_pct: float = RISK_REWARD_STOP_LOSS_PCT,
) -> pd.DataFrame:
    """Sell when price hits the take-profit or stop-loss of a 1:reward_ratio trade.

    Each buy signal enters at that day's close (if the ticker has no open
    position). Risk is the distance to a stop ``stop_loss_pct`` below entry and
    the target sits ``reward_ratio`` times that distance above entry. If a
    bar's range touches both levels, the stop is assumed to fill first.
    """
    if reward_ratio <= 0:
        raise ValueError("reward_ratio must be positive")
    if not 0 < stop_loss_pct < 1:
        raise ValueError("stop_loss_pct must be between 0 and 1")

    closes = ohlcv_data["Close"].reindex(index=buy_signals.index, columns=buy_signals.columns)
    highs = ohlcv_data["High"].reindex_like(closes)
    lows = ohlcv_data["Low"].reindex_like(closes)

    sell_signals = pd.DataFrame(False, index=buy_signals.index, columns=buy_signals.columns)

    for ticker in buy_signals.columns:
        stop = target = None
        for date in buy_signals.index:
            if stop is not None:
                if lows.at[date, ticker] <= stop or highs.at[date, ticker] >= target:
                    sell_signals.at[date, ticker] = True
                    stop = target = None
                continue

            entry = closes.at[date, ticker]
            if buy_signals.at[date, ticker] and pd.notna(entry):
                risk = entry * stop_loss_pct
                stop = entry - risk
                target = entry + risk * reward_ratio

    return sell_signals


def make_exit_rule(reward_ratio: float, stop_loss_pct: float, **_):
    """Backtest exit rule: sell at the stop or target set from the real entry.

    Fills at the level itself, or at the open if the price gapped through it.
    If a bar touches both levels, the stop is assumed to fill first.
    """
    if reward_ratio <= 0:
        raise ValueError("reward_ratio must be positive")
    if not 0 < stop_loss_pct < 1:
        raise ValueError("stop_loss_pct must be between 0 and 1")

    def rule(position, date, bar):
        risk = position.entry_price * stop_loss_pct
        stop = position.entry_price - risk
        target = position.entry_price + risk * reward_ratio
        if bar.low <= stop:
            return (min(stop, bar.open) if pd.notna(bar.open) else stop), "Stop loss"
        if bar.high >= target:
            return (max(target, bar.open) if pd.notna(bar.open) else target), "Take profit"
        return None

    return rule


def main(reward_ratio: float, stop_loss_pct: float) -> pd.DataFrame:
    if not isinstance(buy_signals, pd.DataFrame):
        raise TypeError("buy_signals must be a pandas DataFrame")
    if not isinstance(ohlcv_data, pd.DataFrame):
        raise TypeError("ohlcv_data must be a pandas DataFrame")

    global sell_signals
    sell_signals = generate_sell_signals(
        buy_signals, ohlcv_data, reward_ratio, stop_loss_pct
    )
    return sell_signals


if __name__ == "__main__":
    main(reward_ratio, stop_loss_pct)
