import pandas as pd

buy_signals = globals().get("buy_signals")
sell_signals = None
days_to_hold = globals().get("days_to_hold", 5)


def generate_sell_signals(
    buy_signals: pd.DataFrame, days_to_hold: int
) -> pd.DataFrame:
    if isinstance(days_to_hold, bool) or not isinstance(days_to_hold, int):
        raise TypeError("days_to_hold must be a non-negative integer")
    if days_to_hold < 0:
        raise ValueError("days_to_hold must be a non-negative integer")

    return buy_signals.shift(periods=days_to_hold, fill_value=False)


def main(days_to_hold: int) -> pd.DataFrame:
    if not isinstance(buy_signals, pd.DataFrame):
        raise TypeError("buy_signals must be a pandas DataFrame")

    global sell_signals
    sell_signals = generate_sell_signals(buy_signals, days_to_hold)
    return sell_signals


if __name__ == "__main__":
    main(days_to_hold)
