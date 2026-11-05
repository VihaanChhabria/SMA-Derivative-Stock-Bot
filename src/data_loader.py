"""Market data download helpers."""

from datetime import date, datetime
from typing import Sequence  # noqa: UP035

import pandas as pd
import yfinance as yf

from src.data_store import (
    find_covering_download,
    normalize_tickers,
    save_download,
    to_date,
)


def download_ohlcv(
    tickers: str | Sequence[str],
    start: str | date | datetime,
    end: str | date | datetime,
    *,
    save: bool = True,
    **download_kwargs,
) -> pd.DataFrame:
    """Download OHLCV data using yfinance's standard column layout.

    The result is saved under ``data/downloads/`` unless ``save`` is False,
    so it can be selected later when backtesting.
    """

    data = yf.download(tickers, start=start, end=end, **download_kwargs)
    if save and not data.empty:
        saved = save_download(data, tickers, start, end)
        print(f"Saved downloaded data to '{saved.data_path}'.")
    return data


def load_or_download_ohlcv(
    tickers: str | Sequence[str],
    start: str | date | datetime,
    end: str | date | datetime,
) -> pd.DataFrame:
    """Reuse a saved download covering the request, otherwise download it."""

    saved = find_covering_download(tickers, start, end)
    if saved is None:
        return download_ohlcv(tickers, start=start, end=end)

    print(f"Using saved download '{saved.data_path.name}'.")
    data = saved.load()
    start_ts, end_ts = pd.Timestamp(to_date(start)), pd.Timestamp(to_date(end))
    rows = (data.index >= start_ts) & (data.index < end_ts)
    columns = data.columns.get_level_values(-1).isin(normalize_tickers(tickers))
    return data.loc[rows, columns]
