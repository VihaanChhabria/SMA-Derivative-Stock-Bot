"""Market data download helpers."""

from datetime import date, datetime
from typing import Sequence

import pandas as pd
import yfinance as yf


def download_ohlcv(
    tickers: str | Sequence[str],
    *,
    start: str | date | datetime,
    end: str | date | datetime,
    progress: bool | None = None,
) -> pd.DataFrame:
    """Download OHLCV data using yfinance's standard column layout."""
    if progress is None:
        return yf.download(tickers, start=start, end=end)
    return yf.download(tickers, start=start, end=end, progress=progress)
