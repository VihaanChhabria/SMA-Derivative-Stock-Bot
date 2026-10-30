"""Market data download helpers."""

from datetime import date, datetime
from typing import Sequence  # noqa: UP035

import pandas as pd
import yfinance as yf


def download_ohlcv(
    tickers: str | Sequence[str],
    start: str | date | datetime,
    end: str | date | datetime,
) -> pd.DataFrame:
    """Download OHLCV data using yfinance's standard column layout."""

    return yf.download(tickers, start=start, end=end)
