"""Local storage for downloaded OHLCV data so backtests can reuse it."""

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence  # noqa: UP035

import pandas as pd

from src.paths import PROJECT_ROOT

DOWNLOADS_DIR = PROJECT_ROOT / "data" / "downloads"


def to_date(value: str | date | datetime) -> date:
    """Normalize a YYYY-MM-DD string, date, or datetime to a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def normalize_tickers(tickers: str | Sequence[str]) -> tuple[str, ...]:
    """Return tickers as a sorted, de-duplicated tuple."""
    if isinstance(tickers, str):
        tickers = re.split(r"[\s,]+", tickers)
    return tuple(sorted({t.strip().upper() for t in tickers if t.strip()}))


@dataclass(frozen=True)
class SavedDownload:
    """A downloaded OHLCV dataset stored on disk with its metadata."""

    data_path: Path
    start: date
    end: date
    tickers: tuple[str, ...]
    downloaded_at: datetime

    @property
    def label(self) -> str:
        return (
            f"{self.start} to {self.end} | {len(self.tickers)} tickers | "
            f"downloaded {self.downloaded_at:%Y-%m-%d %H:%M}"
        )

    def covers(
        self,
        tickers: str | Sequence[str],
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> bool:
        """Whether this download contains every ticker over [start, end)."""
        return (
            self.start <= to_date(start)
            and self.end >= to_date(end)
            and set(normalize_tickers(tickers)) <= set(self.tickers)
        )

    def load(self) -> pd.DataFrame:
        return pd.read_pickle(self.data_path)


def save_download(
    data: pd.DataFrame,
    tickers: str | Sequence[str],
    start: str | date | datetime,
    end: str | date | datetime,
) -> SavedDownload:
    """Write downloaded data plus a JSON metadata sidecar to DOWNLOADS_DIR."""
    start, end = to_date(start), to_date(end)
    tickers = normalize_tickers(tickers)
    downloaded_at = datetime.now().replace(microsecond=0)

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    stem = (
        f"ohlcv_{start:%Y%m%d}_{end:%Y%m%d}_"
        f"{len(tickers)}tickers_{downloaded_at:%Y%m%d_%H%M%S}"
    )
    data_path = DOWNLOADS_DIR / f"{stem}.pkl"
    data.to_pickle(data_path)
    data_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "tickers": list(tickers),
                "downloaded_at": downloaded_at.isoformat(),
            },
            indent=2,
        )
    )
    return SavedDownload(data_path, start, end, tickers, downloaded_at)


def list_downloads() -> list[SavedDownload]:
    """Return all saved downloads, most recently downloaded first."""
    downloads = []
    for meta_path in DOWNLOADS_DIR.glob("*.json"):
        data_path = meta_path.with_suffix(".pkl")
        if not data_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            downloads.append(
                SavedDownload(
                    data_path=data_path,
                    start=to_date(meta["start"]),
                    end=to_date(meta["end"]),
                    tickers=tuple(meta["tickers"]),
                    downloaded_at=datetime.fromisoformat(meta["downloaded_at"]),
                )
            )
        except (ValueError, KeyError, TypeError):
            print(f"Skipping unreadable download metadata: '{meta_path}'")
    return sorted(downloads, key=lambda d: d.downloaded_at, reverse=True)


def find_covering_download(
    tickers: str | Sequence[str],
    start: str | date | datetime,
    end: str | date | datetime,
) -> SavedDownload | None:
    """Return the newest saved download that covers the tickers and range."""
    for download in list_downloads():
        if download.covers(tickers, start, end):
            return download
    return None
