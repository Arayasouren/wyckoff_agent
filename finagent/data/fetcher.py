# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Market-data client.

All OHLCV is served by the remote Wyckoff service (/v1/prices); this module
holds NO local data backend (no akshare/yfinance/Wind). The service fetches the
data (it may be backed by Wind WDS) and returns it to the client.

Public API (unchanged shapes, so callers like rolling_window/chart need no edits):
  fetch_ohlcv_df(symbol, days, end_date, source, data_source_type) -> pd.DataFrame
      DataFrame with columns [open, high, low, close, volume] and a DatetimeIndex.
  fetch_ohlcv(symbol, days, end_date, source, data_source_type) -> (dates, prices)
      dates: list["YYYY-MM-DD"]; prices: {"YYYY-MM-DD": close}
"""
from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

from finagent.service import service_post

logger = logging.getLogger(__name__)

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _payload_to_df(data: dict) -> pd.DataFrame:
    """Rebuild an OHLCV DataFrame from the service's /v1/prices JSON."""
    dates = data.get("dates", []) or []
    cols = {c: (data.get(c) or []) for c in _OHLCV_COLS}
    df = pd.DataFrame(cols)
    if not len(df):
        return df
    df.index = pd.to_datetime(dates)
    return df[_OHLCV_COLS]


def _df_to_result(df: pd.DataFrame, days: int) -> tuple[list[str], dict]:
    """Convert a standardised OHLCV DataFrame to (dates, prices)."""
    df = df.sort_index()
    if len(df) > days:
        df = df.iloc[-days:]
    dates = [str(d.date()) if hasattr(d, "date") else str(d)[:10] for d in df.index]
    prices = {
        (str(d.date()) if hasattr(d, "date") else str(d)[:10]): float(row["close"])
        for d, row in df.iterrows()
    }
    return dates, prices


def fetch_ohlcv_df(
    symbol: str,
    days: int = 2000,
    end_date: Optional[str] = None,
    source: Optional[str] = None,          # accepted for signature compat; ignored (server decides)
    data_source_type: str = "stock",
) -> pd.DataFrame:
    """Fetch OHLCV for a symbol from the Wyckoff service. Returns a DataFrame."""
    data = service_post("/v1/prices", {
        "symbol": symbol,
        "days": days,
        "end_date": end_date,
        "data_source_type": data_source_type,
    })
    return _payload_to_df(data)


def fetch_ohlcv(
    symbol: str,
    days: int = 2000,
    end_date: Optional[str] = None,
    source: Optional[str] = None,
    data_source_type: str = "stock",
) -> tuple[list[str], dict]:
    """Fetch OHLCV and return (sorted_dates, {date: close})."""
    df = fetch_ohlcv_df(symbol, days=days, end_date=end_date, data_source_type=data_source_type)
    return _df_to_result(df, days)
