# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Date-based rolling window generator.
Works with trading calendar dates rather than integer indices,
so it naturally handles holidays and suspension gaps.

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Generator, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindowSlice:
    window_end_date: str       # Last date of context window (analysis as-of date)
    horizon_start_date: str    # First date of prediction horizon
    horizon_end_date: str      # Last date of prediction horizon
    is_last: bool = False


class InsufficientDataError(Exception):
    pass


def generate_date_windows(
    trading_dates: list[str],   # sorted ISO date strings from full history
    horizon_days: int = 20,
    step_days: int = 5,
    warmup_days: int = 60,      # skip first N dates (indicators need warmup)
) -> Generator[WindowSlice, None, None]:
    """
    Generate rolling window slices over a list of trading dates.

    For each window:
    - window_end_date: the as-of date for Wyckoff analysis
    - horizon_start_date: first date after window_end
    - horizon_end_date: last date of horizon (window_end + horizon_days trading days)

    Minimum requirement: warmup_days + horizon_days + 1 dates
    """
    n = len(trading_dates)
    min_required = warmup_days + horizon_days + 1
    if n < min_required:
        raise InsufficientDataError(
            f"Need at least {min_required} trading dates, got {n}"
        )

    # First valid window_end is at warmup_days index
    # (so the Wyckoff engine has warmup_days bars before this date)
    start_idx = warmup_days
    # Last valid window_end: must have horizon_days more dates after it
    end_idx = n - horizon_days - 1

    if start_idx > end_idx:
        raise InsufficientDataError("Not enough data for even one window after warmup")

    step = step_days
    idx = start_idx
    while idx <= end_idx:
        window_end = trading_dates[idx]
        horizon_start = trading_dates[idx + 1]
        horizon_end_idx = min(idx + horizon_days, n - 1)
        horizon_end = trading_dates[horizon_end_idx]
        is_last = (idx + step > end_idx)

        yield WindowSlice(
            window_end_date=window_end,
            horizon_start_date=horizon_start,
            horizon_end_date=horizon_end,
            is_last=is_last,
        )
        idx += step


def get_horizon_prices(
    price_series: dict,   # {date_str: close_price}
    horizon_start: str,
    horizon_end: str,
    trading_dates: list[str],
) -> list[float]:
    """
    Extract closing prices for the horizon period.
    Returns list of close prices in date order.
    """
    prices = []
    in_range = False
    for d in trading_dates:
        if d == horizon_start:
            in_range = True
        if in_range:
            if d in price_series:
                prices.append(price_series[d])
            if d == horizon_end:
                break
    return prices


def generate_monthly_windows(
    trading_dates: list[str],
    warmup_months: int = 12,
) -> list[WindowSlice]:
    """
    Generate one window per calendar month.

    - window_end_date  = last trading day of month M
    - horizon_start    = first trading day of month M+1
    - horizon_end      = last  trading day of month M+1
    - Skips first `warmup_months` months (Wyckoff indicators need warmup)
    - Final month (no "next month" to verify against) is omitted
    """
    import pandas as pd
    if not trading_dates:
        raise InsufficientDataError("No trading dates supplied")

    df = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
    df["ym"] = df["date"].dt.to_period("M")
    months = df.groupby("ym")["date"].agg(["min", "max"]).reset_index()

    if len(months) < warmup_months + 2:
        raise InsufficientDataError(
            f"Need at least {warmup_months + 2} months of data, got {len(months)}"
        )

    windows: list[WindowSlice] = []
    last_idx = len(months) - 2  # last index that still has a "next month"
    for i in range(warmup_months, last_idx + 1):
        window_end = months.loc[i, "max"].strftime("%Y-%m-%d")
        horizon_start = months.loc[i + 1, "min"].strftime("%Y-%m-%d")
        horizon_end = months.loc[i + 1, "max"].strftime("%Y-%m-%d")
        is_last = (i == last_idx)
        windows.append(WindowSlice(
            window_end_date=window_end,
            horizon_start_date=horizon_start,
            horizon_end_date=horizon_end,
            is_last=is_last,
        ))
    return windows


def fetch_trading_dates_and_prices(
    symbol: str,
    end_date: Optional[str] = None,
    data_source_type: str = "stock",
) -> tuple[list[str], dict]:
    """
    Fetch full history and return (sorted_trading_dates, {date: close_price}).
    Market data is served by the remote Wyckoff service (/v1/prices); retries on
    transient service errors before giving up.

    data_source_type:
      - "stock" (default): days=2000 trading days back
      - "index":          entire available history (days param ignored)
    """
    import time as _time
    from finagent.data.fetcher import fetch_ohlcv_df

    days = 100_000 if data_source_type == "index" else 2000
    for _attempt in range(4):
        try:
            df = fetch_ohlcv_df(
                symbol, days=days, end_date=end_date,
                data_source_type=data_source_type,
            )
            if df.empty:
                raise ValueError(f"Empty result for {symbol}")
            dates = [d.strftime("%Y-%m-%d") for d in df.index]
            prices = dict(zip(dates, df["close"].astype(float).tolist()))
            return dates, prices
        except Exception as _e:
            if _attempt < 3:
                wait = 60 * (2 ** _attempt)
                logger.warning(f"fetch_ohlcv_df failed (attempt {_attempt+1}/4): {_e}. Retrying in {wait}s...")
                _time.sleep(wait)
            else:
                raise InsufficientDataError(str(_e)) from _e
