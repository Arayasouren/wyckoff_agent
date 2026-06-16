"""
Unified OHLCV data fetcher with pluggable backends.

Supported DATA_SOURCE values (set via env or config):
  yfinance  — Yahoo Finance (default, no setup needed, rate-limited)
  akshare   — AkShare (A-shares only, stable, no rate limit)
  wind      — Wind WDS Oracle DB (requires Oracle Instant Client)

All backends return the same shape:
  (sorted_dates: list[str], prices: dict[str, float])
  where prices maps "YYYY-MM-DD" -> adjusted close price
"""
from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _normalize_symbol(symbol: str) -> dict:
    """Parse symbol into components for each provider."""
    code = symbol.split(".")[0]
    suffix = symbol.upper().split(".")[-1] if "." in symbol else ""
    is_ashare = suffix in ("SS", "SH", "SZ", "BJ")
    return {"raw": symbol, "code": code, "suffix": suffix, "is_ashare": is_ashare}


def _df_to_result(df: pd.DataFrame, days: int) -> tuple[list[str], dict]:
    """Convert a standardised OHLCV DataFrame to (dates, prices)."""
    df = df.sort_index()
    if len(df) > days:
        df = df.iloc[-days:]
    dates = [str(d.date()) if hasattr(d, "date") else str(d)[:10] for d in df.index]
    prices = {
        str(d.date()) if hasattr(d, "date") else str(d)[:10]: float(row["close"])
        for d, row in df.iterrows()
    }
    return dates, prices


# ─── yfinance backend ─────────────────────────────────────────────────────────

def _fetch_yfinance(symbol: str, days: int, end_date: Optional[str]) -> tuple[list[str], dict]:
    import datetime
    import yfinance as yf

    # yfinance uses .SS for Shanghai A-shares, but internal codes use .SH
    yf_symbol = symbol.replace(".SH", ".SS") if symbol.upper().endswith(".SH") else symbol

    if end_date:
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = datetime.datetime.today()
    start_dt = end_dt - datetime.timedelta(days=int(days * 1.5))

    df = yf.download(
        yf_symbol,
        start=start_dt.strftime("%Y-%m-%d"),
        end=(end_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"yfinance returned no data for {yf_symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return _df_to_result(df, days)


# ─── akshare backend ──────────────────────────────────────────────────────────

def _fetch_akshare(symbol: str, days: int, end_date: Optional[str]) -> tuple[list[str], dict]:
    import datetime
    import akshare as ak

    info = _normalize_symbol(symbol)
    if not info["is_ashare"]:
        raise ValueError(f"akshare only supports A-shares, got {symbol}")

    code = info["code"]
    end_str = end_date or datetime.date.today().strftime("%Y%m%d")
    end_str = end_str.replace("-", "")

    # calculate start date
    start_dt = datetime.datetime.strptime(end_str, "%Y%m%d") - datetime.timedelta(days=int(days * 1.5))
    start_str = start_dt.strftime("%Y%m%d")

    # adjust=hfq (后复权) — aligns with Wind's full back-adjusted prices
    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_str,
        end_date=end_str,
        adjust="hfq",
    )
    if df is None or df.empty:
        raise ValueError(f"akshare returned no data for {symbol}")

    df = df.rename(columns={
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]].dropna()
    return _df_to_result(df, days)


# ─── Wind WDS (Oracle) backend ────────────────────────────────────────────────

import threading as _threading
_wind_conn_local = _threading.local()   # per-thread connection (Oracle conns are NOT thread-safe)


def _get_wind_connection():
    """
    Return a thread-local Oracle connection to Wind WDS.
    Each worker thread (from asyncio executor pool) gets its own connection,
    preventing concurrent cursor operations from serializing on a single conn.
    """
    import oracledb
    from finagent.config import WIND_HOST, WIND_PORT, WIND_SERVICE, WIND_USER, WIND_PASSWORD

    conn = getattr(_wind_conn_local, "conn", None)

    if conn is not None:
        try:
            conn.ping()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            conn = None
            _wind_conn_local.conn = None

    if conn is None:
        from finagent.config import ORACLE_CLIENT_DIR
        try:
            kwargs = {"lib_dir": ORACLE_CLIENT_DIR} if ORACLE_CLIENT_DIR else {}
            oracledb.init_oracle_client(**kwargs)
        except Exception as _e:
            logger.debug(f"Oracle thick mode unavailable: {_e}, using thin mode")

        dsn = f"{WIND_HOST}:{WIND_PORT}/{WIND_SERVICE}"
        conn = oracledb.connect(user=WIND_USER, password=WIND_PASSWORD, dsn=dsn)
        _wind_conn_local.conn = conn
        thr_name = _threading.current_thread().name
        logger.info(f"Wind WDS connected [{thr_name}]: Oracle {conn.version}")

    return conn


def _wind_code(symbol: str) -> str:
    """Convert to Wind code format: 600519.SS → 600519.SH, others pass through."""
    info = _normalize_symbol(symbol)
    suffix_map = {"SS": "SH"}   # SS (Yahoo) → SH (Wind); SZ/SH/BJ unchanged
    wind_suffix = suffix_map.get(info["suffix"], info["suffix"])
    return f"{info['code']}.{wind_suffix}"


def _fetch_wind_index_df(symbol: str, end_date: Optional[str]) -> pd.DataFrame:
    """
    Fetch full available history of an A-share index. Returns DataFrame indexed by
    date with columns open/high/low/close/volume.

    Table routing (see below): CITIC industry indices (code 'CI...') →
    winddb.aindexindustrieseodcitics; broad-market indices → winddb.aindexeodprices.

    - All OHLCV fields must be NOT NULL (rows with missing fields are excluded).
    - No `days` cap: returns earliest valid date through `end_date`.
    - Indexes don't have adjustment factors, so uses raw S_DQ_OPEN/HIGH/LOW/CLOSE.
    """
    import datetime
    wind_code = _wind_code(symbol)
    if end_date:
        end_str = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
    else:
        end_str = datetime.datetime.today().strftime("%Y%m%d")

    # Table routing: CITIC industry indices (code starts with 'CI', e.g. CI005021.WI
    # 中信一级行业) live in winddb.aindexindustrieseodcitics; broad-market indices
    # (000300.SH, 000001.SH, 932000.CSI, ...) live in winddb.aindexeodprices.
    # Both share the raw S_DQ_OPEN/HIGH/LOW/CLOSE/VOLUME schema (no adjustment factors).
    _table = ("winddb.aindexindustrieseodcitics"
              if wind_code.upper().startswith("CI") else "winddb.aindexeodprices")

    sql = f"""
        SELECT TRADE_DT,
               S_DQ_OPEN   AS open,
               S_DQ_HIGH   AS high,
               S_DQ_LOW    AS low,
               S_DQ_CLOSE  AS close,
               S_DQ_VOLUME AS volume
        FROM   {_table}
        WHERE  S_INFO_WINDCODE = :1
          AND  TRADE_DT <= :2
          AND  S_DQ_OPEN   IS NOT NULL
          AND  S_DQ_HIGH   IS NOT NULL
          AND  S_DQ_LOW    IS NOT NULL
          AND  S_DQ_CLOSE  IS NOT NULL
          AND  S_DQ_VOLUME IS NOT NULL
        ORDER  BY TRADE_DT
    """
    conn = None
    rows = []
    for _attempt in range(4):
        try:
            conn = _get_wind_connection()
            cur = conn.cursor()
            cur.execute(sql, [wind_code, end_str])
            rows = cur.fetchall()
            cur.close()
            break
        except Exception as _e:
            _wind_conn_local.conn = None
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
            if _attempt < 3:
                import time as _t
                _wait = 30 * (2 ** _attempt)  # 30s, 60s, 120s — survives ~3.5min Wind hiccup
                logger.warning(f"Wind index query/connect failed ({_e}), retry {_attempt + 1}/4 in {_wait}s...")
                _t.sleep(_wait)
                continue
            raise

    if not rows:
        raise ValueError(f"{_table} returned no data for {wind_code} up to {end_str}")

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df[["open", "high", "low", "close", "volume"]]


def _fetch_wind(symbol: str, days: int, end_date: Optional[str]) -> tuple[list[str], dict]:
    import datetime
    from finagent.config import WIND_TABLE

    info = _normalize_symbol(symbol)
    if not info["is_ashare"]:
        raise ValueError(f"Wind WDS backend only supports A-shares, got {symbol}")

    wind_code = _wind_code(symbol)

    if end_date:
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = datetime.datetime.today()
    start_dt = end_dt - datetime.timedelta(days=int(days * 1.5))

    # winddb.ashareeodprices columns (standard Wind schema):
    #   S_INFO_WINDCODE, TRADE_DT, S_DQ_ADJOPEN, S_DQ_ADJHIGH, S_DQ_ADJLOW,
    #   S_DQ_ADJCLOSE, S_DQ_VOLUME, S_DQ_AMOUNT, OPDATE, OPMODE
    # Use full back-adjusted prices (全复权)
    sql = f"""
        SELECT TRADE_DT,
               S_DQ_ADJOPEN  AS open,
               S_DQ_ADJHIGH  AS high,
               S_DQ_ADJLOW   AS low,
               S_DQ_ADJCLOSE AS close,
               S_DQ_VOLUME   AS volume
        FROM   {WIND_TABLE}
        WHERE  S_INFO_WINDCODE = :1
          AND  TRADE_DT >= :2
          AND  TRADE_DT <= :3
        ORDER  BY TRADE_DT
    """
    start_str = start_dt.strftime("%Y%m%d")
    end_str   = end_dt.strftime("%Y%m%d")

    conn = None
    rows = []
    for _attempt in range(4):
        try:
            conn = _get_wind_connection()
            cursor = conn.cursor()
            cursor.execute(sql, [wind_code, start_str, end_str])
            rows = cursor.fetchall()
            cursor.close()
            break
        except Exception as _e:
            _wind_conn_local.conn = None
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
            if _attempt < 3:
                import time as _t
                _wait = 30 * (2 ** _attempt)
                logger.warning(f"Wind query/connect failed ({_e}), retry {_attempt + 1}/4 in {_wait}s...")
                _t.sleep(_wait)
                continue
            raise

    if not rows:
        raise ValueError(f"Wind WDS returned no data for {wind_code} ({start_str}~{end_str})")

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")
    df = df.astype(float)
    df = df.dropna(subset=["close"])

    return _df_to_result(df, days)


# ─── public API ───────────────────────────────────────────────────────────────

def _get_providers(chosen: str, is_ashare: bool) -> list[str]:
    if chosen == "wind":
        return ["wind", "akshare", "yfinance"] if is_ashare else ["yfinance"]
    elif chosen == "akshare":
        return ["akshare", "yfinance"] if is_ashare else ["yfinance"]
    return ["yfinance"]


def _fetch_df(
    symbol: str,
    days: int,
    end_date: Optional[str],
    source: Optional[str],
) -> pd.DataFrame:
    """Internal: fetch full OHLCV DataFrame from best available provider."""
    from finagent.config import DATA_SOURCE
    chosen = (source or DATA_SOURCE).lower()
    info = _normalize_symbol(symbol)
    providers = _get_providers(chosen, info["is_ashare"])

    # Each backend returns (dates, prices) — we need the full df.
    # Re-implement per-backend to return DataFrame directly.
    last_err: Exception = ValueError("No providers available")
    for provider in providers:
        try:
            logger.debug(f"Fetching {symbol} df via {provider}")
            if provider == "wind":
                dates, prices = _fetch_wind(symbol, days, end_date)
            elif provider == "akshare":
                dates, prices = _fetch_akshare(symbol, days, end_date)
            else:
                dates, prices = _fetch_yfinance(symbol, days, end_date)
            # Rebuild minimal df from (dates, prices) — close only for non-wind
            df = pd.DataFrame({"date": dates, "close": [prices[d] for d in dates]})
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            for col in ("open", "high", "low"):
                if col not in df.columns:
                    df[col] = df["close"]
            if "volume" not in df.columns:
                df["volume"] = 0.0
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.warning(f"{provider} fetch failed for {symbol}: {e}")
            last_err = e
    raise last_err


def fetch_ohlcv_df(
    symbol: str,
    days: int = 2000,
    end_date: Optional[str] = None,
    source: Optional[str] = None,
    data_source_type: str = "stock",
) -> pd.DataFrame:
    """
    Fetch OHLCV and return a DataFrame with columns: open, high, low, close, volume.
    Index is DatetimeIndex. Uses Wind full-adj prices when DATA_SOURCE=wind.

    data_source_type:
      - "stock" (default): use ashareeodprices with adjusted prices (S_DQ_ADJ*)
      - "index":           use aindexeodprices with raw OHLCV, full available history
                           (days parameter is ignored — entire history up to end_date)
    """
    if data_source_type == "index":
        df = _fetch_wind_index_df(symbol, end_date)
        # Honor days only if it shrinks the result (mostly used for snapshot context window)
        if days and len(df) > days:
            df = df.iloc[-days:]
        return df

    from finagent.config import DATA_SOURCE
    chosen = (source or DATA_SOURCE).lower()
    info = _normalize_symbol(symbol)
    providers = _get_providers(chosen, info["is_ashare"])

    last_err: Exception = ValueError("No providers available")
    for provider in providers:
        try:
            logger.debug(f"Fetching {symbol} ohlcv_df via {provider}")
            if provider == "wind":
                import datetime
                from finagent.config import WIND_TABLE
                wind_code = _wind_code(symbol)
                if end_date:
                    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
                else:
                    end_dt = datetime.datetime.today()
                start_dt = end_dt - datetime.timedelta(days=int(days * 1.5))
                sql = f"""
                    SELECT TRADE_DT,
                           S_DQ_ADJOPEN  AS open,
                           S_DQ_ADJHIGH  AS high,
                           S_DQ_ADJLOW   AS low,
                           S_DQ_ADJCLOSE AS close,
                           S_DQ_VOLUME   AS volume
                    FROM   {WIND_TABLE}
                    WHERE  S_INFO_WINDCODE = :1
                      AND  TRADE_DT >= :2
                      AND  TRADE_DT <= :3
                    ORDER  BY TRADE_DT
                """
                conn = _get_wind_connection()
                cur = conn.cursor()
                cur.execute(sql, [wind_code, start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")])
                rows = cur.fetchall()
                cur.close()
                if not rows:
                    raise ValueError(f"Wind WDS returned no data for {wind_code}")
                df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
                df = df.set_index("date").astype(float).dropna(subset=["close"])
                if len(df) > days:
                    df = df.iloc[-days:]
                return df[["open", "high", "low", "close", "volume"]]
            else:
                # For non-Wind, re-fetch via existing functions
                if provider == "akshare":
                    dates, prices = _fetch_akshare(symbol, days, end_date)
                else:
                    dates, prices = _fetch_yfinance(symbol, days, end_date)
                df = pd.DataFrame({"date": dates, "close": [prices[d] for d in dates]})
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                df["open"] = df["high"] = df["low"] = df["close"]
                df["volume"] = 0.0
                return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.warning(f"{provider} ohlcv_df failed for {symbol}: {e}")
            last_err = e
    raise last_err


def fetch_ohlcv(
    symbol: str,
    days: int = 2000,
    end_date: Optional[str] = None,
    source: Optional[str] = None,
) -> tuple[list[str], dict]:
    """
    Fetch OHLCV and return (sorted_dates, {date: adj_close}).

    source overrides DATA_SOURCE env var for this call.
    Falls back to next provider if primary fails (wind → akshare → yfinance).
    """
    from finagent.config import DATA_SOURCE

    chosen = (source or DATA_SOURCE).lower()
    info = _normalize_symbol(symbol)

    # Build ordered provider list
    if chosen == "wind":
        providers = ["wind", "akshare", "yfinance"] if info["is_ashare"] else ["yfinance"]
    elif chosen == "akshare":
        providers = ["akshare", "yfinance"] if info["is_ashare"] else ["yfinance"]
    else:
        providers = ["yfinance"]

    last_err: Exception = ValueError("No providers available")
    for provider in providers:
        try:
            logger.debug(f"Fetching {symbol} via {provider}")
            if provider == "wind":
                return _fetch_wind(symbol, days, end_date)
            elif provider == "akshare":
                return _fetch_akshare(symbol, days, end_date)
            else:
                return _fetch_yfinance(symbol, days, end_date)
        except Exception as e:
            logger.warning(f"{provider} fetch failed for {symbol}: {e}")
            last_err = e
            continue

    raise last_err
