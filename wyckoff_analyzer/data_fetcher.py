"""
Data fetcher for OHLCV data.
Supports: US stocks (AAPL), indices (^SPX), A-shares (600519.SS / 000001.SZ), HK (0700.HK)
"""
import pandas as pd
try:
    import yfinance as yf
except Exception:
    yf = None  # Will fail at call time if used; patch replaces fetch_ohlcv anyway


def _normalize_code(code: str) -> str:
    """Auto-append exchange suffix for A-share codes."""
    code = code.strip()
    # Already has suffix
    if '.' in code or '^' in code:
        return code
    # 6-digit Shanghai codes: 6xxxxx, 5xxxxx
    if len(code) == 6 and code[0] in ('6', '5'):
        return code + '.SS'
    # 6-digit Shenzhen codes: 0xxxxx, 1xxxxx, 2xxxxx, 3xxxxx
    if len(code) == 6 and code[0] in ('0', '1', '2', '3'):
        return code + '.SZ'
    return code


def fetch_ohlcv(code: str, days: int = 500, end_date: str = None) -> pd.DataFrame:
    """
    Fetch OHLCV daily data.

    Args:
      code: ticker symbol
      days: lookback days
      end_date: optional end date 'YYYY-MM-DD' (defaults to today)

    Returns DataFrame with columns: open, high, low, close, volume
    Index: DatetimeIndex
    """
    import datetime
    ticker = _normalize_code(code)

    # Always use start/end date range — period="Nd" is unreliable for some A-share tickers
    if end_date:
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_dt = datetime.datetime.today()
    start_dt = end_dt - datetime.timedelta(days=int(days * 1.5))  # buffer for weekends/holidays
    df = yf.download(ticker, start=start_dt.strftime('%Y-%m-%d'),
                     end=(end_dt + datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
                     auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol.")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    df.index.name = 'date'

    # Trim to requested number of days
    if len(df) > days:
        df = df.iloc[-days:]

    return df
