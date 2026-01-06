# gold_data_pipeline.py
# Fixed: Consistent timezone handling to prevent tz-aware/tz-naive mixing

from datetime import datetime, timedelta, time, timezone
import logging
import pandas as pd
import numpy as np

try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

try:
    from dateutil import tz
except Exception:
    tz = None

logger = logging.getLogger(__name__)

UTC = timezone.utc
ET = tz.gettz("America/New_York") if tz else timezone(timedelta(hours=-5))


# ---------------------------------------------------------------------
# Internal helpers (timezone-safe)
# ---------------------------------------------------------------------

def _ensure_utc_datetime(dt) -> datetime:
    """
    Ensure datetime is UTC-aware.
    """
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    return dt


def _normalize_to_date_index(idx: pd.DatetimeIndex) -> pd.Index:
    """
    Convert DatetimeIndex to date-only Index (tz-naive dates).
    """
    if hasattr(idx, 'tz') and idx.tz is not None:
        # Convert to UTC first, then extract date
        return pd.Index(idx.tz_convert('UTC').normalize().date, name='date')
    else:
        # Already tz-naive
        return pd.Index(pd.to_datetime(idx).normalize().date, name='date')


def _latest_complete_cme_date(now_utc: datetime) -> datetime.date:
    """
    Conservative CME GC settlement logic.
    Settlement ~13:30 ET. Use yesterday until +30 min buffer passes.
    """
    try:
        now_et = now_utc.astimezone(ET)
        settlement = datetime.combine(
            now_et.date(),
            time(13, 30),
            tzinfo=ET
        )
        if now_et < settlement + timedelta(minutes=30):
            return now_et.date() - timedelta(days=1)
        return now_et.date()
    except Exception as e:
        logger.warning(f"Timezone conversion failed: {e}, using UTC-1 day")
        return (now_utc - timedelta(days=1)).date()


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def fetch_1y_history(days: int = 365) -> pd.DataFrame:
    """
    Fetch historical daily gold prices.
    Returns DataFrame with date-only index (not datetime).
    """
    if YahooTicker is None:
        raise ImportError("yahooquery is required")

    now_utc = datetime.now(UTC)
    last_complete = _latest_complete_cme_date(now_utc)

    # Use date objects for start/end to avoid timezone issues
    end_date = last_complete
    start_date = end_date - timedelta(days=days)

    tq = YahooTicker("GC=F")
    
    try:
        df = tq.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d"
        )
    except Exception as e:
        logger.error(f"Yahoo query failed: {e}")
        df = None

    if df is None or df.empty:
        # Return empty DataFrame with proper structure
        idx = pd.date_range(start=start_date, end=end_date, freq="D")
        df_empty = pd.DataFrame(
            index=pd.Index(idx.date, name="date"),
            columns=["open", "high", "low", "close", "volume"]
        )
        return df_empty

    # Handle MultiIndex (symbol, date)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    # Normalize index to date-only (removes timezone info)
    df.index = _normalize_to_date_index(df.index)

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan

    # Create full date range for reindexing (tz-naive dates)
    full_idx = pd.date_range(start=start_date, end=end_date, freq="D")
    full_idx_dates = pd.Index(full_idx.date, name="date")
    
    # Reindex to fill missing dates
    df = df.reindex(full_idx_dates)
    df.index.name = "date"

    # Ensure volume is numeric
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    return df[["open", "high", "low", "close", "volume"]]


def fetch_yesterday_settlement(symbol: str = "GC=F") -> float:
    """
    Fetch most recent settlement close price.
    """
    if YahooTicker is None:
        raise ImportError("yahooquery is required")

    ticker = YahooTicker(symbol)
    
    try:
        hist = ticker.history(period="7d", interval="1d")
    except Exception as e:
        raise RuntimeError(f"Yahoo history query failed: {e}")

    if hist is None or hist.empty:
        raise RuntimeError("No daily history returned from Yahoo")

    # Handle MultiIndex
    if isinstance(hist.index, pd.MultiIndex):
        hist = hist.reset_index(level=0, drop=True)

    # Normalize to date index
    hist.index = _normalize_to_date_index(hist.index)

    if "close" not in hist.columns:
        raise RuntimeError("No 'close' column in Yahoo history")

    # Get most recent valid close
    hist_valid = hist[hist["close"].notna()]
    if hist_valid.empty:
        raise RuntimeError("No valid settlement close found in Yahoo data")

    return float(hist_valid["close"].iloc[-1])


# Alias for compatibility
fetch_yesterday_settlement_close = fetch_yesterday_settlement


def build_incomplete_today_bar() -> dict:
    """
    Build incomplete bar for today using yesterday's close as open.
    """
    try:
        open_px = fetch_yesterday_settlement_close()
    except Exception as e:
        logger.warning(f"Could not fetch yesterday settlement: {e}")
        open_px = np.nan
    
    return {
        "open": open_px,
        "high": np.nan,
        "low": np.nan,
        "close": np.nan,
        "volume": np.nan,
    }


def fetch_snapshot() -> dict:
    """
    Fetch current market snapshot.
    """
    if YahooTicker is None:
        raise ImportError("yahooquery is required")

    tq = YahooTicker("GC=F")
    try:
        px = tq.price.get("GC=F", {}) or {}
    except Exception as e:
        logger.warning(f"Price fetch failed: {e}")
        px = {}

    return {
        "price": px.get("regularMarketPrice"),
        "high": px.get("regularMarketDayHigh"),
        "low": px.get("regularMarketDayLow"),
        "volume": px.get("regularMarketVolume"),
    }


def merge_snapshot_into_today(incomplete_bar: dict) -> dict:
    """
    Merge live snapshot data into incomplete bar.
    """
    try:
        snap = fetch_snapshot()
    except Exception as e:
        logger.warning(f"Snapshot fetch failed: {e}")
        snap = {}
    
    bar = dict(incomplete_bar)

    if snap.get("high") is not None:
        bar["high"] = snap["high"]
    if snap.get("low") is not None:
        bar["low"] = snap["low"]
    if snap.get("price") is not None:
        bar["close"] = snap["price"]
    if snap.get("volume") is not None:
        bar["volume"] = snap["volume"]

    return bar


def get_365_with_today() -> pd.DataFrame:
    """
    Get 365 days of history plus incomplete today bar.
    """
    hist = fetch_1y_history(365)
    today_bar = merge_snapshot_into_today(build_incomplete_today_bar())

    # Use date object for today (not datetime)
    today_date = datetime.now(UTC).date()
    today_df = pd.DataFrame([today_bar], index=pd.Index([today_date], name="date"))

    # Concatenate (both have date-only indices now)
    final = pd.concat([hist, today_df], axis=0)

    # Ensure all required columns exist
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in final.columns:
            final[c] = np.nan

    # Mark complete bars
    final["is_complete"] = final["close"].notna()
    
    return final[["open", "high", "low", "close", "volume", "is_complete"]]
