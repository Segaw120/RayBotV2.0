# gold_data_pipeline.py
# Logic unchanged – timezone handling fixed and made robust
# Minor fix: provide fetch_yesterday_settlement and alias fetch_yesterday_settlement_close
# so callers using either name work.

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
ET = tz.gettz("America/New_York") if tz else None


# ---------------------------------------------------------------------
# Internal helpers (timezone-safe)
# ---------------------------------------------------------------------

def _ensure_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Ensure DatetimeIndex is UTC-aware.
    Handles tz-naive and tz-aware safely.
    """
    idx = pd.to_datetime(idx)

    if idx.tz is None:
        return idx.tz_localize(UTC)
    else:
        return idx.tz_convert(UTC)


def _latest_complete_cme_date(now_utc: datetime) -> datetime.date:
    """
    Conservative CME GC settlement logic.
    Settlement ~13:30 ET. Use yesterday until +30 min buffer passes.
    """
    if ET:
        now_et = now_utc.astimezone(ET)
        settlement = datetime.combine(
            now_et.date(),
            time(13, 30),
            tzinfo=ET
        )
        if now_et < settlement + timedelta(minutes=30):
            return now_et.date() - timedelta(days=1)
        return now_et.date()
    else:
        return (now_utc - timedelta(days=1)).date()


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def fetch_1y_history(days: int = 365) -> pd.DataFrame:
    """
    Fetch last `days` COMPLETE daily GC=F futures candles.
    Returns DataFrame indexed by date with columns open, high, low, close, volume.
    """
    if YahooTicker is None:
        raise ImportError("yahooquery is required")

    now_utc = datetime.utcnow().replace(tzinfo=UTC)
    last_complete = _latest_complete_cme_date(now_utc)

    end = datetime.combine(last_complete, time(0, 0), tzinfo=UTC)
    start = end - timedelta(days=days)

    tq = YahooTicker("GC=F")
    df = tq.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d"
    )

    if df is None or df.empty:
        idx = pd.date_range(start=start, end=end, freq="D")
        df_empty = pd.DataFrame(index=idx, columns=["open", "high", "low", "close", "volume"])
        df_empty.index.name = "date"
        return df_empty

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    df.index = _ensure_utc_index(df.index)
    # convert index to date-only for reindexing consistency
    dates = df.index.normalize().date
    df.index = pd.Index(dates)
    df.index.name = "date"

    df.columns = [c.lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan

    full_idx = pd.date_range(start=start, end=end, freq="D").date
    df = df.reindex(full_idx)
    df.index.name = "date"

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    return df[["open", "high", "low", "close", "volume"]]


def fetch_yesterday_settlement(symbol: str = "GC=F") -> float:
    """
    Fetch last available daily settlement close from Yahoo.
    Falls back gracefully if today's settlement is not yet published.
    Returns a float close price.
    """
    if YahooTicker is None:
        raise ImportError("yahooquery is required")

    ticker = YahooTicker(symbol)
    hist = ticker.history(period="7d", interval="1d")

    if hist is None or hist.empty:
        raise RuntimeError("No daily history returned from Yahoo")

    if isinstance(hist.index, pd.MultiIndex):
        hist = hist.reset_index(level=0, drop=True)

    # Normalize index to UTC-aware then to date
    try:
        idx = _ensure_utc_index(hist.index)
        dates = idx.normalize().date
        hist.index = pd.Index(dates)
        hist.index.name = "date"
    except Exception:
        # fallback: coerce index to dates
        hist.index = pd.to_datetime(hist.index).date
        hist.index.name = "date"

    # Keep only rows with a valid 'close'
    if "close" not in hist.columns:
        raise RuntimeError("No 'close' column in Yahoo history")
    hist_valid = hist[hist["close"].notna()]

    if hist_valid.empty:
        raise RuntimeError("No valid settlement close found in Yahoo data")

    # Last available completed daily close
    settlement_close = float(hist_valid["close"].iloc[-1])
    return settlement_close


# Alias to maintain compatibility with callers expecting fetch_yesterday_settlement_close
fetch_yesterday_settlement_close = fetch_yesterday_settlement


def build_incomplete_today_bar() -> dict:
    """
    Create incomplete daily bar:
    open = yesterday close
    others empty
    """
    open_px = fetch_yesterday_settlement_close()
    return {
        "open": open_px,
        "high": np.nan,
        "low": np.nan,
        "close": np.nan,
        "volume": np.nan,
    }


def fetch_snapshot() -> dict:
    """
    Fetch latest snapshot from Yahoo.
    """
    if YahooTicker is None:
        raise ImportError("yahooquery is required")

    tq = YahooTicker("GC=F")
    px = {}
    try:
        px = tq.price.get("GC=F", {}) or {}
    except Exception:
        # best-effort access
        try:
            p = tq.price
            px = p.get("GC=F", {}) if isinstance(p, dict) else {}
        except Exception:
            px = {}

    return {
        "price": px.get("regularMarketPrice"),
        "high": px.get("regularMarketDayHigh"),
        "low": px.get("regularMarketDayLow"),
        "volume": px.get("regularMarketVolume"),
    }


def merge_snapshot_into_today(incomplete_bar: dict) -> dict:
    """
    Fill incomplete daily bar with snapshot values.
    """
    snap = fetch_snapshot()

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
    Final dataset:
    - 365 complete daily candles
    - 1 synthetic 'today' candle
    """
    hist = fetch_1y_history(365)

    today_bar = merge_snapshot_into_today(build_incomplete_today_bar())

    # ensure hist index is date index
    hist_out = hist.copy()
    if hist_out.index.dtype == object or not hasattr(hist_out.index, "date"):
        try:
            hist_out = hist_out.reset_index()
            hist_out["date"] = pd.to_datetime(hist_out["date"]).dt.date
            hist_out = hist_out.set_index("date")
        except Exception:
            pass

    today_date = datetime.utcnow().date()
    today_df = pd.DataFrame([today_bar], index=pd.Index([today_date], name="date"))

    final = pd.concat([hist_out, today_df], axis=0)
    # ensure columns exist
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in final.columns:
            final[c] = np.nan

    final["is_complete"] = final["close"].notna()
    final = final[["open", "high", "low", "close", "volume", "is_complete"]]

    return final