# gold_data_pipeline.py
# Logic unchanged – timezone handling fixed and made robust

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

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    df.index = _ensure_utc_index(df.index)
    df.columns = [c.lower() for c in df.columns]

    return df[["open", "high", "low", "close", "volume"]].dropna()


def fetch_yesterday_settlement_close() -> float:
    """
    Return yesterday's settlement close price.
    """
    df = fetch_1y_history(days=2)
    return float(df["close"].iloc[-1])


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
    tq = YahooTicker("GC=F")
    px = tq.price.get("GC=F", {})

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
    bar["high"] = snap["high"]
    bar["low"] = snap["low"]
    bar["close"] = snap["price"]
    bar["volume"] = snap["volume"]

    return bar


def get_365_with_today() -> pd.DataFrame:
    """
    Final dataset:
    - 365 complete daily candles
    - 1 synthetic 'today' candle
    """
    hist = fetch_1y_history(365)

    today_bar = merge_snapshot_into_today(
        build_incomplete_today_bar()
    )

    today_idx = pd.DatetimeIndex(
        [datetime.utcnow().date()],
        tz=UTC
    )

    today_df = pd.DataFrame([today_bar], index=today_idx)

    return pd.concat([hist, today_df])