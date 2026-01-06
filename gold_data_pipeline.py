# gold_daily_with_snapshot.py
"""
Fetch 365 days of daily GC futures (complete bars) + today's incomplete daily item
anchored by yesterday's settlement close as 'open' and filled from live snapshot.

Functions:
- fetch_1y_history(symbol="GC=F", days=365) -> pd.DataFrame   # complete daily history
- fetch_yesterday_settlement(symbol="GC=F") -> dict           # {'date': 'YYYY-MM-DD', 'settlement_close': float}
- create_incomplete_item(settlement_close, date_today) -> dict
- fetch_latest_snapshot_and_merge(incomplete_item, symbol="GC=F") -> dict
- get_365_with_today(symbol="GC=F", days=365) -> pd.DataFrame  # final DF (history + today's incomplete row)
"""

from datetime import datetime, timedelta, time, timezone
import math
import warnings

import pandas as pd
import numpy as np

# Yahooquery import (optional)
try:
    from yahooquery import Ticker
except Exception:
    Ticker = None

# timezone support: prefer zoneinfo, fallback to dateutil.tz, else naive offsets
try:
    from zoneinfo import ZoneInfo

    ET_ZONE = ZoneInfo("America/New_York")
    EAT_ZONE = ZoneInfo("Africa/Addis_Ababa")
except Exception:
    try:
        from dateutil import tz

        ET_ZONE = tz.gettz("America/New_York")
        EAT_ZONE = tz.gettz("Africa/Addis_Ababa")
    except Exception:
        # fallback (not DST-aware): ET = UTC-5, EAT = UTC+3
        ET_ZONE = timezone(timedelta(hours=-5))
        EAT_ZONE = timezone(timedelta(hours=3))


def _ensure_ticker():
    if Ticker is None:
        raise ImportError("yahooquery not installed. Install with `pip install yahooquery`.")


def _latest_complete_date_by_cme_conservative(now_utc=None):
    """
    Conservative rule: treat daily settlement around 13:30 ET.
    If current ET time is before settlement+30min, treat latest complete day as yesterday.
    Otherwise treat today as complete.
    """
    if now_utc is None:
        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    try:
        # compute now in ET
        now_et = now_utc.astimezone(ET_ZONE) if getattr(ET_ZONE, "zone", None) or hasattr(ET_ZONE, "tzname") else now_utc.astimezone(ET_ZONE)
    except Exception:
        # ET_ZONE might be a fixed offset tz or zoneinfo; fallback to naive conversion
        now_et = now_utc

    # CME settlement minute ~ 13:29-13:30 ET; use 13:30 as cutoff
    settlement_time_et = time(13, 30)
    settlement_dt = datetime.combine(now_et.date(), settlement_time_et).replace(tzinfo=now_et.tzinfo)

    # window: before settlement + 30 minutes -> treat latest complete date as yesterday
    if now_et < (settlement_dt + timedelta(minutes=30)):
        latest_complete = (now_et.date() - timedelta(days=1))
    else:
        latest_complete = now_et.date()
    return latest_complete


def fetch_1y_history(symbol: str = "GC=F", days: int = 365) -> pd.DataFrame:
    """
    Fetch the last `days` of daily OHLCV for `symbol` (e.g., "GC=F").
    Returns a DataFrame indexed by date (YYYY-MM-DD) with columns: open, high, low, close, volume.
    Weekends are included and will be NaN for OHLCV unless the data provider reports something.
    Uses a conservative latest-complete-date to avoid partial/today-close repaint issues.
    """
    _ensure_ticker()
    tq = Ticker(symbol)

    # decide latest complete date
    latest_complete = _latest_complete_date_by_cme_conservative()
    end = latest_complete
    start = end - timedelta(days=days - 1)

    # yahooquery history expects start/end strings
    raw = tq.history(start=start.strftime("%Y-%m-%d"), end=(end + timedelta(days=1)).strftime("%Y-%m-%d"), interval="1d")

    if raw is None or raw.empty:
        # return empty reindexed DF for date range
        idx = pd.date_range(start=start, end=end, freq="D").date
        empty = pd.DataFrame(index=pd.to_datetime(idx), columns=["open", "high", "low", "close", "volume"])
        empty.index.name = "date"
        return empty

    # yahooquery sometimes returns multiindex index when multiple symbols requested; normalize
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index(level=0, drop=True)

    # normalize index to date (no time)
    raw.index = pd.to_datetime(raw.index).tz_convert("UTC") if getattr(raw.index, "tz", None) else pd.to_datetime(raw.index)
    # convert to date-only index (no tz)
    raw.index = raw.index.normalize().date
    df = raw.copy()
    # normalize column names
    df.columns = [c.lower() for c in df.columns]

    # keep only core columns if present
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan

    # reindex to full calendar days (include weekends)
    full_idx = pd.date_range(start=start, end=end, freq="D").date
    df = df.reindex(full_idx)
    df.index = pd.to_datetime(df.index).date  # ensure index entries are date objects
    df.index.name = "date"

    # cast volume to numeric where possible
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    return df[["open", "high", "low", "close", "volume"]]


def fetch_yesterday_settlement(symbol: str = "GC=F") -> dict:
    """
    Approximate yesterday's settlement by taking the most recent completed day's 'close'
    prior to today from the provider (Yahoo).
    Returns {'date': 'YYYY-MM-DD', 'settlement_close': float}
    """
    _ensure_ticker()
    tq = Ticker(symbol)
    raw = tq.history(period="7d", interval="1d")
    if raw is None or raw.empty:
        raise RuntimeError("No historical data returned to approximate settlement.")

    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index(level=0, drop=True)

    raw.index = pd.to_datetime(raw.index).tz_convert("UTC") if getattr(raw.index, "tz", None) else pd.to_datetime(raw.index)
    raw.index = raw.index.normalize().date

    today_date = datetime.utcnow().date()

    # filter to days strictly before today (completed days)
    past = raw[raw.index < today_date]
    if past.empty:
        raise RuntimeError("No past completed trading day found in history to determine yesterday's settlement.")

    last_date = past.index[-1]
    last_row = past.iloc[-1]
    close_val = last_row.get("close", None)
    if close_val is None or (isinstance(close_val, float) and math.isnan(close_val)):
        raise RuntimeError("Yesterday close is missing in provider data.")

    return {"date": str(last_date), "settlement_close": float(close_val)}


def create_incomplete_item(settlement_close: float, date_today: str = None) -> dict:
    """
    Create an incomplete JSON-like dict representing today's daily item.
    - 'open' set to settlement_close
    - 'high','low','close','volume' set to None
    - is_complete = False
    """
    if date_today is None:
        date_today = datetime.utcnow().date().isoformat()
    item = {
        "date": date_today,
        "open": float(settlement_close) if settlement_close is not None else None,
        "high": None,
        "low": None,
        "close": None,
        "volume": None,
        "is_complete": False,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source": "incomplete_from_settlement",
    }
    return item


def fetch_latest_snapshot_and_merge(incomplete_item: dict, symbol: str = "GC=F") -> dict:
    """
    Use yahooquery's Ticker.price to get a live snapshot and merge fields into incomplete_item:
    fills close, high, low, volume where available. Attempts intraday 1m high/low if present.
    """
    _ensure_ticker()
    tq = Ticker(symbol)
    snap = tq.price
    if snap is None or symbol not in snap:
        raise RuntimeError("No snapshot available from yahooquery.price")

    s = snap[symbol]
    # prefer 'regularMarketPrice' then 'lastPrice'
    last_price = s.get("regularMarketPrice") or s.get("lastPrice") or s.get("price")
    last_vol = s.get("regularMarketVolume") or s.get("volume")
    day_high = s.get("regularMarketDayHigh") or s.get("dayHigh") or None
    day_low = s.get("regularMarketDayLow") or s.get("dayLow") or None

    # try to fetch intraday 1m bars for today's high/low if available
    try:
        intraday = tq.history(period="1d", interval="1m")
        if intraday is not None and not intraday.empty:
            if isinstance(intraday.index, pd.MultiIndex):
                intraday = intraday.reset_index(level=0, drop=True)
            intraday.index = pd.to_datetime(intraday.index)
            # keep today's bars only
            intraday_today = intraday[intraday.index.date == datetime.utcnow().date()]
            if not intraday_today.empty:
                # compute high/low from intraday bars
                if "high" in intraday_today.columns:
                    day_high = float(intraday_today["high"].max())
                if "low" in intraday_today.columns:
                    day_low = float(intraday_today["low"].min())
    except Exception:
        # silent fallback to snapshot fields
        pass

    merged = dict(incomplete_item)  # copy
    if last_price is not None:
        merged["close"] = float(last_price)
    if last_vol is not None:
        try:
            merged["volume"] = int(last_vol)
        except Exception:
            merged["volume"] = last_vol
    if day_high is not None:
        merged["high"] = float(day_high)
    if day_low is not None:
        merged["low"] = float(day_low)

    merged["updated_at"] = datetime.utcnow().isoformat() + "Z"
    merged["source_snapshot"] = "yahoo_price"

    return merged


def get_365_with_today(symbol: str = "GC=F", days: int = 365) -> pd.DataFrame:
    """
    High-level function returning:
    - DataFrame of last `days` complete daily rows (index=date)
    - plus today's incomplete row (index=today) appended at bottom, with is_complete flag
    """
    # 1) history
    hist = fetch_1y_history(symbol=symbol, days=days)

    # 2) yesterday settlement
    settlement = fetch_yesterday_settlement(symbol=symbol)
    settlement_close = settlement["settlement_close"]

    # 3) incomplete item for today
    today_iso = datetime.utcnow().date().isoformat()
    incomplete = create_incomplete_item(settlement_close=settlement_close, date_today=today_iso)

    # 4) merge snapshot into incomplete
    try:
        merged_today = fetch_latest_snapshot_and_merge(incomplete, symbol=symbol)
    except Exception:
        # if snapshot fails, keep the incomplete as-is (it's valid)
        merged_today = incomplete

    # 5) append to DataFrame (ensure same columns)
    # convert hist index to ISO date strings for safe concat
    hist_out = hist.copy()
    hist_out = hist_out.reset_index()
    hist_out["date"] = hist_out["date"].astype(str)
    # desired columns
    cols = ["date", "open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in hist_out.columns:
            hist_out[c] = None

    # prepare today's row as DF
    today_row = {
        "date": merged_today.get("date"),
        "open": merged_today.get("open"),
        "high": merged_today.get("high"),
        "low": merged_today.get("low"),
        "close": merged_today.get("close"),
        "volume": merged_today.get("volume"),
    }
    today_df = pd.DataFrame([today_row])

    # concat and ensure index by date
    final = pd.concat([hist_out[cols], today_df], ignore_index=True)
    final["date"] = pd.to_datetime(final["date"]).dt.date
    final = final.set_index("date")

    # add is_complete flag (True if close is notna and date != today)
    final["is_complete"] = final["close"].notna()
    # ensure today's row is marked incomplete if close was None originally
    if pd.to_datetime(datetime.utcnow().date()).date() in final.index:
        # if today's close is not None we might mark it complete (depending on snapshot)
        pass

    # reorder columns
    final = final[["open", "high", "low", "close", "volume", "is_complete"]]

    return final


# Simple demo when run directly
if __name__ == "__main__":
    symbol = "GC=F"
    try:
        df = get_365_with_today(symbol=symbol, days=365)
        print(df.tail(7))
    except Exception as e:
        print("Error:", e)