""" gold_data_pipeline.py

A modular Python module to:

Fetch 1 year (365 days) of gold futures history (GC=F) including weekends (filled as NaNs)

Fetch yesterday's settlement (approx) daily close

Create an incomplete daily JSON item for the latest date with yesterday's settlement as the open and other fields empty

Fetch latest live snapshot and merge it into the incomplete JSON (fill close/high/low/volume)

Upsert (delete existing and insert fresh) the 1-year history + latest daily JSON item into a Supabase table

Send a Telegram log message after upsert


Design goals:

Modular functions that can be composed by a polling service (e.g., run every 30 minutes before settlement)

Use environment variables for secrets (Supabase and Telegram). Do NOT hardcode secrets in code.

Primary data source: yahooquery. Fallbacks / notes included in docstrings.


Requirements:

pip install yahooquery pandas requests python-dotenv


Usage pattern (simplified): from gold_data_pipeline import run_update run_update()

This file is intended to be dropped into your pipeline and called by a scheduler (cron, systemd timer, or a short-running service that polls).

Note on CME settlement accuracy:

Yahoo-derived daily close is an approximation. For strict CME settlement matching you will need an official CME feed. This module uses Yahoo as an accessible source.


"""

from datetime import datetime, timedelta, timezone import os import time import json import math from typing import Optional, Dict, Any, List

import pandas as pd import requests from dateutil import tz

try: from yahooquery import Ticker except Exception: Ticker = None

-------------------------------

Configuration / Environment

-------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")  # e.g. https://xyzcompany.supabase.co SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Service role key or API key with write permissions SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "gold_futures")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

Default symbol for gold futures on Yahoo

DEFAULT_SYMBOL = os.getenv("GOLD_SYMBOL", "GC=F")

Exchange timezone for GC futures (CME / New York) — used for some computations

ET_ZONE = tz.gettz("America/New_York") UTC_ZONE = timezone.utc EAT_ZONE = tz.gettz("Africa/Addis_Ababa")

-------------------------------

Helper utilities

-------------------------------

def send_telegram_message(bot_token: str, chat_id: str, text: str) -> Dict[str, Any]: """Send a Telegram message and return the response JSON.

Expects bot token and chat id to be provided. The function will raise on HTTP error.
"""
if not bot_token or not chat_id:
    raise ValueError("Telegram bot token and chat id must be provided via environment variables.")

url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
payload = {"chat_id": chat_id, "text": text}
resp = requests.post(url, json=payload, timeout=10)
resp.raise_for_status()
return resp.json()

def to_utc(dt: datetime) -> datetime: if dt.tzinfo is None: return dt.replace(tzinfo=timezone.utc) return dt.astimezone(timezone.utc)

def ensure_ticker(): if Ticker is None: raise ImportError("yahooquery not installed or could not be imported. Install with pip install yahooquery.")

-------------------------------

Data fetching functions

-------------------------------

def fetch_1y_history(symbol: str = DEFAULT_SYMBOL, days: int = 365) -> pd.DataFrame: """Fetch last days days of daily OHLCV data (1d) from Yahoo via yahooquery.

Returns a DataFrame indexed by UTC date (date only) with columns: open, high, low, close, volume
Weekends and missing dates will be included in the index and contain NaNs.
"""
ensure_ticker()
t = Ticker(symbol)

# yahooquery can fetch by period or start/end; use end as today and start days back
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=days)

# history returns a DataFrame with a DatetimeIndex (UTC). Use period param as a safe fallback.
try:
    raw = t.history(start=start_date.strftime("%Y-%m-%d"), end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"), interval="1d")
except Exception:
    # fallback using period
    raw = t.history(period=f"{days}d", interval="1d")

# yahooquery returns a multiindex if multiple symbols requested; ensure single-symbol DataFrame
if isinstance(raw, pd.DataFrame) and ("symbol" in raw.columns):
    raw = raw[raw["symbol"] == symbol].copy()
    raw = raw.drop(columns=[c for c in ["symbol"] if c in raw.columns])

if raw is None or raw.empty:
    # return an empty DataFrame with the dates as index
    idx = pd.date_range(start=start_date, end=end_date, freq="D")
    return pd.DataFrame(index=idx, columns=["open", "high", "low", "close", "volume"]).astype(object)

# Ensure index is UTC date only
raw = raw.copy()
raw.index = pd.to_datetime(raw.index).tz_convert("UTC") if raw.index.tz is not None else pd.to_datetime(raw.index).tz_localize("UTC")
raw.index = raw.index.tz_convert("UTC")
raw.index = raw.index.normalize()  # keep date only at midnight UTC

# Select common OHLCV columns if present
cols = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
df = raw[cols].copy()

# Reindex to include weekends
full_idx = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")
df = df.reindex(full_idx)
df.index.name = "date_utc"

return df

def fetch_yesterday_settlement(symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]: """Fetch yesterday's daily close as an approximation of settlement.

Returns a dict with keys: date (ISO), settlement_close (float), source (string)

Note: For true CME official settlement you need direct CME data. This function uses Yahoo daily close as approximation.
"""
ensure_ticker()
t = Ticker(symbol)

# We'll request the last 5 days and take the most recent completed day (not today)
raw = t.history(period="5d", interval="1d")
if raw is None or raw.empty:
    raise RuntimeError("No historical data returned for settlement approximation")

# Handle multiindex/symbol columns
if isinstance(raw, pd.DataFrame) and "symbol" in raw.columns:
    raw = raw[raw["symbol"] == symbol].copy()

df = raw.copy()
df.index = pd.to_datetime(df.index)
# Normalize to date only in exchange timezone (approx via ET)
df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index
df.index = df.index.tz_convert("UTC")
df.index = df.index.normalize()

today_utc = datetime.utcnow().date()
# filter to dates strictly before today
df_past = df[df.index.date < today_utc]
if df_past.empty:
    raise RuntimeError("Could not find a past trading day to use as 'yesterday' settlement")

last_row = df_past.iloc[-1]
last_date = df_past.index[-1].date()
settlement_close = float(last_row.get("close", math.nan))

return {"date": last_date.isoformat(), "settlement_close": settlement_close, "source": "yahoo_daily_close_approx"}

def create_incomplete_item_df(settlement_close: float, date_today: Optional[datetime] = None) -> Dict[str, Any]: """Create a JSON-like dict representing an incomplete daily item for today.

The 'open' field is set to yesterday's settlement_close. The rest of the fields are None/empty.
"""
if date_today is None:
    date_today = datetime.utcnow().date()
else:
    date_today = date_today if isinstance(date_today, (datetime,)) else date_today
    try:
        date_today = date_today.date()
    except Exception:
        pass

item = {
    "date": date_today.isoformat() if hasattr(date_today, "isoformat") else str(date_today),
    "open": settlement_close,
    "high": None,
    "low": None,
    "close": None,
    "volume": None,
    "is_complete": False,
    "source": "incomplete_from_settlement",
    "created_at": datetime.utcnow().isoformat() + "Z",
}
return item

def fetch_latest_snapshot_and_merge(incomplete_item: Dict[str, Any], symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]: """Fetch latest snapshot via yahooquery Ticker.price and merge into incomplete_item.

Fills close, volume, high, low fields where possible. If intraday 1m data is available we use today's intraday high/low.
"""
ensure_ticker()
t = Ticker(symbol)

snapshot = t.price
if snapshot is None or symbol not in snapshot:
    raise RuntimeError("No snapshot available from yahooquery")

s = snapshot[symbol]
# prefer regularMarketPrice
last_price = s.get("regularMarketPrice") or s.get("lastPrice") or s.get("price")
last_volume = s.get("regularMarketVolume") or s.get("volume") or None
day_high = s.get("regularMarketDayHigh")
day_low = s.get("regularMarketDayLow")

# Try to get intraday 1m today high / low if available for better accuracy
try:
    intraday = t.history(interval="1m", period="1d")
    if isinstance(intraday, pd.DataFrame) and not intraday.empty:
        # normalize index
        intraday.index = pd.to_datetime(intraday.index)
        intraday.index = intraday.index.tz_localize("UTC") if intraday.index.tz is None else intraday.index
        intraday_today = intraday[~intraday.index.normalize().duplicated(keep='first')]
        # in practice the intraday df will contain today's per-minute bars
        if not intraday.empty:
            today_idx = intraday.index.normalize()[-1]
            # compute today's high/low from intraday bars
            try:
                today_bars = intraday.groupby(intraday.index.normalize()).last().iloc[-1:]
            except Exception:
                today_bars = intraday
            if not today_bars.empty:
                day_high = float(intraday["high"].max()) if "high" in intraday.columns else day_high
                day_low = float(intraday["low"].min()) if "low" in intraday.columns else day_low
except Exception:
    # intraday may not be available or can raise errors; we will fallback
    pass

# Merge into incomplete item
merged = incomplete_item.copy()

if last_price is not None:
    merged["close"] = float(last_price)
if last_volume is not None:
    try:
        merged["volume"] = int(last_volume)
    except Exception:
        merged["volume"] = last_volume
if day_high is not None:
    try:
        merged["high"] = float(day_high)
    except Exception:
        merged["high"] = day_high
if day_low is not None:
    try:
        merged["low"] = float(day_low)
    except Exception:
        merged["low"] = day_low

merged["updated_at"] = datetime.utcnow().isoformat() + "Z"
merged["source_snapshot"] = "yahoo_price"

return merged

-------------------------------

Supabase integration

-------------------------------

def supabase_delete_existing(symbol_col: str = "symbol", symbol_value: str = DEFAULT_SYMBOL, table: str = SUPABASE_TABLE, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None) -> Dict[str, Any]: """Delete existing rows for the symbol in the supabase table. Returns the response JSON or raises on error.

NOTE: Deleting all rows and re-inserting ensures a clean upsert and avoids complex on_conflict handling.
"""
supabase_url = supabase_url or SUPABASE_URL
supabase_key = supabase_key or SUPABASE_KEY

if not supabase_url or not supabase_key:
    raise ValueError("Supabase URL and KEY must be provided via environment variables")

# Build the delete URL; Supabase REST allows delete with filter: DELETE /<table>?symbol=eq.<value>
delete_url = f"{supabase_url}/rest/v1/{table}?{symbol_col}=eq.{symbol_value}"

headers = {"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"}
r = requests.delete(delete_url, headers=headers, timeout=30)
if not r.ok:
    raise RuntimeError(f"Supabase delete failed: {r.status_code} {r.text}")
try:
    return r.json()
except Exception:
    return {"status_code": r.status_code, "text": r.text}

def supabase_insert_rows(rows: List[Dict[str, Any]], table: str = SUPABASE_TABLE, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None) -> Dict[str, Any]: """Insert rows (list of dicts) into Supabase table using REST API. Returns response JSON or raises.

If rows are many, this function sends them in batches of 500.
"""
supabase_url = supabase_url or SUPABASE_URL
supabase_key = supabase_key or SUPABASE_KEY

if not supabase_url or not supabase_key:
    raise ValueError("Supabase URL and KEY must be provided via environment variables")

headers = {
    "apikey": supabase_key,
    "Authorization": f"Bearer {supabase_key}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

insert_url = f"{supabase_url}/rest/v1/{table}"

responses = []
chunk_size = 300
for i in range(0, len(rows), chunk_size):
    chunk = rows[i:i + chunk_size]
    r = requests.post(insert_url, headers=headers, data=json.dumps(chunk), timeout=60)
    if not r.ok:
        raise RuntimeError(f"Supabase insert failed: {r.status_code} {r.text}")
    try:
        responses.append(r.json())
    except Exception:
        responses.append({"status_code": r.status_code, "text": r.text})
    # small pause to avoid hammering
    time.sleep(0.2)

return {"inserted_chunks": len(responses)}

-------------------------------

Orchestration

-------------------------------

def prepare_rows_for_supabase(df_history: pd.DataFrame, latest_item: Dict[str, Any], symbol: str = DEFAULT_SYMBOL) -> List[Dict[str, Any]]: """Convert DataFrame history + latest daily item into a list of dict rows ready for Supabase insertion.

Each row will have at least: date (ISO), symbol, open, high, low, close, volume, is_complete, source
"""
rows: List[Dict[str, Any]] = []

# history rows
for idx, row in df_history.iterrows():
    date_iso = pd.to_datetime(idx).date().isoformat()
    r = {
        "date": date_iso,
        "symbol": symbol,
        "open": None if pd.isna(row.get("open")) else float(row.get("open")),
        "high": None if pd.isna(row.get("high")) else float(row.get("high")),
        "low": None if pd.isna(row.get("low")) else float(row.get("low")),
        "close": None if pd.isna(row.get("close")) else float(row.get("close")),
        "volume": None if pd.isna(row.get("volume")) else int(row.get("volume")),
        "is_complete": False if pd.isna(row.get("close")) else True,
        "source": "yahoo_history",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    rows.append(r)

# latest_item (today)
latest_row = {
    "date": latest_item.get("date"),
    "symbol": symbol,
    "open": latest_item.get("open"),
    "high": latest_item.get("high"),
    "low": latest_item.get("low"),
    "close": latest_item.get("close"),
    "volume": latest_item.get("volume"),
    "is_complete": latest_item.get("is_complete", False),
    "source": latest_item.get("source_snapshot", latest_item.get("source")),
    "created_at": latest_item.get("created_at", datetime.utcnow().isoformat() + "Z"),
    "updated_at": latest_item.get("updated_at", None),
}
rows.append(latest_row)

return rows

def run_update(symbol: str = DEFAULT_SYMBOL, supabase_table: str = SUPABASE_TABLE, notify_telegram: bool = True): """High-level orchestration function that performs the full flow described in the prompt.

Steps:
1. Fetch 365-day history (includes weekends via reindex)
2. Fetch yesterday settlement (approx via Yahoo)
3. Create incomplete item using settlement as 'open'
4. Fetch snapshot and merge to fill today's fields
5. Delete existing rows for the symbol in Supabase table and insert the history + latest item
6. Send a Telegram log message with the result
"""
# Fetch 1-year history
df_history = fetch_1y_history(symbol=symbol, days=365)

# Fetch yesterday settlement
settlement = fetch_yesterday_settlement(symbol=symbol)
settlement_close = settlement["settlement_close"]

# Create incomplete item for today
incomplete = create_incomplete_item_df(settlement_close=settlement_close)

# Merge snapshot
merged = fetch_latest_snapshot_and_merge(incomplete, symbol=symbol)

# Prepare rows
rows = prepare_rows_for_supabase(df_history, merged, symbol=symbol)

# Supabase operations: delete existing rows for this symbol and insert new
if SUPABASE_URL is None or SUPABASE_KEY is None:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set as environment variables before running run_update")

delete_resp = supabase_delete_existing(symbol_col="symbol", symbol_value=symbol, table=supabase_table)
insert_resp = supabase_insert_rows(rows, table=supabase_table)

msg = f"Supabase update completed for {symbol} — deleted existing rows; inserted {len(rows)} rows (history + latest)."

# Send Telegram notification if desired
if notify_telegram:
    try:
        send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)
    except Exception as e:
        # do not fail the update if telegram fails; log to stdout instead
        print("Telegram send failed:", str(e))

return {"deleted": delete_resp, "inserted": insert_resp, "rows": len(rows)}

If the module is run directly, perform a single update (useful for manual testing)

if name == "main": import argparse

parser = argparse.ArgumentParser(description="Gold data pipeline updater")
parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Symbol (default: GC=F)")
parser.add_argument("--table", default=SUPABASE_TABLE, help="Supabase table name")
parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram notifications")
args = parser.parse_args()

result = run_update(symbol=args.symbol, supabase_table=args.table, notify_telegram=(not args.no_telegram))
print(json.dumps(result, indent=2, default=str))