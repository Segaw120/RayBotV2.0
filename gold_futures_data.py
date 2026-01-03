import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import yahooquery as yq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_latest_complete_date() -> str:
    """
    Calculate yesterday's date as the latest complete trading day with close price.
    Returns date in 'YYYY-MM-DD' format.
    """
    yesterday = datetime.now().date() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    logger.info(f"Latest complete date calculated: {yesterday_str}")
    return yesterday_str

def fetch_365d_gold_prices() -> pd.DataFrame:
    """
    Fetch 365 days of gold price data ending with yesterday's complete date.
    
    Returns:
        pd.DataFrame: Cleaned gold price DataFrame (UTC+3) with exact 365-day range.
    """
    try:
        # Calculate date range: yesterday as end, 365 days before as start
        end_date = get_latest_complete_date()
        start_date = (datetime.strptime(end_date, '%Y-%m-%d').date() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching 365 days gold data: {start_date} to {end_date}")
        
        # Fetch using main function
        df = fetch_gold_prices(start_date, end_date)
        
        # Verify we got data up to latest complete date
        latest_date = df.index.max().normalize().strftime('%Y-%m-%d')
        if latest_date != end_date:
            logger.warning(f"Latest data date {latest_date} != expected {end_date}")
        
        logger.info(f"Successfully fetched {len(df)} days from {start_date} to {latest_date}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching 365d gold prices: {str(e)}")
        raise

def fetch_gold_prices(start_date: str, end_date: str = None, interval: str = '1d') -> pd.DataFrame:
    """
    Fetch gold price data (GC=F) using yahooquery for specified date range.
    """
    try:
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        if start_dt >= end_dt:
            raise ValueError("start_date must be before end_date")
        
        logger.info(f"Fetching gold prices (GC=F) from {start_date} to {end_date}, interval={interval}")
        
        ticker = yq.Ticker('GC=F')
        raw = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if raw.empty:
            raise ValueError("No data returned from Yahoo Finance")
        
        logger.info(f"Fetched {len(raw)} rows of raw data")
        df_clean = _clean_standardize_gold(raw, start_dt.date(), end_dt.date())
        
        actual_start = df_clean.index.min().normalize()
        actual_end = df_clean.index.max().normalize()
        logger.info(f"Clean DataFrame: {len(df_clean)} rows from {actual_start} to {actual_end}")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error fetching gold prices: {str(e)}")
        raise

def _clean_standardize_gold(raw: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Clean and standardize per exact specifications - NO forward fill."""
    logger.info("Starting cleaning and standardization process")
    
    # 3. Clean and Standardize
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index(level=0, drop=True)

    # Robustly parse/normalize the index and remove timezones, then convert to ET (+3)
    dt_index = pd.to_datetime(raw.index)
    # If tz-aware, convert to UTC naive first
    if getattr(dt_index, "tz", None) is not None:
        try:
            dt_index = dt_index.tz_convert("UTC").tz_localize(None)
        except Exception:
            dt_index = dt_index.tz_localize(None)
    # Now shift timestamps from UTC to UTC+3 (Ethiopian time alignment)
    dt_index = dt_index + pd.Timedelta(hours=3)
    raw.index = dt_index

    raw.columns = [c.lower() for c in raw.columns]         # Normalize casing

    # Select required columns, ensure close exists
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df = raw[required_cols].dropna(subset=['close']).dropna(how='all')
    
    df.sort_index(inplace=True)
    logger.info(f"Final clean DataFrame: {len(df)} rows")
    return df

# Example usage
if __name__ == "__main__":
    print("Latest complete date:", get_latest_complete_date())
    
    try:
        gold_df = fetch_365d_gold_prices()
        print("
First 5 rows:")
        print(gold_df.head())
        print(f"
Shape: {gold_df.shape}")
        print(f"Date range: {gold_df.index.min()} to {gold_df.index.max()}")
    except Exception as e:
        logger.error(f"Demo failed: {e}")