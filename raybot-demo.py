# Chunk 1/3: imports, config, UI, fetch + feature helpers, model class & loader
import os
import io
import math
import time
import uuid
import json
import joblib
import pickle
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from yahooquery import Ticker

# supabase import - try common variants
try:
    from supabase import create_client as create_supabase_client
except Exception:
    try:
        import supabase
        create_supabase_client = getattr(supabase, "create_client", None)
    except Exception:
        create_supabase_client = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("inference_app")
logger.setLevel(logging.INFO)

# Streamlit UI configuration
st.set_page_config(page_title="Cascade Trader Inference", layout="wide")
st.title("Cascade Trader Inference App")

# Supabase configuration (hardcoded as requested)
SUPABASE_URL = "https://jubcotqsbvguwzklngzd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp1YmNvdHFzYnZndXd6a2xuZ3pkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1OTU0MjA3MCwiZXhwIjoyMDc1MTE4MDcwfQ.1HV-o9JFa_nCZGXcoap2OgOCKjRSlyFSRvKmYk70eDk"
supabase_client = None
if create_supabase_client is not None:
    try:
        supabase_client = create_supabase_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.warning("Could not create supabase client: %s", e)

# Asset categories and tickers (unchanged)
ASSET_CATEGORIES = {
    "Forex (Major)": [
        {"name": "EUR/USD", "ticker": "EURUSD=X"},
        {"name": "GBP/USD", "ticker": "GBPUSD=X"},
        {"name": "USD/JPY", "ticker": "USDJPY=X"},
        {"name": "AUD/USD", "ticker": "AUDUSD=X"},
        {"name": "USD/CAD", "ticker": "USDCAD=X"},
        {"name": "USD/CHF", "ticker": "USDCHF=X"},
        {"name": "NZD/USD", "ticker": "NZDUSD=X"}
    ],
    "Metals": [
        {"name": "Gold (Futures)", "ticker": "GC=F"},
        {"name": "Silver (Futures)", "ticker": "SI=F"},
        {"name": "Platinum (Futures)", "ticker": "PL=F"},
        {"name": "Palladium (Futures)", "ticker": "PA=F"}
    ],
    "Indices": [
        {"name": "S&P 500", "ticker": "^GSPC"},
        {"name": "Dow Jones Industrial Average", "ticker": "^DJI"},
        {"name": "NASDAQ Composite", "ticker": "^IXIC"},
        {"name": "Russell 2000", "ticker": "^RUT"},
        {"name": "FTSE 100", "ticker": "^FTSE"},
        {"name": "DAX", "ticker": "^GDAXI"},
        {"name": "Nikkei 225", "ticker": "^N225"}
    ],
    "Oil": [
        {"name": "WTI Crude Oil (Futures)", "ticker": "CL=F"},
        {"name": "Brent Crude Oil (Futures)", "ticker": "BZ=F"}
    ],
    "Crypto (Major)": [
        {"name": "Bitcoin / USD", "ticker": "BTC-USD"},
        {"name": "Ethereum / USD", "ticker": "ETH-USD"},
        {"name": "Binance Coin / USD", "ticker": "BNB-USD"},
        {"name": "Solana / USD", "ticker": "SOL-USD"},
        {"name": "XRP / USD", "ticker": "XRP-USD"},
        {"name": "Cardano / USD", "ticker": "ADA-USD"},
        {"name": "Dogecoin / USD", "ticker": "DOGE-USD"}
    ]
}

# Sidebar / UI (unchanged)
st.sidebar.header("App Configuration")
category = st.sidebar.selectbox("Asset Category", list(ASSET_CATEGORIES.keys()))
asset_options = [asset["name"] for asset in ASSET_CATEGORIES[category]]
selected_asset_name = st.sidebar.selectbox("Asset", asset_options)
selected_ticker = next(asset["ticker"] for asset in ASSET_CATEGORIES[category] if asset["name"] == selected_asset_name)

end_date = st.sidebar.date_input("End date", value=datetime.today() - timedelta(days=1))
start_date = st.sidebar.date_input("Start date", value=end_date - timedelta(days=365))
interval = "1d"

st.sidebar.header("Trading Parameters")
account_balance = st.sidebar.number_input("Account Balance ($)", value=10000, step=1000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
atr_multiplier_tp = st.sidebar.slider("TP ATR Multiplier", 1.0, 5.0, 2.0, 0.5)
atr_multiplier_sl = st.sidebar.slider("SL ATR Multiplier", 0.5, 3.0, 1.0, 0.25)

st.sidebar.header("Model Parameters")
seq_len = st.sidebar.slider("Sequence Length", 8, 256, 64, step=8)

st.sidebar.header("Model Checkpoint")
checkpoint_file = st.sidebar.file_uploader(
    "Upload L1 checkpoint file (.pt)",
    type=['pt'],
    help="Upload the L1 model checkpoint file"
)

# Session state
if 'predictions' not in st.session_state: st.session_state.predictions = None
if 'limit_orders' not in st.session_state: st.session_state.limit_orders = None
if 'market_data' not in st.session_state: st.session_state.market_data = None
if 'checkpoint_bytes' not in st.session_state: st.session_state.checkpoint_bytes = None

# Fetch market data (robust to yahooquery return shapes)
def fetch_market_data(ticker: str, start_date: datetime, end_date: datetime, interval: str = "1d") -> pd.DataFrame:
    try:
        st.info(f"Fetching {ticker} {interval} data from Yahoo Finance...")
        ticker_obj = Ticker(ticker)
        data = ticker_obj.history(start=start_date.strftime('%Y-%m-%d'),
                                  end=end_date.strftime('%Y-%m-%d'),
                                  interval=interval.lower())

        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            st.error(f"No data returned for {ticker}")
            return pd.DataFrame()

        # When yahooquery returns a DataFrame with datetime index
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                # try index
                df = df.reset_index()
                if 'index' in df.columns and np.issubdtype(df['index'].dtype, np.datetime64):
                    df['timestamp'] = pd.to_datetime(df['index'])
                elif 'date' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'])
                elif 'period' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['period'])
                else:
                    # fallback to first datetime-like column
                    dt_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
                    if dt_cols:
                        df['timestamp'] = pd.to_datetime(df[dt_cols[0]])
                    else:
                        st.error("Could not find timestamp column in yahooquery response.")
                        return pd.DataFrame()
            # normalize column names
            df.columns = [c.lower() for c in df.columns]
            # ensure ohlcv exist
            for col in ['open','high','low','close','volume']:
                if col not in df.columns:
                    df[col] = np.nan
            df = df.set_index('timestamp').sort_index()
            df = df[['open','high','low','close','volume']]
            st.success(f"Fetched {len(df)} bars for {ticker}")
            return df
        else:
            st.error("Yahooquery returned unsupported data structure.")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

# Feature engineering (keeps same behavior, but returns full features DF)
def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(0.0, index=df.index)

    ret1 = c.pct_change().fillna(0.0)
    f['ret1'] = ret1
    f['logret1'] = np.log1p(ret1.replace(-1, -0.999999))
    tr = (h - l).clip(lower=0)
    f['tr'] = tr.fillna(0.0)
    f['atr'] = tr.rolling(14, min_periods=1).mean().fillna(0.0)

    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'vol_z_{w}'] = (v.rolling(w).mean() - v.rolling(max(1,w*3)).mean()).fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)
    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def to_sequences(features: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    Nrows, F = features.shape
    X = np.zeros((len(indices), seq_len, F), dtype=features.dtype)
    for i, t in enumerate(indices):
        t = int(t)
        t0 = t - seq_len + 1
        if t0 < 0:
            pad_count = -t0
            pad = np.repeat(features[[0]], pad_count, axis=0)
            seq = np.vstack([pad, features[0:t+1]])
        else:
            seq = features[t0:t+1]
        if seq.shape[0] < seq_len:
            pad_needed = seq_len - seq.shape[0]
            pad = np.repeat(seq[[0]], pad_needed, axis=0)
            seq = np.vstack([pad, seq])
        X[i] = seq[-seq_len:]
    return X

# Model building block (same as before)
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, pdrop):
        super().__init__()
        pad = (k - 1) * d // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(pdrop)
        self.res = (c_in == c_out)
    def forward(self, x):
        out = self.conv(x); out = self.bn(out); out = self.act(out); out = self.drop(out)
        if self.res: out = out + x
        return out

class Level1ScopeCNN(nn.Module):
    def __init__(self, in_features=12, channels=(32,64,128), kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1):
        super().__init__()
        chs = [in_features] + list(channels)
        blocks = []
        for i in range(len(channels)):
            k = kernel_sizes[min(i, len(kernel_sizes)-1)]
            d = dilations[min(i, len(dilations)-1)]
            blocks.append(ConvBlock(chs[i], chs[i+1], k, d, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.project = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
        self.head = nn.Linear(chs[-1], 1)
    @property
    def embedding_dim(self): return int(self.blocks[-1].conv.out_channels)
    def forward(self, x):
        z = self.blocks(x)
        z = self.project(z)
        z_pool = z.mean(dim=-1)
        logit = self.head(z_pool)
        return logit, z_pool

# Robust checkpoint loader
def load_model_from_checkpoint(checkpoint_bytes: bytes, device="cpu"):
    """
    Load model and scaler from a checkpoint bytes object.
    Accepts:
      - torch.save(state_dict)    -> OrderedDict
      - torch.save({'model': state_dict, 'scaler': scaler, ...})
      - torch.save({'model': state_dict, 'scaler_seq': pickle.dumps(scaler), ...})
    Returns: model (nn.Module), scaler (StandardScaler or None)
    """
    try:
        map_loc = torch.device(device)
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=map_loc)

        # Determine state_dict
        state_dict = None
        scaler = None

        if isinstance(checkpoint, dict):
            # common keys: 'model', 'state_dict', 'model_state'
            if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
                state_dict = checkpoint['state_dict']
            else:
                # maybe the dict *is* a state_dict (some pickled dicts)
                # Heuristic: if values are tensors, treat as state_dict
                sample_vals = list(checkpoint.values())[:3]
                if all(isinstance(v, torch.Tensor) for v in sample_vals):
                    state_dict = checkpoint
        elif isinstance(checkpoint, (torch.nn.Module,)):
            # unlikely: direct module saved, not state_dict
            try:
                model = checkpoint.to(map_loc).eval()
                return model, None
            except Exception:
                state_dict = None

        # attempt to find scaler inside payload
        if isinstance(checkpoint, dict):
            if 'scaler' in checkpoint:
                scaler = checkpoint['scaler']
            elif 'scaler_seq' in checkpoint:
                sval = checkpoint['scaler_seq']
                # maybe it's pickled bytes
                if isinstance(sval, (bytes, bytearray)):
                    try:
                        scaler = pickle.loads(sval)
                    except Exception:
                        scaler = sval
                else:
                    scaler = sval
            elif 'scaler_tab' in checkpoint:
                sval = checkpoint['scaler_tab']
                if isinstance(sval, (bytes, bytearray)):
                    try:
                        scaler = pickle.loads(sval)
                    except Exception:
                        scaler = sval
                else:
                    scaler = sval

        # If state_dict not found but checkpoint is an OrderedDict-like, treat as state_dict
        if state_dict is None:
            # try heuristic again
            if isinstance(checkpoint, dict):
                sample_vals = list(checkpoint.values())[:3]
                if all(isinstance(v, torch.Tensor) for v in sample_vals):
                    state_dict = checkpoint

        if state_dict is None:
            st.warning("Checkpoint did not contain an identifiable state_dict; attempting to return raw object as model.")
            # return raw object if module
            if isinstance(checkpoint, nn.Module):
                checkpoint.to(map_loc).eval()
                return checkpoint, None
            return None, None

        # infer in_features from conv weight key heuristically
        in_features = None
        conv_keys = [k for k in state_dict.keys() if k.endswith(".conv.weight")]
        if conv_keys:
            # pick first conv and take shape[1]
            k0 = conv_keys[0]
            in_features = state_dict[k0].shape[1]
        else:
            # fallback: try to find any weight with 3 dims (out,in,k)
            tensor_keys = [k for k,v in state_dict.items() if isinstance(v, torch.Tensor) and v.dim() >= 2]
            if tensor_keys:
                # try to infer from first conv-like weight where dim==3
                for k in tensor_keys:
                    v = state_dict[k]
                    if v.dim() == 3:
                        in_features = v.shape[1]
                        break
        if in_features is None:
            in_features = 12
            logger.warning("Could not infer in_features from state_dict; defaulting to %d", in_features)

        model = Level1ScopeCNN(in_features=in_features)
        model.load_state_dict(state_dict, strict=False)
        model.to(map_loc).eval()

        # If scaler is a bytes or pickled object retrieved earlier but not parsed, try to unpickle
        if scaler is None and isinstance(checkpoint, dict):
            # look for pickled bytes anywhere with name containing 'scaler'
            for k in checkpoint.keys():
                if 'scaler' in k.lower():
                    sval = checkpoint[k]
                    if isinstance(sval, (bytes, bytearray)):
                        try:
                            scaler = pickle.loads(sval)
                        except Exception:
                            scaler = None
                    elif isinstance(sval, StandardScaler):
                        scaler = sval
        # final sanity: scaler must be sklearn StandardScaler (or similar)
        if scaler is not None and not hasattr(scaler, 'transform'):
            logger.warning("Loaded scaler object doesn't expose a transform() method; ignoring scaler.")
            scaler = None

        return model, scaler

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None


# Chunk 2/3: prediction, order generation, supabase logging helpers

def predict_with_model(model: nn.Module, scaler, df: pd.DataFrame, seq_len: int = 64):
    """
    Make predictions using the loaded model.
    - merges engineered features back into df so ATR is available
    - scales features with scaler
    - moves tensors to model device for inference
    """
    if model is None:
        st.error("Model not loaded")
        return None
    if scaler is None:
        st.warning("Scaler not provided — predictions may be invalid. Attempting to proceed.")

    try:
        # Compute features and merge into df (so atr/tr are available later)
        features = compute_engineered_features(df)
        # attach features back to df for ATR availability
        merged = df.copy()
        merged = merged.join(features, how='left')

        # Build feature matrix used during training: seq columns + select micro cols
        seq_cols = ['open', 'high', 'low', 'close', 'volume']
        micro_cols = ['ret1', 'tr', 'vol_5', 'mom_5', 'chanpos_10']
        present_micro = [c for c in micro_cols if c in features.columns]
        feat_seq_df = pd.concat([merged[seq_cols].astype(float), features[present_micro]], axis=1, sort=False).fillna(0.0)

        # Scale
        X_seq_all = feat_seq_df.values
        if scaler is not None:
            try:
                X_seq_all_scaled = scaler.transform(X_seq_all)
            except Exception as e:
                st.warning("Scaler transform failed: %s. Proceeding without scaling." % e)
                X_seq_all_scaled = X_seq_all
        else:
            X_seq_all_scaled = X_seq_all

        # Create sequences
        indices = np.arange(len(merged))
        Xseq = to_sequences(X_seq_all_scaled, indices, seq_len=seq_len)

        # inference in batches; ensure tensors are on model device
        device = next(model.parameters()).device
        logits = []
        batch = 256
        model.eval()
        with torch.no_grad():
            for i in range(0, len(Xseq), batch):
                sub = Xseq[i:i+batch]
                xb = torch.tensor(sub.transpose(0,2,1), dtype=torch.float32, device=device)
                out = model(xb)
                logit = out[0] if isinstance(out, tuple) else out
                logits.append(logit.detach().cpu().numpy())
        logits = np.concatenate(logits, axis=0).reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))

        # Build predictions df aligned to index
        preds = pd.DataFrame({
            'timestamp': merged.index,
            'probability': probs
        }).set_index('timestamp')

        # Store merged df (with atr) back so order generator can use it
        return preds, merged

    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def generate_limit_orders(df_with_features: pd.DataFrame, predictions: pd.DataFrame,
                          atr_multiplier_tp: float = 2.0, atr_multiplier_sl: float = 1.0,
                          risk_per_trade: float = 0.02, account_balance: float = 10000.0,
                          prob_threshold: float = 0.5):
    """
    Generate limit orders by joining predictions to market rows.
    For each prediction >= threshold, create an order using that row's close and atr.
    """
    if predictions is None or predictions.empty:
        return pd.DataFrame()

    # ensure index types match
    pred = predictions.copy()
    dfm = df_with_features.copy()

    # join predictions onto dfm by timestamp index
    merged = dfm.join(pred, how='right')
    merged = merged.dropna(subset=['probability']).copy()

    orders = []
    for ts, row in merged.iterrows():
        prob = float(row['probability'])
        if prob < prob_threshold:
            continue

        # current price and atr: prefer 'atr' then 'tr'
        current_price = float(row.get('close', np.nan))
        atr = float(row.get('atr', np.nan) if 'atr' in row and not pd.isna(row.get('atr')) else row.get('tr', np.nan))
        if pd.isna(current_price) or pd.isna(atr) or atr == 0:
            # skip if missing required numbers
            continue

        direction = "BUY" if prob >= prob_threshold else "SELL"  # essentially always BUY for prob >= threshold
        if direction == "BUY":
            entry_price = current_price
            tp_price = entry_price + (atr * atr_multiplier_tp)
            sl_price = entry_price - (atr * atr_multiplier_sl)
        else:
            entry_price = current_price
            tp_price = entry_price - (atr * atr_multiplier_tp)
            sl_price = entry_price + (atr * atr_multiplier_sl)

        # position sizing: risk_amount in dollars / stop_distance in price units
        risk_amount = account_balance * risk_per_trade
        stop_distance = abs(entry_price - sl_price)
        position_size = (risk_amount / stop_distance) if stop_distance > 0 else 0.0

        reward = abs(tp_price - entry_price)
        risk = stop_distance
        rr_ratio = (reward / risk) if risk > 0 else 0.0

        order = {
            "timestamp": ts.isoformat(),
            "direction": direction,
            "entry_price": float(round(entry_price, 6)),
            "tp_price": float(round(tp_price, 6)),
            "sl_price": float(round(sl_price, 6)),
            "position_size": float(round(position_size, 6)),
            "risk_amount": float(round(risk_amount, 2)),
            "risk_pct": float(round(risk_per_trade * 100, 3)),
            "rr_ratio": float(round(rr_ratio, 3)),
            "probability": float(round(prob, 6)),
            "atr": float(round(atr, 6)),
            "symbol": selected_ticker
        }
        orders.append(order)

    return pd.DataFrame(orders)

def log_to_supabase(predictions: pd.DataFrame, orders: pd.DataFrame):
    """Log predictions and orders to Supabase if client available"""
    if supabase_client is None:
        st.warning("Supabase client not configured or import failed; skipping log.")
        return
    try:
        # predictions: convert index to column
        pred_df = predictions.reset_index().copy()
        pred_df['created_at'] = datetime.utcnow().isoformat()
        pred_df['symbol'] = selected_ticker
        # convert records
        pred_recs = pred_df.to_dict('records')
        if pred_recs:
            supabase_client.table("predictions").insert(pred_recs).execute()

        if orders is not None and not orders.empty:
            orders_df = orders.copy()
            orders_df['created_at'] = datetime.utcnow().isoformat()
            order_recs = orders_df.to_dict('records')
            if order_recs:
                supabase_client.table("limit_orders").insert(order_recs).execute()

        st.success("Successfully logged data to Supabase")
    except Exception as e:
        st.error(f"Error logging to Supabase: {e}")
        logger.error(traceback.format_exc())


# Chunk 3/3: Streamlit main UI flow and wiring

def main():
    st.header("Model Inference and Order Management")

    # Fetch market data
    if st.button("Fetch Market Data"):
        with st.spinner(f"Fetching {selected_ticker} data..."):
            market_data = fetch_market_data(selected_ticker, start_date, end_date, interval)
            if not market_data.empty:
                # compute features and attach so ATR is present for orders
                feats = compute_engineered_features(market_data)
                market_data = market_data.join(feats, how='left')
                st.session_state.market_data = market_data
                st.success(f"Successfully fetched {len(market_data)} bars")
                st.dataframe(market_data.tail())

    # Save checkpoint bytes to session_state once uploaded (so we can reuse)
    if checkpoint_file is not None:
        # read bytes once and keep
        st.session_state.checkpoint_bytes = checkpoint_file.read()
        st.info("Checkpoint bytes loaded into session state")

    # Load model from checkpoint
    model = None
    scaler = None
    if st.session_state.checkpoint_bytes is not None:
        if st.button("Load Model from Checkpoint"):
            with st.spinner("Loading model..."):
                model, scaler = load_model_from_checkpoint(st.session_state.checkpoint_bytes, device="cpu")
                if model is None:
                    st.error("Failed to load model from checkpoint.")
                else:
                    st.success("Model loaded.")
                    st.write("Model device:", next(model.parameters()).device)
                    if scaler is not None:
                        st.write("Scaler detected.")
                    else:
                        st.warning("No scaler found in checkpoint. Predictions may be off.")

    # If model was loaded earlier in this run, keep references
    if 'loaded_model' not in st.session_state: st.session_state.loaded_model = None
    if 'loaded_scaler' not in st.session_state: st.session_state.loaded_scaler = None

    # Persist newly loaded model
    if model is not None:
        st.session_state.loaded_model = model
        st.session_state.loaded_scaler = scaler

    # Make predictions
    if st.session_state.loaded_model is not None:
        if st.session_state.market_data is None or st.session_state.market_data.empty:
            st.info("Please fetch market data first.")
        else:
            if st.button("Generate Predictions"):
                with st.spinner("Making predictions..."):
                    preds, merged_df = predict_with_model(st.session_state.loaded_model, st.session_state.loaded_scaler, st.session_state.market_data, seq_len=seq_len)
                    if preds is None:
                        st.error("Prediction failed.")
                    else:
                        st.session_state.predictions = preds
                        # merged_df contains market data + features (including atr)
                        st.session_state.market_data = merged_df
                        st.success("Predictions generated!")
                        st.subheader("Predictions (tail)")
                        st.dataframe(preds.tail(20))

                        # Generate limit orders
                        if st.button("Generate Limit Orders"):
                            with st.spinner("Generating limit orders..."):
                                orders = generate_limit_orders(
                                    st.session_state.market_data,
                                    st.session_state.predictions,
                                    atr_multiplier_tp=atr_multiplier_tp,
                                    atr_multiplier_sl=atr_multiplier_sl,
                                    risk_per_trade=risk_per_trade,
                                    account_balance=account_balance,
                                    prob_threshold=0.5
                                )
                                if orders is None or orders.empty:
                                    st.warning("No trading signals generated (no predictions above threshold or missing ATR).")
                                else:
                                    st.session_state.limit_orders = orders
                                    st.success(f"Generated {len(orders)} limit orders")
                                    st.subheader("Limit Orders")
                                    st.dataframe(orders)

                                    if st.button("Log to Supabase"):
                                        with st.spinner("Logging to Supabase..."):
                                            log_to_supabase(st.session_state.predictions, orders)

    else:
        st.info("Upload and load a model checkpoint to begin inference.")

if __name__ == "__main__":
    main()
