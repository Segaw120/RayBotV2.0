import os
import io
import math
import time
import json
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
import supabase
from metaapi_cloud_sdk import MetaApi

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
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

# MetaAPI configuration (demo account)
METAAPI_LOGIN = "5043812126"
METAAPI_PASSWORD = "U*Ox8cJt"
METAAPI_ACCOUNT_ID = "Demo"  # You'll need to provide this

# Initialize MetaAPI client
metaapi = MetaApi("your_api_token")  # You'll need to provide this
account = metaapi.metaapi(account_id=METAAPI_ACCOUNT_ID).get_account()
if account is None:
    st.error("Failed to connect to MetaAPI account")
    st.stop()

# App configuration
st.sidebar.header("App Configuration")
symbol = st.sidebar.text_input("Symbol", value="GC=F")
start_date = st.sidebar.date_input("Start date", value=datetime.today() - timedelta(days=7))
end_date = st.sidebar.date_input("End date", value=datetime.today())
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "15m", "5m", "1m"], index=0)
account_balance = st.sidebar.number_input("Account Balance ($)", value=10000, step=1000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
atr_multiplier_tp = st.sidebar.slider("TP ATR Multiplier", 1.0, 5.0, 2.0, 0.5)
atr_multiplier_sl = st.sidebar.slider("SL ATR Multiplier", 0.5, 3.0, 1.0, 0.25)

# File uploader for model checkpoint
st.sidebar.header("Model Checkpoint")
checkpoint_file = st.sidebar.file_uploader(
    "Upload L1 checkpoint file (.pt)",
    type=['pt'],
    help="Upload the L1 model checkpoint file"
)

# Session state for storing predictions and orders
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'limit_orders' not in st.session_state:
    st.session_state.limit_orders = None

def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    """Compute a compact set of engineered features from OHLCV."""
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
    """
    Build sequences ending at each index t: [t-seq_len+1, ..., t]
    Returns shape [N, seq_len, F]
    """
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


def load_model_from_checkpoint(checkpoint_file, device="cpu"):
    """Load the L1 model from a checkpoint file"""
    try:
        device = torch.device(device)
        checkpoint = torch.load(checkpoint_file, map_location=device)

        # Get input features from the model architecture
        in_features = checkpoint['model']['blocks.0.conv.weight'].shape[1]

        # Initialize model
        model = Level1ScopeCNN(in_features=in_features)
        model.load_state_dict(checkpoint['model'])
        model.to(device).eval()

        # Load scaler if available
        scaler = None
        if 'scaler' in checkpoint:
            scaler = checkpoint['scaler']
        elif 'scaler_seq' in checkpoint:
            scaler = checkpoint['scaler_seq']

        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def predict_with_model(model, scaler, df, seq_len=64):
    """Make predictions using the loaded model"""
    if model is None or scaler is None:
        st.error("Model or scaler not loaded")
        return None

    # Compute features
    features = compute_engineered_features(df)

    # Get sequence columns
    seq_cols = ['open', 'high', 'low', 'close', 'volume']
    micro_cols = ['ret1', 'tr', 'vol_5', 'mom_5', 'chanpos_10']

    # Prepare feature dataframe
    feat_seq_df = pd.concat([
        df[seq_cols].astype(float),
        features[[c for c in micro_cols if c in features.columns]]
    ], axis=1, sort=False).fillna(0.0)

    # Scale features
    X_seq_all = feat_seq_df.values
    X_seq_all_scaled = scaler.transform(X_seq_all)

    # Create sequences for all indices
    indices = np.arange(len(df))
    Xseq = to_sequences(X_seq_all_scaled, indices, seq_len=seq_len)

    # Predict
    model.eval()
    logits = []
    batch = 256
    with torch.no_grad():
        for i in range(0, len(Xseq), batch):
            sub = Xseq[i:i+batch]
            xb = torch.tensor(sub.transpose(0, 2, 1), dtype=torch.float32, device=device)
            logit, _ = model(xb)
            logits.append(logit.detach().cpu().numpy())

    logits = np.concatenate(logits, axis=0).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Create predictions dataframe
    predictions = pd.DataFrame({
        'timestamp': df.index,
        'probability': probs
    })

    return predictions

def generate_limit_orders(df, predictions, atr_multiplier_tp=2.0, atr_multiplier_sl=1.0,
                         risk_per_trade=0.02, account_balance=10000):
    """Generate limit orders based on predictions"""
    orders = []

    for i, (_, row) in enumerate(predictions.iterrows()):
        current_price = df['close'].iloc[i]
        atr = df['atr'].iloc[i] if 'atr' in df.columns else df['tr'].iloc[i]

        if row['probability'] >= 0.5:  # Using 0.5 as threshold
            direction = "BUY" if row['probability'] > 0.5 else "SELL"

            if direction == "BUY":
                entry_price = current_price
                tp_price = entry_price + (atr * atr_multiplier_tp)
                sl_price = entry_price - (atr * atr_multiplier_sl)
            else:
                entry_price = current_price
                tp_price = entry_price - (atr * atr_multiplier_tp)
                sl_price = entry_price + (atr * atr_multiplier_sl)

            # Calculate position size based on risk
            risk_amount = account_balance * risk_per_trade
            stop_distance = abs(entry_price - sl_price)
            position_size = risk_amount / stop_distance if stop_distance > 0 else 0

            # Calculate R:R ratio
            risk = abs(entry_price - sl_price)
            reward = abs(tp_price - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0

            order = {
                "timestamp": row['timestamp'],
                "direction": direction,
                "entry_price": round(entry_price, 2),
                "tp_price": round(tp_price, 2),
                "sl_price": round(sl_price, 2),
                "position_size": round(position_size, 4),
                "risk_amount": round(risk_amount, 2),
                "risk_pct": round(risk_per_trade * 100, 2),
                "rr_ratio": round(rr_ratio, 2),
                "probability": round(row['probability'], 4),
                "atr": round(atr, 2)
            }

            orders.append(order)

    return pd.DataFrame(orders)

def log_to_supabase(predictions, orders):
    """Log predictions and orders to Supabase"""
    try:
        # Log predictions
        predictions_data = predictions.copy()
        predictions_data['created_at'] = datetime.utcnow().isoformat()
        supabase_client.table("predictions").insert(predictions_data.to_dict('records')).execute()

        # Log orders
        if orders is not None and not orders.empty:
            orders_data = orders.copy()
            orders_data['created_at'] = datetime.utcnow().isoformat()
            supabase_client.table("limit_orders").insert(orders_data.to_dict('records')).execute()

        st.success("Successfully logged data to Supabase")
    except Exception as e:
        st.error(f"Error logging to Supabase: {str(e)}")
        logger.error(f"Error logging to Supabase: {str(e)}")
        logger.error(traceback.format_exc())

def enter_limit_orders(orders):
    """Enter limit orders using MetaAPI"""
    if orders is None or orders.empty:
        st.warning("No orders to enter")
        return

    try:
        for _, order in orders.iterrows():
            # Convert order to MetaAPI format
            trade = {
                "symbol": symbol,
                "volume": order['position_size'],
                "type": "ORDER_TYPE_BUY_LIMIT" if order['direction'] == "BUY" else "ORDER_TYPE_SELL_LIMIT",
                "price": order['entry_price'],
                "sl": order['sl_price'],
                "tp": order['tp_price'],
                "deviation": 10,  # Allow 10 points deviation
                "magic": 123456,  # Unique identifier for the order
                "comment": f"CascadeTrader order - prob: {order['probability']}"
            }

            # Place the order
            result = account.create_order(trade)
            logger.info(f"Order placed: {result}")

        st.success(f"Successfully placed {len(orders)} orders on MetaAPI")
    except Exception as e:
        st.error(f"Error placing orders: {str(e)}")
        logger.error(f"Error placing orders: {str(e)}")
        logger.error(traceback.format_exc())


def main():
    """Main application logic"""
    st.header("Model Inference and Order Management")

    # Load model if checkpoint is uploaded
    if checkpoint_file is not None:
        st.info("Loading model from checkpoint...")
        model, scaler = load_model_from_checkpoint(checkpoint_file)
        if model is not None and scaler is not None:
            st.success("Model loaded successfully!")

            # File uploader for market data
            st.header("Market Data")
            data_file = st.file_uploader(
                "Upload market data (CSV)",
                type=['csv'],
                help="CSV with columns: timestamp, open, high, low, close, volume"
            )

            if data_file is not None:
                try:
                    # Load data
                    df = pd.read_csv(data_file, parse_dates=['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df = df.sort_index()
                    st.success(f"Loaded {len(df)} bars")

                    # Add ATR calculation
                    if 'atr' not in df.columns:
                        df['tr'] = (df['high'] - df['low']).clip(lower=0)
                        df['atr'] = df['tr'].rolling(14, min_periods=1).mean()

                    # Make predictions
                    st.info("Making predictions...")
                    predictions = predict_with_model(model, scaler, df)

                    if predictions is not None:
                        st.session_state.predictions = predictions
                        st.success("Predictions generated!")

                        # Show predictions
                        st.subheader("Predictions")
                        st.dataframe(predictions.head(20))

                        # Generate limit orders
                        st.info("Generating limit orders...")
                        orders = generate_limit_orders(
                            df,
                            predictions,
                            atr_multiplier_tp=atr_multiplier_tp,
                            atr_multiplier_sl=atr_multiplier_sl,
                            risk_per_trade=risk_per_trade,
                            account_balance=account_balance
                        )

                        if not orders.empty:
                            st.session_state.limit_orders = orders
                            st.success(f"Generated {len(orders)} limit orders")

                            # Show orders
                            st.subheader("Limit Orders")
                            st.dataframe(orders)

                            # Log to Supabase
                            if st.button("Log to Supabase"):
                                log_to_supabase(predictions, orders)

                            # Enter orders on MetaAPI
                            if st.button("Enter Orders on MetaAPI"):
                                enter_limit_orders(orders)
                        else:
                            st.warning("No trading signals generated")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    logger.error(f"Error processing data: {str(e)}")
                    logger.error(traceback.format_exc())
    else:
        st.info("Please upload a model checkpoint file to begin")

if __name__ == "__main__":
    main()
