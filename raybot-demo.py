# l1_inference_app.py
import os
import io
import math
import time
import json
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# torch + safe load helpers
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals

# sklearn scaler (for optional fallback)
from sklearn.preprocessing import StandardScaler

# yahooquery (used to fetch market data)
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

# -----------------------------------------------------------------------------
# Logging / UI
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("l1_inference_app")

st.set_page_config(page_title="Cascade Trader — L1 Inference & Limit Orders", layout="wide")
st.title("Cascade Trader — L1 Inference & Limit Orders (Bare-bones)")

# -----------------------------------------------------------------------------
# Minimal model architecture (must match L1 training architecture)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Utilities: safe model load (handles older checkpoints using add_safe_globals)
# -----------------------------------------------------------------------------
def safe_load_checkpoint_from_bytes(file_bytes: bytes, map_location="cpu"):
    """
    Safely attempt to load a PyTorch checkpoint from raw bytes.
    - Adds safe globals for common numpy/sklearn objects that appear in pickled payloads.
    - Tries torch.load with weights_only=False (for legacy checkpoints) if available.
    Returns the loaded object (could be Module, dict, or dict-like state).
    """
    buf = io.BytesIO(file_bytes)

    # allow some commonly-needed globals for legacy checkpoints
    safe_globals = []
    try:
        import numpy as _np
        safe_globals.append(_np.core.multiarray.scalar)
    except Exception:
        pass
    # StandardScaler sometimes appears inside checkpoints as an object
    try:
        import sklearn.preprocessing._data as _sd
        safe_globals.append(_sd.StandardScaler)
    except Exception:
        try:
            from sklearn.preprocessing import StandardScaler as _ss
            safe_globals.append(_ss)
        except Exception:
            pass

    # register safe globals if available
    if safe_globals:
        add_safe_globals(safe_globals)

    # Attempt torch.load with weights_only=False (PyTorch 2.6+); fallback to plain torch.load
    try:
        try:
            loaded = torch.load(buf, map_location=map_location, weights_only=False)
        except TypeError:
            # older PyTorch doesn't accept weights_only
            buf.seek(0)
            loaded = torch.load(buf, map_location=map_location)
    except Exception as e:
        # re-raise with contextual message
        raise RuntimeError(f"torch.load failed: {e}")
    return loaded

# -----------------------------------------------------------------------------
# Feature engineering (keeps parity with your training function)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Fetch market data (Gold futures GC=F)
# -----------------------------------------------------------------------------
def fetch_gold(symbol="GC=F", start=None, end=None, interval="1d"):
    if YahooTicker is None:
        raise RuntimeError("yahooquery not installed. Install via pip install yahooquery")
    if end is None:
        end = datetime.utcnow()
    if start is None:
        start = end - timedelta(days=365)
    tq = YahooTicker(symbol)
    raw = tq.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval=interval)
    if raw is None or (isinstance(raw, dict) and not raw):
        return pd.DataFrame()
    if isinstance(raw, dict):
        raw = pd.DataFrame(raw)
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index(level=0, drop=True)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    raw.columns = [c.lower() for c in raw.columns]
    if "close" not in raw.columns and "adjclose" in raw.columns:
        raw["close"] = raw["adjclose"]
    raw = raw[~raw.index.duplicated(keep="first")]
    # ensure columns
    for col in ["open","high","low","close","volume"]:
        if col not in raw.columns:
            raw[col] = 0.0
    return raw[["open","high","low","close","volume"]]

# -----------------------------------------------------------------------------
# Limit order generator
# -----------------------------------------------------------------------------
def generate_limit_order_from_prediction(df: pd.DataFrame, prob: float,
                                         atr_window=14,
                                         sl_atr_mult=1.0, tp_atr_mult=2.0,
                                         account_balance=10000.0, risk_pct=0.02):
    """
    Create a single-sided (LONG) limit order suggestion using ATR and risk.
    Assumes latest bar is the entry reference.
    """
    if df is None or df.empty:
        return None

    # compute TR and ATR
    tr = pd.DataFrame({
        "hl": df["high"] - df["low"],
        "hc": (df["high"] - df["close"].shift(1)).abs(),
        "lc": (df["low"] - df["close"].shift(1)).abs()
    }).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=1).mean().iloc[-1]

    entry_price = float(df["close"].iloc[-1])
    sl_distance = atr * sl_atr_mult
    tp_distance = atr * tp_atr_mult

    sl_price = entry_price - sl_distance
    tp_price = entry_price + tp_distance

    # position sizing by risk
    risk_amount = account_balance * float(risk_pct)
    if sl_distance <= 0:
        position_size = 0.0
    else:
        position_size = risk_amount / sl_distance

    return {
        "side": "LONG" if prob >= 0.5 else "SHORT",
        "entry_price": entry_price,
        "stop_loss": sl_price,
        "take_profit": tp_price,
        "atr": float(atr),
        "position_size": float(position_size),
        "risk_amount": float(risk_amount),
        "risk_pct": float(risk_pct),
        "probability": float(prob)
    }

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.sidebar.header("Config")
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "15m"], index=0)
lookback_days = st.sidebar.number_input("Lookback days", min_value=30, max_value=3650, value=365)
seq_len = st.sidebar.number_input("Sequence length", min_value=8, max_value=256, value=64, step=8)
account_balance = st.sidebar.number_input("Account balance ($)", min_value=100.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 2.0) / 100.0
sl_atr_mult = st.sidebar.slider("SL ATR multiplier", 0.5, 3.0, 1.0, step=0.1)
tp_atr_mult = st.sidebar.slider("TP ATR multiplier", 1.0, 5.0, 2.0, step=0.1)

st.sidebar.header("Model checkpoint (L1 only)")
checkpoint_file = st.sidebar.file_uploader("Upload L1 .pt checkpoint", type=["pt","pth","bin"])

st.sidebar.markdown("---")
st.sidebar.markdown("Bare-bones: fetch GC=F, load L1, infer latest bar, output limit order (LONG).")

# session state
if "market" not in st.session_state:
    st.session_state.market = None
if "model_obj" not in st.session_state:
    st.session_state.model_obj = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "prob" not in st.session_state:
    st.session_state.prob = None
if "order" not in st.session_state:
    st.session_state.order = None

# Fetch data
if st.button("Fetch latest GC=F data"):
    with st.spinner("Fetching data..."):
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=int(lookback_days))
            df = fetch_gold(symbol="GC=F", start=start, end=end, interval=interval)
            if df is None or df.empty:
                st.error("No data returned. Check internet / yahooquery.")
            else:
                st.session_state.market = df
                st.success(f"Fetched {len(df)} bars (last: {df.index[-1]})")
                st.dataframe(df.tail(10))
        except Exception as e:
            st.error(f"Fetch failed: {e}")
            logger.exception("fetch error")

# Load model
if checkpoint_file is not None:
    st.info("Loading checkpoint (L1) — this may take a moment.")
    raw = checkpoint_file.read()
    try:
        loaded = safe_load_checkpoint_from_bytes(raw, map_location="cpu")
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        st.session_state.model_obj = None
        loaded = None
        logger.exception("checkpoint load failed")
    else:
        # handle a few shapes:
        model_obj = None
        scaler_obj = None

        # If the checkpoint is a dict that contains 'model' or 'state_dict'
        if isinstance(loaded, dict):
            # many training pipelines saved dict with keys like {'model': model_state_dict, 'scaler_seq': scaler_bytes, ...}
            # try to find model state or full model inside dict
            if "model" in loaded:
                candidate = loaded["model"]
            elif "state_dict" in loaded:
                candidate = loaded["state_dict"]
            else:
                candidate = loaded

            # If candidate is a torch.nn.Module object (rare), use it directly
            if isinstance(candidate, nn.Module):
                model_obj = candidate
            # If candidate looks like a state_dict (mapping of tensors)
            elif isinstance(candidate, dict) and all(isinstance(v, (torch.Tensor, np.ndarray, int, float)) or isinstance(v, torch.Tensor) for v in candidate.values()):
                # infer in_features from first conv weight if available
                # common key path: 'blocks.0.conv.weight' or 'blocks.0.conv.weight'
                possible_keys = [
                    "blocks.0.conv.weight",
                    "blocks.0.conv.weight",
                    "blocks.0.conv.weight"
                ]
                found = None
                for k in candidate.keys():
                    if k.endswith(".conv.weight") and "blocks.0" in k:
                        found = k; break
                if found is None:
                    # try first conv weight by heuristic
                    for k in candidate.keys():
                        if ".conv.weight" in k:
                            found = k; break
                if found is not None:
                    w = candidate[found]
                    # handle numpy arrays present in state dict
                    if isinstance(w, np.ndarray):
                        in_features = w.shape[1]
                    elif isinstance(w, torch.Tensor):
                        in_features = w.shape[1]
                    else:
                        # fallback: require user to provide in_features (not present)
                        in_features = None
                    if in_features is None:
                        st.warning("Could not infer 'in_features' from state dict; attempting default 12.")
                        in_features = 12
                    model = Level1ScopeCNN(in_features=in_features)
                    model.load_state_dict(candidate)
                    model_obj = model
                else:
                    # If the dict itself may be state dict for the module with conventional keys
                    try:
                        model = Level1ScopeCNN()
                        model.load_state_dict(candidate)
                        model_obj = model
                    except Exception as ex:
                        st.warning("Checkpoint dict didn't match expected L1 state_dict. Try saving model as state_dict or full model.")
                        logger.exception("state_dict->model load failed")
                        model_obj = None
            else:
                # loaded dict but ambiguous; user might have saved entire model object at top level
                try:
                    # sometimes torch.save(model) yields a Module directly
                    if isinstance(loaded, nn.Module):
                        model_obj = loaded
                except Exception:
                    model_obj = None

            # scaler extraction: trained pipelines sometimes include 'scaler_seq' or 'scaler'
            if "scaler_seq" in loaded:
                try:
                    scaler_obj = loaded["scaler_seq"]
                except Exception:
                    scaler_obj = None
            elif "scaler" in loaded:
                scaler_obj = loaded["scaler"]
            elif "scaler_seq.pkl" in loaded:
                scaler_obj = loaded["scaler_seq.pkl"]

        else:
            # loaded is not dict: could be a Module or state_dict
            if isinstance(loaded, nn.Module):
                model_obj = loaded
            elif isinstance(loaded, dict):
                # try to load as state_dict
                try:
                    model = Level1ScopeCNN()
                    model.load_state_dict(loaded)
                    model_obj = model
                except Exception:
                    model_obj = None

        if model_obj is None:
            st.error("Failed to construct L1 model from checkpoint. See logs.")
            st.session_state.model_obj = None
        else:
            model_obj.eval()
            st.session_state.model_obj = model_obj
            st.session_state.scaler = scaler_obj  # may be None
            st.success("L1 model loaded into session (CPU).")
            # display basic info
            try:
                st.write("Model embedding dim:", model_obj.embedding_dim)
            except Exception:
                pass

# Optional: allow user to upload scaler pickle if checkpoint didn't include it
scaler_upload = st.sidebar.file_uploader("Optional: upload scaler_seq.pkl (pickle)", type=["pkl","pickle"])
if scaler_upload is not None:
    try:
        scaler_bytes = scaler_upload.read()
        scaler_obj = pickle.loads(scaler_bytes)
        st.session_state.scaler = scaler_obj
        st.success("Scaler loaded from uploaded file.")
    except Exception as e:
        st.error(f"Failed to load scaler: {e}")
        logger.exception("scaler load error")

# Run inference on latest bars
if st.button("Run L1 inference on latest data"):
    if st.session_state.market is None or st.session_state.market.empty:
        st.error("No market data present. Click 'Fetch latest GC=F data' first.")
    elif st.session_state.model_obj is None:
        st.error("No L1 model loaded. Upload checkpoint.")
    else:
        try:
            df = st.session_state.market.copy()
            # compute features
            eng = compute_engineered_features(df)
            # prepare seq cols + micro cols used in training
            seq_cols = ['open','high','low','close','volume']
            micro_cols = ['ret1','tr','vol_5','mom_5','chanpos_10']
            use_micro = [c for c in micro_cols if c in eng.columns]
            feat_seq_df = pd.concat([df[seq_cols].astype(float), eng[use_micro]], axis=1).fillna(0.0)
            X_seq_all = feat_seq_df.values

            # scaler: if present, use it; else fit a StandardScaler on the X_seq_all (quick fallback)
            scaler = st.session_state.scaler
            if scaler is None:
                st.warning("No scaler available in checkpoint — fitting a fresh StandardScaler on the fetched data (fast fallback).")
                scaler = StandardScaler()
                scaler.fit(X_seq_all)
            X_seq_scaled = scaler.transform(X_seq_all)

            # build sequences for last index (we run inference for latest bar)
            last_idx = np.array([len(X_seq_scaled)-1])
            Xseq = to_sequences(X_seq_scaled, last_idx, seq_len=int(seq_len))

            # run model (ensure correct device = cpu)
            model = st.session_state.model_obj
            model.to(torch.device("cpu"))
            model.eval()
            with torch.no_grad():
                xb = torch.tensor(Xseq.transpose(0,2,1), dtype=torch.float32, device="cpu")
                out = model(xb)
                # model returns (logit, emb) or
