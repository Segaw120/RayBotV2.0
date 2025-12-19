# -------------------------
# Chunk 1/3: imports, UI, data fetch, feature engineering, sequences
# -------------------------
import io
import os
import math
import time
import pickle
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# PyTorch
import torch
import torch.nn as nn

# sklearn scaler class for safe_globals allowlisting
from sklearn.preprocessing import StandardScaler

# Optional Yahoo fetcher
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cascade_l1_inference")
logger.setLevel(logging.INFO)

# Streamlit page
st.set_page_config(page_title="Cascade L1 Inference", layout="wide")
st.title("Cascade Trader — L1 Inference & Limit Orders")

# Sidebar: asset + date + model upload + params
st.sidebar.header("App Configuration")
ASSET = st.sidebar.text_input("Ticker (Yahoo)", value="GC=F")
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "15m", "5m", "1m"], index=0)

st.sidebar.header("Dates")
end_date = st.sidebar.date_input("End date", value=datetime.utcnow().date())
start_date = st.sidebar.date_input("Start date", value=end_date - timedelta(days=365))

st.sidebar.header("Model Checkpoint (L1)")
checkpoint_file = st.sidebar.file_uploader("Upload L1 checkpoint (.pt or .pth)", type=["pt","pth"], help="Either state_dict or saved payload")

st.sidebar.header("Trading / Risk")
account_balance = st.sidebar.number_input("Account Balance ($)", value=10000.0, step=1000.0)
risk_per_trade_pct = st.sidebar.slider("Risk per Trade (%)", 0.1, 10.0, 2.0, 0.1) / 100.0
atr_mult_tp = st.sidebar.slider("TP ATR Multiplier", 1.0, 5.0, 2.0, 0.1)
atr_mult_sl = st.sidebar.slider("SL ATR Multiplier", 0.5, 3.0, 1.0, 0.1)
seq_len = st.sidebar.slider("Sequence length", 8, 256, 64, step=8)

st.sidebar.markdown("**Run controls**")
fetch_btn = st.sidebar.button("Fetch Market Data")
predict_btn = st.sidebar.button("Predict (L1 only)")
orders_btn = st.sidebar.button("Generate Limit Orders (from last predictions)")

# ---------- Data fetch ----------
def fetch_market_data(ticker: str, start_date: datetime, end_date: datetime, interval: str="1d") -> pd.DataFrame:
    """Fetch OHLCV using yahooquery, normalize, return DataFrame indexed by timestamp."""
    if YahooTicker is None:
        st.warning("yahooquery not installed — fetch disabled. Please install yahooquery or provide data via upload.")
        return pd.DataFrame()
    try:
        t = YahooTicker(ticker)
        raw = t.history(start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        interval=interval)
        if raw is None:
            return pd.DataFrame()
        if isinstance(raw, dict):
            raw = pd.DataFrame(raw)
        # flatten multi-index if present
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index(level=0, drop=True)
        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index()
        raw.columns = [c.lower() for c in raw.columns]
        if "close" not in raw.columns and "adjclose" in raw.columns:
            raw["close"] = raw["adjclose"]
        # ensure required columns
        for col in ["open","high","low","close","volume"]:
            if col not in raw.columns:
                raw[col] = 0.0
        raw = raw[~raw.index.duplicated(keep="first")]
        return raw[["open","high","low","close","volume"]]
    except Exception as e:
        logger.exception("Fetch failed: %s", e)
        st.error(f"Fetch failed: {e}")
        return pd.DataFrame()

# ---------- Feature engineering ----------
def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    """Compute engineered features - must match training."""
    df = df.copy()
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(0.0, index=df.index)

    f = pd.DataFrame(index=df.index)
    ret1 = c.pct_change().fillna(0.0)
    f['ret1'] = ret1
    f['logret1'] = np.log1p(ret1.replace(-1, -0.999999))
    tr = (h - l).clip(lower=0.0)
    f['tr'] = tr.fillna(0.0)
    # optional ATR
    f['atr'] = tr.rolling(14, min_periods=1).mean().fillna(0.0)

    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'vol_z_{w}'] = (v.rolling(w).mean() - v.rolling(max(1, w*3)).mean()).fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)
    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)

# ---------- Sequence builder ----------
def to_sequences(features: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    """Build sequences ending at each index t: [t-seq_len+1, ..., t] -> shape [N, seq_len, F]."""
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

# Public placeholders in session state
if "market_data" not in st.session_state:
    st.session_state.market_data = pd.DataFrame()
if "predictions" not in st.session_state:
    st.session_state.predictions = pd.DataFrame()
if "l1_model" not in st.session_state:
    st.session_state.l1_model = None
if "scaler_seq" not in st.session_state:
    st.session_state.scaler_seq = None

# Fetch button handler
if fetch_btn:
    st.info(f"Fetching {ASSET} bars...")
    df = fetch_market_data(ASSET, start_date, end_date, interval)
    if df is None or df.empty:
        st.error("No market data fetched.")
    else:
        st.success(f"Fetched {len(df)} bars.")
        st.session_state.market_data = df
        st.dataframe(df.tail(10))


# -------------------------
# Chunk 2/3: Level1 model class, robust checkpoint loader, L1-only prediction
# -------------------------
# (Re-declare architecture here; must match training)
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
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        if self.res:
            out = out + x
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
    def embedding_dim(self):
        return int(self.blocks[-1].conv.out_channels)
    def forward(self, x):
        z = self.blocks(x)
        z = self.project(z)
        z_pool = z.mean(dim=-1)
        logit = self.head(z_pool)
        return logit, z_pool

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))
    def forward(self, logits):
        T = torch.exp(self.log_temp)
        return logits / T
    def transform(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            device = next(self.parameters()).device
            logits_t = torch.tensor(logits.reshape(-1,1), dtype=torch.float32, device=device)
            scaled = self.forward(logits_t).cpu().numpy()
        return scaled.reshape(-1)

# ---------- Robust loader for single L1 checkpoint ----------
def load_l1_from_bytes(checkpoint_bytes: bytes, device: str="cpu") -> Tuple[Optional[nn.Module], Optional[StandardScaler], Optional[TemperatureScaler]]:
    """
    Accepts uploaded checkpoint bytes. Handles:
      - state_dict saved via torch.save(model.state_dict())
      - payload dict saved via torch.save(payload) where payload may contain 'model', 'state_dict', 'scaler_seq', etc.
    Returns (model, scaler_seq, temp_scaler)
    """
    device = torch.device(device)
    temp_scaler = None
    scaler_seq = None
    model = None
    stream = io.BytesIO(checkpoint_bytes)

    # Try safe load with allowlisting sklearn StandardScaler (PyTorch 2.6+ safe globals)
    loaded = None
    try:
        # prefer to add StandardScaler to allowed globals to allow payloads that contain pickled scaler objects
        if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
            try:
                torch.serialization.add_safe_globals([StandardScaler])
            except Exception:
                # ignore if not available or already set
                pass
        loaded = torch.load(stream, map_location='cpu')
    except Exception as e:
        # fallback: try direct pickle (some users saved payload via joblib or pickle inside payload)
        logger.warning("torch.load failed, attempting pickle.loads fallback: %s", e)
        try:
            stream.seek(0)
            loaded = pickle.loads(stream.read())
        except Exception as e2:
            logger.exception("Failed to load checkpoint with torch.load and pickle: %s", e2)
            raise RuntimeError(f"Failed to load checkpoint: {e2}")

    # Loaded can be state_dict or a payload dict
    if isinstance(loaded, dict):
        # Case A: direct state_dict of model (keys like 'blocks.0.conv.weight')
        # Heuristic: if many tensor values and conv.weight key exists -> treat as state_dict
        if any(k.startswith("blocks.") for k in loaded.keys()):
            state_dict = loaded
            # infer in_features from first conv weight if present
            first_conv = state_dict.get("blocks.0.conv.weight")
            if first_conv is None:
                # try alternate key name
                for k in state_dict.keys():
                    if ".conv.weight" in k and "blocks.0" in k:
                        first_conv = state_dict[k]
                        break
            if first_conv is None:
                raise RuntimeError("Could not infer in_features from state_dict")
            in_features = first_conv.shape[1]
            model = Level1ScopeCNN(in_features=in_features)
            model.load_state_dict(state_dict)
            model.to(device).eval()
            # no scaler present in this case
            return model, None, temp_scaler
        # Case B: payload wrapper (common pattern: {'model': state_dict, 'scaler_seq': <obj> ...})
        else:
            # common keys
            state_dict = None
            if "model" in loaded and isinstance(loaded["model"], dict):
                state_dict = loaded["model"]
            elif "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                state_dict = loaded["state_dict"]
            elif any(k.startswith("blocks.") for k in loaded.keys()):
                state_dict = loaded
            # get scaler if present
            if "scaler_seq" in loaded:
                sc = loaded["scaler_seq"]
                # sometimes scaler is bytes (pickled) or already object
                if isinstance(sc, (bytes, bytearray)):
                    try:
                        scaler_seq = pickle.loads(sc)
                    except Exception:
                        scaler_seq = sc  # leave as-is
                else:
                    scaler_seq = sc
            # temperature scaler might be present
            if "l1_temp" in loaded and isinstance(loaded["l1_temp"], TemperatureScaler):
                temp_scaler = loaded["l1_temp"]
            # if we have state_dict load model
            if state_dict is not None:
                # infer in_features
                first_conv = state_dict.get("blocks.0.conv.weight")
                if first_conv is None:
                    # try to search keys
                    for k in state_dict.keys():
                        if ".conv.weight" in k and "blocks.0" in k:
                            first_conv = state_dict[k]
                            break
                if first_conv is None:
                    raise RuntimeError("Could not infer in_features from payload state_dict")
                in_features = first_conv.shape[1]
                model = Level1ScopeCNN(in_features=in_features)
                model.load_state_dict(state_dict)
                model.to(device).eval()
                return model, scaler_seq, temp_scaler
    # If we reach here, unsupported format
    raise RuntimeError("Unsupported checkpoint format for L1. Expect a state_dict or payload with 'model'/'state_dict' keys.")

# ---------- L1-only predict function ----------
def predict_l1_all(model: nn.Module, scaler_seq: StandardScaler, df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """
    Predict probabilities for all bars in df using L1 only.
    Returns DataFrame with timestamp, logit, prob.
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    if scaler_seq is None:
        raise RuntimeError("Scaler not loaded")
    if df is None or df.empty:
        return pd.DataFrame()
    eng = compute_engineered_features(df)
    seq_cols = ['open','high','low','close','volume']
    micro_cols = [c for c in ['ret1','tr','vol_5','mom_5','chanpos_10','logret1','rmean_5'] if c in eng.columns]
    # Build combined seq features exactly in same order used during training (we attempt best-effort)
    feat_seq_df = pd.concat([df[seq_cols].astype(float), eng[[c for c in micro_cols if c in eng.columns]]], axis=1, sort=False).fillna(0.0)
    # Validate feature count matches scaler expectation
    X_seq_all = feat_seq_df.values
    if hasattr(scaler_seq, "n_features_in_") and scaler_seq.n_features_in_ != X_seq_all.shape[1]:
        # mismatch — attempt to align by using scaler.feature_names_in_ if available
        if hasattr(scaler_seq, "feature_names_in_"):
            fnames = list(scaler_seq.feature_names_in_)
            try:
                combined = pd.concat([df[seq_cols].astype(float), eng.reindex(columns=fnames[len(seq_cols):])], axis=1, sort=False)
                X_seq_all = combined.values
            except Exception:
                raise RuntimeError(f"Feature mismatch: scaler expects {scaler_seq.n_features_in_} features but found {X_seq_all.shape[1]}. Consider training-compatible scaler or uploading scaler used in training.")
        else:
            raise RuntimeError(f"Feature mismatch: scaler expects {scaler_seq.n_features_in_} features but found {X_seq_all.shape[1]}.")
    # scale
    X_seq_all_scaled = scaler_seq.transform(X_seq_all)
    # build sequences for every index we want to infer (use full range)
    indices = np.arange(len(df))
    Xseq = to_sequences(X_seq_all_scaled, indices, seq_len=seq_len)
    # batch predict
    model.eval()
    logits = []
    batch = 256
    with torch.no_grad():
        for i in range(0, len(Xseq), batch):
            sub = Xseq[i:i+batch]
            xb = torch.tensor(sub.transpose(0,2,1), dtype=torch.float32)  # [B, F, T] expected
            logit, _ = model(xb)
            logits.append(logit.detach().cpu().numpy().reshape(-1))
    logits = np.concatenate(logits, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    out = pd.DataFrame({"timestamp": df.index, "logit": logits, "prob": probs})
    return out


# -------------------------
# Chunk 3/3: limit order generation, UI wiring, running predict/export
# -------------------------
# ---------- Limit order generation ----------
def generate_limit_orders_from_predictions(df: pd.DataFrame, preds: pd.DataFrame,
                                           prob_threshold: float = 0.5,
                                           risk_per_trade: float = 0.02,
                                           account_value: float = 10000.0,
                                           atr_window: int = 14,
                                           sl_mult: float = 1.0,
                                           tp_mult: float = 2.0,
                                           max_lookback: int = 64) -> pd.DataFrame:
    """
    Build limit orders from predictions DataFrame (timestamp, prob).
    For each predicted 'prob >= threshold' in the last `max_lookback` bars, create an order.
    """
    if df is None or df.empty or preds is None or preds.empty:
        return pd.DataFrame()
    # ensure 'atr' available
    eng = compute_engineered_features(df)
    if 'atr' not in eng.columns:
        eng['atr'] = eng['tr'].rolling(atr_window, min_periods=1).mean().fillna(0.0)
    # join preds onto df by timestamp index
    preds_indexed = preds.set_index('timestamp')
    df2 = df.copy()
    df2 = df2.join(preds_indexed[['prob']], how='left')
    df2['prob'] = df2['prob'].fillna(0.0)
    # Only look at last max_lookback bars
    tail = df2.iloc[-max_lookback:].copy()
    orders = []
    for idx, row in tail.iterrows():
        p = float(row['prob'])
        if p < prob_threshold:
            continue
        entry_price = float(row['close'])
        atr = float(eng.loc[idx, 'atr']) if idx in eng.index else float(row.get('atr', row['tr']))
        if atr <= 0 or math.isnan(atr):
            continue
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
        # position sizing
        risk_amount = account_value * risk_per_trade
        if sl_distance <= 0:
            continue
        position_size = risk_amount / sl_distance
        # conviction scaler (map prob [0.5,1.0] -> [0.0,1.0])
        conviction = max(0.0, min(1.0, (p - 0.5) / 0.5))
        adjusted_position_size = position_size * conviction
        if adjusted_position_size <= 0:
            continue
        rr = (tp_distance / sl_distance) if sl_distance>0 else float("inf")
        order = {
            "timestamp": idx,
            "symbol": ASSET,
            "direction": "LONG",
            "entry_price": round(entry_price, 6),
            "tp_price": round(tp_price, 6),
            "sl_price": round(sl_price, 6),
            "position_size": round(adjusted_position_size, 6),
            "risk_amount": round(risk_amount, 2),
            "risk_pct": round(risk_per_trade*100, 3),
            "conviction": round(conviction, 4),
            "prob": round(p, 4),
            "atr": round(atr, 6),
            "rr": round(rr, 3)
        }
        orders.append(order)
    return pd.DataFrame(orders)

# ---------- UI wiring: model load / predict / orders ----------
# Load checkpoint if uploaded
if checkpoint_file is not None:
    try:
        raw_bytes = checkpoint_file.read()
        st.info("Loading checkpoint (L1) — this may take a moment.")
        model, scaler_seq, temp_scaler = load_l1_from_bytes(raw_bytes, device="cpu")
        st.session_state.l1_model = model
        st.session_state.scaler_seq = scaler_seq
        # if scaler present, show shape
        if scaler_seq is not None and hasattr(scaler_seq, "n_features_in_"):
            st.success(f"L1 model loaded. Scaler expects {scaler_seq.n_features_in_} features.")
        else:
            st.success("L1 model loaded. No scaler found in checkpoint; you must upload training scaler or ensure your feature matrix matches the model input.")
    except Exception as e:
        st.error(f"Failed to load L1 checkpoint: {e}")
        st.error(traceback.format_exc())
        st.session_state.l1_model = None
        st.session_state.scaler_seq = None

# Predict button
if predict_btn:
    try:
        model = st.session_state.l1_model
        scaler = st.session_state.scaler_seq
        df = st.session_state.market_data
        if model is None:
            st.error("L1 model not loaded. Upload checkpoint first.")
        elif scaler is None:
            st.error("Scaler not loaded. The checkpoint did not include a scaler. Upload the scaler used in training or provide a checkpoint that includes it.")
        elif df is None or df.empty:
            st.error("No market data present. Fetch market data first.")
        else:
            with st.spinner("Running L1 predictions for all bars..."):
                preds = predict_l1_all(model, scaler, df, seq_len=seq_len)
                st.session_state.predictions = preds
                st.success(f"Predicted {len(preds)} bars.")
                st.dataframe(preds.tail(20))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        logger.exception("Prediction failed: %s", e)

# Orders button: use last predictions
if orders_btn:
    try:
        df = st.session_state.market_data
        preds = st.session_state.predictions
        if preds is None or preds.empty:
            st.error("No predictions available. Run Predict first.")
        elif df is None or df.empty:
            st.error("No market data available.")
        else:
            with st.spinner("Generating limit orders from predictions..."):
                orders_df = generate_limit_orders_from_predictions(
                    df=df,
                    preds=preds,
                    prob_threshold=0.5,
                    risk_per_trade=risk_per_trade_pct,
                    account_value=account_balance,
                    atr_window=14,
                    sl_mult=atr_mult_sl,
                    tp_mult=atr_mult_tp,
                    max_lookback=seq_len
                )
                if orders_df.empty:
                    st.warning("No orders generated above threshold in the lookback window.")
                else:
                    st.session_state.limit_orders = orders_df
                    st.success(f"Generated {len(orders_df)} orders.")
                    st.dataframe(orders_df)
    except Exception as e:
        st.error(f"Order generation failed: {e}")
        logger.exception("Order generation failed: %s", e)

# Small helpers / download buttons
if "predictions" in st.session_state and not st.session_state.predictions.empty:
    st.download_button("Download predictions CSV", data=st.session_state.predictions.to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{ASSET}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv")

if "limit_orders" in st.session_state and st.session_state.limit_orders is not None and not st.session_state.limit_orders.empty:
    st.download_button("Download orders CSV", data=st.session_state.limit_orders.to_csv(index=False).encode("utf-8"),
                       file_name=f"orders_{ASSET}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv")

st.info("L1-only inference ready. Upload L1 checkpoint (state_dict or payload), fetch market data, then press Predict → Generate Limit Orders.")
