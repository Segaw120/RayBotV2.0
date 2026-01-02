import io
import re
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("l1_inference")

st.set_page_config(page_title="Cascade Trader — L1 Inference", layout="wide")
st.title("Cascade Trader — L1 Inference & Limit Orders (Auto-arch loader)")

# ---------------------------
# Flexible Level1 model (accepts channels tuple)
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, pdrop=0.1):
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
        return out + x if self.res else out

class Level1ScopeCNN(nn.Module):
    def __init__(self, in_features=12, channels=(32,64,128), kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1):
        super().__init__()
        chs = [in_features] + list(channels)
        blocks = []
        for i in range(len(channels)):
            k = kernel_sizes[min(i, len(kernel_sizes)-1)]
            d = dilations[min(i, len(dilations)-1)]
            blocks.append(ConvBlock(chs[i], chs[i+1], k=k, d=d, pdrop=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
        self.head = nn.Linear(chs[-1], 1)
    @property
    def embedding_dim(self):
        return int(self.blocks[-1].conv.out_channels)
    def forward(self, x):
        z = self.blocks(x)
        z = self.proj(z)
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
    def load_state(self, st_dict):
        try:
            self.load_state_dict(st_dict)
        except Exception:
            logger.warning("Temp scaler load failed.")

# ---------------------------
# Feature engineering
# ---------------------------
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

# ---------------------------
# UPDATED: Gold fetch helper (Ethiopia Alignment)
# ---------------------------
def fetch_gold_history(days=365, interval="1d") -> pd.DataFrame:
    if YahooTicker is None:
        raise RuntimeError("yahooquery not installed")
    
    # 1. Calculate time window
    now_utc = datetime.utcnow()
    # Align to Ethiopian time (UTC+3)
    now_ethiopia = now_utc + timedelta(hours=3)

    # If daily interval, choose the latest *complete* daily date (use yesterday in ET)
    if str(interval).lower() in ("1d", "daily", "d"):
        latest_complete_date = (now_ethiopia.date() - timedelta(days=1))
        end = datetime.combine(latest_complete_date, datetime.min.time())
        start = end - timedelta(days=days)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
    else:
        # For intraday intervals we keep the original behaviour (use now UTC)
        end = now_utc
        start = end - timedelta(days=days)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

    # Safety: ensure start is not after end
    if start > end:
        start = end - timedelta(days=days)
        start_str = start.strftime("%Y-%m-%d")

    logger.info(f"Fetching data from {start_str} to {end_str}")

    # 2. API Call (Yahoo Finance via yahooquery)
    tq = YahooTicker("GC=F")
    raw = tq.history(start=start_str, end=end_str, interval=interval)

    if raw is None or (isinstance(raw, dict) and not raw):
        logger.warning("No data returned from Yahoo Finance")
        return pd.DataFrame()

    if isinstance(raw, dict):
        raw = pd.DataFrame(raw)
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index(level=0, drop=True)

    # 3. Clean and Standardize
    # Robustly parse/normalize the index and remove timezones, then convert to ET (+3)
    dt_index = pd.to_datetime(raw.index)
    
    if getattr(dt_index, "tz", None) is not None:
        try:
            dt_index = dt_index.tz_convert("UTC").tz_localize(None)
        except Exception:
            dt_index = dt_index.tz_localize(None)
            
    # Shift timestamps from UTC to UTC+3 (Ethiopian time alignment)
    dt_index = dt_index + pd.Timedelta(hours=3)
    raw.index = dt_index

    raw.columns = [c.lower() for c in raw.columns]
    if "close" not in raw.columns and "adjclose" in raw.columns:
        raw["close"] = raw["adjclose"]
    
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()
    
    # Ensure all required columns exist
    required = ['open','high','low','close','volume']
    for col in required:
        if col not in raw.columns:
            raw[col] = 0.0

    logger.info(f"Fetched {len(raw)} records aligned to ET")
    return raw[required]

# ---------------------------
# Checkpoint robust loader helpers
# ---------------------------
def _is_state_dict_like(d: dict) -> bool:
    if not isinstance(d, dict): return False
    keys = list(d.keys())
    for k in keys[:20]:
        if any(sub in k for sub in ("conv.weight","bn.weight","head.weight","proj.weight")):
            return True
    return False

def extract_state_dict(container):
    if container is None: return None, {}
    if isinstance(container, dict) and _is_state_dict_like(container):
        return container, {}
    for key in ("model_state_dict","state_dict","model","model_weights"):
        if isinstance(container, dict) and key in container and _is_state_dict_like(container[key]):
            return container[key], {k:v for k,v in container.items() if k != key}
    return None, {}

def strip_module_prefix(state):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

def infer_arch_from_state(state):
    blocks = {}
    conv_re = re.compile(r"blocks\.(\d+)\.conv\.weight")
    for k,v in state.items():
        m = conv_re.search(k)
        if m and hasattr(v, "shape"):
            idx = int(m.group(1))
            blocks[idx] = (int(v.shape[0]), int(v.shape[1]))
    if not blocks: return None, None
    ordered = [blocks[i] for i in sorted(blocks.keys())]
    return ordered[0][1], tuple(b[0] for b in ordered)

def load_checkpoint_bytes_safe(raw_bytes: bytes):
    buf = io.BytesIO(raw_bytes)
    try:
        return torch.load(buf, map_location="cpu", weights_only=False)
    except Exception:
        buf.seek(0)
        return torch.load(buf, map_location="cpu", weights_only=True)

# ---------------------------
# Streamlit UI Logic
# ---------------------------
st.sidebar.header("Config")
seq_len = st.sidebar.slider("Sequence length", 8, 256, 64, step=8)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 2.0) / 100.0
tp_mult = st.sidebar.slider("TP ATR multiplier", 1.0, 5.0, 1.0)
sl_mult = st.sidebar.slider("SL ATR multiplier", 0.5, 3.0, 1.0)
account_balance = st.sidebar.number_input("Account balance ($)", value=10000.0)

ckpt = st.sidebar.file_uploader("Upload L1 checkpoint", type=["pt","pth","bin"])

if "market_df" not in st.session_state: st.session_state.market_df = None
if "l1_model" not in st.session_state: st.session_state.l1_model = None
if "scaler_seq" not in st.session_state: st.session_state.scaler_seq = None
if "temp_scaler" not in st.session_state: st.session_state.temp_scaler = None

if st.button("Fetch latest Gold (GC=F) - ET Aligned"):
    try:
        df = fetch_gold_history(days=365, interval="1d")
        if df.empty:
            st.error("No data returned")
        else:
            st.session_state.market_df = df
            st.success(f"Fetched {len(df)} bars. Last index: {df.index[-1]}")
            st.dataframe(df.tail(10))
    except Exception as e:
        st.error(f"Fetch failed: {e}")

if ckpt is not None:
    try:
        raw = ckpt.read()
        loaded = load_checkpoint_bytes_safe(raw)
        state_dict, extras = extract_state_dict(loaded)
        if state_dict:
            state_dict = strip_module_prefix(state_dict)
            in_f, chs = infer_arch_from_state(state_dict)
            model = Level1ScopeCNN(in_features=in_f or 12, channels=chs or (32,64,128))
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            st.session_state.l1_model = model
            st.session_state.scaler_seq = extras.get("scaler_seq") or loaded.get("scaler_seq")
            st.success(f"Model loaded (Inferred In: {in_f})")
    except Exception as e:
        st.error(f"Load failed: {e}")

if st.button("Run L1 inference"):
    if st.session_state.market_df is None or st.session_state.l1_model is None:
        st.error("Data or Model missing.")
    else:
        df = st.session_state.market_df.copy()
        feats = compute_engineered_features(df)
        cols = ['open','high','low','close','volume','ret1','tr','vol_5','mom_5','chanpos_10']
        data_to_scale = pd.concat([df, feats], axis=1)[[c for c in cols if c in df.columns or c in feats.columns]].fillna(0.0)
        
        scaler = st.session_state.scaler_seq or StandardScaler().fit(data_to_scale.values)
        X_scaled = scaler.transform(data_to_scale.values)
        Xseq = to_sequences(X_scaled, np.array([len(X_scaled)-1]), seq_len)
        xb = torch.tensor(Xseq.transpose(0,2,1), dtype=torch.float32)
        
        with torch.no_grad():
            logit, _ = st.session_state.l1_model(xb)
            prob = float(torch.sigmoid(logit).item())
        
        atr = feats['atr'].iloc[-1]
        entry = float(df['close'].iloc[-1])
        st.metric("Buy Probability", f"{prob:.4%}")
        
        st.json({
            "entry": round(entry, 2),
            "sl": round(entry - atr * sl_mult, 2),
            "tp": round(entry + atr * tp_mult, 2),
            "size": (account_balance * risk_pct) / (atr * sl_mult)
        })
