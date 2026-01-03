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

# Import gold futures module
from gold_futures_data import fetch_365d_gold_prices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("l1_inference")

st.set_page_config(page_title="Cascade Trader — L1 Inference", layout="wide")
st.title("🚀 Cascade Trader — L1 Inference & Limit Orders")

# ---------------------------
# Neural Network Models
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

# ---------------------------
# Feature Engineering (Training Pipeline Match)
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
# Checkpoint Robust Loaders
# ---------------------------
def _is_state_dict_like(d: dict) -> bool:
    if not isinstance(d, dict): return False
    keys = list(d.keys())
    for k in keys[:20]:
        if any(sub in k for sub in ("conv.weight","bn.weight","head.weight","proj.weight","blocks.0.conv.weight")):
            return True
    vals = list(d.values())[:10]
    if all(isinstance(v, (torch.Tensor, np.ndarray)) for v in vals): return True
    return False

def extract_state_dict(container):
    if container is None: return None, {}
    if isinstance(container, dict) and _is_state_dict_like(container): return container, {}
    
    for key in ("model_state_dict","state_dict","model","model_state","model_weights"):
        if isinstance(container, dict) and key in container and _is_state_dict_like(container[key]):
            extras = {k:v for k,v in container.items() if k != key}
            return container[key], extras
    
    if isinstance(container, dict):
        for k,v in container.items():
            if isinstance(v, dict) and _is_state_dict_like(v):
                extras = {kk:vv for kk,vv in container.items() if kk != k}
                return v, extras
    return None, {}

def strip_module_prefix(state):
    new = {}
    for k,v in state.items():
        nk = k if not k.startswith("module.") else k[len("module."):]
        new[nk] = v
    return new

_conv_key_re = re.compile(r"blocks.(d+).conv.weight")

def infer_arch_from_state(state):
    blocks = {}
    for k,v in state.items():
        m = _conv_key_re.search(k)
        if m and hasattr(v, "shape"):
            idx = int(m.group(1))
            out_ch, in_ch = int(v.shape[0]), int(v.shape[1])
            blocks[idx] = (out_ch, in_ch, tuple(v.shape))
    
    if blocks:
        ordered = [blocks[i] for i in sorted(blocks.keys())]
        return ordered[0][1], tuple(b[0] for b in ordered)
    return 12, (32,64,128)

def load_checkpoint_bytes_safe(raw_bytes: bytes):
    buf = io.BytesIO(raw_bytes)
    try: 
        return torch.load(buf, map_location="cpu", weights_only=False)
    except:
        buf.seek(0)
        try: 
            return torch.load(buf, map_location="cpu", weights_only=True)
        except:
            buf.seek(0)
            import pickle
            return pickle.loads(buf.read())

# ---------------------------
# UI Configuration & Session State
# ---------------------------
st.sidebar.header("⚙️ Config")
seq_len = st.sidebar.slider("Sequence length", 8, 256, 64, step=8)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 2.0) / 100
tp_mult = st.sidebar.slider("TP ATR multiplier", 1.0, 5.0, 1.0)
sl_mult = st.sidebar.slider("SL ATR multiplier", 0.5, 3.0, 1.0)
account_balance = st.sidebar.number_input("Account balance ($)", value=10000.0)

ckpt = st.sidebar.file_uploader("Upload L1 checkpoint (.pt/.pth/.bin)", type=["pt","pth","bin"])

# Initialize session state
for key in ["market_df", "l1_model", "scaler_seq", "temp_scaler"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------
# Data Fetching with Gold Module
# ---------------------------
if st.button("🔄 Fetch 365d Gold Futures (Yesterday Complete)", use_container_width=True):
    try:
        with st.spinner("Fetching latest complete gold data (UTC+3)..."):
            df = fetch_365d_gold_prices()
        
        if df.empty:
            st.error("❌ No data returned from gold_futures_data module")
        else:
            st.session_state.market_df = df
            latest_date = df.index.max().strftime('%Y-%m-%d %H:%M UTC+3')
            st.success(f"✅ Fetched {len(df)} complete daily bars
Latest: {latest_date}")
            st.dataframe(df[['open','high','low','close','volume']].tail(10), 
                        use_container_width=True)
    except Exception as e:
        st.error(f"❌ Fetch failed: {str(e)}")
        logger.error(f"Gold fetch error: {e}")

# ---------------------------
# Checkpoint Loading
# ---------------------------
if ckpt is not None:
    try:
        raw = ckpt.read()
        loaded = load_checkpoint_bytes_safe(raw)
        state_dict, extras = extract_state_dict(loaded)
        
        if state_dict is None:
            if isinstance(loaded, nn.Module):
                st.session_state.l1_model = loaded
                st.success("✅ Loaded L1 model directly")
            else:
                st.error("❌ No valid state_dict found in checkpoint")
        else:
            state_dict = strip_module_prefix(state_dict)
            in_features, channels = infer_arch_from_state(state_dict)
            
            model = Level1ScopeCNN(in_features=in_features, channels=channels)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            st.session_state.l1_model = model
            st.success(f"✅ Model loaded: in_features={in_features}, channels={channels}")
            
            # Try to load scaler
            scaler_key = "scaler_seq" if "scaler_seq" in extras else "scaler"
            if scaler_key in extras:
                st.session_state.scaler_seq = extras[scaler_key]
                st.success("✅ Scaler loaded")
    
    except Exception as e:
        st.error(f"❌ Checkpoint load failed: {e}")

# ---------------------------
# L1 Inference & Trading Proposal
# ---------------------------
if st.button("🚀 Run L1 Inference & Propose Limit Order", type="primary", use_container_width=True):
    if not st.session_state.market_df:
        st.error("❌ No market data. Fetch gold data first.")
    elif not st.session_state.l1_model:
        st.error("❌ No model loaded. Upload checkpoint first.")
    else:
        with st.spinner("Running L1 inference pipeline..."):
            df = st.session_state.market_df.copy()
            feats = compute_engineered_features(df)
            
            # Feature selection (OHLCV + key engineered features)
            seq_cols = ['open','high','low','close','volume']
            micro_cols = ['ret1','tr','vol_5','mom_5','chanpos_10']
            use_cols = [c for c in seq_cols + micro_cols 
                       if c in list(df.columns) + list(feats.columns)]
            
            feat_seq_df = pd.concat([
                df[seq_cols].astype(float), 
                feats[[c for c in micro_cols if c in feats.columns]]
            ], axis=1)[use_cols].fillna(0.0)
            
            X_all = feat_seq_df.values.astype('float32')
            scaler = st.session_state.scaler_seq
            
            if scaler is None:
                st.warning("⚠️ No scaler found; fitting temporary scaler")
                scaler = StandardScaler().fit(X_all)
            
            X_scaled = scaler.transform(X_all)
            last_idx = np.array([len(X_scaled)-1], dtype=int)
            Xseq = to_sequences(X_scaled, last_idx, seq_len)
            xb = torch.tensor(Xseq.transpose(0,2,1), dtype=torch.float32)
            
            # Model inference
            model = st.session_state.l1_model
            model.eval()
            with torch.no_grad():
                logit, embedding = model(xb)
                prob = float(torch.sigmoid(logit).cpu().numpy().reshape(-1)[0])
        
        # Trading calculations
        atr = feats['atr'].iloc[-1]
        entry = float(df['close'].iloc[-1])
        sl_price = entry - atr * sl_mult
        tp_price = entry + atr * tp_mult
        risk_amount = account_balance * risk_pct
        stop_distance = abs(entry - sl_price)
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0.0
        
        # Results
        col1, col2 = st.columns(2)
        col1.success(f"🎯 **L1 Buy Probability**
{prob:.4f}")
        col2.metric("Latest Close", f"${entry:.2f}", delta=f"{atr:.2f} ATR")
        
        st.subheader("📈 Proposed LONG Limit Order")
        st.json({
            "entry_price": round(entry, 2),
            "stop_loss": round(sl_price, 2),
            "take_profit": round(tp_price, 2),
            "atr_current": round(float(atr), 2),
            "position_size": round(float(position_size), 4),
            "risk_amount_usd": round(float(risk_amount), 2),
            "buy_probability": round(float(prob), 4),
            "latest_close_time": df.index[-1].strftime('%Y-%m-%d %H:%M UTC+3'),
            "data_points": len(df)
        })

st.caption("✅ Powered by gold_futures_data.py module | 365 days ending yesterday (complete daily closes)")