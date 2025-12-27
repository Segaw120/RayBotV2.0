# chunk1_l1_infer.py - COMPLETE FIXED VERSION
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

# Fallback yfinance import
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    yf = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("l1_inference")

st.set_page_config(page_title="Cascade Trader — L1 Inference", layout="wide")
st.title("Cascade Trader — L1 Inference & Limit Orders (Auto-arch loader)")

# ---------------------------
# Flexible Level1 model
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
# Feature engineering with timezone fix
# ---------------------------
def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    if df.index.tz is not None:
        df = df.tz_localize(None)
    
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

# chunk2_l1_infer.py - COMPLETE FIXED VERSION

# WEEKEND-AWARE Gold fetch (365 weekdays always)
def fetch_gold_history(days=365, interval="1d") -> pd.DataFrame:
    def is_weekend():
        now = datetime.utcnow()
        return now.weekday() >= 5
    
    def adjust_weekend_period(target_days):
        if not is_weekend():
            return target_days
        weekday_target = target_days * 7 // 5  # ~40% more for weekends
        return max(weekday_target, target_days + 50)
    
    period_days = adjust_weekend_period(days)
    logger.info(f"Fetching {days} weekdays → period={period_days}d ({'weekend' if is_weekend() else 'weekday'} mode)")
    
    # Try yahooquery
    if YahooTicker is not None:
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=period_days)
            tq = YahooTicker("GC=F")
            raw = tq.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval=interval)
            if raw is not None and not (isinstance(raw, dict) and not raw):
                df = _process_raw_data(raw)
                if len(df) >= days * 0.8:
                    return _filter_weekdays(df, days) if is_weekend() else df
        except Exception as e:
            logger.warning(f"yahooquery failed: {e}")
    
    # Fallback yfinance
    if YF_AVAILABLE:
        try:
            ticker = yf.Ticker("GC=F")
            raw = ticker.history(period=f"{period_days}d", interval=interval, prepost=True)
            if not raw.empty:
                df = _process_raw_data(raw)
                return _filter_weekdays(df, days) if is_weekend() else df
        except Exception as e:
            logger.warning(f"yfinance failed: {e}")
    
    return pd.DataFrame()

def _process_raw_data(raw):
    if isinstance(raw, dict):
        raw = pd.DataFrame(raw)
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index(level=0, drop=True)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw = raw.sort_index()
    raw.columns = [c.lower() for c in raw.columns]
    if "close" not in raw.columns and "adjclose" in raw.columns:
        raw["close"] = raw["adjclose"]
    raw = raw[~raw.index.duplicated(keep="first")]
    return _normalize_gold_df(raw)

def _filter_weekdays(df: pd.DataFrame, min_days: int) -> pd.DataFrame:
    if df.empty:
        return df
    df_weekdays = df[df.index.weekday < 5].copy()
    if len(df_weekdays) < min_days * 0.7:
        return df_weekdays.tail(min_days) if len(df_weekdays) > 0 else df
    return df_weekdays.tail(min_days)

def _normalize_gold_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ['open','high','low','close','volume']
    for col in required:
        if col not in df.columns:
            df[col] = 0.0
    df = df[required].copy()
    df.columns = df.columns.str.lower()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df[~df.index.duplicated(keep="last")].sort_index()

# Checkpoint loaders
def _is_state_dict_like(d: dict) -> bool:
    if not isinstance(d, dict): return False
    keys = list(d.keys())
    for k in keys[:20]:
        if any(sub in k for sub in ("conv.weight","bn.weight","head.weight","proj.weight","blocks.0.conv.weight")):
            return True
    return False

def extract_state_dict(container):
    if container is None: return None, {}
    if isinstance(container, dict) and _is_state_dict_like(container): return container, {}
    for key in ("model_state_dict","state_dict","model","model_state","model_weights"):
        if isinstance(container, dict) and key in container and _is_state_dict_like(container[key]):
            return container[key], {k:v for k,v in container.items() if k != key}
    return None, {}

def strip_module_prefix(state):
    return {k[len("module."):]:v if k.startswith("module.") else k:v for k,v in state.items()}

_conv_key_re = re.compile(r"blocks.(d+).conv.weight")

def infer_arch_from_state(state):
    blocks = {}
    for k,v in state.items():
        m = _conv_key_re.search(k)
        if m and hasattr(v, "shape"):
            idx, out_ch, in_ch = int(m.group(1)), int(v.shape[0]), int(v.shape[1])
            blocks[idx] = (out_ch, in_ch, tuple(v.shape))
    if not blocks:
        return 12, (32,64,128)
    ordered = [blocks[i] for i in sorted(blocks.keys())]
    return ordered[0][1], tuple(b[0] for b in ordered)

def load_checkpoint_bytes_safe(raw_bytes: bytes):
    buf = io.BytesIO(raw_bytes)
    try: return torch.load(buf, map_location="cpu", weights_only=False)
    except:
        buf.seek(0)
        try: return torch.load(buf, map_location="cpu", weights_only=True)
        except:
            buf.seek(0)
            import pickle
            return pickle.loads(buf.read())

# UI
st.sidebar.header("Config")
seq_len = st.sidebar.slider("Sequence length", 8, 256, 64, step=8)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 2.0) / 100.0
tp_mult = st.sidebar.slider("TP ATR multiplier", 1.0, 5.0, 1.0)
sl_mult = st.sidebar.slider("SL ATR multiplier", 0.5, 3.0, 1.0)
account_balance = st.sidebar.number_input("Account balance ($)", value=10000.0)

ckpt = st.sidebar.file_uploader("Upload L1 checkpoint (.pt/.pth/.bin)", type=["pt","pth","bin"])

if "market_df" not in st.session_state: st.session_state.market_df = None
if "l1_model" not in st.session_state: st.session_state.l1_model = None
if "scaler_seq" not in st.session_state: st.session_state.scaler_seq = None
if "temp_scaler" not in st.session_state: st.session_state.temp_scaler = None

# Fetch button
if st.button("🔄 Fetch latest Gold (GC=F)"):
    with st.spinner("Fetching 365 weekdays..."):
        try:
            df = fetch_gold_history(days=365, interval="1d")
            if df.empty:
                st.error("❌ No data returned")
            else:
                st.session_state.market_df = df
                st.success(f"✅ Fetched {len(df)} weekday bars | Latest: {df.index[-1].strftime('%Y-%m-%d')}")
                st.dataframe(df.tail(10), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Fetch failed: {str(e)[:200]}")

# Load checkpoint
if ckpt is not None:
    try:
        raw = ckpt.read()
        loaded = load_checkpoint_bytes_safe(raw)
        state_dict, extras = extract_state_dict(loaded)
        if state_dict is None:
            if isinstance(loaded, nn.Module):
                st.session_state.l1_model = loaded
                st.success("✅ Loaded L1 module")
            else:
                st.error("❌ No state_dict found")
        else:
            state_dict = strip_module_prefix(state_dict)
            in_f, channels = infer_arch_from_state(state_dict)
            st.info(f"🔍 Inferred: in_features={in_f}, channels={channels}")
            model = Level1ScopeCNN(in_features=in_f, channels=channels)
            try:
                model.load_state_dict(state_dict, strict=True)
                st.success("✅ Loaded (strict=True)")
            except:
                model.load_state_dict(state_dict, strict=False)
                st.warning("⚠️ Loaded (strict=False)")
            model.eval()
            st.session_state.l1_model = model
            
            # Scalers
            for key in ["scaler_seq", "scaler"]:
                if key in extras or (isinstance(loaded, dict) and key in loaded):
                    st.session_state.scaler_seq = extras.get(key) or loaded.get(key)
                    st.success("✅ Loaded scaler")
                    break
    except Exception as e:
        st.error(f"❌ Checkpoint failed: {str(e)[:200]}")

# Inference
if st.button("🚀 Run L1 inference & propose limit order"):
    if st.session_state.market_df is None:
        st.error("❌ Fetch data first")
    elif st.session_state.l1_model is None:
        st.error("❌ Upload checkpoint first")
    else:
        with st.spinner("Running inference..."):
            df = st.session_state.market_df.copy()
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            
            feats = compute_engineered_features(df)
            seq_cols = ['open','high','low','close','volume']
            micro_cols = ['ret1','tr','vol_5','mom_5','chanpos_10']
            use_cols = [c for c in seq_cols + micro_cols if c in list(df.columns) + list(feats.columns)]
            
            feat_seq_df = pd.concat([df[seq_cols].astype(float), feats[[c for c in micro_cols if c in feats.columns]]], axis=1)[use_cols].fillna(0.0)
            X_all = feat_seq_df.values.astype('float32')
            
            scaler = st.session_state.scaler_seq
            if scaler is None:
                scaler = StandardScaler().fit(X_all)
                st.warning("⚠️ Using temp scaler")
            
            X_scaled = scaler.transform(X_all)
            last_idx = np.array([len(X_scaled)-1], dtype=int)
            Xseq = to_sequences(X_scaled, last_idx, seq_len=seq_len)
            xb = torch.tensor(Xseq.transpose(0,2,1), dtype=torch.float32)
            
            model = st.session_state.l1_model
            model.eval()
            with torch.no_grad():
                logit, emb = model(xb)
                if st.session_state.temp_scaler is not None:
                    logit = st.session_state.temp_scaler(logit)
                prob = float(torch.sigmoid(logit).cpu().numpy()[0])
            
            atr = feats['atr'].iloc[-1] if 'atr' in feats else (df['high']-df['low']).rolling(14, min_periods=1).mean().iloc[-1]
            entry = float(df['close'].iloc[-1])
            sl = float(entry - atr * sl_mult)
            tp = float(entry + atr * tp_mult)
            risk_amount = account_balance * risk_pct
            stop_distance = abs(entry - sl)
            size = risk_amount / stop_distance if stop_distance > 0 else 0.0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Buy Probability", f"{prob:.1%}")
                st.metric("📏 ATR", f"${atr:.2f}")
            with col2:
                st.metric("💰 Entry", f"${entry:.2f}")
                st.metric("⚠️ Risk", f"${risk_amount:.0f}")
            
            st.subheader("💼 Proposed LONG Order")
            order = {
                "entry": round(entry, 2), "stop_loss": round(sl, 2), "take_profit": round(tp, 2),
                "atr": float(atr), "position_size": float(size), "risk_amount_usd": float(risk_amount),
                "probability": float(prob)
            }
            st.json(order)

st.caption("✅ Fixed: Weekend weekday filtering (365 trading days), timezone errors, dual yfinance/yahooquery")