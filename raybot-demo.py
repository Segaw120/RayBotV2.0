# chunk1_l1_infer.py - FIXED VERSION
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
# Flexible Level1 model (unchanged)
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
# FIXED Feature engineering with timezone handling
# ---------------------------
def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    # Normalize timezone first
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

# ---------------------------
# FIXED Gold fetch with yfinance fallback + weekend handling
# ---------------------------
def fetch_gold_history(days=365, interval="1d") -> pd.DataFrame:
    """Robust gold fetch with yfinance fallback and timezone normalization"""
    
    # Try yahooquery first
    if YahooTicker is not None:
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=days)
            tq = YahooTicker("GC=F")
            raw = tq.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval=interval)
            if raw is not None and not (isinstance(raw, dict) and not raw):
                if isinstance(raw, dict):
                    raw = pd.DataFrame(raw)
                if isinstance(raw.index, pd.MultiIndex):
                    raw = raw.reset_index(level=0, drop=True)
                raw.index = pd.to_datetime(raw.index).tz_localize(None)  # FIX: Remove timezone
                raw = raw.sort_index()
                raw.columns = [c.lower() for c in raw.columns]
                if "close" not in raw.columns and "adjclose" in raw.columns:
                    raw["close"] = raw["adjclose"]
                raw = raw[~raw.index.duplicated(keep="first")]
                if len(raw) > 0:
                    return _normalize_gold_df(raw)
        except Exception as e:
            logger.warning(f"yahooquery failed: {e}")
    
    # Fallback to yfinance (more reliable on weekends)
    if YF_AVAILABLE:
        try:
            ticker = yf.Ticker("GC=F")
            # Use period instead of start/end for weekend robustness
            period = min(days//30 + 1, 2*days//30)  # 2y max for stability
            raw = ticker.history(period=f"{period}d", interval=interval, prepost=True)
            if not raw.empty:
                raw.index = pd.to_datetime(raw.index).tz_localize(None)  # FIX: Remove timezone
                return _normalize_gold_df(raw)
        except Exception as e:
            logger.warning(f"yfinance failed: {e}")
    
    st.warning("Both yahooquery & yfinance failed. Markets may be closed (weekend). Try weekdays.")
    return pd.DataFrame()

def _normalize_gold_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV dataframe with timezone safety"""
    required = ['open','high','low','close','volume']
    for col in required:
        if col not in df.columns:
            df[col] = 0.0
    df = df[required].copy()
    df.columns = df.columns.str.lower()
    # Ensure numeric and drop NaNs
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    # Remove exact duplicates and sort
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

# ---------------------------
# Checkpoint loaders (unchanged but with better error handling)
# ---------------------------
def _is_state_dict_like(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    keys = list(d.keys())
    for k in keys[:20]:
        if any(sub in k for sub in ("conv.weight","bn.weight","head.weight","proj.weight","blocks.0.conv.weight")):
            return True
    vals = list(d.values())[:10]
    if all(isinstance(v, (torch.Tensor, np.ndarray)) for v in vals):
        return True
    return False

def extract_state_dict(container):
    if container is None:
        return None, {}
    if isinstance(container, dict) and _is_state_dict_like(container):
        return container, {}
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
        nk = k
        if k.startswith("module."):
            nk = k[len("module."):]
        new[nk] = v
    return new

_conv_key_re = re.compile(r"blocks.(d+).conv.weight")

def infer_arch_from_state(state):
    blocks = {}
    for k,v in state.items():
        m = _conv_key_re.search(k)
        if m and hasattr(v, "shape"):
            idx = int(m.group(1))
            out_ch = int(v.shape[0])
            in_ch = int(v.shape[1])
            blocks[idx] = (out_ch, in_ch, tuple(v.shape))
    if not blocks:
        for k,v in state.items():
            if ".conv.weight" in k and hasattr(v, "shape"):
                parts = k.split(".")
                try:
                    idx = int(parts[1]) if parts[0]=='blocks' else 0
                except Exception:
                    idx = 0
                out_ch = int(v.shape[0]); in_ch = int(v.shape[1])
                blocks[idx] = (out_ch, in_ch, tuple(v.shape))
    if not blocks:
        return 12, (32,64,128)
    ordered = [blocks[i] for i in sorted(blocks.keys())]
    channels = [b[0] for b in ordered]
    in_features = ordered[0][1]
    return int(in_features), tuple(channels)

def load_checkpoint_bytes_safe(raw_bytes: bytes):
    buf = io.BytesIO(raw_bytes)
    try:
        obj = torch.load(buf, map_location="cpu", weights_only=False)
        return obj
    except Exception:
        buf.seek(0)
        try:
            obj = torch.load(buf, map_location="cpu", weights_only=True)
            return obj
        except Exception:
            buf.seek(0)
            import pickle
            try:
                obj = pickle.loads(buf.read())
                return obj
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint: {e}")

# chunk2_l1_infer.py - FIXED VERSION (continue after chunk1)
# Sidebar configuration
st.sidebar.header("Config")
seq_len = st.sidebar.slider("Sequence length", 8, 256, 64, step=8)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 2.0) / 100.0
tp_mult = st.sidebar.slider("TP ATR multiplier", 1.0, 5.0, 1.0)
sl_mult = st.sidebar.slider("SL ATR multiplier", 0.5, 3.0, 1.0)
account_balance = st.sidebar.number_input("Account balance ($)", value=10000.0)

ckpt = st.sidebar.file_uploader("Upload L1 checkpoint (.pt/.pth/.bin)", type=["pt","pth","bin"])

# session state
if "market_df" not in st.session_state:
    st.session_state.market_df = None
if "l1_model" not in st.session_state:
    st.session_state.l1_model = None
if "scaler_seq" not in st.session_state:
    st.session_state.scaler_seq = None
if "temp_scaler" not in st.session_state:
    st.session_state.temp_scaler = None

# FIXED: Fetch Gold data with better weekend handling
if st.button("🔄 Fetch latest Gold (GC=F)"):
    with st.spinner("Fetching GC=F data... (works on weekends)"):
        try:
            df = fetch_gold_history(days=365, interval="1d")
            if df.empty:
                st.error("❌ No data returned. Try weekdays or check internet.")
            else:
                st.session_state.market_df = df
                st.success(f"✅ Fetched {len(df)} bars | Latest: {df.index[-1].strftime('%Y-%m-%d')}")
                st.dataframe(df.tail(10), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Fetch failed: {str(e)[:200]}")

# Load checkpoint (improved error messages)
if ckpt is not None:
    try:
        raw = ckpt.read()
        loaded = load_checkpoint_bytes_safe(raw)
        state_dict, extras = extract_state_dict(loaded)
        if state_dict is None:
            if isinstance(loaded, nn.Module):
                st.session_state.l1_model = loaded
                st.success("✅ Loaded L1 as module object")
            else:
                st.error("❌ No state_dict found. Save with torch.save(model.state_dict(), 'model.pt')")
        else:
            state_dict = strip_module_prefix(state_dict)
            inferred_in, inferred_channels = infer_arch_from_state(state_dict)
            st.info(f"🔍 Inferred: in_features={inferred_in}, channels={inferred_channels}")
            
            model = Level1ScopeCNN(in_features=inferred_in, channels=inferred_channels)
            try:
                model.load_state_dict(state_dict, strict=True)
                st.success("✅ Loaded state_dict (strict=True)")
            except:
                model.load_state_dict(state_dict, strict=False)
                st.warning("⚠️ Loaded with strict=False (partial match)")
            
            model.eval()
            st.session_state.l1_model = model
            
            # Load scaler/temp_scaler (best effort)
            scaler_keys = ["scaler_seq", "scaler", "scaler_seq.pkl"]
            for key in scaler_keys:
                if key in extras or (isinstance(loaded, dict) and key in loaded):
                    st.session_state.scaler_seq = extras.get(key) or loaded.get(key)
                    st.success("✅ Loaded scaler")
                    break
            
            temp_keys = ["temp_scaler_state"]
            for key in temp_keys:
                temp_state = extras.get(key) or (loaded.get(key) if isinstance(loaded, dict) else None)
                if temp_state is not None:
                    ts = TemperatureScaler()
                    ts.load_state_dict(temp_state, strict=False)
                    st.session_state.temp_scaler = ts
                    st.success("✅ Loaded temp scaler")
                    break
    except Exception as e:
        st.error(f"❌ Checkpoint load failed: {str(e)[:200]}")

# Run inference
if st.button("🚀 Run L1 inference & propose limit order"):
    if st.session_state.market_df is None:
        st.error("❌ No market data. Click 'Fetch latest Gold' first.")
    elif st.session_state.l1_model is None:
        st.error("❌ No model loaded. Upload checkpoint first.")
    else:
        with st.spinner("Running inference..."):
            df = st.session_state.market_df.copy()
            
            # FIXED: Normalize timezone before features
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            feats = compute_engineered_features(df)
            
            seq_cols = ['open','high','low','close','volume']
            micro_cols = ['ret1','tr','vol_5','mom_5','chanpos_10']
            use_cols = [c for c in seq_cols + micro_cols if c in list(df.columns) + list(feats.columns)]
            
            feat_seq_df = pd.concat([
                df[seq_cols].astype(float), 
                feats[[c for c in micro_cols if c in feats.columns]]
            ], axis=1)[use_cols].fillna(0.0)
            
            X_all = feat_seq_df.values.astype('float32')
            scaler = st.session_state.scaler_seq
            if scaler is None:
                st.warning("⚠️ No scaler - fitting temp StandardScaler")
                scaler = StandardScaler().fit(X_all)
            
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
            
            # Trading logic
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
                st.metric("⚠️ Risk Amount", f"${risk_amount:.0f}")
            
            st.subheader("💼 Proposed LONG Order")
            order = {
                "entry": round(entry, 2),
                "stop_loss": round(sl, 2),
                "take_profit": round(tp, 2),
                "atr": float(atr),
                "position_size": float(size),
                "risk_amount_usd": float(risk_amount),
                "probability": float(prob)
            }
            st.json(order)
            
            st.code(f"""
# Copy for Supabase/Telegram bot:
curl -X POST https://your-project.supabase.co/functions/v1/raybot-inference \\
  -H "Authorization: Bearer YOUR_ANON_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"account_balance": {account_balance}, "risk_pct": {risk_pct*100}, "tp_mult": {tp_mult}, "sl_mult": {sl_mult}}}'
            """)

st.caption("✅ Fixed: Timezone errors, weekend data, yfinance fallback, robust checkpoint loading")