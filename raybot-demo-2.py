# streamlit_app.py
# Fixed: Import name and timezone handling

import io
import re
import logging
from datetime import datetime, timedelta, time, timezone

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

# IMPORT FETCHER - Updated to match actual filename
# Make sure gold_data_pipeline.py is in the same folder or on PYTHONPATH
from gold_data_pipeline import get_365_with_today as fetch_gold_history

# Safe timezone fallback
try:
    import gold_data_pipeline as _gds
    if not hasattr(_gds, 'ET') or _gds.ET is None:
        try:
            from dateutil import tz as _tz
            _gds.ET = _tz.gettz("America/New_York")
        except Exception:
            _gds.ET = timezone(timedelta(hours=-5))
except Exception:
    pass

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
    """
    Flexible L1 CNN which accepts a channels tuple, kernel_sizes and dilations.
    """
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
# Feature engineering (same as training)
# ---------------------------
def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    """
    Compute technical features from OHLCV data.
    """
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
    """
    Convert features to sequences for model input.
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

# ---------------------------
# Checkpoint robust loader helpers
# ---------------------------
def _is_state_dict_like(d: dict) -> bool:
    """Check if dict looks like a state_dict."""
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
    """Extract state_dict from various checkpoint formats."""
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
    """Remove 'module.' prefix from state dict keys."""
    new = {}
    for k,v in state.items():
        nk = k
        if k.startswith("module."):
            nk = k[len("module."):]
        new[nk] = v
    return new

_conv_key_re = re.compile(r"blocks\.(\d+)\.conv\.weight")

def infer_arch_from_state(state):
    """
    Inspect state_dict to infer architecture.
    Returns (in_features, channels_tuple)
    """
    blocks = {}
    for k,v in state.items():
        m = _conv_key_re.search(k)
        if m and hasattr(v, "shape"):
            idx = int(m.group(1))
            out_ch = int(v.shape[0])
            in_ch = int(v.shape[1])
            blocks[idx] = (out_ch, in_ch, tuple(v.shape))
    
    if not blocks:
        # Fallback: search for any key with '.conv.weight'
        for k,v in state.items():
            if ".conv.weight" in k and hasattr(v, "shape"):
                parts = k.split(".")
                try:
                    idx = int(parts[1]) if parts[0]=='blocks' else None
                except Exception:
                    idx = None
                out_ch = int(v.shape[0])
                in_ch = int(v.shape[1])
                if idx is None:
                    blocks[0] = (out_ch, in_ch, tuple(v.shape))
                else:
                    blocks[idx] = (out_ch, in_ch, tuple(v.shape))
    
    if not blocks:
        return None, None
    
    ordered = [blocks[i] for i in sorted(blocks.keys())]
    channels = [b[0] for b in ordered]
    in_features = ordered[0][1]
    
    return int(in_features), tuple(int(x) for x in channels)

def load_checkpoint_bytes_safe(raw_bytes: bytes):
    """Load checkpoint with multiple fallback strategies."""
    buf = io.BytesIO(raw_bytes)
    
    try:
        obj = torch.load(buf, map_location="cpu", weights_only=False)
        return obj
    except Exception as e:
        logger.info(f"torch.load direct failed: {e}")
        buf.seek(0)
        try:
            obj = torch.load(buf, map_location="cpu", weights_only=True)
            return obj
        except Exception as e2:
            buf.seek(0)
            import pickle
            try:
                obj = pickle.loads(buf.read())
                return obj
            except Exception as e3:
                logger.exception("All checkpoint load attempts failed")
                raise RuntimeError(f"Failed to load checkpoint: {e3}") from e3

# ---------------------------
# Streamlit UI
# ---------------------------
st.sidebar.header("Config")
seq_len = st.sidebar.slider("Sequence length", 8, 256, 64, step=8)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 2.0) / 100.0
tp_mult = st.sidebar.slider("TP ATR multiplier", 1.0, 5.0, 1.0)
sl_mult = st.sidebar.slider("SL ATR multiplier", 0.5, 3.0, 1.0)
account_balance = st.sidebar.number_input("Account balance ($)", value=10000.0)

ckpt = st.sidebar.file_uploader("Upload L1 checkpoint (.pt/.pth/.bin)", type=["pt","pth","bin"])

# Session state
if "market_df" not in st.session_state:
    st.session_state.market_df = None
if "l1_model" not in st.session_state:
    st.session_state.l1_model = None
if "scaler_seq" not in st.session_state:
    st.session_state.scaler_seq = None
if "temp_scaler" not in st.session_state:
    st.session_state.temp_scaler = None

# Fetch Gold data
if st.button("Fetch latest Gold (GC=F)"):
    try:
        with st.spinner("Fetching gold data..."):
            df = fetch_gold_history()
        
        if df.empty:
            st.error("No data returned")
        else:
            st.session_state.market_df = df
            st.success(f"✅ Fetched {len(df)} bars (index type: {type(df.index[0]).__name__})")
            st.dataframe(df.tail(10))
    except Exception as e:
        st.error(f"❌ Fetch failed: {e}")
        logger.exception("Fetch error")

# Load checkpoint
if ckpt is not None:
    try:
        with st.spinner("Loading checkpoint..."):
            raw = ckpt.read()
            loaded = load_checkpoint_bytes_safe(raw)
            state_dict, extras = extract_state_dict(loaded)
        
        if state_dict is None:
            if isinstance(loaded, nn.Module):
                st.session_state.l1_model = loaded
                st.success("✅ Loaded L1 as module object from checkpoint.")
            else:
                st.error("❌ Could not find state_dict inside checkpoint.")
        else:
            state_dict = strip_module_prefix(state_dict)
            inferred_in, inferred_channels = infer_arch_from_state(state_dict)
            
            if inferred_in is None or inferred_channels is None:
                st.warning("⚠️ Could not infer architecture; using defaults (in=12, channels=(32,64,128))")
                inferred_in = inferred_in or 12
                inferred_channels = inferred_channels or (32,64,128)
            
            st.info(f"📊 Inferred: in_features={inferred_in}, channels={inferred_channels}")
            
            model = Level1ScopeCNN(in_features=inferred_in, channels=inferred_channels)
            
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=True)
                st.success("✅ Loaded state_dict with strict=True")
            except Exception as e_strict:
                st.warning(f"⚠️ strict=True failed: {e_strict}. Trying strict=False...")
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                st.success("✅ Loaded state_dict with strict=False (some params may differ)")
            
            model.eval()
            st.session_state.l1_model = model
            
            # Load scaler
            scaler_candidate = None
            if isinstance(loaded, dict):
                scaler_candidate = loaded.get("scaler_seq") or loaded.get("scaler")
            if not scaler_candidate and isinstance(extras, dict):
                scaler_candidate = extras.get("scaler_seq") or extras.get("scaler")
            
            if scaler_candidate is not None:
                st.session_state.scaler_seq = scaler_candidate
                st.success("✅ Loaded scaler from checkpoint")
            else:
                st.warning("⚠️ No scaler found; will fit temporary scaler at inference")
            
            # Load temp scaler
            temp_state = None
            if isinstance(loaded, dict):
                temp_state = loaded.get("temp_scaler_state")
            if not temp_state and isinstance(extras, dict):
                temp_state = extras.get("temp_scaler_state")
            
            if temp_state is not None:
                ts = TemperatureScaler()
                try:
                    ts.load_state_dict(temp_state)
                    st.session_state.temp_scaler = ts
                    st.success("✅ Loaded temperature scaler")
                except Exception:
                    st.warning("⚠️ Failed to load temperature scaler")
    
    except Exception as e:
        st.error(f"❌ Failed to load checkpoint: {e}")
        logger.exception("Checkpoint load error")

# Run inference
if st.button("🚀 Run L1 inference & propose limit order"):
    if st.session_state.market_df is None:
        st.error("❌ No market data. Fetch first.")
    elif st.session_state.l1_model is None:
        st.error("❌ No model loaded. Upload checkpoint.")
    else:
        try:
            with st.spinner("Running inference..."):
                df = st.session_state.market_df.copy()
                feats = compute_engineered_features(df)
                
                seq_cols = ['open','high','low','close','volume']
                micro_cols = ['ret1','tr','vol_5','mom_5','chanpos_10']
                use_cols = [c for c in seq_cols + micro_cols if c in list(df.columns) + list(feats.columns)]
                
                feat_seq_df = pd.concat(
                    [df[seq_cols].astype(float), feats[[c for c in micro_cols if c in feats.columns]]], 
                    axis=1
                )[use_cols].fillna(0.0)
                
                X_all = feat_seq_df.values.astype('float32')
                
                scaler = st.session_state.scaler_seq
                if scaler is None:
                    st.warning("⚠️ Fitting temporary StandardScaler (not recommended for production)")
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
                        try:
                            logit_np = logit.cpu().numpy().reshape(-1,1)
                            temp = st.session_state.temp_scaler
                            scaled = temp(torch.tensor(logit_np)).cpu().numpy().reshape(-1)
                            prob = float(1.0 / (1.0 + np.exp(-scaled))[0])
                        except Exception:
                            prob = float(torch.sigmoid(logit).cpu().numpy().reshape(-1)[0])
                    else:
                        prob = float(torch.sigmoid(logit).cpu().numpy().reshape(-1)[0])
            
            st.subheader("📊 L1 Result")
            st.metric("Buy Probability", f"{prob:.4f}", delta=f"{(prob-0.5)*100:.1f}% vs neutral")
            
            # Compute limit order
            atr = feats['atr'].iloc[-1] if 'atr' in feats.columns else (df['high']-df['low']).rolling(14, min_periods=1).mean().iloc[-1]
            entry = float(df['close'].iloc[-1])
            sl = float(entry - atr * sl_mult)
            tp = float(entry + atr * tp_mult)
            
            risk_amount = account_balance * risk_pct
            stop_distance = abs(entry - sl)
            size = risk_amount / stop_distance if stop_distance > 0 else 0.0
            
            st.subheader("📈 Proposed Limit Order (LONG)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entry", f"${entry:.2f}")
                st.metric("Position Size", f"{size:.4f} oz")
            with col2:
                st.metric("Stop Loss", f"${sl:.2f}", delta=f"-{sl_mult}× ATR")
                st.metric("Risk Amount", f"${risk_amount:.2f}")
            with col3:
                st.metric("Take Profit", f"${tp:.2f}", delta=f"+{tp_mult}× ATR")
                st.metric("ATR", f"${float(atr):.2f}")
            
            st.json({
                "entry": round(entry, 6),
                "stop_loss": round(sl, 6),
                "take_profit": round(tp, 6),
                "atr": float(atr),
                "position_size": float(size),
                "risk_amount_usd": float(risk_amount),
                "probability": float(prob),
                "risk_reward_ratio": round(abs(tp-entry)/abs(entry-sl), 2) if abs(entry-sl) > 0 else 0
            })
            
        except Exception as e:
            st.error(f"❌ Inference failed: {e}")
            logger.exception("Inference error")

st.caption("✨ Auto-detects model architecture from checkpoint. Uses date-only indices to prevent timezone mixing errors.")
