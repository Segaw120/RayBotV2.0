# chunk1_l1_inference.py
import io
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

# Optional: allowlist problematic types for torch.load when necessary
try:
    from torch.serialization import add_safe_globals
    add_safe_globals([np.core.multiarray.scalar, StandardScaler])
except Exception:
    # ignore if not available (older torch)
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("l1_inference")

st.set_page_config(page_title="Cascade Trader — L1 Inference", layout="wide")
st.title("Cascade Trader — L1 Inference & Limit Orders (Robust Loader)")

# ---------------------------
# Model architecture (L1)
# ---------------------------
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
        return out + x if self.res else out

class Level1ScopeCNN(nn.Module):
    def __init__(self, in_features=12, channels=(32,64,128)):
        super().__init__()
        chs = [in_features] + list(channels)
        blocks = []
        blocks.append(ConvBlock(chs[0], chs[1], 5, 1, 0.1))
        blocks.append(ConvBlock(chs[1], chs[2], 3, 2, 0.1))
        blocks.append(ConvBlock(chs[2], chs[2], 3, 4, 0.1))
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
        self.head = nn.Linear(chs[-1], 1)
    @property
    def embedding_dim(self): return int(self.blocks[-1].conv.out_channels)
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
    def load_state(self, state):
        try:
            self.load_state_dict(state)
        except Exception:
            logger.warning("Temp scaler load failed (state mismatch)")

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
# Data fetch (gold futures)
# ---------------------------
def fetch_gold_history(days=365, interval="1d") -> pd.DataFrame:
    if YahooTicker is None:
        raise RuntimeError("yahooquery not installed")
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    tq = YahooTicker("GC=F")
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
    required = ['open','high','low','close','volume']
    for col in required:
        if col not in raw.columns:
            raw[col] = 0.0
    return raw[required]

# ---------------------------
# Robust checkpoint extraction
# ---------------------------
def _is_state_dict_like(d: dict) -> bool:
    # heuristic: contains some typical param names
    if not isinstance(d, dict):
        return False
    keys = list(d.keys())
    if len(keys) == 0:
        return False
    sample_keys = keys[:10]
    # look for conv/bn/weight substrings
    for k in sample_keys:
        if any(sub in k for sub in ("conv.weight","bn.weight","head.weight","proj.weight","blocks.0.conv.weight","blocks")):
            return True
    # also accept if values are tensors/arrays
    sample_vals = list(d.values())[:5]
    if all(isinstance(v, (torch.Tensor, np.ndarray)) for v in sample_vals):
        return True
    return False

def extract_state_dict(container):
    """Try multiple ways to find the actual state_dict inside the loaded object."""
    if container is None:
        return None, {}
    # already a state_dict
    if isinstance(container, dict) and _is_state_dict_like(container):
        return container, {}
    # common wrapper keys
    for key in ("model_state_dict","state_dict","model","model_state","model_weights","model_state_dict"):
        if isinstance(container, dict) and key in container and _is_state_dict_like(container[key]):
            extras = {k:v for k,v in container.items() if k != key}
            return container[key], extras
    # nested search
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

def find_first_conv_in_features(state):
    # find a key that endswith conv.weight and take shape[1]
    for k,v in state.items():
        if "conv.weight" in k and hasattr(v, "shape"):
            # v is tensor with shape [out, in, k]
            try:
                return int(v.shape[1])
            except Exception:
                continue
    # fallback: search for any '.weight' and try to infer
    for k,v in state.items():
        if k.endswith(".weight") and hasattr(v, "shape"):
            # many weights are linear with shape [out,in]
            return int(v.shape[1]) if len(v.shape) > 1 else int(v.shape[0])
    return None

def load_checkpoint_bytes_safe(raw_bytes: bytes):
    """
    Try torch.load with safe global fixes; return loaded object or raise.
    """
    buf = io.BytesIO(raw_bytes)
    try:
        obj = torch.load(buf, map_location="cpu", weights_only=False)
        return obj
    except Exception as e:
        logger.info("torch.load direct failed, attempting fallback torch.load with weights_only=True: %s", e)
        buf.seek(0)
        try:
            obj = torch.load(buf, map_location="cpu", weights_only=True)
            return obj
        except Exception as e2:
            logger.info("weights_only load also failed: %s", e2)
            buf.seek(0)
            # final fallback: try python pickle load
            import pickle
            try:
                obj = pickle.loads(buf.read())
                return obj
            except Exception as e3:
                logger.exception("All checkpoint load attempts failed")
                raise RuntimeError(f"Failed to load checkpoint: {e3}") from e3



# chunk2_l1_inference.py
# (Continue from chunk 1: run after importing or concatenating both chunks)

# Sidebar config
st.sidebar.header("Config")
seq_len = st.sidebar.slider("Sequence length", 8, 256, 64, step=8)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 2.0) / 100.0
tp_mult = st.sidebar.slider("TP ATR multiplier", 1.0, 5.0, 2.0)
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

# Fetch gold
if st.button("Fetch latest Gold (GC=F)"):
    try:
        df = fetch_gold_history(days=365, interval="1d")
        if df.empty:
            st.error("No data returned")
        else:
            st.session_state.market_df = df
            st.success(f"Fetched {len(df)} bars")
            st.dataframe(df.tail(10))
    except Exception as e:
        st.error(f"Fetch failed: {e}")
        st.error(str(e))

# Load checkpoint and extract state dict
if ckpt is not None:
    try:
        raw = ckpt.read()
        loaded = load_checkpoint_bytes_safe(raw)
        # try to extract state dict and extras
        state_dict, extras = extract_state_dict(loaded)
        if state_dict is None:
            # maybe the loaded object is a torch.nn.Module saved directly
            if isinstance(loaded, nn.Module):
                # rare case: user saved whole module
                st.session_state.l1_model = loaded
                st.success("Loaded L1 as module from checkpoint.")
            else:
                raise RuntimeError("Could not find a state_dict inside checkpoint (keys: {}).".format(list(loaded.keys()) if isinstance(loaded, dict) else type(loaded)))
        else:
            # strip module prefix
            state_dict = strip_module_prefix(state_dict)
            in_features = find_first_conv_in_features(state_dict)
            if in_features is None:
                st.warning("Could not infer in_features from checkpoint; falling back to 12")
                in_features = 12
            model = Level1ScopeCNN(in_features=in_features)
            # load with strict=False to tolerate missing/unexpected keys; report them
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            st.session_state.l1_model = model
            st.session_state.scorer_missing = missing
            st.session_state.scorer_unexpected = unexpected
            # load scaler if present in extras or loaded dict
            scaler = None
            if isinstance(loaded, dict) and "scaler_seq" in loaded:
                scaler = loaded["scaler_seq"]
            elif "scaler_seq" in extras:
                scaler = extras.get("scaler_seq")
            if scaler is not None:
                st.session_state.scaler_seq = scaler
            # load temp scaler if present
            temp = None
            if isinstance(loaded, dict) and "temp_scaler_state" in loaded:
                temp = loaded["temp_scaler_state"]
            elif "temp_scaler_state" in extras:
                temp = extras.get("temp_scaler_state")
            if temp is not None:
                ts = TemperatureScaler()
                try:
                    ts.load_state_dict(temp)
                    st.session_state.temp_scaler = ts
                except Exception:
                    st.session_state.temp_scaler = None
            st.success("L1 model loaded (strict=False). Check console for missing/unexpected keys.")
            if missing:
                st.warning(f"Missing keys (not loaded): {len(missing)} example: {missing[:5]}")
            if unexpected:
                st.info(f"Unexpected keys present in checkpoint but not used by model: {len(unexpected)} example: {unexpected[:5]}")
    except Exception as e:
        st.error(f"Failed to load L1 checkpoint: {e}")
        st.error(str(e))

# Run inference (single-bar latest)
if st.button("Run L1 inference & propose limit order"):
    if st.session_state.market_df is None:
        st.error("No market data. Fetch first.")
    elif st.session_state.l1_model is None:
        st.error("No model loaded. Upload checkpoint.")
    else:
        df = st.session_state.market_df.copy()
        feats = compute_engineered_features(df)
        seq_cols = ['open','high','low','close','volume']
        micro_cols = ['ret1','tr','vol_5','mom_5','chanpos_10']
        use_cols = [c for c in seq_cols + micro_cols if c in list(df.columns) + list(feats.columns)]
        feat_seq_df = pd.concat([df[seq_cols].astype(float), feats[[c for c in micro_cols if c in feats.columns]]], axis=1)[use_cols].fillna(0.0)
        X_all = feat_seq_df.values.astype('float32')
        scaler = st.session_state.scaler_seq
        if scaler is None:
            st.warning("No scaler found in checkpoint — fitting a temporary StandardScaler (NOT ideal).")
            scaler = StandardScaler().fit(X_all)
        X_scaled = scaler.transform(X_all)
        # build sequence ending at last bar
        idx = np.array([len(X_scaled)-1], dtype=int)
        Xseq = to_sequences(X_scaled, idx, seq_len=seq_len)
        xb = torch.tensor(Xseq.transpose(0,2,1), dtype=torch.float32)
        model = st.session_state.l1_model
        model.eval()
        with torch.no_grad():
            logit, emb = model(xb)
            # apply temp scaler if present
            if st.session_state.temp_scaler is not None:
                try:
                    logit_np = logit.cpu().numpy().reshape(-1,1)
                    scaled = st.session_state.temp_scaler.forward(torch.tensor(logit_np)).cpu().numpy().reshape(-1)
                    prob = 1.0 / (1.0 + np.exp(-scaled))[0]
                except Exception:
                    prob = float(torch.sigmoid(logit).cpu().numpy().reshape(-1)[0])
            else:
                prob = float(torch.sigmoid(logit).cpu().numpy().reshape(-1)[0])
        st.subheader("L1 result")
        st.write(f"Probability (buy): {prob:.4f}")
        # compute ATR and limit order
        atr = feats['atr'].iloc[-1] if 'atr' in feats.columns else (df['high']-df['low']).rolling(14, min_periods=1).mean().iloc[-1]
        entry = float(df['close'].iloc[-1])
        sl = float(entry - atr * sl_mult)
        tp = float(entry + atr * tp_mult)
        # position sizing
        risk_amount = account_balance * risk_pct
        stop_distance = abs(entry - sl)
        size = risk_amount / stop_distance if stop_distance > 0 else 0.0
        st.subheader("Proposed limit order (LONG)")
        st.json({
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "atr": float(atr),
            "position_size": float(size),
            "risk_amount_usd": float(risk_amount),
            "probability": float(prob)
        })

# small status info
st.markdown("---")
st.caption("Loader notes: this loader will accept nested checkpoint wrappers (model_state_dict, scaler_seq, temp_scaler_state) and will try to load tolerantly. If you still see missing keys, your checkpoint uses a different naming/architecture — share checkpoint structure and I can adapt.")
