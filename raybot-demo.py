# l1_inference_app.py
import io
import math
import pickle
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn
from torch.serialization import add_safe_globals

from sklearn.preprocessing import StandardScaler

try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

# -----------------------------------------------------------------------------
# Logging / UI
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("l1_inference")

st.set_page_config(page_title="Cascade Trader — L1 Inference", layout="wide")
st.title("Cascade Trader — L1 Inference & Limit Orders")

# -----------------------------------------------------------------------------
# L1 Model Architecture (must match training)
# -----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, pdrop):
        super().__init__()
        pad = (k - 1) * d // 2
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=d, padding=pad)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(pdrop)
        self.res = c_in == c_out

    def forward(self, x):
        y = self.drop(self.act(self.bn(self.conv(x))))
        return y + x if self.res else y


class Level1ScopeCNN(nn.Module):
    def __init__(self, in_features=12):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_features, 32, 5, 1, 0.1),
            ConvBlock(32, 64, 3, 2, 0.1),
            ConvBlock(64, 128, 3, 4, 0.1),
        )
        self.proj = nn.Conv1d(128, 128, 1)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        z = self.proj(self.blocks(x))
        z = z.mean(dim=-1)
        return self.head(z), z


# -----------------------------------------------------------------------------
# Safe checkpoint loader (PyTorch ≥2.6 compatible)
# -----------------------------------------------------------------------------
def load_checkpoint_bytes(raw: bytes):
    add_safe_globals([
        np.core.multiarray.scalar,
        StandardScaler
    ])
    buf = io.BytesIO(raw)
    try:
        return torch.load(buf, map_location="cpu", weights_only=False)
    except TypeError:
        buf.seek(0)
        return torch.load(buf, map_location="cpu")


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
def engineer_features(df):
    f = pd.DataFrame(index=df.index)
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)

    ret = c.pct_change().fillna(0)
    tr = (h - l).abs()

    f["ret1"] = ret
    f["tr"] = tr
    f["atr"] = tr.rolling(14, min_periods=1).mean()

    f["mom_5"] = c - c.rolling(5).mean()
    f["vol_5"] = ret.rolling(5).std()
    f["chanpos_10"] = (c - c.rolling(10).min()) / (
        c.rolling(10).max() - c.rolling(10).min()
    )

    return f.replace([np.inf, -np.inf], 0).fillna(0)


def make_sequence(x, seq_len):
    if len(x) < seq_len:
        pad = np.repeat(x[[0]], seq_len - len(x), axis=0)
        x = np.vstack([pad, x])
    return x[-seq_len:]


# -----------------------------------------------------------------------------
# Market data
# -----------------------------------------------------------------------------
def fetch_gold(days=365, interval="1d"):
    if YahooTicker is None:
        raise RuntimeError("yahooquery not installed")

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    t = YahooTicker("GC=F")
    df = t.history(start=start.strftime("%Y-%m-%d"),
                   end=end.strftime("%Y-%m-%d"),
                   interval=interval)

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]]




# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")

seq_len = st.sidebar.slider("Sequence Length", 8, 256, 64)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.1, 5.0, 2.0) / 100
tp_mult = st.sidebar.slider("TP ATR Multiplier", 1.0, 5.0, 2.0)
sl_mult = st.sidebar.slider("SL ATR Multiplier", 0.5, 3.0, 1.0)
balance = st.sidebar.number_input("Account Balance ($)", 100.0, value=10_000.0)

ckpt_file = st.sidebar.file_uploader("Upload L1 checkpoint (.pt)", type=["pt"])

# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None


# -----------------------------------------------------------------------------
# Fetch data
# -----------------------------------------------------------------------------
if st.button("Fetch latest Gold futures data"):
    st.session_state.df = fetch_gold()
    st.success("Market data loaded")
    st.dataframe(st.session_state.df.tail(10))


# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
if ckpt_file:
    try:
        obj = load_checkpoint_bytes(ckpt_file.read())

        if isinstance(obj, dict) and "state_dict" in obj:
            state = obj["state_dict"]
        elif isinstance(obj, dict):
            state = obj
        else:
            state = None

        model = Level1ScopeCNN()
        if state is not None:
            model.load_state_dict(state)

        model.eval()
        st.session_state.model = model

        if isinstance(obj, dict) and "scaler_seq" in obj:
            st.session_state.scaler = obj["scaler_seq"]

        st.success("L1 model loaded")

    except Exception as e:
        st.error(f"Model load failed: {e}")


# -----------------------------------------------------------------------------
# Inference + Limit order
# -----------------------------------------------------------------------------
if st.button("Run L1 Inference"):
    if st.session_state.df is None or st.session_state.model is None:
        st.error("Missing data or model")
    else:
        df = st.session_state.df.copy()

        feats = engineer_features(df)
        seq_df = pd.concat([df[["open","high","low","close","volume"]], feats], axis=1)

        X = seq_df.values.astype(np.float32)

        scaler = st.session_state.scaler
        if scaler is None:
            scaler = StandardScaler().fit(X)

        Xs = scaler.transform(X)
        Xseq = make_sequence(Xs, seq_len)

        xb = torch.tensor(Xseq.T[None, ...])
        with torch.no_grad():
            logit, _ = st.session_state.model(xb)
            prob = torch.sigmoid(logit).item()

        atr = feats["atr"].iloc[-1]
        entry = df["close"].iloc[-1]

        sl = entry - atr * sl_mult
        tp = entry + atr * tp_mult

        risk_amt = balance * risk_pct
        size = risk_amt / (entry - sl) if entry > sl else 0

        st.subheader("L1 Output")
        st.metric("Buy Probability", f"{prob:.2%}")

        st.subheader("Limit Order Proposal")
        st.json({
            "side": "LONG",
            "entry": float(entry),
            "stop_loss": float(sl),
            "take_profit": float(tp),
            "position_size": float(size),
            "risk_$": float(risk_amt)
        })
