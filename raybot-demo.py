# chunk1_l1_infer.py
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
    """
    Flexible L1 CNN which accepts a channels tuple, kernel_sizes and dilations.
    channels: tuple of out-channels per block, e.g. (32,64,128)
    in_features: number of input features (channels) to the first conv
    """
    def __init__(
        self,
        in_features=12,
        channels=(32, 64, 128),
        kernel_sizes=(5, 3, 3),
        dilations=(1, 2, 4),
        dropout=0.1,
    ):
        super().__init__()
        chs = [in_features] + list(channels)
        blocks = []
        for i in range(len(channels)):
            k = kernel_sizes[min(i, len(kernel_sizes) - 1)]
            d = dilations[min(i, len(dilations) - 1)]
            blocks.append(ConvBlock(chs[i], chs[i + 1], k=k, d=d, pdrop=dropout))
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
def compute_engineered_features(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = (
        df["volume"].astype(float)
        if "volume" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    ret1 = c.pct_change().fillna(0.0)
    f["ret1"] = ret1
    f["logret1"] = np.log1p(ret1.replace(-1, -0.999999))

    tr = (h - l).clip(lower=0)
    f["tr"] = tr.fillna(0.0)
    f["atr"] = tr.rolling(14, min_periods=1).mean().fillna(0.0)

    for w in windows:
        f[f"rmean_{w}"] = c.pct_change(w).fillna(0.0)
        f[f"vol_{w}"] = ret1.rolling(w).std().fillna(0.0)
        f[f"tr_mean_{w}"] = tr.rolling(w).mean().fillna(0.0)
        f[f"vol_z_{w}"] = (
            v.rolling(w).mean() - v.rolling(max(1, w * 3)).mean()
        ).fillna(0.0)
        f[f"mom_{w}"] = (c - c.rolling(w).mean()).fillna(0.0)

        roll_max = c.rolling(w).max().fillna(method="bfill")
        roll_min = c.rolling(w).min().fillna(method="bfill")
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f"chanpos_{w}"] = ((c - roll_min) / denom).fillna(0.5)

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
            seq = np.vstack([pad, features[0 : t + 1]])
        else:
            seq = features[t0 : t + 1]

        if seq.shape[0] < seq_len:
            pad_needed = seq_len - seq.shape[0]
            pad = np.repeat(seq[[0]], pad_needed, axis=0)
            seq = np.vstack([pad, seq])

        X[i] = seq[-seq_len:]

    return X


# ---------------------------
# Gold fetch helper
# ---------------------------
def fetch_gold_history(days=365, interval="1d") -> pd.DataFrame:
    if YahooTicker is None:
        raise RuntimeError("yahooquery not installed")

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    tq = YahooTicker("GC=F")
    raw = tq.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
    )

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

    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in raw.columns:
            raw[col] = 0.0

    return raw[required]


# ---------------------------
# Checkpoint robust loader helpers
# ---------------------------
def _is_state_dict_like(d: dict) -> bool:
    if not isinstance(d, dict):
        return False

    keys = list(d.keys())
    for k in keys[:20]:
        if any(
            sub in k
            for sub in (
                "conv.weight",
                "bn.weight",
                "head.weight",
                "proj.weight",
                "blocks.0.conv.weight",
            )
        ):
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

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "model_state",
        "model_weights",
    ):
        if (
            isinstance(container, dict)
            and key in container
            and _is_state_dict_like(container[key])
        ):
            extras = {k: v for k, v in container.items() if k != key}
            return container[key], extras

    if isinstance(container, dict):
        for k, v in container.items():
            if isinstance(v, dict) and _is_state_dict_like(v):
                extras = {kk: vv for kk, vv in container.items() if kk != k}
                return v, extras

    return None, {}


def strip_module_prefix(state):
    new = {}
    for k, v in state.items():
        nk = k[len("module.") :] if k.startswith("module.") else k
        new[nk] = v
    return new


_conv_key_re = re.compile(r"blocks\.(\d+)\.conv\.weight")


def infer_arch_from_state(state):
    """
    Inspect state_dict to infer:
      - in_features (channels input to first conv)
      - channels tuple (out-channels for each block)
    """
    blocks = {}
    for k, v in state.items():
        m = _conv_key_re.search(k)
        if m and hasattr(v, "shape"):
            idx = int(m.group(1))
            out_ch = int(v.shape[0])
            in_ch = int(v.shape[1])
            blocks[idx] = (out_ch, in_ch, tuple(v.shape))

    if not blocks:
        for k, v in state.items():
            if ".conv.weight" in k and hasattr(v, "shape"):
                parts = k.split(".")
                try:
                    idx = int(parts[1]) if parts[0] == "blocks" else None
                except Exception:
                    idx = None
                out_ch = int(v.shape[0])
                in_ch = int(v.shape[1])
                blocks[idx or 0] = (out_ch, in_ch, tuple(v.shape))

    if not blocks:
        return None, None

    ordered = [blocks[i] for i in sorted(blocks.keys())]
    channels = [b[0] for b in ordered]
    in_features = ordered[0][1]

    return int(in_features), tuple(int(x) for x in channels)


def load_checkpoint_bytes_safe(raw_bytes: bytes):
    buf = io.BytesIO(raw_bytes)
    try:
        return torch.load(buf, map_location="cpu", weights_only=False)
    except Exception:
        buf.seek(0)
        try:
            return torch.load(buf, map_location="cpu", weights_only=True)
        except Exception:
            buf.seek(0)
            import pickle

            return pickle.loads(buf.read())


# chunk2_l1_infer.py
# continue running after chunk1

st.sidebar.header("Config")
seq_len = st.sidebar.slider("Sequence length", 8, 256, 64, step=8)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 2.0) / 100.0
tp_mult = st.sidebar.slider("TP ATR multiplier", 1.0, 5.0, 1.0)
sl_mult = st.sidebar.slider("SL ATR multiplier", 0.5, 3.0, 1.0)
account_balance = st.sidebar.number_input("Account balance ($)", value=10000.0)

ckpt = st.sidebar.file_uploader(
    "Upload L1 checkpoint (.pt/.pth/.bin)", type=["pt", "pth", "bin"]
)

if "market_df" not in st.session_state:
    st.session_state.market_df = None
if "l1_model" not in st.session_state:
    st.session_state.l1_model = None
if "scaler_seq" not in st.session_state:
    st.session_state.scaler_seq = None
if "temp_scaler" not in st.session_state:
    st.session_state.temp_scaler = None

if st.button("Fetch latest Gold (GC=F)"):
    df = fetch_gold_history(days=365, interval="1d")
    st.session_state.market_df = df
    st.dataframe(df.tail(10))

if ckpt is not None:
    raw = ckpt.read()
    loaded = load_checkpoint_bytes_safe(raw)
    state_dict, extras = extract_state_dict(loaded)
    state_dict = strip_module_prefix(state_dict)
    in_f, ch = infer_arch_from_state(state_dict)
    model = Level1ScopeCNN(in_features=in_f, channels=ch)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    st.session_state.l1_model = model

if st.button("Run L1 inference & propose limit order"):
    df = st.session_state.market_df.copy()
    feats = compute_engineered_features(df)

    seq_cols = ["open", "high", "low", "close", "volume"]
    micro_cols = ["ret1", "tr", "vol_5", "mom_5", "chanpos_10"]

    feat_df = pd.concat(
        [df[seq_cols], feats[micro_cols]], axis=1
    ).fillna(0.0)

    X = feat_df.values.astype("float32")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    Xseq = to_sequences(Xs, np.array([len(Xs) - 1]), seq_len)
    xb = torch.tensor(Xseq.transpose(0, 2, 1))

    with torch.no_grad():
        logit, _ = st.session_state.l1_model(xb)
        prob = float(torch.sigmoid(logit).item())

    atr = feats["atr"].iloc[-1]
    entry = df["close"].iloc[-1]
    sl = entry - atr * sl_mult
    tp = entry + atr * tp_mult

    risk_amount = account_balance * risk_pct
    size = risk_amount / abs(entry - sl)

    st.json(
        {
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "position_size": size,
            "probability": prob,
        }
    )