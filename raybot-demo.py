import streamlit as st
import torch
import numpy as np
import pandas as pd
from yahooquery import Ticker
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
ASSET = "GC=F"          # Gold Futures
LOOKBACK = 20           # Bars for feature window
RISK_PCT = 0.005        # 0.5% risk
RR = 2.0                # Risk:Reward

st.set_page_config(page_title="L1 Gold Inference", layout="centered")

# -----------------------------
# SAFE TORCH LOAD (PYTORCH 2.6 FIX)
# -----------------------------
def safe_load_model(file_bytes):
    import numpy as np
    from torch.serialization import add_safe_globals

    # Allow numpy scalar objects
    add_safe_globals([np.core.multiarray.scalar])

    buffer = torch.BytesIO(file_bytes)

    model = torch.load(
        buffer,
        map_location="cpu",
        weights_only=False  # IMPORTANT
    )

    model.eval()
    return model


# -----------------------------
# FETCH GOLD DATA
# -----------------------------
@st.cache_data(ttl=300)
def fetch_gold():
    ticker = Ticker(ASSET)
    df = ticker.history(period="5d", interval="5m")

    if df.empty:
        raise RuntimeError("Failed to fetch Gold data")

    df = df.reset_index()
    return df


# -----------------------------
# FEATURE ENGINEERING (MINIMAL)
# -----------------------------
def prepare_features(df):
    """
    Very minimal feature set:
    - returns
    - range
    - volume z-score
    """
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()

    df = df.dropna()

    features = df[["return", "range", "vol_z"]].tail(LOOKBACK)

    return torch.tensor(features.values, dtype=torch.float32).unsqueeze(0)


# -----------------------------
# LIMIT ORDER LOGIC
# -----------------------------
def generate_order(price, signal):
    direction = "LONG" if signal > 0 else "SHORT"

    if direction == "LONG":
        entry = price * (1 - 0.0005)
        sl = entry * (1 - RISK_PCT)
        tp = entry * (1 + RISK_PCT * RR)
    else:
        entry = price * (1 + 0.0005)
        sl = entry * (1 + RISK_PCT)
        tp = entry * (1 - RISK_PCT * RR)

    return {
        "direction": direction,
        "entry": round(entry, 2),
        "stop_loss": round(sl, 2),
        "take_profit": round(tp, 2)
    }


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🟡 Gold Futures — L1 Model Inference")

st.markdown("**Bare-bones L1 inference only**")

# Upload model
uploaded = st.file_uploader("Upload L1 model (.pt)", type=["pt"])

# Fetch data
with st.spinner("Fetching Gold futures data..."):
    data = fetch_gold()

st.success("Gold data loaded")

st.dataframe(data.tail(5), use_container_width=True)

if uploaded:
    try:
        with st.spinner("Loading L1 model..."):
            model = safe_load_model(uploaded.read())

        st.success("Model loaded successfully")

        # Prepare features
        X = prepare_features(data)

        # Inference
        with torch.no_grad():
            signal = model(X).item()

        last_price = data["close"].iloc[-1]
        order = generate_order(last_price, signal)

        st.subheader("📌 Inference Result")
        st.metric("Model Signal", round(signal, 4))

        st.subheader("📈 Limit Order")
        st.json(order)

    except Exception as e:
        st.error(f"Model inference failed: {e}")
