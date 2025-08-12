import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fredapi import Fred

FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

# Series codes
series = {
    "real_yield": "DFII10",
    "breakeven": "T10YIE",
    "usd": "DTWEXBGS",
    "hy_oas": "BAMLH0A0HYM2",
    "ig_oas": "BAMLC0A0CM",
    "walcl": "WALCL",
    "rrp": "RRPONTSYD",
    "tga": "WTREGEN",
    "claims": "ICSA",
    "spx": "SP500",
    "vix": "VIXCLS",
    "vix3m": "VXVCLS"
}

# date range: 2 years
end = datetime.today()
start = end - timedelta(days=730)

def fetch_series(code):
    try:
        data = fred.get_series(code, observation_start=start, observation_end=end)
    except Exception:
        data = pd.Series(dtype=float)
    return data

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def zscore(series):
    return (series - series.mean()) / series.std()

signals = []

# Real yield
real = fetch_series(series["real_yield"]).dropna()
if not real.empty:
    val = float(real.iloc[-1])
    z = float(zscore(real).iloc[-1])
    desc = "10-year TIPS real yield; rising real yields pressure growth and long-duration assets."
    signals.append({"id": "real_yield", "value": val, "z": z, "desc": desc})

# Breakeven
breakeven = fetch_series(series["breakeven"]).dropna()
if not breakeven.empty:
    val = float(breakeven.iloc[-1])
    z = float(zscore(breakeven).iloc[-1])
    desc = "10-year inflation breakeven; higher breakevens imply rising inflation expectations."
    signals.append({"id": "breakeven", "value": val, "z": z, "desc": desc})

# USD index
usd = fetch_series(series["usd"]).dropna()
if not usd.empty:
    val = float(usd.iloc[-1])
    z = float(zscore(usd).iloc[-1])
    desc = "Trade-weighted USD (broad); a stronger dollar can be a headwind for risk assets and commodities."
    signals.append({"id": "usd", "value": val, "z": z, "desc": desc})

# HY OAS
hy = fetch_series(series["hy_oas"]).dropna()
if not hy.empty:
    val = float(hy.iloc[-1])
    z = float(zscore(hy).iloc[-1])
    desc = "High-yield credit option-adjusted spread; widening spreads suggest stress in credit markets."
    signals.append({"id": "hy_oas", "value": val, "z": z, "desc": desc})

# IG OAS
ig = fetch_series(series["ig_oas"]).dropna()
if not ig.empty:
    val = float(ig.iloc[-1])
    z = float(zscore(ig).iloc[-1])
    desc = "Investment-grade credit spread; widening indicates risk aversion and tightening conditions."
    signals.append({"id": "ig_oas", "value": val, "z": z, "desc": desc})

# Net liquidity
walcl = fetch_series(series["walcl"])
rrp = fetch_series(series["rrp"])
tga = fetch_series(series["tga"])
if not walcl.empty:
    df = pd.DataFrame({"walcl": walcl, "rrp": rrp, "tga": tga})
    df = df.resample("D").ffill()
    df["net_liquidity"] = df["walcl"] * 1e6 - df["rrp"] * 1e9 - df["tga"] * 1e9
    nl = df["net_liquidity"].dropna()
    if not nl.empty:
        val = float(nl.iloc[-1])
        z = float(zscore(nl).iloc[-1])
        desc = "Net liquidity = Fed balance sheet minus reverse repo and Treasury cash; rising liquidity supports risk assets."
        signals.append({"id": "net_liquidity", "value": val, "z": z, "desc": desc})

# Initial claims
claims = fetch_series(series["claims"]).dropna()
if not claims.empty:
    val = float(claims.iloc[-1])
    z = float(zscore(claims).iloc[-1])
    desc = "Weekly initial unemployment claims; rising claims may signal labor market weakness."
    signals.append({"id": "claims", "value": val, "z": z, "desc": desc})

# S&P 500 RSI(14)
spx = fetch_series(series["spx"]).dropna()
if not spx.empty:
    rsi_series = compute_rsi(spx).dropna()
    if not rsi_series.empty:
        val = float(rsi_series.iloc[-1])
        z = float(zscore(rsi_series).iloc[-1])
        desc = "S&P 500 RSI (14-day); measures momentum; overbought >70, oversold <30."
        signals.append({"id": "spx_rsi14", "value": val, "z": z, "desc": desc})

# VIX level
vix = fetch_series(series["vix"]).dropna()
if not vix.empty:
    val = float(vix.iloc[-1])
    z = float(zscore(vix).iloc[-1])
    desc = "CBOE Volatility Index (VIX); higher levels signal market stress."
    signals.append({"id": "vix_level", "value": val, "z": z, "desc": desc})

# VIX term (VIX minus VIX3M)
vix3m = fetch_series(series["vix3m"]).dropna()
if not vix.empty and not vix3m.empty:
    df_vix = pd.DataFrame({"vix": vix, "vix3m": vix3m}).dropna()
    spread = df_vix["vix"] - df_vix["vix3m"]
    val = float(spread.iloc[-1])
    z = float(zscore(spread).iloc[-1])
    desc = "VIX term structure (VIX minus VIX3M); positive spread indicates backwardation (stress)."
    signals.append({"id": "vix_term", "value": val, "z": z, "desc": desc})

# Write signals to docs/signals.json
os.makedirs("docs", exist_ok=True)
with open("docs/signals.json", "w") as f:
    json.dump(signals, f, indent=2)
