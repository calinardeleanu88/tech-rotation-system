import os
import json
import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# v3.2.1 — Official version
# DEF = XLV + IEF (50/50)
# Permanent DEF = 25%
# Execution: Fridays + cooldown 15 trading days + min-change 5%
# =========================

PLATFORMS = ["MSFT", "GOOGL"]
SEMIS     = ["NVDA", "KLAC", "AMD"]
INFRA     = ["VRT"]
DEF       = ["XLV", "IEF"]
TECH_KILL = "XLK"

ALL = sorted(set(PLATFORMS + SEMIS + INFRA + DEF + [TECH_KILL]))

START = "2006-01-01"
END = None

# Allocation
W_PLATFORM = 0.35
W_SEMIS    = 0.25
W_INFRA    = 0.15
W_DEF_BASE = 0.25

AMD_CAP_WITHIN_SEMIS = 0.12

# DEF split 50/50
DEF_XLV = 0.50
DEF_IEF = 0.50

# Signals
WEEKLY_EMA_W = 30
DAILY_SMA    = 20
CONF_WIN     = 10
CONF_ON      = 7
CONF_OFF     = 3
RS_WIN       = 63

# Execution controls
EXECUTE_WEEKLY = True
MIN_TRADING_DAYS_BETWEEN_TRADES = 15
MIN_CHANGE_TO_COUNT_AS_TRADE    = 0.05

# Costs
TC_BPS_PER_1X = 5
RFR = 0.0

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())

def annualized_vol(daily_ret: pd.Series) -> float:
    return float(daily_ret.std() * np.sqrt(252))

def cagr_calendar(eq: pd.Series) -> float:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return float((eq.iloc[-1] / eq.iloc[0])**(1/years) - 1)

def sharpe(daily_ret: pd.Series, rfr=0.0) -> float:
    excess = daily_ret - rfr/252
    vol = daily_ret.std()
    return np.nan if vol == 0 else float((excess.mean() / vol) * np.sqrt(252))

def sortino(daily_ret: pd.Series, rfr=0.0) -> float:
    excess = daily_ret - rfr/252
    downside = daily_ret[daily_ret < 0].std()
    return np.nan if downside == 0 else float((excess.mean() / downside) * np.sqrt(252))

def normalize_positive(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(s.sum())
    if total <= 0:
        return s * 0.0
    return s / total

def trading_days_between(idx: pd.DatetimeIndex, a: pd.Timestamp, b: pd.Timestamp) -> int:
    if a is None:
        return 10**9
    return int(((idx > a) & (idx <= b)).sum())

def add_def(w_row: pd.Series, amount: float):
    if amount <= 0:
        return
    w_row[DEF[0]] += amount * DEF_XLV
    w_row[DEF[1]] += amount * DEF_IEF

# -------------------------
# Data
# -------------------------
px = yf.download(ALL, start=START, end=END, auto_adjust=True, progress=False)["Close"]
px = px.dropna(how="all").ffill()

ret_d = px.pct_change().fillna(0.0)
px_w = px.resample("W-FRI").last().dropna(how="all").ffill()

# Weekly gate mapped to daily
weekly_gate = {}
for t in ALL:
    w = px_w[t]
    ema = w.ewm(span=WEEKLY_EMA_W, adjust=False).mean()
    gate_w = (w > ema).astype("boolean").fillna(False)
    gate_d = gate_w.reindex(px.index, method="ffill").astype("boolean").fillna(False)
    weekly_gate[t] = gate_d

weekly_gate = pd.DataFrame(weekly_gate, index=px.index)
tech_on = weekly_gate[TECH_KILL].astype(bool)

# Daily confirm
sma = px.rolling(DAILY_SMA).mean()
above_sma = (px > sma).astype("boolean").fillna(False)
above_count = above_sma.rolling(CONF_WIN).sum()

confirm_on  = (above_count >= CONF_ON).astype("boolean").fillna(False)
confirm_off = (above_count <= CONF_OFF).astype("boolean").fillna(False)

# State machine
def build_state(ticker: str) -> pd.Series:
    gate = weekly_gate[ticker].astype(bool)
    on_sig  = confirm_on[ticker].astype(bool) & gate
    off_sig = confirm_off[ticker].astype(bool) | (~gate)

    st = pd.Series(False, index=px.index, dtype=bool)
    on = False
    for dt in px.index:
        if bool(off_sig.loc[dt]):
            on = False
        elif bool(on_sig.loc[dt]):
            on = True
        st.loc[dt] = on
    return st

state = {t: build_state(t) for t in (PLATFORMS + SEMIS + INFRA)}
state = pd.DataFrame(state, index=px.index)

# Relative strength
rs = px.pct_change(RS_WIN).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Target weights (daily)
assets = PLATFORMS + SEMIS + INFRA + DEF
w_target = pd.DataFrame(0.0, index=px.index, columns=assets)

for dt in px.index:
    row = w_target.loc[dt]

    # Permanent defensive buffer
    add_def(row, W_DEF_BASE)

    # Kill-switch => 100% DEF
    if not bool(tech_on.loc[dt]):
        w_target.loc[dt, :] = 0.0
        w_target.loc[dt, DEF[0]] = DEF_XLV
        w_target.loc[dt, DEF[1]] = DEF_IEF
        continue

    plat_on  = [x for x in PLATFORMS if bool(state.loc[dt, x])]
    semis_on = [x for x in SEMIS if bool(state.loc[dt, x])]
    infra_on = [x for x in INFRA if bool(state.loc[dt, x])]

    # Platforms
    if plat_on:
        scores = normalize_positive(rs.loc[dt, plat_on])
        if float(scores.sum()) == 0:
            for x in plat_on:
                row[x] += W_PLATFORM / len(plat_on)
        else:
            for x in plat_on:
                row[x] += W_PLATFORM * float(scores.loc[x])
    else:
        add_def(row, W_PLATFORM)

    # Infra
    if infra_on:
        row[infra_on[0]] += W_INFRA
    else:
        add_def(row, W_INFRA)

    # Semis with AMD cap
    if semis_on:
        semis_budget = W_SEMIS
        amd_w = min(AMD_CAP_WITHIN_SEMIS, semis_budget) if "AMD" in semis_on else 0.0
        remaining = semis_budget - amd_w

        non_amd = [x for x in semis_on if x != "AMD"]
        if remaining > 0 and non_amd:
            scores = normalize_positive(rs.loc[dt, non_amd])
            if float(scores.sum()) == 0:
                for x in non_amd:
                    row[x] += remaining / len(non_amd)
            else:
                for x in non_amd:
                    row[x] += remaining * float(scores.loc[x])
        elif remaining > 0 and not non_amd:
            add_def(row, remaining)

        if amd_w > 0:
            row["AMD"] += amd_w
    else:
        add_def(row, W_SEMIS)

    # Normalize drift
    s = float(row.sum())
    if abs(s - 1.0) > 1e-10 and s > 0:
        w_target.loc[dt] = row / s

# Execution: Fridays + cooldown
exec_days = px.index[px.index.weekday == 4] if EXECUTE_WEEKLY else px.index
w_exec = pd.DataFrame(index=exec_days, columns=assets, data=np.nan)

last_trade_day = None
prev_w = None
churn_flags = 0

for dt in exec_days:
    proposed = w_target.loc[dt].copy()
    hard_risk_off = (not bool(tech_on.loc[dt]))

    if prev_w is None:
        w_exec.loc[dt] = proposed.values
        prev_w = proposed
        last_trade_day = dt
        continue

    change = float((proposed - prev_w).abs().sum())
    td = trading_days_between(px.index, last_trade_day, dt)

    if (not hard_risk_off) and (td < MIN_TRADING_DAYS_BETWEEN_TRADES) and (change >= MIN_CHANGE_TO_COUNT_AS_TRADE):
        w_exec.loc[dt] = prev_w.values
        churn_flags += 1
    else:
        w_exec.loc[dt] = proposed.values
        if change >= MIN_CHANGE_TO_COUNT_AS_TRADE:
            last_trade_day = dt
        prev_w = proposed

# Expand to daily executed weights
w_d = pd.DataFrame(index=px.index, columns=assets, data=np.nan)
w_d.loc[w_exec.index, :] = w_exec.values
w_d = w_d.infer_objects(copy=False).ffill().fillna(0.0)

# Performance with costs
tc = TC_BPS_PER_1X / 10_000.0
dw = w_d.diff().abs().sum(axis=1).fillna(0.0)
cost = tc * dw

port_ret = (w_d * ret_d[assets]).sum(axis=1) - cost
equity = (1 + port_ret).cumprod()

annual_turnover = float(dw.mean() * 252 / 2)
avg_def_w = float(w_d[DEF].sum(axis=1).mean())
avg_risk_w = float(1.0 - avg_def_w)

summary = {
    "period_start": str(equity.index[0].date()),
    "period_end": str(equity.index[-1].date()),
    "cagr": cagr_calendar(equity),
    "maxdd": max_drawdown(equity),
    "vol": annualized_vol(port_ret),
    "sharpe": sharpe(port_ret, RFR),
    "sortino": sortino(port_ret, RFR),
    "turnover_annual": annual_turnover,
    "avg_def_weight": avg_def_w,
    "avg_risk_weight": avg_risk_w,
    "churn_flags": int(churn_flags),
    "latest_exec_date": str(w_exec.index[-1].date()),
}

# Save outputs
with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

pd.DataFrame([summary]).to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)

# last weights
w_exec.tail(30).to_csv(os.path.join(OUTDIR, "recent_exec_weights.csv"))

# equity + returns
pd.DataFrame({
    "equity": equity,
    "port_ret": port_ret,
}).to_csv(os.path.join(OUTDIR, "equity.csv"))

print("OK. Saved outputs/summary.json, outputs/summary.csv, outputs/recent_exec_weights.csv, outputs/equity.csv")
print(json.dumps(summary, indent=2))
