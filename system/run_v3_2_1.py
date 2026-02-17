import os, json, time
import numpy as np
import pandas as pd
import yfinance as yf

# ====== CONFIG ======
PLATFORMS = ["MSFT", "GOOGL"]
SEMIS     = ["NVDA", "KLAC", "AMD"]
INFRA     = ["VRT"]
DEF       = ["XLV", "IEF"]
TECH_KILL = "XLK"
ALL = sorted(set(PLATFORMS + SEMIS + INFRA + DEF + [TECH_KILL]))

START = "2006-01-01"
END = None

W_PLATFORM = 0.35
W_SEMIS    = 0.25
W_INFRA    = 0.15
W_DEF_BASE = 0.25
AMD_CAP_WITHIN_SEMIS = 0.12

DEF_XLV = 0.50
DEF_IEF = 0.50

WEEKLY_EMA_W = 30
DAILY_SMA    = 20
CONF_WIN     = 10
CONF_ON      = 7
CONF_OFF     = 3
RS_WIN       = 63

EXECUTE_WEEKLY = True
MIN_TRADING_DAYS_BETWEEN_TRADES = 15
MIN_CHANGE_TO_COUNT_AS_TRADE    = 0.05

TC_BPS_PER_1X = 5
RFR = 0.0

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ====== HELPERS ======
def max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    return float((eq/peak - 1.0).min())

def annualized_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(252))

def cagr_calendar(eq: pd.Series) -> float:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return float((eq.iloc[-1]/eq.iloc[0])**(1/years) - 1) if years > 0 else np.nan

def sharpe(r: pd.Series, rfr=0.0) -> float:
    ex = r - rfr/252
    v = r.std()
    return float((ex.mean()/v) * np.sqrt(252)) if v != 0 else np.nan

def sortino(r: pd.Series, rfr=0.0) -> float:
    ex = r - rfr/252
    d = r[r < 0].std()
    return float((ex.mean()/d) * np.sqrt(252)) if d != 0 else np.nan

def normalize_positive(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    tot = float(s.sum())
    return s*0.0 if tot <= 0 else s/tot

def trading_days_between(idx: pd.DatetimeIndex, a, b) -> int:
    if a is None: return 10**9
    return int(((idx > a) & (idx <= b)).sum())

def add_def(row: pd.Series, amount: float):
    if amount <= 0: return
    row[DEF[0]] += amount * DEF_XLV
    row[DEF[1]] += amount * DEF_IEF

def download_prices(tickers, start, end, tries=3, sleep_s=2):
    last_err = None
    for i in range(tries):
        try:
            df = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False, threads=False
            )
            if df is None or len(df) == 0:
                raise RuntimeError("yfinance returned empty dataframe")
            # handle multiindex columns
            if isinstance(df.columns, pd.MultiIndex):
                if "Close" not in df.columns.get_level_values(0):
                    raise RuntimeError(f"yfinance columns missing Close: {df.columns.levels[0].tolist()}")
                px = df["Close"]
            else:
                # sometimes returns single-level with Close already
                px = df.get("Close", df)
            if isinstance(px, pd.Series):
                px = px.to_frame()
            return px
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to download prices after {tries} tries: {last_err}")

# ====== DOWNLOAD ======
px = download_prices(ALL, START, END)
px = px.dropna(how="all")
if px.empty:
    raise RuntimeError("Price data is empty after dropna(how='all').")

px = px.ffill()

# Diagnostics: coverage & last price
coverage = px.notna().mean()
diag = pd.DataFrame({
    "ticker": coverage.index,
    "coverage": coverage.values,
    "last_price": [px[c].dropna().iloc[-1] if px[c].notna().any() else np.nan for c in px.columns],
    "last_date": [str(px[c].dropna().index[-1].date()) if px[c].notna().any() else "" for c in px.columns],
})
diag.to_csv(os.path.join(OUTDIR, "diagnostics.csv"), index=False)

bad = diag[diag["coverage"] < 0.80]
if len(bad) > 0:
    raise RuntimeError(f"Data coverage too low for: {bad['ticker'].tolist()} — see outputs/diagnostics.csv")

ret_d = px.pct_change().fillna(0.0)
px_w = px.resample("W-FRI").last().dropna(how="all").ffill()

# ====== WEEKLY GATE ======
weekly_gate = {}
for t in ALL:
    w = px_w[t]
    ema = w.ewm(span=WEEKLY_EMA_W, adjust=False).mean()
    gate_w = (w > ema).astype("boolean").fillna(False)
    weekly_gate[t] = gate_w.reindex(px.index, method="ffill").fillna(False)

weekly_gate = pd.DataFrame(weekly_gate, index=px.index)
tech_on = weekly_gate[TECH_KILL].astype(bool)

# ====== DAILY CONFIRM ======
sma = px.rolling(DAILY_SMA).mean()
above = (px > sma).astype("boolean").fillna(False)
cnt = above.rolling(CONF_WIN).sum()

confirm_on  = (cnt >= CONF_ON).astype("boolean").fillna(False)
confirm_off = (cnt <= CONF_OFF).astype("boolean").fillna(False)

def build_state(ticker: str) -> pd.Series:
    gate = weekly_gate[ticker].astype(bool)
    on_sig  = confirm_on[ticker].astype(bool) & gate
    off_sig = confirm_off[ticker].astype(bool) | (~gate)
    st = pd.Series(False, index=px.index, dtype=bool)
    on = False
    for dt in px.index:
        if bool(off_sig.loc[dt]): on = False
        elif bool(on_sig.loc[dt]): on = True
        st.loc[dt] = on
    return st

state = pd.DataFrame({t: build_state(t) for t in (PLATFORMS + SEMIS + INFRA)}, index=px.index)
rs = px.pct_change(RS_WIN).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ====== TARGET WEIGHTS ======
assets = PLATFORMS + SEMIS + INFRA + DEF
w_target = pd.DataFrame(0.0, index=px.index, columns=assets)

for dt in px.index:
    row = w_target.loc[dt]
    add_def(row, W_DEF_BASE)

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
            for x in plat_on: row[x] += W_PLATFORM / len(plat_on)
        else:
            for x in plat_on: row[x] += W_PLATFORM * float(scores.loc[x])
    else:
        add_def(row, W_PLATFORM)

    # Infra
    if infra_on:
        row[infra_on[0]] += W_INFRA
    else:
        add_def(row, W_INFRA)

    # Semis
    if semis_on:
        budget = W_SEMIS
        amd_w = min(AMD_CAP_WITHIN_SEMIS, budget) if "AMD" in semis_on else 0.0
        rem = budget - amd_w
        non_amd = [x for x in semis_on if x != "AMD"]

        if rem > 0 and non_amd:
            scores = normalize_positive(rs.loc[dt, non_amd])
            if float(scores.sum()) == 0:
                for x in non_amd: row[x] += rem / len(non_amd)
            else:
                for x in non_amd: row[x] += rem * float(scores.loc[x])
        elif rem > 0 and not non_amd:
            add_def(row, rem)

        if amd_w > 0: row["AMD"] += amd_w
    else:
        add_def(row, W_SEMIS)

    s = float(row.sum())
    if abs(s - 1.0) > 1e-10 and s > 0:
        w_target.loc[dt] = row / s

# ====== EXECUTION ======
exec_days = px.index[px.index.weekday == 4] if EXECUTE_WEEKLY else px.index
w_exec = pd.DataFrame(index=exec_days, columns=assets, data=np.nan)

last_trade = None
prev = None
churn = 0

for dt in exec_days:
    prop = w_target.loc[dt].copy()
    hard_off = (not bool(tech_on.loc[dt]))

    if prev is None:
        w_exec.loc[dt] = prop
        prev = prop
        last_trade = dt
        continue

    change = float((prop - prev).abs().sum())
    td = trading_days_between(px.index, last_trade, dt)

    if (not hard_off) and (td < MIN_TRADING_DAYS_BETWEEN_TRADES) and (change >= MIN_CHANGE_TO_COUNT_AS_TRADE):
        w_exec.loc[dt] = prev
        churn += 1
    else:
        w_exec.loc[dt] = prop
        if change >= MIN_CHANGE_TO_COUNT_AS_TRADE:
            last_trade = dt
        prev = prop

if (w_exec.fillna(0.0).sum(axis=1) == 0).all():
    raise RuntimeError("Executed weights are all zero — aborting.")

w_daily = pd.DataFrame(index=px.index, columns=assets, data=np.nan)
w_daily.loc[w_exec.index, :] = w_exec.values
w_daily = w_daily.infer_objects(copy=False).ffill().fillna(0.0)

# ====== PERFORMANCE ======
tc = TC_BPS_PER_1X / 10_000.0
dw = w_daily.diff().abs().sum(axis=1).fillna(0.0)
cost = tc * dw

port_ret = (w_daily * ret_d[assets]).sum(axis=1) - cost
equity = (1 + port_ret).cumprod()

summary = {
    "period_start": str(equity.index[0].date()),
    "period_end": str(equity.index[-1].date()),
    "cagr": cagr_calendar(equity),
    "maxdd": max_drawdown(equity),
    "vol": annualized_vol(port_ret),
    "sharpe": sharpe(port_ret, RFR),
    "sortino": sortino(port_ret, RFR),
    "turnover_annual": float(dw.mean() * 252 / 2),
    "avg_def_weight": float(w_daily[DEF].sum(axis=1).mean()),
    "avg_risk_weight": float(1.0 - w_daily[DEF].sum(axis=1).mean()),
    "churn_flags": int(churn),
    "latest_exec_date": str(w_exec.index[-1].date()),
}

with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

pd.DataFrame([summary]).to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)
w_exec.tail(30).to_csv(os.path.join(OUTDIR, "recent_exec_weights.csv"))
pd.DataFrame({"equity": equity, "port_ret": port_ret}).to_csv(os.path.join(OUTDIR, "equity.csv"))

print("OK — outputs written. Latest exec:", summary["latest_exec_date"])
print(json.dumps(summary, indent=2))
