import os, json, time
import numpy as np
import pandas as pd
import yfinance as yf

# ====== CONFIG ======
PLATFORMS = ["MSFT", "GOOGL"]
SEMIS     = ["NVDA", "KLAC", "AMD"]
INFRA     = ["VRT"]                 # optional (may be missing on GH)
DEF       = ["XLV", "IEF"]
TECH_KILL = "XLK"

START = "2006-01-01"
END = None

# Allocation (official)
W_PLATFORM = 0.35
W_SEMIS    = 0.25
W_INFRA    = 0.15
W_DEF_BASE = 0.25
AMD_CAP_WITHIN_SEMIS = 0.12

# DEF split 50/50 (official)
DEF_XLV = 0.50
DEF_IEF = 0.50

# Signals
WEEKLY_EMA_W = 30
DAILY_SMA    = 20
CONF_WIN     = 10
CONF_ON      = 7
CONF_OFF     = 3
RS_WIN       = 63

# Execution
EXECUTE_WEEKLY = True
MIN_TRADING_DAYS_BETWEEN_TRADES = 15
MIN_CHANGE_TO_COUNT_AS_TRADE    = 0.05

# Costs
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
    for _ in range(tries):
        try:
            df = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False, threads=False
            )
            if df is None or len(df) == 0:
                raise RuntimeError("yfinance returned empty dataframe")

            if isinstance(df.columns, pd.MultiIndex):
                px = df["Close"]
            else:
                px = df.get("Close", df)

            if isinstance(px, pd.Series):
                px = px.to_frame()

            return px
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to download prices after {tries} tries: {last_err}")

# ====== DOWNLOAD ======
ALL = sorted(set(PLATFORMS + SEMIS + INFRA + DEF + [TECH_KILL]))
px = download_prices(ALL, START, END).dropna(how="all").ffill()
if px.empty:
    raise RuntimeError("Price data empty.")

# Diagnostics
coverage = px.notna().mean()
diag = pd.DataFrame({
    "ticker": coverage.index,
    "coverage": coverage.values,
    "last_price": [px[c].dropna().iloc[-1] if px[c].notna().any() else np.nan for c in px.columns],
    "last_date":  [str(px[c].dropna().index[-1].date()) if px[c].notna().any() else "" for c in px.columns],
})
diag.to_csv(os.path.join(OUTDIR, "diagnostics.csv"), index=False)

# Hard-fail only if CORE missing badly
CORE_TICKERS = set(PLATFORMS + SEMIS + DEF + [TECH_KILL])
bad_core = [t for t in coverage.index if (t in CORE_TICKERS and float(coverage[t]) < 0.80)]
if bad_core:
    raise RuntimeError(f"Data coverage too low for CORE tickers: {bad_core} — see outputs/diagnostics.csv")

# INFRA optional
infra_enabled = True
for t in INFRA:
    if t in coverage.index and float(coverage[t]) < 0.80:
        infra_enabled = False

W_INFRA_EFFECTIVE = W_INFRA if infra_enabled else 0.0
W_DEF_BASE_EFFECTIVE = W_DEF_BASE + (W_INFRA if not infra_enabled else 0.0)

ret_d = px.pct_change().fillna(0.0)
px_w = px.resample("W-FRI").last().dropna(how="all").ffill()

# ====== WEEKLY GATE ======
weekly_gate = {}
for t in px.columns:
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

STATE_UNI = PLATFORMS + SEMIS + (INFRA if infra_enabled else [])
state = pd.DataFrame({t: build_state(t) for t in STATE_UNI}, index=px.index)
rs = px.pct_change(RS_WIN).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ====== TARGET WEIGHTS ======
assets = PLATFORMS + SEMIS + (INFRA if infra_enabled else []) + DEF
w_target = pd.DataFrame(0.0, index=px.index, columns=assets)

for dt in px.index:
    row = w_target.loc[dt]
    add_def(row, W_DEF_BASE_EFFECTIVE)

    # Kill-switch
    if not bool(tech_on.loc[dt]):
        w_target.loc[dt, :] = 0.0
        w_target.loc[dt, DEF[0]] = DEF_XLV
        w_target.loc[dt, DEF[1]] = DEF_IEF
        continue

    plat_on  = [x for x in PLATFORMS if bool(state.loc[dt, x])]
    semis_on = [x for x in SEMIS if bool(state.loc[dt, x])]
    infra_on = [x for x in INFRA if infra_enabled and bool(state.loc[dt, x])]

    # Platforms
    if plat_on:
        scores = normalize_positive(rs.loc[dt, plat_on])
        if float(scores.sum()) == 0:
            for x in plat_on: row[x] += W_PLATFORM / len(plat_on)
        else:
            for x in plat_on: row[x] += W_PLATFORM * float(scores.loc[x])
    else:
        add_def(row, W_PLATFORM)

    # Infra (optional)
    if infra_enabled and infra_on:
        row[infra_on[0]] += W_INFRA_EFFECTIVE
    else:
        add_def(row, W_INFRA_EFFECTIVE)

    # Semis (AMD cap)
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

    # Normalize (critical)
    s = float(row.sum())
    if s > 0 and abs(s - 1.0) > 1e-10:
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

# ---- HARD NORMALIZE EXEC (the GH fix) ----
row_sums = w_exec.fillna(0.0).sum(axis=1)

# if any exec row sum == 0 => set to 100% DEF
zero_rows = row_sums == 0
if zero_rows.any():
    w_exec.loc[zero_rows, :] = 0.0
    w_exec.loc[zero_rows, DEF[0]] = DEF_XLV
    w_exec.loc[zero_rows, DEF[1]] = DEF_IEF
    row_sums = w_exec.fillna(0.0).sum(axis=1)

# normalize any non-1 sums
need_norm = (row_sums > 0) & (abs(row_sums - 1.0) > 1e-8)
w_exec.loc[need_norm, :] = w_exec.loc[need_norm, :].div(row_sums[need_norm], axis=0)

# verify last row
last_sum = float(w_exec.iloc[-1].fillna(0.0).sum())
if not (0.9999 <= last_sum <= 1.0001):
    raise RuntimeError(f"Executed weights do not sum to 1 (last_sum={last_sum}).")

# Expand to daily
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
    "infra_enabled": bool(infra_enabled),
    "latest_exec_sum": last_sum,
}

with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

pd.DataFrame([summary]).to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)

# add sum column to make debugging obvious
w_out = w_exec.copy()
w_out["WEIGHT_SUM"] = w_out.fillna(0.0).sum(axis=1)
w_out.tail(30).to_csv(os.path.join(OUTDIR, "recent_exec_weights.csv"), float_format="%.6f")

pd.DataFrame({"equity": equity, "port_ret": port_ret}).to_csv(os.path.join(OUTDIR, "equity.csv"))

# also save latest exec weights json
latest = w_exec.iloc[-1].fillna(0.0).to_dict()
with open(os.path.join(OUTDIR, "latest_exec_weights.json"), "w") as f:
    json.dump(latest, f, indent=2)

print("OK — outputs written. infra_enabled =", infra_enabled, "| latest_exec =", summary["latest_exec_date"], "| sum =", last_sum)
print(json.dumps(summary, indent=2))
