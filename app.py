"""Does the risk label survive a crash? A dashboard for the FRA group project.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analytics as an
from config import (BENCHMARK, DATA_START, INVESTMENT, PERIODS, PORTFOLIO_A, PORTFOLIO_B,
                    RISK_FREE, TODAY_YEARS, DEFAULT_CALM, DEFAULT_CRISIS)
from demo_data import synthetic_prices

st.set_page_config(page_title="Risk labels under stress", page_icon="📉", layout="wide")

COL_A, COL_B, COL_N = "#D9480F", "#1C7ED6", "#868E96"
CSV_PATH = Path(__file__).parent / "data" / "prices.csv"

st.markdown(
    """
<style>
.block-container {padding-top: 2.2rem; max-width: 1250px;}
.verdict {border-left: 4px solid #1C7ED6; background: rgba(28,126,214,.07); padding: .7rem 1rem;
          border-radius: 0 6px 6px 0; margin: .4rem 0 .8rem 0;}
.verdict b {font-weight: 650;}
.small {font-size: .85rem; opacity: .75;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- formatting
def inr(x: float) -> str:
    """Rupees with Indian digit grouping, e.g. 1,50,000."""
    s = f"{abs(round(x)):d}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if x < 0 else "") + "₹" + s


def inr_short(x: float) -> str:
    a = abs(x)
    if a >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if a >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    return inr(x)


def pct(x: float, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading prices...", ttl=6 * 3600)
def load_prices(mode: str):
    """Returns (prices, source_label). source_label: csv | yahoo | demo | failed."""
    if mode.startswith("Demo"):
        return synthetic_prices(), "demo"
    if mode.startswith("Auto") and CSV_PATH.exists():
        return pd.read_csv(CSV_PATH, index_col=0, parse_dates=True), "csv"
    try:
        import yfinance as yf

        tickers = list(PORTFOLIO_A) + list(PORTFOLIO_B) + [BENCHMARK]
        px = yf.download(tickers, start=DATA_START, auto_adjust=True, progress=False)["Close"]
        px = px.dropna(how="all")
        if px.shape[0] > 500 and px.shape[1] >= 10:
            return px, "yahoo"
    except Exception:
        pass
    return synthetic_prices(), "failed"


@st.cache_data(show_spinner="Building efficient frontier...")
def cached_frontier(mu: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    return an.frontier_curve(mu, cov)


# ----------------------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header("Controls")
default_mode = os.environ.get("FRA_DATA_MODE", "Auto (CSV, else Yahoo)")
modes = ["Auto (CSV, else Yahoo)", "Live from Yahoo", "Demo (synthetic, NOT real)"]
mode = sb.selectbox("Data source", modes, index=modes.index(default_mode) if default_mode in modes else 0)

investment = sb.number_input("Investment per portfolio (₹)", min_value=10_000, value=INVESTMENT, step=100_000)
cl = sb.slider("VaR / ES confidence level", 90.0, 99.5, 95.0, 0.5, format="%.1f%%") / 100
horizon = sb.slider("Horizon (trading days)", 1, 10, 1, help="1-day figures are scaled by sqrt(days).")
rf = sb.number_input("Risk-free rate (annual, %)", value=RISK_FREE * 100, step=0.25) / 100
today_years = sb.slider("'Today' window (years of recent data)", 1, 5, TODAY_YEARS)


def period_picker(label: str, default: str):
    names = list(PERIODS) + ["Custom dates"]
    pick = sb.selectbox(label, names, index=names.index(default))
    if pick == "Custom dates":
        c1, c2 = sb.columns(2)
        s = c1.date_input(f"{label}: start", pd.Timestamp("2011-01-01"), key=label + "s")
        e = c2.date_input(f"{label}: end", pd.Timestamp("2011-12-31"), key=label + "e")
        return f"Custom: {s:%b %Y} to {e:%b %Y}", (pd.Timestamp(s), pd.Timestamp(e))
    return pick, tuple(map(pd.Timestamp, PERIODS[pick]))


crisis_name, crisis_win = period_picker("Crisis period", DEFAULT_CRISIS)
calm_name, calm_win = period_picker("Calm period", DEFAULT_CALM)

# ----------------------------------------------------------------------------- load + prepare
prices, source = load_prices(mode)
if source == "failed":
    st.error("Could not reach Yahoo Finance and no data/prices.csv was found, so this page is showing "
             "**synthetic demo data (not real)**. Run `python fetch_data.py` and commit data/prices.csv.")
elif source == "demo":
    st.warning("Synthetic demo data is selected. These are NOT real prices; they only show how the dashboard works.")

all_tickers = list(PORTFOLIO_A) + list(PORTFOLIO_B)
missing_cols = [t for t in all_tickers + [BENCHMARK] if t not in prices.columns]
if BENCHMARK in missing_cols:
    st.error("Benchmark (Nifty 50) is missing from the data.")
    st.stop()
rets_all = an.returns_from_prices(prices)
bench = rets_all[BENCHMARK].dropna()
end = rets_all.index.max()
today_name = f"Today (last {today_years}y)"
today_win = (end - pd.DateOffset(years=today_years) + pd.Timedelta(days=1), end)

era_opts = [today_name, crisis_name, calm_name] + [p for p in PERIODS if p not in (crisis_name, calm_name)]
era_sel = sb.multiselect("Eras for the optimiser test", era_opts, default=era_opts,
                         help="Each era gets its own efficient frontier.")
if today_name not in era_sel:
    era_sel = [today_name] + era_sel
era_windows = {today_name: today_win, crisis_name: crisis_win, calm_name: calm_win}
for p, w in PERIODS.items():
    era_windows.setdefault(p, tuple(map(pd.Timestamp, w)))
needed = [era_windows[e] for e in set(era_sel) | {crisis_name, calm_name}]

present = [t for t in all_tickers if t in prices.columns]
usable = an.usable_tickers(rets_all[present], needed)
colsA = [t for t in PORTFOLIO_A if t in usable]
colsB = [t for t in PORTFOLIO_B if t in usable]

# data coverage table
first_dates = prices[present].apply(lambda s: s.first_valid_index())
coverage = pd.DataFrame({
    "Portfolio": ["A" if t in PORTFOLIO_A else "B" for t in present],
    "First price": [first_dates[t].date() if pd.notna(first_dates[t]) else None for t in present],
    "Used": ["yes" if t in usable else "dropped (incomplete data in a chosen period)" for t in present],
}, index=present)

# optional custom weights
with sb.expander("Custom weights (optional)"):
    st.caption("Default is equal weight inside each portfolio. Edit and the whole dashboard updates.")
    w_df = pd.DataFrame({"Ticker": all_tickers,
                         "Weight %": [100 / len(PORTFOLIO_A)] * len(PORTFOLIO_A) + [100 / len(PORTFOLIO_B)] * len(PORTFOLIO_B)})
    w_edit = st.data_editor(w_df, hide_index=True, disabled=["Ticker"], key="wedit", height=300)
raw_w = w_edit.set_index("Ticker")["Weight %"]
wA = an.normalise(raw_w.reindex(colsA).fillna(0)) if colsA else pd.Series(dtype=float)
wB = an.normalise(raw_w.reindex(colsB).fillna(0)) if colsB else pd.Series(dtype=float)

target_override = sb.checkbox("Override target return for the optimiser test")
target_manual = sb.number_input("Target annual return (%)", value=15.0, step=1.0) / 100 if target_override else None

# ----------------------------------------------------------------------------- guard
st.title("Did the risk label survive the crash?")
st.caption(f"Two ₹{investment/1e5:.0f}-lakh portfolios of Indian equities, stress-tested through a crisis and a calm market, "
           f"then rebuilt with the Markowitz optimiser using each era's data. Data to {end:%d %b %Y}.")

if len(colsA) < 3 or len(colsB) < 3:
    st.error("Too few stocks have clean data for all chosen periods. Pick later periods or swap tickers in config.py.")
    st.dataframe(coverage, width="stretch")
    st.stop()
if len(colsA) < 10 or len(colsB) < 10:
    st.warning(f"Portfolio A has {len(colsA)} and B has {len(colsB)} usable stocks; the brief asks for at least 10 each. "
               "See the data coverage table in the Portfolios tab and swap tickers in config.py.")

PORTS = {"A": (colsA, wA, COL_A, "Portfolio A (high risk)"), "B": (colsB, wB, COL_B, "Portfolio B (low risk)")}


# ----------------------------------------------------------------------------- Test 1 maths
def summarise(win):
    s, e = win
    rb = an.window(bench, s, e)
    out, series = {}, {}
    for k, (cols, w, _, _) in PORTS.items():
        assets = an.window(rets_all[cols], s, e).fillna(0)
        pr = assets @ w
        out[k] = an.period_summary(pr, rb, assets, cl, rf)
        series[k] = pr
    out["Nifty"] = an.period_summary(rb, rb, rb.to_frame(), cl, rf)
    series["Nifty"] = rb
    return out, series


T1 = {"crisis": summarise(crisis_win), "calm": summarise(calm_win)}
scale = math.sqrt(horizon)
es_amt = lambda per, k: T1[per][0][k]["ES hist"] * investment * scale
var_amt = lambda per, k, key="VaR hist": T1[per][0][k][key] * investment * scale

ratio_calm = T1["calm"][0]["A"]["ES hist"] / T1["calm"][0]["B"]["ES hist"]
ratio_crisis = T1["crisis"][0]["A"]["ES hist"] / T1["crisis"][0]["B"]["ES hist"]
A_riskier_both = all(T1[p][0]["A"]["ES hist"] > T1[p][0]["B"]["ES hist"] and
                     T1[p][0]["A"]["Ann. volatility"] > T1[p][0]["B"]["Ann. volatility"] for p in T1)
stress_mult = {k: T1["crisis"][0][k]["ES hist"] / T1["calm"][0][k]["ES hist"] for k in ("A", "B")}
breach = {k: an.breach_rate(T1["crisis"][1][k], T1["calm"][0][k]["VaR hist"]) for k in ("A", "B")}
B_vs_nifty = T1["crisis"][0]["B"]["ES hist"] / T1["crisis"][0]["Nifty"]["ES hist"]
corr_move = {k: (T1["calm"][0][k]["Avg pairwise corr."], T1["crisis"][0][k]["Avg pairwise corr."]) for k in ("A", "B")}

# ----------------------------------------------------------------------------- Test 2 maths
mu_today, cov_today = an.era_stats(an.window(rets_all[colsA + colsB], *today_win).fillna(0))
T2 = {}
for k, (cols, w, _, _) in PORTS.items():
    base_target = float(w @ mu_today[cols])
    target = target_manual if target_manual is not None else base_target
    eras = {}
    for name in era_sel:
        er = an.window(rets_all[cols], *era_windows[name]).fillna(0)
        mu, cov = an.era_stats(er)
        res = an.optimiser_at_target(mu, cov, target)
        res.update(mu=mu, cov=cov, n=len(er),
                   your_ret=float(w @ mu), your_vol=float(np.sqrt(w.values @ cov.values @ w.values)))
        if res["weights"] is not None:
            res["turnover"] = an.turnover(w, res["weights"])
        eras[name] = res
    T2[k] = {"target": target, "base_target": base_target, "eras": eras}

hist_eras = [e for e in era_sel if e != today_name]
tov = {k: [T2[k]["eras"][e]["turnover"] for e in hist_eras if T2[k]["eras"][e].get("turnover") is not None] for k in PORTS}
infeasible = {k: [e for e in hist_eras if T2[k]["eras"][e]["status"] == "infeasible"] for k in PORTS}

# ----------------------------------------------------------------------------- tabs
tab_over, tab_port, tab_t1, tab_t2, tab_method = st.tabs(
    ["Overview", "1. The portfolios", "2. Test 1: risk label", "3. Test 2: allocation", "Method & caveats"])

# ============================================================================ OVERVIEW
with tab_over:
    held = "held up" if A_riskier_both else "did not fully hold"
    st.markdown(
        f"<div class='verdict'><b>Test 1.</b> The high-risk label {held}: in the crisis ({crisis_name}) Portfolio A's "
        f"{cl*100:.1f}% Expected Shortfall was {pct(T1['crisis'][0]['A']['ES hist'])} of value per day against "
        f"{pct(T1['crisis'][0]['B']['ES hist'])} for B, i.e. {ratio_crisis:.1f}x (calm: {ratio_calm:.1f}x). "
        f"The A-to-B gap {'widened' if ratio_crisis > ratio_calm else 'narrowed'} in the crisis. "
        f"B's crisis ES was {stress_mult['B']:.1f}x its own calm ES and {B_vs_nifty:.2f}x the Nifty's, so the low-risk label "
        f"{'protected against the market' if B_vs_nifty < 1 else 'did NOT protect better than the market'}.</div>",
        unsafe_allow_html=True)
    msg = []
    for k in PORTS:
        part = f"Portfolio {k}: "
        if tov[k]:
            lo, hi = min(tov[k]), max(tov[k])
            span = pct(hi, 0) if round(lo * 100) == round(hi * 100) else f"{pct(lo, 0)} to {pct(hi, 0)}"
            part += f"in the eras where the target was reachable the optimiser would move {span} of the portfolio"
            if infeasible[k]:
                part += f"; in {len(infeasible[k])} of {len(hist_eras)} historical eras the target return was unreachable"
        elif hist_eras:
            part += f"the target return of {pct(T2[k]['target'])} was unreachable in all {len(hist_eras)} historical eras, so there is no optimal mix to compare"
        else:
            part += "select at least one historical era in the sidebar"
        msg.append(part)
    worst = max([x for k in PORTS for x in tov[k]], default=None)
    lead = ("Today's weights would not have been chosen by the optimiser in other eras. " if worst is not None and worst > 0.10
            else "Today's weights stay close to what the optimiser would pick in other eras. " if worst is not None else "")
    st.markdown("<div class='verdict'><b>Test 2.</b> " + lead + ". ".join(msg) + ".</div>", unsafe_allow_html=True)

    c = st.columns(4)
    c[0].metric("Per portfolio", inr_short(investment))
    c[1].metric("Stocks (A / B)", f"{len(colsA)} / {len(colsB)}")
    c[2].metric(f"Crisis {cl*100:.1f}% ES, A", inr_short(es_amt("crisis", "A")), help=f"{horizon}-day horizon")
    c[3].metric(f"Crisis {cl*100:.1f}% ES, B", inr_short(es_amt("crisis", "B")), help=f"{horizon}-day horizon")

    fig = go.Figure()
    for k, col in (("A", COL_A), ("B", COL_B)):
        fig.add_bar(name=PORTS[k][3], x=[f"Calm: {calm_name}", f"Crisis: {crisis_name}"],
                    y=[es_amt("calm", k), es_amt("crisis", k)], marker_color=col,
                    text=[inr_short(es_amt("calm", k)), inr_short(es_amt("crisis", k))], textposition="outside")
    fig.update_layout(title=f"Expected Shortfall ({cl*100:.1f}%, {horizon}-day) in rupees", yaxis_title="₹ loss in the bad tail",
                      barmode="group", height=380, margin=dict(t=60, b=40), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("Use the sidebar to change the rupee amount, confidence level, horizon, periods and weights. Everything recalculates.")

# ============================================================================ PORTFOLIOS
with tab_port:
    st.subheader("Why A is high risk and B is low risk")
    st.write("Labels are set from **today's data** (last "
             f"{today_years} years) using three numbers: annualised volatility, beta against the Nifty 50, and maximum drawdown. "
             "A is cyclical, leveraged-to-growth names (metals, PSU banks, real estate, capital goods); B is defensive "
             "(FMCG, pharma, IT, utilities).")
    today_rets = an.window(rets_all, *today_win)
    stock_tbl = an.stock_risk_table(today_rets[colsA + colsB].fillna(0), an.window(bench, *today_win))
    label_rows = []
    for k, (cols, w, _, nm) in PORTS.items():
        pr = an.window(rets_all[cols], *today_win).fillna(0) @ w
        label_rows.append({"Portfolio": nm, "Ann. volatility": an.ann_vol(pr), "Beta vs Nifty": an.beta(pr, an.window(bench, *today_win)),
                           "Max drawdown": an.max_drawdown(pr), "Avg stock volatility": stock_tbl.loc[cols, "Ann. volatility"].mean(),
                           "Avg stock beta": stock_tbl.loc[cols, "Beta vs Nifty"].mean()})
    lab = pd.DataFrame(label_rows).set_index("Portfolio")
    checks = {"Volatility": lab["Ann. volatility"].iloc[0] > lab["Ann. volatility"].iloc[1],
              "Beta": lab["Beta vs Nifty"].iloc[0] > lab["Beta vs Nifty"].iloc[1],
              "Drawdown": lab["Max drawdown"].iloc[0] < lab["Max drawdown"].iloc[1]}
    fmt = lab.copy()
    for c_ in ("Ann. volatility", "Max drawdown", "Avg stock volatility"):
        fmt[c_] = fmt[c_].map(pct)
    for c_ in ("Beta vs Nifty", "Avg stock beta"):
        fmt[c_] = fmt[c_].map(lambda v: f"{v:.2f}")
    st.dataframe(fmt, width="stretch")
    st.markdown("Label test: " + " · ".join(f"{k} {'✔' if v else '✘'}" for k, v in checks.items()) +
                ("  → A is riskier on every measure." if all(checks.values()) else "  → mixed; consider re-picking stocks."))

    left, right = st.columns([3, 2])
    with left:
        tbl = pd.DataFrame({
            "Portfolio": ["A" if t in colsA else "B" for t in colsA + colsB],
            "Company": [(PORTFOLIO_A | PORTFOLIO_B)[t][0] for t in colsA + colsB],
            "Sector": [(PORTFOLIO_A | PORTFOLIO_B)[t][1] for t in colsA + colsB],
            "Weight": [pct(wA[t] if t in wA else wB[t]) for t in colsA + colsB],
            "Volatility": stock_tbl["Ann. volatility"].map(pct).values,
            "Beta": stock_tbl["Beta vs Nifty"].map(lambda v: f"{v:.2f}").values,
            "Max DD": stock_tbl["Max drawdown"].map(pct).values,
        }, index=colsA + colsB)
        st.dataframe(tbl, width="stretch", height=480)
    with right:
        fv = go.Figure()
        for k, (cols, _, col, nm) in PORTS.items():
            fv.add_trace(go.Box(y=stock_tbl.loc[cols, "Ann. volatility"] * 100, name=f"{k}", marker_color=col, boxpoints="all"))
        fv.update_layout(title="Stock volatility by portfolio (%)", height=480, showlegend=False, yaxis_title="annualised %")
        st.plotly_chart(fv, width="stretch")
    st.markdown("**Data coverage** (stocks need clean prices in every chosen period)")
    st.dataframe(coverage, width="stretch", height=250)

# ============================================================================ TEST 1
with tab_t1:
    st.subheader("Test 1: did 'high risk' behave like high risk when it mattered?")
    st.caption(f"Crisis: {crisis_name}  |  Calm: {calm_name}  |  {cl*100:.1f}% confidence  |  {horizon}-day horizon  |  "
               f"Portfolio value {inr(investment)}. Historical VaR/ES use that window's own daily returns.")

    rows = []
    for per, label in (("calm", calm_name), ("crisis", crisis_name)):
        for k in ("A", "B", "Nifty"):
            s = T1[per][0][k]
            rows.append({"Period": ("Calm" if per == "calm" else "Crisis"), "Portfolio": {"A": "A (high)", "B": "B (low)", "Nifty": "Nifty 50"}[k],
                         "Days": s["Observations"], "Period return": pct(s["Period return"]), "Ann. vol": pct(s["Ann. volatility"]),
                         "Beta": f"{s['Beta vs Nifty']:.2f}", "Max drawdown": pct(s["Max drawdown"]), "Worst day": pct(s["Worst day"]),
                         "Sharpe": f"{s['Sharpe']:.2f}", "Avg corr": pct(s["Avg pairwise corr."], 0) if k != "Nifty" else "-",
                         "VaR (hist) ₹": inr(var_amt(per, k)), "VaR (normal) ₹": inr(var_amt(per, k, "VaR normal")),
                         "VaR (Cornish-Fisher) ₹": inr(var_amt(per, k, "VaR Cornish-Fisher")),
                         "ES (hist) ₹": inr(es_amt(per, k)), "ES (normal) ₹": inr(var_amt(per, k, "ES normal")),
                         "Tail days": s["Tail obs (hist.)"]})
    t1_df = pd.DataFrame(rows)
    st.dataframe(t1_df, hide_index=True, width="stretch")
    st.download_button("Download table (CSV)", t1_df.to_csv(index=False), "test1_results.csv", "text/csv")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("A / B expected-shortfall ratio", f"{ratio_crisis:.2f}x", f"{ratio_crisis - ratio_calm:+.2f} vs calm", delta_color="off")
    k2.metric("B crisis ES vs Nifty", f"{B_vs_nifty:.2f}x", "below market" if B_vs_nifty < 1 else "above market", delta_color="off")
    k3.metric("Calm-VaR breach rate in crisis, A", pct(breach["A"]), f"expected {pct(1 - cl)}", delta_color="off",
              help="How often crisis-period daily losses exceeded the 1-day VaR you would have computed in the calm period.")
    k4.metric("Calm-VaR breach rate in crisis, B", pct(breach["B"]), f"expected {pct(1 - cl)}", delta_color="off")

    st.markdown(
        f"**Reading it.** A's ES rose {stress_mult['A']:.1f}x from calm to crisis and B's rose {stress_mult['B']:.1f}x. "
        f"Average pairwise correlation inside A went {pct(corr_move['A'][0], 0)} → {pct(corr_move['A'][1], 0)}, "
        f"inside B {pct(corr_move['B'][0], 0)} → {pct(corr_move['B'][1], 0)}: diversification "
        f"{'weakened' if corr_move['B'][1] > corr_move['B'][0] else 'did not weaken'} in B when it was needed. "
        f"A calm-period VaR would have been breached on {pct(breach['A'])} (A) and {pct(breach['B'])} (B) of crisis days.")

    # return distributions
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Calm: {calm_name}", f"Crisis: {crisis_name}"), shared_yaxes=False)
    for j, per in enumerate(("calm", "crisis"), start=1):
        for k, col in (("A", COL_A), ("B", COL_B)):
            fig.add_trace(go.Histogram(x=T1[per][1][k] * 100, name=PORTS[k][3], marker_color=col, opacity=.6, nbinsx=45,
                                       histnorm="percent", showlegend=(j == 1)), row=1, col=j)
            fig.add_vline(x=-T1[per][0][k]["VaR hist"] * 100, line_dash="dash", line_color=col, row=1, col=j)
        fig.update_xaxes(title_text="daily return (%)", row=1, col=j)
    fig.update_layout(barmode="overlay", height=380, title=f"Daily return distributions (dashed = {cl*100:.1f}% historical VaR)",
                      legend=dict(orientation="h", y=-0.25), margin=dict(t=80))
    st.plotly_chart(fig, width="stretch")

    # growth of money
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Calm: {calm_name}", f"Crisis: {crisis_name}"))
    for j, per in enumerate(("calm", "crisis"), start=1):
        for k, col, nm in (("A", COL_A, "A"), ("B", COL_B, "B"), ("Nifty", COL_N, "Nifty 50")):
            wealth = investment * (1 + T1[per][1][k]).cumprod()
            fig.add_trace(go.Scatter(x=wealth.index, y=wealth.values, name=nm, line=dict(color=col, width=2),
                                     showlegend=(j == 1)), row=1, col=j)
    fig.update_layout(height=380, title="What happened to your money", yaxis_title="portfolio value (₹)",
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=80))
    st.plotly_chart(fig, width="stretch")

# ============================================================================ TEST 2
with tab_t2:
    st.subheader("Test 2: would the optimiser pick your weights in a different era?")
    st.markdown(
        "Take the annual return each portfolio earns on today's data as the **target**. For each era, estimate expected returns "
        "and covariances from that era only and ask the long-only Markowitz optimiser for the **lowest-risk mix that still "
        "reaches the target**. If the answer keeps changing, 'optimal' depends on which past you believe.")
    for k, (cols, w, col, nm) in PORTS.items():
        st.markdown(f"##### {nm}: target return {pct(T2[k]['target'])} a year"
                    + ("" if target_manual is None else f" (manual override; today's equal-weight return is {pct(T2[k]['base_target'])})"))
    pa, pb = st.tabs([PORTS["A"][3], PORTS["B"][3]])
    for tab, k in ((pa, "A"), (pb, "B")):
        cols, w, col, nm = PORTS[k]
        with tab:
            eras = T2[k]["eras"]
            target = T2[k]["target"]
            names = list(eras)
            ncols = min(3, len(names))
            nrows = math.ceil(len(names) / ncols)
            titles = []
            for n in names:
                st_ = eras[n]["status"]
                tag = {"optimal": "optimiser picks a different mix", "gmv": "target below min-risk return: holds min-variance mix",
                       "infeasible": "TARGET UNREACHABLE"}[st_]
                titles.append(f"{n}<br><sup>{tag}</sup>")
            fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles, vertical_spacing=0.2)
            for i, n in enumerate(names):
                r_, c_ = i // ncols + 1, i % ncols + 1
                e = eras[n]
                fr = cached_frontier(e["mu"], e["cov"])
                vols = np.sqrt(np.diag(e["cov"].values)) * 100
                fig.add_trace(go.Scatter(x=vols, y=e["mu"].values * 100, mode="markers", marker=dict(size=6, color="#ADB5BD"),
                                         text=[PORTFOLIO_A.get(t, PORTFOLIO_B.get(t))[0] for t in cols], name="single stocks",
                                         hovertemplate="%{text}<br>vol %{x:.1f}%<br>ret %{y:.1f}%", showlegend=(i == 0)), row=r_, col=c_)
                fig.add_trace(go.Scatter(x=fr["vol"] * 100, y=fr["ret"] * 100, mode="lines", line=dict(color="#343A40", width=2),
                                         name="efficient frontier", showlegend=(i == 0)), row=r_, col=c_)
                fig.add_trace(go.Scatter(x=[e["your_vol"] * 100], y=[e["your_ret"] * 100], mode="markers",
                                         marker=dict(size=14, color=col, symbol="star", line=dict(width=1, color="white")),
                                         name="your weights", showlegend=(i == 0)), row=r_, col=c_)
                if e["weights"] is not None:
                    fig.add_trace(go.Scatter(x=[e["vol"] * 100], y=[e["ret"] * 100], mode="markers",
                                             marker=dict(size=12, color="#212529", symbol="diamond"), name="optimiser at target",
                                             showlegend=(i == 0)), row=r_, col=c_)
                fig.add_hline(y=target * 100, line_dash="dot", line_color=col, row=r_, col=c_)
                fig.update_xaxes(title_text="risk (ann. vol %)", row=r_, col=c_)
                fig.update_yaxes(title_text="return (ann. %)", row=r_, col=c_)
            fig.update_layout(height=390 * nrows, title="Efficient frontier by era (dotted line = target return)",
                              legend=dict(orientation="h", y=-0.12 / nrows * 2), margin=dict(t=100))
            st.plotly_chart(fig, width="stretch")

            # summary table
            rows = []
            for n in names:
                e = eras[n]
                rows.append({
                    "Era": n, "Days": e["n"], "Best single-stock return": pct(e["max_ret"]),
                    "Target reachable?": "yes" if e["status"] != "infeasible" else "NO",
                    "Optimiser risk": pct(e["vol"]) if e["vol"] is not None else "n/a",
                    "Your risk (same era)": pct(e["your_vol"]),
                    "Re-allocation needed": pct(e["turnover"], 0) if e.get("turnover") is not None else "n/a",
                    "Stocks held >1% (optimiser / yours)": f"{int((e['weights'] > .01).sum())} / {int((w > .01).sum())}" if e["weights"] is not None else "n/a",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Re-allocation needed = half the sum of absolute weight differences: the share of the portfolio the optimiser would move.")

            # weight heatmap
            wm = pd.DataFrame({"Your weights": w})
            for n in names:
                wm[n] = eras[n]["weights"] if eras[n]["weights"] is not None else np.nan
            z = wm.values * 100
            text = np.where(np.isnan(z), "n/a", np.char.add(np.round(np.nan_to_num(z), 0).astype(int).astype(str), "%"))
            lbl = [PORTFOLIO_A.get(t, PORTFOLIO_B.get(t))[0] for t in wm.index]
            hm = go.Figure(go.Heatmap(z=z, x=[c.replace(" (", "<br>(") for c in wm.columns], y=lbl, text=text, texttemplate="%{text}",
                                      colorscale=[[0, "#F1F3F5"], [1, col]], zmin=0, zmax=max(30, np.nanmax(z)), showscale=False,
                                      hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>"))
            hm.update_layout(title="Weights: yours vs what the optimiser would hold in each era", height=60 + 34 * len(lbl),
                             yaxis=dict(autorange="reversed"), xaxis=dict(side="top"), margin=dict(t=110))
            st.plotly_chart(hm, width="stretch")
            infe = [n for n in names if eras[n]["status"] == "infeasible"]
            if infe:
                st.info("Target return unreachable in: " + ", ".join(infe) +
                        ". In those eras no long-only mix of these stocks earned that much, so there is no optimal weight to compare.")

# ============================================================================ METHOD
with tab_method:
    st.subheader("Method and caveats")
    st.markdown(f"""
**Data.** Adjusted daily closes (splits and dividends) from Yahoo Finance via `yfinance`, NSE tickers, Nifty 50 (`^NSEI`) as the market.
Simple daily returns; portfolios hold fixed weights, rebalanced daily. Stocks missing clean data in a chosen period are dropped from everything.

**Risk labels.** Annualised volatility (daily std x sqrt 252), beta vs Nifty (regression slope), max drawdown. Computed on the last {today_years} years.

**VaR and Expected Shortfall.** 1-day, scaled by sqrt(days) for longer horizons.
- *Historical*: the loss quantile of that window's own returns; ES is the average of losses beyond it.
- *Normal*: closed form from the window's mean and standard deviation.
- *Cornish-Fisher*: normal VaR adjusted for skew and fat tails.
With ~250 days and 99% confidence only about 3 observations sit in the tail, so treat 99% historical figures as rough (the table shows tail days).

**Extra measures and why.** Beta and Sharpe place the loss in market context; max drawdown shows the path, which VaR ignores;
average pairwise correlation tests whether diversification failed in the crisis; the *breach rate* shows how badly a calm-period VaR underestimates crisis losses.

**Optimiser.** For each era: annualised mean vector and covariance from that era's daily data, long-only, weights sum to 1, minimise variance
subject to expected return >= target. If the target is below the minimum-variance portfolio's return, the optimiser holds the minimum-variance portfolio;
if above the best single stock's return, it is reported as unreachable.

**Caveats.** Survivorship bias (stocks that still trade today); historical means are very noisy, so frontiers from short windows are unstable by construction,
which is part of the point; corporate actions (demergers, mergers) can distort some tickers; no costs, taxes or liquidity; not investment advice.
""")
