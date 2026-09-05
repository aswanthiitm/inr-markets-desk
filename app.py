"""Streamlit desk dashboard: curve, forwards, structures and hedge policy in one place.

    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from imd.curves import ZeroCurve, fit_nss, instruments_from_ccil, nss_from_ccil  # noqa: E402
from imd.data import (  # noqa: E402
    fetch_ccil_tenorwise_yields, fetch_ccil_zcyc_params, fetch_ust_curve, fetch_yahoo_series,
)
from imd.hedging import ExposureSchedule, HedgePolicy, compare_policies, simulate_paths  # noqa: E402
from imd.pricing import cip_forward, forward_points, garman_kohlhagen, zero_cost_collar  # noqa: E402
from imd.pricing.rates import InterestRateSwap  # noqa: E402

st.set_page_config(page_title="INR Markets Desk", layout="wide")
GRID = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30])


@st.cache_data(ttl=3600)
def load():
    yields = fetch_ccil_tenorwise_yields()
    yields = yields[yields["date"] == yields["date"].max()]
    zcyc = fetch_ccil_zcyc_params()
    ust = fetch_ust_curve([2026]).iloc[-1]
    fx = fetch_yahoo_series("USDINR", "10y")
    return yields, zcyc, ust, fx


yields, zcyc, ust_last, fx = load()
official = nss_from_ccil(zcyc.iloc[0])
inr = ZeroCurve.from_nss(official, "continuous", "CCIL ZCYC")
usd = ZeroCurve.from_nodes(np.asarray(ust_last.index, float), ust_last.to_numpy(), "continuous")
spot_live = float(fx.iloc[-1])
returns = np.log(fx).diff().dropna()

st.title("INR Markets Desk")
st.caption(
    f"INR sovereign curve from CCIL's published ZCYC (as of {zcyc.iloc[0]['date'].date()}), "
    f"USD curve from the US Treasury daily par curve, USD/INR spot from live quotes. "
    "Every number below is computed at load time — nothing is hard-coded."
)

spot = st.sidebar.number_input("USD/INR spot", value=round(spot_live, 4), step=0.05, format="%.4f")
vol = st.sidebar.slider("USD/INR volatility (%)", 3.0, 15.0,
                        float(round(returns.tail(252).std() * np.sqrt(252) * 100, 1)), 0.1) / 100
tab1, tab2, tab3, tab4 = st.tabs(["Curve", "FX forwards", "Structures", "Hedge policy"])

with tab1:
    fitted, diag = fit_nss(instruments_from_ccil(yields), compounding="continuous")
    mine = ZeroCurve.from_nss(fitted, "continuous")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1Y zero", f"{inr.zero(1.0):.2f}%")
    c2.metric("10Y zero", f"{inr.zero(10.0):.2f}%")
    c3.metric("1s10s slope", f"{(inr.zero(10.0) - inr.zero(1.0)) * 100:.0f} bp")
    c4.metric("fit error vs benchmarks", f"{diag['rmse_bp']:.1f} bp")

    fig = go.Figure()
    fig.add_scatter(x=GRID, y=inr.zero(GRID), name="CCIL official ZCYC", line=dict(width=3))
    fig.add_scatter(x=GRID, y=mine.zero(GRID), name="fitted NSS", line=dict(dash="dash"))
    fig.add_scatter(x=yields["tenor_years"], y=yields["ytm"], mode="markers",
                    name="observed YTM", marker=dict(size=9))
    fig.update_layout(height=430, xaxis_title="maturity (years)", yaxis_title="rate (%)",
                      margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Par INR OIS rates implied by the curve")
    st.dataframe(pd.DataFrame([
        {"tenor": f"{t}Y", "par rate (%)": round(inr.par_swap_rate(t), 4),
         "PV01 (₹ per ₹100 cr)": round(InterestRateSwap(t, inr.par_swap_rate(t), 1e9).pv01(inr), 0)}
        for t in (1, 2, 3, 5, 7, 10)
    ]), hide_index=True, width="stretch")

with tab2:
    tenors = np.array([1, 2, 3, 6, 9, 12]) / 12
    fwd = cip_forward(spot, inr.df(tenors), usd.df(tenors))
    frame = pd.DataFrame({
        "tenor": [f"{int(t*12)}M" for t in tenors],
        "forward": np.round(fwd, 4),
        "points (paise)": np.round(forward_points(spot, fwd), 1),
        "premium (% p.a.)": np.round((fwd / spot - 1) / tenors * 100, 3),
        "INR zero (%)": np.round(inr.zero(tenors), 3),
        "USD zero (%)": np.round(usd.zero(tenors), 3),
    })
    st.dataframe(frame, hide_index=True, width="stretch")
    st.info(
        "The premium is the rate differential and nothing else. If a broker quotes you a "
        "forward away from this, the difference is the **forward basis** — the price of "
        "onshore dollar funding, not a view on the rupee."
    )

with tab3:
    tenor = st.slider("option tenor (years)", 0.25, 3.0, 1.0, 0.25)
    r_inr, r_usd = inr.zero(tenor) / 100, usd.zero(tenor) / 100
    fwd = float(cip_forward(spot, inr.df(tenor), usd.df(tenor))[0])
    cap_pct = st.slider("collar cap, % above forward", 0.5, 6.0, 2.0, 0.25) / 100
    cap = fwd * (1 + cap_pct)
    floor = zero_cost_collar(spot, fwd, r_inr, r_usd, vol, tenor, cap)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{tenor:g}Y forward", f"{fwd:.4f}")
    c2.metric("collar cap / floor", f"{cap:.3f} / {floor:.3f}")
    c3.metric("ATMF call premium",
              f"{float(garman_kohlhagen(spot, fwd, r_inr, r_usd, vol, tenor, 'call')):.3f}")
    grid = np.linspace(spot * 0.85, spot * 1.15, 250)
    fig = go.Figure()
    fig.add_scatter(x=grid, y=grid, name="unhedged", line=dict(dash="dot"))
    fig.add_scatter(x=grid, y=np.full_like(grid, fwd), name="forward")
    fig.add_scatter(x=grid, y=np.clip(grid, floor, cap), name="zero-cost collar",
                    line=dict(width=3))
    fig.update_layout(height=420, xaxis_title="USD/INR at maturity",
                      yaxis_title="effective rate paid", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

with tab4:
    monthly = st.number_input("monthly USD payable", value=1_000_000, step=100_000)
    exposure = ExposureSchedule.monthly(12, monthly)
    fwd_curve = cip_forward(spot, inr.df(exposure.tenors_years), usd.df(exposure.tenors_years))
    paths = simulate_paths(spot, exposure.tenors_years, 12_000, returns=returns, seed=11)
    table = compare_policies(
        [HedgePolicy("Unhedged", "unhedged"),
         HedgePolicy("100% forward", "forward", 1.0),
         HedgePolicy("Layered 100% to 30%", "forward", np.linspace(1.0, 0.3, 12)),
         HedgePolicy("Zero-cost collar", "collar", 1.0, cap_over_forward=0.02, vol=vol),
         HedgePolicy("50% fwd + 50% call", "blend", 1.0, option_share=0.5, vol=vol)],
        exposure, spot, fwd_curve, inr.zero(1.0) / 100, usd.zero(1.0) / 100, paths,
        budget_rate=float(np.mean(fwd_curve)),
    )
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    st.caption("CFaR = 95th-percentile rupee cost above the budget rate, in ₹ crore.")
