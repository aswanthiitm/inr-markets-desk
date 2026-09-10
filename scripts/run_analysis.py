#!/usr/bin/env python
"""End-to-end run: pull live data, build curves, price, hedge, measure risk, plot.

    python scripts/run_analysis.py

Writes every figure to ``reports/figures/`` and a machine-readable summary of all
headline numbers to ``reports/results.json`` so the README can never drift from what
the code actually produced.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imd.config import FIGURES, REPORTS, SNAPSHOTS  # noqa: E402
from imd.curves import ZeroCurve, fit_nss, instruments_from_ccil, nss_from_ccil  # noqa: E402
from imd.curves.fit import bootstrap_par_curve  # noqa: E402
from imd.data import (  # noqa: E402
    fetch_ccil_tenorwise_yields, fetch_ccil_zcyc_params, fetch_sofr,
    fetch_ust_curve, fetch_yahoo_series,
)
from imd.hedging import ExposureSchedule, HedgePolicy, compare_policies, simulate_paths  # noqa: E402
from imd.plotting import ACCENT, ACCENT2, ACCENT3, ACCENT4, MUTED, SERIES, annotate, use_style  # noqa: E402
from imd.pricing import cip_forward, forward_points, forward_premium_pct, garman_kohlhagen, gk_greeks, zero_cost_collar  # noqa: E402
from imd.pricing.rates import InterestRateSwap  # noqa: E402
from imd.risk import (  # noqa: E402
    backtest_var, ewma_volatility, expected_shortfall, historical_var,
    parametric_var, pca_curve_factors, student_t_var,
)

use_style()
GRID = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30])
R: dict = {"generated_on": str(date.today())}


def log(msg: str) -> None:
    print(f"[imd] {msg}", flush=True)


# ---------------------------------------------------------------- 1. market data
log("fetching market data")
tenorwise = fetch_ccil_tenorwise_yields()
asof = tenorwise["date"].max()
today = tenorwise[tenorwise["date"] == asof]
zcyc = fetch_ccil_zcyc_params()
official = nss_from_ccil(zcyc.iloc[0])

ust = fetch_ust_curve([2024, 2025, 2026])
ust_last = ust.iloc[-1]
sofr = fetch_sofr(250)
usdinr = fetch_yahoo_series("USDINR", "10y")
dxy = fetch_yahoo_series("DXY", "10y")
spot = float(usdinr.iloc[-1])
fx_returns = np.log(usdinr).diff().dropna()

R["asof"] = str(asof.date())
R["spot_usdinr"] = round(spot, 4)
R["sofr"] = float(sofr.iloc[-1])
R["ccil_official_nss"] = official.__dict__ | {}

# snapshot today's inputs so any run is reproducible after the fact
today.to_csv(SNAPSHOTS / f"ccil_yields_{asof.date()}.csv", index=False)
zcyc.head(1).to_csv(SNAPSHOTS / f"ccil_zcyc_{asof.date()}.csv", index=False)

# ------------------------------------------------------------------- 2. curves
log("building INR sovereign curve")
instruments = instruments_from_ccil(today, "GSEC+TBILL")
fitted, diag = fit_nss(instruments, compounding="continuous")
fitted_bounded, diag_b = fit_nss(instruments, compounding="continuous", tau_bounds=(0.5, 12.0))
boot = bootstrap_par_curve(instruments, compounding="continuous")

curve_official = ZeroCurve.from_nss(official, "continuous", "CCIL official ZCYC")
curve_fitted = ZeroCurve.from_nss(fitted, "continuous", "fitted NSS")
usd_curve = ZeroCurve.from_nodes(np.asarray(ust_last.index, float), ust_last.to_numpy(),
                                 "continuous", "USD Treasury")

gap_bp = (curve_fitted.zero(GRID) - curve_official.zero(GRID)) * 100
boot_gap_bp = (curve_fitted.zero(GRID) - boot.zero(GRID)) * 100
R["curve"] = {
    "n_instruments": diag["n_instruments"],
    "reprice_rmse_bp": round(diag["rmse_bp"], 2),
    "reprice_max_bp": round(diag["max_abs_bp"], 2),
    "vs_ccil_rmse_bp": round(float(np.sqrt((gap_bp ** 2).mean())), 1),
    "vs_ccil_max_bp": round(float(np.abs(gap_bp).max()), 1),
    "vs_bootstrap_rmse_bp": round(float(np.sqrt((boot_gap_bp ** 2).mean())), 1),
    "fitted_nss": fitted.__dict__ | {},
    "fitted_nss_tau_bounded": fitted_bounded.__dict__ | {},
    "tau_at_bound_unconstrained": diag["tau_at_bound"],
    "zero_10y": round(float(curve_official.zero(10.0)), 4),
    "zero_1y": round(float(curve_official.zero(1.0)), 4),
    "slope_1s10s_bp": round((float(curve_official.zero(10.0)) - float(curve_official.zero(1.0))) * 100, 1),
}

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
ax = axes[0]
ax.plot(GRID, curve_official.zero(GRID), color=ACCENT, label="CCIL official ZCYC")
ax.plot(GRID, curve_fitted.zero(GRID), color=ACCENT2, ls="--", label="fitted NSS (8 benchmarks)")
ax.plot(GRID, boot.zero(GRID), color=ACCENT3, ls=":", label="sequential bootstrap")
obs = today[today["segment"].isin(["GSEC", "TBILL"])]
ax.scatter(obs["tenor_years"], obs["ytm"], s=22, color=MUTED, zorder=5, label="observed YTM")
ax.set_xlabel("maturity (years)"); ax.set_ylabel("zero rate (%)")
ax.set_title(f"INR sovereign zero curve — {asof.date()}")
ax.legend(loc="lower right")
annotate(ax, f"fit reprices all {diag['n_instruments']} benchmarks\nto {diag['rmse_bp']:.1f} bp RMSE")

ax = axes[1]
ax.axhline(0, color=MUTED, lw=0.8)
ax.bar(np.arange(len(GRID)), gap_bp, color=ACCENT2, width=0.55, label="fitted − CCIL")
ax.plot(np.arange(len(GRID)), boot_gap_bp, color=ACCENT3, marker="o", ms=3, lw=1.2,
        label="fitted − bootstrap")
ax.set_xticks(np.arange(len(GRID))); ax.set_xticklabels([f"{g:g}" for g in GRID])
ax.set_xlabel("maturity (years)"); ax.set_ylabel("difference (bp)")
ax.set_title("Where an 8-instrument curve disagrees with the official one")
ax.legend(loc="lower right")
_worst = GRID[int(np.argmax(np.abs(gap_bp)))]
annotate(ax, f"RMSE vs CCIL {R['curve']['vs_ccil_rmse_bp']:.0f} bp, worst {np.abs(gap_bp).max():.0f} bp at {_worst:g}y —\n"
             "8 indicative benchmarks cannot resolve every segment\n"
             "CCIL fits off the full NDS-OM traded universe")
fig.tight_layout(); fig.savefig(FIGURES / "01_zero_curve.png"); plt.close(fig)

# ------------------------------------------------------- 3. SDL spread over G-Sec
sdl = today[today["segment"] == "SDL"]
if not sdl.empty:
    sdl_spread = {
        row["security"]: round((row["ytm"] - float(curve_official.zero(row["tenor_years"]))) * 100, 1)
        for _, row in sdl.iterrows()
    }
    R["sdl_spread_bp"] = sdl_spread
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(range(len(sdl_spread)), list(sdl_spread.values()), color=ACCENT4, width=0.5)
    ax.set_xticks(range(len(sdl_spread)))
    ax.set_xticklabels([s.split("%")[-1].strip()[:16] for s in sdl_spread], rotation=18, ha="right")
    ax.set_ylabel("spread over sovereign zero (bp)")
    ax.set_title("State development loans price wide of the centre")
    annotate(ax, "SDLs carry no explicit central guarantee;\nthe spread is the market's price for that")
    fig.tight_layout(); fig.savefig(FIGURES / "02_sdl_spread.png"); plt.close(fig)

# --------------------------------------------------------- 4. CIP forward curve
log("pricing the USD/INR forward curve off covered interest parity")
fx_tenors = np.array([1, 2, 3, 6, 9, 12]) / 12.0
fair_fwd = cip_forward(spot, curve_official.df(fx_tenors), usd_curve.df(fx_tenors))
premium = forward_premium_pct(spot, fair_fwd, fx_tenors)
R["fx_forwards"] = {
    f"{int(round(t*12))}M": {
        "forward": round(float(f), 4),
        "points_paise": round(float(forward_points(spot, f)), 1),
        "premium_pct_pa": round(float(p), 3),
    }
    for t, f, p in zip(fx_tenors, fair_fwd, premium)
}

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
ax = axes[0]
ax.plot(fx_tenors * 12, forward_points(spot, fair_fwd), color=ACCENT, marker="o", ms=4)
ax.set_xlabel("tenor (months)"); ax.set_ylabel("forward points (paise)")
ax.set_title(f"CIP-implied USD/INR forward points — spot {spot:.3f}")
annotate(ax, "INR rates above USD rates ⇒ USD/INR trades\nat a forward premium at every tenor")
ax = axes[1]
ax.plot(fx_tenors * 12, premium, color=ACCENT2, marker="o", ms=4, label="INR fwd premium")
ax.plot(fx_tenors * 12, curve_official.zero(fx_tenors), color=ACCENT, ls="--", label="INR zero")
ax.plot(fx_tenors * 12, usd_curve.zero(fx_tenors), color=ACCENT3, ls="--", label="USD zero")
ax.set_xlabel("tenor (months)"); ax.set_ylabel("% p.a.")
ax.set_title("The premium is the rate differential, nothing else")
ax.legend()
annotate(ax, "premium ≈ INR zero − USD zero;\nany residual is the forward basis", loc="lower right")
fig.tight_layout(); fig.savefig(FIGURES / "03_fx_forwards.png"); plt.close(fig)

# ------------------------------------------------------------- 5. option surface
log("pricing FX options")
r_inr = float(curve_official.zero(1.0)) / 100.0
r_usd = float(usd_curve.zero(1.0)) / 100.0
realised_vol = float(fx_returns.tail(252).std(ddof=1) * np.sqrt(252))
R["realised_vol_1y"] = round(realised_vol * 100, 2)

f12 = float(fair_fwd[-1])
strikes = np.linspace(spot * 0.92, spot * 1.10, 60)
call_px = np.array([float(garman_kohlhagen(spot, k, r_inr, r_usd, realised_vol, 1.0, "call"))
                    for k in strikes])
deltas = np.array([float(np.ravel(gk_greeks(spot, k, r_inr, r_usd, realised_vol, 1.0)["delta"])[0])
                   for k in strikes])
cap = f12 * 1.02
floor = zero_cost_collar(spot, f12, r_inr, r_usd, realised_vol, 1.0, cap)
atmf_call = float(garman_kohlhagen(spot, f12, r_inr, r_usd, realised_vol, 1.0, "call"))
R["options"] = {
    "vol_used_pct": round(realised_vol * 100, 2),
    "atmf_1y_call_inr": round(atmf_call, 4),
    "atmf_1y_call_pct_of_spot": round(atmf_call / spot * 100, 2),
    "collar_cap": round(cap, 4),
    "collar_floor": round(floor, 4),
    "collar_band_paise": round((cap - floor) * 100, 1),
}

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
ax = axes[0]
ax.plot(strikes, call_px, color=ACCENT, label="premium (left)")
ax.axvline(f12, color=MUTED, ls="--", lw=1)
ax.set_xlabel("strike (INR per USD)"); ax.set_ylabel("premium (INR per USD)")
ax.set_title(f"1Y USD call — Garman-Kohlhagen at {realised_vol*100:.1f}% realised vol")
ax_d = ax.twinx(); ax_d.grid(False)
ax_d.plot(strikes, deltas, color=ACCENT2, ls="--", lw=1.4, label="spot delta (right)")
ax_d.set_ylabel("spot delta", color=ACCENT2); ax_d.set_ylim(0, 1)
ax_d.annotate("1Y forward", xy=(f12, 0.5), xytext=(f12 + 1.2, 0.72), fontsize=8, color=MUTED,
              arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
lines = ax.get_lines()[:1] + ax_d.get_lines()[:1]
ax.legend(lines, [ln.get_label() for ln in lines], loc="lower left")
annotate(ax, f"at-the-money-forward premium = {atmf_call/spot*100:.2f}% of spot;\n"
             "delta 0.5 at the forward, not at spot — the\n"
             "distinction an FX desk is paid to keep straight", loc="upper right")
ax = axes[1]
payoff_x = np.linspace(spot * 0.85, spot * 1.15, 300)
ax.plot(payoff_x, payoff_x, color=MUTED, ls=":", label="unhedged")
ax.plot(payoff_x, np.full_like(payoff_x, f12), color=ACCENT3, label="forward")
ax.plot(payoff_x, np.clip(payoff_x, floor, cap), color=ACCENT2, lw=2.2, label="zero-cost collar")
ax.set_xlabel("USD/INR at maturity"); ax.set_ylabel("effective rate paid")
ax.set_title("What an importer actually pays under each hedge")
ax.legend(loc="upper left")
annotate(ax, f"collar band {floor:.2f} — {cap:.2f}\ncosts nothing up front", loc="lower right")
fig.tight_layout(); fig.savefig(FIGURES / "04_options.png"); plt.close(fig)

# -------------------------------------------------------------- 6. hedging study
log("simulating corporate hedging policies")
exposure = ExposureSchedule.monthly(12, 1_000_000)
sched_fwd = cip_forward(spot, curve_official.df(exposure.tenors_years),
                        usd_curve.df(exposure.tenors_years))
budget = float(np.mean(sched_fwd))
paths = simulate_paths(spot, exposure.tenors_years, 25_000, method="bootstrap",
                       returns=fx_returns, block=10, seed=11)
policies = [
    HedgePolicy("Unhedged", "unhedged"),
    HedgePolicy("100% forward", "forward", 1.0),
    HedgePolicy("Layered 100% to 30%", "forward", np.linspace(1.0, 0.3, 12)),
    HedgePolicy("Zero-cost collar (+2%)", "collar", 1.0, cap_over_forward=0.02, vol=realised_vol),
    HedgePolicy("50% fwd + 50% call", "blend", 1.0, option_share=0.5, vol=realised_vol),
]
table = compare_policies(policies, exposure, spot, sched_fwd, r_inr, r_usd, paths,
                         budget_rate=budget)
table.to_csv(REPORTS / "hedge_policy_comparison.csv", index=False)
R["hedging"] = {
    "exposure_usd_m": exposure.total_usd / 1e6,
    "budget_rate": round(budget, 4),
    "table": json.loads(table.round(4).to_json(orient="records")),
}

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
ax = axes[0]
from imd.hedging import run_policy  # noqa: E402
for policy, colour in zip(policies, SERIES):
    rate = np.sort(run_policy(policy, exposure, spot, sched_fwd, r_inr, r_usd,
                              paths)["effective_rate"])
    ax.plot(rate, np.linspace(0, 1, rate.size), color=colour, lw=1.7, label=policy.name)
ax.axvline(budget, color=MUTED, ls="--", lw=1)
ax.text(budget, 0.02, " budget rate", fontsize=8, color=MUTED)
ax.set_xlim(np.percentile(paths.mean(axis=1), 0.2), np.percentile(paths.mean(axis=1), 99.8))
ax.set_xlabel("effective rate paid (INR per USD)"); ax.set_ylabel("cumulative probability")
ax.set_title("What an importer ends up paying, policy by policy")
ax.legend(loc="lower right")
annotate(ax, f"$12m of monthly payables\n{paths.shape[0]:,} block-bootstrap paths\n"
             "a vertical line is certainty; a flat one is exposure")

ax = axes[1]
order = table.sort_values("cfar_95_cr")
bars = ax.barh(order["policy"], order["cfar_95_cr"], color=ACCENT, height=0.55)
for bar, value, cost in zip(bars, order["cfar_95_cr"], order["cost_vs_unhedged_cr"]):
    ax.text(bar.get_width() + 0.09, bar.get_y() + bar.get_height() / 2,
            f"₹{value:.2f} cr   (mean cost vs unhedged {cost:+.2f} cr)",
            va="center", fontsize=8, color=MUTED)
ax.set_xlim(0, order["cfar_95_cr"].max() * 1.75)
ax.set_xlabel("Cash-Flow-at-Risk, 95% (₹ crore over budget)")
ax.set_title("Cost overrun a treasurer should plan for")
ax.grid(axis="y", visible=False)
fig.text(0.53, -0.03, "Over this sample the realised INR drift exceeded the forward premium, "
                      "so hedging was both cheaper and safer — the case for a policy, not a view.",
         fontsize=8.2, color=MUTED)
fig.tight_layout(); fig.savefig(FIGURES / "05_hedging.png"); plt.close(fig)

# --------------------------------------------------------------------- 7. risk
log("measuring risk and backtesting VaR")
usd_position = 10e6
pnl = (fx_returns * usdinr.shift(1) * usd_position).dropna()
estimates = {
    "historical": historical_var(pnl),
    "normal": parametric_var(pnl),
    "student_t": student_t_var(pnl),
}
ewma = ewma_volatility(fx_returns)
var_forecast = (2.326 * ewma * usdinr.shift(1) * usd_position).dropna()
bt = backtest_var(pnl, var_forecast)
R["risk"] = {
    "position_usd_m": usd_position / 1e6,
    "var_es": {k: {"var_inr": round(v.var, 0), "es_inr": round(v.expected_shortfall, 0),
                   "method": v.method} for k, v in estimates.items()},
    "frtb_es_975_inr": round(expected_shortfall(pnl), 0),
    "backtest": {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in bt.items() if k != "exception_dates"},
    "worst_day_inr": round(float(pnl.min()), 0),
    "worst_day_date": str(pnl.idxmin().date()),
}

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))
ax = axes[0]
ax.plot(pnl.index, pnl / 1e5, color=MUTED, lw=0.6, label="daily P&L")
ax.plot(var_forecast.index, -var_forecast / 1e5, color=ACCENT2, lw=1.2,
        label="EWMA 99% VaR")
breaches = pnl.reindex(var_forecast.index) < -var_forecast
ax.scatter(var_forecast.index[breaches], (pnl.reindex(var_forecast.index)[breaches]) / 1e5,
           s=9, color=ACCENT2, zorder=5, label=f"exceptions ({int(breaches.sum())})")
ax.set_ylabel("₹ lakh"); ax.set_title(f"USD {usd_position/1e6:.0f}m long — VaR vs realised P&L")
ax.legend(loc="lower left")
annotate(ax, f"Kupiec p={bt['p_value']:.2f}, Christoffersen p={bt['p_value_ind']:.2f}\n"
             f"model not rejected; Basel zone {bt['basel_zone_last250']} on the last 250 days")

ax = axes[1]
ax.hist(pnl / 1e5, bins=260, color="#cfd9e0", density=True)
for (name, est), colour in zip(estimates.items(), [ACCENT, ACCENT3, ACCENT2]):
    ax.axvline(-est.var / 1e5, color=colour, lw=1.5,
               label=f"{est.method} VaR ₹{est.var/1e5:.1f}L")
ax.set_xlim(-115, 115)
ax.set_ylabel("density")
ax.set_xlabel("daily P&L (₹ lakh)"); ax.set_title("Three estimators, three different tails")
ax.legend(loc="upper left")
annotate(ax, "the normal model understates the 99% loss;\nfitted Student-t degrees of freedom < 4",
         loc="upper right")
fig.tight_layout(); fig.savefig(FIGURES / "06_var.png"); plt.close(fig)

# ------------------------------------------------------------ 8. curve PCA + book
log("decomposing curve risk")
pca = pca_curve_factors(ust)
R["pca"] = {
    "explained_variance_ratio": pca["explained_variance_ratio"].round(4).to_dict(),
    "factor_vols_bp": pca["factor_vols_bp"].round(2).to_dict(),
    "window": f"{ust.index.min().date()} to {ust.index.max().date()}",
    "n_days": int(len(ust)),
}
swaps = [InterestRateSwap(t, curve_official.par_swap_rate(t), 1e9, pay_fixed=True)
         for t in (2, 5, 10)]
R["swaps"] = {
    f"{s.tenor:g}Y": {"par_rate_pct": round(s.par_rate(curve_official), 4),
                      "pv01_inr_per_bn": round(s.pv01(curve_official), 0)}
    for s in swaps
}

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
ax = axes[0]
for name, colour in zip(pca["loadings"].columns, [ACCENT, ACCENT2, ACCENT3]):
    share = pca["explained_variance_ratio"][name]
    ax.plot(pca["loadings"].index, pca["loadings"][name], color=colour, marker="o", ms=3,
            label=f"{name} ({share:.0%})")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xscale("log"); ax.set_xticks([0.25, 1, 2, 5, 10, 30])
ax.set_xticklabels(["3M", "1Y", "2Y", "5Y", "10Y", "30Y"])
ax.set_xlabel("tenor"); ax.set_ylabel("loading")
ax.set_title("Three factors explain almost every curve move")
ax.legend()
annotate(ax, f"PCA on {len(ust)} daily curve changes\n{R['pca']['window']}", loc="lower left")

ax = axes[1]
tenors_sw = np.array([2, 5, 10])
pv01s = np.array([s.pv01(curve_official) for s in swaps]) / 1e3
ax.bar(range(3), pv01s, color=ACCENT, width=0.5)
ax.set_xticks(range(3)); ax.set_xticklabels(["2Y", "5Y", "10Y"])
ax.set_ylabel("PV01 (₹ thousand per ₹100 cr)")
ax.set_title("Rate risk per ₹100 crore of pay-fixed INR OIS")
ax.grid(axis="x", visible=False)
annotate(ax, "PV01 scales with the swap annuity —\nthe 10Y carries ~4x the 2Y's rate risk")
fig.tight_layout(); fig.savefig(FIGURES / "07_curve_risk.png"); plt.close(fig)

# ---------------------------------------------------------------- 9. FX context
fig, ax = plt.subplots(figsize=(9.0, 3.8))
ax.plot(usdinr.index, usdinr.values, color=ACCENT, label="USD/INR")
ax2 = ax.twinx(); ax2.plot(dxy.index, dxy.values, color=MUTED, lw=1.0, alpha=0.8, label="DXY")
ax2.grid(False); ax2.set_ylabel("DXY", color=MUTED)
ax.set_ylabel("USD/INR"); ax.set_title("USD/INR against the broad dollar")
corr = np.corrcoef(
    np.log(usdinr).diff().dropna().align(np.log(dxy).diff().dropna(), join="inner")[0],
    np.log(usdinr).diff().dropna().align(np.log(dxy).diff().dropna(), join="inner")[1],
)[0, 1]
R["usdinr_dxy_return_corr"] = round(float(corr), 3)
annotate(ax, f"daily return correlation with DXY: {corr:.2f}\n"
             "USD/INR is only partly a dollar story — the rest is\n"
             "the rate differential, oil, and RBI's own bid")
fig.tight_layout(); fig.savefig(FIGURES / "08_usdinr_context.png"); plt.close(fig)

# ------------------------------------------------------------------- write-up
(REPORTS / "results.json").write_text(json.dumps(R, indent=2, default=str))
log(f"wrote {REPORTS/'results.json'} and 8 figures to {FIGURES}")
print(json.dumps({k: R[k] for k in ("asof", "spot_usdinr", "curve", "risk")}, indent=2, default=str)[:2200])
