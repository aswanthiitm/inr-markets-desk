"""USD/INR forward and option pricing.

Everything here rests on one no-arbitrage identity — covered interest parity. Borrow
a dollar, sell it spot for rupees, invest the rupees to T and simultaneously buy the
dollars back forward: the trade is riskless, so the forward must satisfy

.. math:: F(T) = S \\cdot \\frac{DF_{USD}(T)}{DF_{INR}(T)}

Because INR rates sit well above USD rates, ``F > S`` and USD/INR trades at a *forward
premium* — the annualised premium is what an importer pays to hedge and what an
exporter earns. When the traded forward differs from this fair value the gap is the
**forward basis**, and on USD/INR it is not noise: it moves with RBI's own forward-book
intervention, with FCNR(B) flows, and with onshore dollar funding stress.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# ------------------------------------------------------------------ forwards
def cip_forward(spot: float, df_dom, df_for) -> np.ndarray:
    """Fair forward FX rate from domestic (INR) and foreign (USD) discount factors.

    Quoted USD/INR style: number of rupees per dollar, so INR is *domestic*.
    """
    return spot * np.asarray(df_for, float) / np.asarray(df_dom, float)


def forward_points(spot: float, forward) -> np.ndarray:
    """Forward points in paise (1 paisa = 0.01 INR), the desk's quoting unit."""
    return (np.asarray(forward, float) - spot) * 100.0


def forward_premium_pct(spot: float, forward, tenor_years) -> np.ndarray:
    """Annualised forward premium in percent."""
    return (np.asarray(forward, float) / spot - 1.0) / np.asarray(tenor_years, float) * 100.0


def implied_basis_bp(market_forward, fair_forward, spot: float, tenor_years) -> np.ndarray:
    """Traded-minus-fair forward premium, in basis points per annum.

    A positive number means the market is paying *more* than covered interest parity
    says it should — the classic signature of onshore dollar scarcity.
    """
    mkt = forward_premium_pct(spot, market_forward, tenor_years)
    fair = forward_premium_pct(spot, fair_forward, tenor_years)
    return (mkt - fair) * 100.0


# ------------------------------------------------------------------- options
def _d1_d2(spot, strike, r_dom, r_for, vol, t):
    spot, strike, vol, t = map(lambda v: np.asarray(v, float), (spot, strike, vol, t))
    vt = vol * np.sqrt(t)
    d1 = (np.log(spot / strike) + (r_dom - r_for + 0.5 * vol ** 2) * t) / vt
    return d1, d1 - vt


def garman_kohlhagen(spot, strike, r_dom, r_for, vol, t, option: str = "call") -> np.ndarray:
    """Garman-Kohlhagen premium, in INR per USD of notional.

    Black-Scholes adapted to FX: the foreign currency earns its own interest rate, so
    USD behaves like a dividend-paying asset with dividend yield ``r_for``. Rates and
    vol are decimals (0.0595 not 5.95); ``t`` is in years.

    A ``call`` is the right to *buy* USD against INR — the hedge an importer buys.
    A ``put`` is the right to *sell* USD — the exporter's hedge.
    """
    d1, d2 = _d1_d2(spot, strike, r_dom, r_for, vol, t)
    df_dom, df_for = np.exp(-r_dom * np.asarray(t, float)), np.exp(-r_for * np.asarray(t, float))
    if option.lower().startswith("c"):
        return spot * df_for * norm.cdf(d1) - strike * df_dom * norm.cdf(d2)
    return strike * df_dom * norm.cdf(-d2) - spot * df_for * norm.cdf(-d1)


def gk_greeks(spot, strike, r_dom, r_for, vol, t, option: str = "call") -> dict:
    """Spot delta, gamma, vega (per 1 vol point) and theta (per day)."""
    d1, d2 = _d1_d2(spot, strike, r_dom, r_for, vol, t)
    t = np.asarray(t, float)
    df_dom, df_for = np.exp(-r_dom * t), np.exp(-r_for * t)
    pdf = norm.pdf(d1)
    is_call = option.lower().startswith("c")

    delta = df_for * (norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0)
    gamma = df_for * pdf / (spot * vol * np.sqrt(t))
    vega = spot * df_for * pdf * np.sqrt(t) / 100.0
    theta_common = -spot * df_for * pdf * vol / (2 * np.sqrt(t))
    if is_call:
        theta = theta_common + r_for * spot * df_for * norm.cdf(d1) - r_dom * strike * df_dom * norm.cdf(d2)
    else:
        theta = theta_common - r_for * spot * df_for * norm.cdf(-d1) + r_dom * strike * df_dom * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta_per_day": theta / 365.0}


def implied_vol(price, spot, strike, r_dom, r_for, t, option: str = "call") -> float:
    """Back out the volatility that reprices a quoted premium."""
    def objective(v: float) -> float:
        return float(garman_kohlhagen(spot, strike, r_dom, r_for, v, t, option)) - price
    return brentq(objective, 1e-6, 5.0, xtol=1e-10)


# ---------------------------------------------------------------- structures
def zero_cost_collar(spot, forward, r_dom, r_for, vol, t, cap_strike: float,
                     bracket: tuple[float, float] | None = None) -> float:
    """Floor strike that makes an importer's collar cost zero.

    The importer buys a USD call struck at ``cap_strike`` (the worst rate they will
    ever pay) and sells a USD put struck at the returned floor (giving up the benefit
    below it). Solving for a zero net premium is what makes the structure free, and
    the width of the resulting band is the honest price of "free".
    """
    lo, hi = bracket or (spot * 0.70, cap_strike * 0.999)

    def net(floor: float) -> float:
        long_call = float(garman_kohlhagen(spot, cap_strike, r_dom, r_for, vol, t, "call"))
        short_put = float(garman_kohlhagen(spot, floor, r_dom, r_for, vol, t, "put"))
        return long_call - short_put

    return brentq(net, lo, hi, xtol=1e-8)


def seagull(spot, r_dom, r_for, vol, t, cap_strike: float, lower_put_strike: float) -> float:
    """Floor strike of a zero-cost importer seagull (long call, short put, long lower put).

    Buying back a deep out-of-the-money put caps the tail loss the collar's short put
    leaves open, and is paid for by moving the sold floor higher. Corporates with a
    board-mandated worst case use this instead of a plain collar.
    """
    long_call = float(garman_kohlhagen(spot, cap_strike, r_dom, r_for, vol, t, "call"))
    long_low_put = float(garman_kohlhagen(spot, lower_put_strike, r_dom, r_for, vol, t, "put"))
    target = long_call + long_low_put

    def net(floor: float) -> float:
        return float(garman_kohlhagen(spot, floor, r_dom, r_for, vol, t, "put")) - target

    return brentq(net, lower_put_strike * 1.001, cap_strike * 0.999, xtol=1e-8)
