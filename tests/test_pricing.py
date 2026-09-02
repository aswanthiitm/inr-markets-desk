"""Option and forward tests, all of them arbitrage relations."""
import numpy as np
import pytest

from imd.pricing import (
    cip_forward, forward_premium_pct, garman_kohlhagen, gk_greeks, implied_vol,
    zero_cost_collar,
)

SPOT, R_INR, R_USD, VOL, T = 95.0, 0.0575, 0.0415, 0.06, 1.0


def test_covered_interest_parity_holds_by_construction():
    df_inr, df_usd = np.exp(-R_INR * T), np.exp(-R_USD * T)
    fwd = float(cip_forward(SPOT, df_inr, df_usd))
    assert fwd == pytest.approx(SPOT * np.exp((R_INR - R_USD) * T), rel=1e-12)
    assert float(forward_premium_pct(SPOT, fwd, T)) > 0  # INR rates above USD rates


def test_put_call_parity():
    """C - P = S.DF_for - K.DF_dom. If this breaks, every structure built on it does."""
    strike = 97.0
    call = float(garman_kohlhagen(SPOT, strike, R_INR, R_USD, VOL, T, "call"))
    put = float(garman_kohlhagen(SPOT, strike, R_INR, R_USD, VOL, T, "put"))
    expected = SPOT * np.exp(-R_USD * T) - strike * np.exp(-R_INR * T)
    assert call - put == pytest.approx(expected, abs=1e-10)


def test_option_prices_respect_their_no_arbitrage_bounds():
    strike = 92.0
    call = float(garman_kohlhagen(SPOT, strike, R_INR, R_USD, VOL, T, "call"))
    intrinsic = max(SPOT * np.exp(-R_USD * T) - strike * np.exp(-R_INR * T), 0.0)
    assert intrinsic <= call <= SPOT * np.exp(-R_USD * T)


def test_premium_rises_with_volatility_and_falls_with_strike():
    base = float(garman_kohlhagen(SPOT, 96.0, R_INR, R_USD, VOL, T, "call"))
    assert float(garman_kohlhagen(SPOT, 96.0, R_INR, R_USD, VOL * 1.5, T, "call")) > base
    assert float(garman_kohlhagen(SPOT, 99.0, R_INR, R_USD, VOL, T, "call")) < base


def test_implied_vol_inverts_the_pricer():
    price = float(garman_kohlhagen(SPOT, 96.0, R_INR, R_USD, 0.0812, T, "call"))
    assert implied_vol(price, SPOT, 96.0, R_INR, R_USD, T) == pytest.approx(0.0812, abs=1e-8)


def test_call_delta_is_bounded_and_matches_a_bumped_price():
    greeks = gk_greeks(SPOT, 96.0, R_INR, R_USD, VOL, T, "call")
    delta = float(np.ravel(greeks["delta"])[0])
    assert 0.0 < delta < 1.0
    h = 1e-5
    up = float(garman_kohlhagen(SPOT + h, 96.0, R_INR, R_USD, VOL, T, "call"))
    down = float(garman_kohlhagen(SPOT - h, 96.0, R_INR, R_USD, VOL, T, "call"))
    assert (up - down) / (2 * h) == pytest.approx(delta, abs=1e-6)


def test_zero_cost_collar_really_costs_zero():
    fwd = SPOT * np.exp((R_INR - R_USD) * T)
    cap = fwd * 1.02
    floor = zero_cost_collar(SPOT, fwd, R_INR, R_USD, VOL, T, cap)
    call = float(garman_kohlhagen(SPOT, cap, R_INR, R_USD, VOL, T, "call"))
    put = float(garman_kohlhagen(SPOT, floor, R_INR, R_USD, VOL, T, "put"))
    assert call - put == pytest.approx(0.0, abs=1e-8)
    assert floor < fwd < cap  # the band must straddle the forward
