"""Curve tests. These are no-arbitrage identities, not regression snapshots —
they must hold for any curve, on any day, or the pricing library is wrong.
"""
import numpy as np
import pytest

from imd.curves import CouponBond, NSSParams, TBill, ZeroCurve, bootstrap_par_curve, fit_nss
from imd.curves.nss import nss_zero

PARAMS = NSSParams(7.0, -1.5, -3.0, 4.0, 1.8, 8.0)


def test_nss_converges_to_beta0_at_the_long_end():
    """As t -> infinity every loading dies and only the level survives.

    Convergence is O(tau/t), so 500 years is nowhere near enough for a 1 bp check —
    which is itself the point: the NSS long end is anchored by beta0 only
    asymptotically, and inside any traded maturity the slope term is still live.
    """
    assert nss_zero(np.array([500.0]), PARAMS)[0] == pytest.approx(PARAMS.beta0, abs=0.1)
    assert nss_zero(np.array([1e6]), PARAMS)[0] == pytest.approx(PARAMS.beta0, abs=1e-4)


def test_nss_short_rate_is_beta0_plus_beta1():
    """At t -> 0 the slope loading tends to 1 and both curvature loadings to 0."""
    assert nss_zero(np.array([1e-8]), PARAMS)[0] == pytest.approx(
        PARAMS.beta0 + PARAMS.beta1, abs=1e-4
    )


def test_discount_factors_are_positive_and_decreasing():
    curve = ZeroCurve.from_nss(PARAMS)
    dfs = curve.df(np.linspace(0.01, 30, 400))
    assert np.all(dfs > 0)
    assert np.all(np.diff(dfs) < 0)


@pytest.mark.parametrize("compounding", ["annual", "continuous", "semiannual"])
def test_forward_rates_compound_back_to_the_spot_curve(compounding):
    """DF(0,t2) must equal DF(0,t1) discounted by the t1->t2 forward."""
    curve = ZeroCurve.from_nss(PARAMS, compounding=compounding)
    t1, t2 = 2.0, 5.0
    fwd = float(curve.forward(t1, t2)[0]) / 100.0
    dt = t2 - t1
    implied = float(curve.df(t1)[0]) * (
        np.exp(-fwd * dt) if compounding == "continuous" else (1 + fwd) ** -dt
    )
    assert implied == pytest.approx(float(curve.df(t2)[0]), rel=1e-9)


def test_bond_price_ytm_roundtrip():
    bond = CouponBond(9.8, 6.94)
    price = bond.price_from_ytm(6.9551)
    assert bond.ytm_from_price(price) == pytest.approx(6.9551, abs=1e-8)


def test_a_bond_priced_at_its_coupon_trades_at_par():
    bond = CouponBond(10.0, 7.0)
    assert bond.price_from_ytm(7.0) == pytest.approx(100.0, abs=1e-8)


def test_bootstrap_reprices_every_input_exactly():
    """The defining property of a bootstrap: zero repricing error, by construction."""
    instruments = [
        (TBill(0.25, "91D"), 5.21),
        (TBill(0.5, "182D"), 5.62),
        (TBill(1.0, "364D"), 5.91),
        (CouponBond(4.8, 6.36, "5y"), 6.50),
        (CouponBond(9.8, 6.94, "10y"), 6.96),
        (CouponBond(14.8, 7.06, "15y"), 7.11),
    ]
    curve = bootstrap_par_curve(instruments, compounding="continuous")
    for instrument, market_ytm in instruments:
        modelled = float(np.sum(instrument.flows * curve.df(instrument.times)))
        assert modelled == pytest.approx(instrument.price_from_ytm(market_ytm), abs=1e-6)


def test_nss_fit_recovers_a_curve_it_generated():
    """Fit against synthetic yields from a known curve; errors must be sub-basis-point."""
    truth = ZeroCurve.from_nss(PARAMS, compounding="continuous")
    instruments = []
    for maturity, coupon in [(0.25, None), (1.0, None), (2.0, 6.5), (5.0, 6.8),
                             (10.0, 7.0), (20.0, 7.1), (30.0, 7.1)]:
        inst = TBill(maturity) if coupon is None else CouponBond(maturity, coupon)
        price = float(np.sum(inst.flows * truth.df(inst.times)))
        instruments.append((inst, inst.ytm_from_price(price)))
    _, diagnostics = fit_nss(instruments, compounding="continuous")
    assert diagnostics["rmse_bp"] < 1.0


def test_par_swap_rate_makes_the_swap_worth_nothing():
    from imd.pricing.rates import InterestRateSwap

    curve = ZeroCurve.from_nss(PARAMS, compounding="continuous")
    par = curve.par_swap_rate(5.0)
    assert InterestRateSwap(5.0, par, 1e9).npv(curve) == pytest.approx(0.0, abs=1e-6)


def test_flat_curve_has_flat_forwards():
    curve = ZeroCurve.from_nodes([1, 5, 10, 30], [6.0] * 4, compounding="continuous")
    assert float(curve.forward(2.0, 7.0)[0]) == pytest.approx(6.0, abs=1e-9)
