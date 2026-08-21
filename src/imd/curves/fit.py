"""Curve construction: calibrate NSS to traded yields, and a classical bootstrap.

Two independent constructions are provided on purpose. Agreement between a
parametric fit and a sequential bootstrap is the standard desk sanity check that the
curve is driven by the market rather than by the estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .curve import Compounding, ZeroCurve
from .instruments import CouponBond, TBill
from .nss import NSSParams

# Wide but economically sensible box. beta0 + beta1 is the instantaneous short rate,
# so the bounds below admit any INR short rate between roughly -4% and +25%.
_LOWER = np.array([0.0, -15.0, -40.0, -40.0, 0.10, 0.10])
_UPPER = np.array([20.0, 15.0, 40.0, 40.0, 30.0, 30.0])

_STARTS = [
    np.array([7.0, -1.5, -2.0, 2.0, 1.5, 6.0]),
    np.array([7.5, -2.0, -8.0, 9.0, 5.0, 7.0]),
    np.array([6.5, -1.0, 1.0, -1.0, 0.8, 3.0]),
    np.array([8.0, -3.0, -17.0, 18.0, 6.8, 7.1]),
    np.array([7.0, 0.5, 3.0, -3.0, 2.5, 12.0]),
]


def instruments_from_ccil(tenorwise: pd.DataFrame, segment: str = "GSEC+TBILL") -> list:
    """Turn CCIL's indicative-yield table into priceable instruments.

    `segment` selects which curve to build: ``"GSEC+TBILL"`` for the sovereign curve
    (the default) or ``"SDL"`` for the state-development-loan curve.
    """
    wanted = {"GSEC", "TBILL"} if segment == "GSEC+TBILL" else {segment}
    rows = tenorwise[tenorwise["segment"].isin(wanted)]
    out = []
    for _, r in rows.iterrows():
        if r["segment"] == "TBILL":
            out.append((TBill(float(r["tenor_years"]), r["security"]), float(r["ytm"])))
        else:
            out.append(
                (CouponBond(float(r["tenor_years"]), float(r["coupon"]), r["security"]),
                 float(r["ytm"]))
            )
    return out


def _yield_residuals(x: np.ndarray, instruments, compounding: Compounding) -> np.ndarray:
    """Model-minus-market yield error in basis points, one entry per instrument.

    Fitting in *yield* space rather than price space stops the 30-year bond — whose
    price is ~15x more rate-sensitive than a 3-month bill — from dominating the
    objective. This is the same inverse-duration weighting central banks use when
    they publish a fitted curve.
    """
    curve = ZeroCurve.from_nss(NSSParams.from_array(x), compounding=compounding)
    res = []
    for inst, mkt_ytm in instruments:
        model_price = float(np.sum(inst.flows * curve.df(inst.times)))
        if not np.isfinite(model_price) or model_price <= 1e-6:
            res.append(1e4)
            continue
        try:
            model_ytm = inst.ytm_from_price(model_price)
        except (ValueError, RuntimeError):
            res.append(1e4)
            continue
        res.append((model_ytm - mkt_ytm) * 100.0)
    return np.asarray(res, float)


def fit_nss(instruments, compounding: Compounding = "annual",
            starts: list[np.ndarray] | None = None,
            tau_bounds: tuple[float, float] = (0.10, 30.0),
            prior: NSSParams | None = None,
            ridge_bp: float = 0.0) -> tuple[NSSParams, dict]:
    """Calibrate NSS to a set of ``(instrument, market_ytm)`` pairs.

    The NSS objective is non-convex and, worse, *weakly identified* when the input set
    is small: beta2 and tau1 trade off almost exactly against beta3 and tau2, so an
    unconstrained fit to a handful of benchmarks routinely pins a tau on its bound and
    collapses the matching beta to zero. The fitted *curve* is still fine; the fitted
    *parameters* are not comparable across days, which breaks any use that reads
    economic meaning into them (level/slope/curvature risk factors, day-on-day
    parameter deltas, hedging by factor).

    Two levers are provided against that:

    ``tau_bounds``
        Restrict the hump locations to an economically meaningful window — the INR
        curve's humps live in the 1-10 year segment, so ``(0.5, 12.0)`` is a defensible
        prior that removes most of the degeneracy.
    ``ridge_bp``
        Tikhonov penalty pulling the solution toward ``prior`` (typically yesterday's
        parameters, or the published CCIL set). ``ridge_bp`` is expressed in the same
        basis-point units as the pricing residuals, so ``ridge_bp=5`` says "a one-unit
        parameter move must buy at least 5 bp of repricing improvement".

    Returns the fitted parameters and a diagnostics dict of per-instrument yield errors
    in basis points, RMSE and the worst absolute error.
    """
    lower = _LOWER.copy()
    upper = _UPPER.copy()
    lower[4] = lower[5] = tau_bounds[0]
    upper[4] = upper[5] = tau_bounds[1]

    prior_vec = prior.as_array() if prior is not None else None

    def residuals(x: np.ndarray) -> np.ndarray:
        base = _yield_residuals(x, instruments, compounding)
        if ridge_bp > 0.0 and prior_vec is not None:
            return np.concatenate([base, ridge_bp * (x - prior_vec)])
        return base

    best_x, best_cost = None, np.inf
    candidate_starts = list(starts or _STARTS)
    if prior_vec is not None:
        candidate_starts.insert(0, prior_vec)

    for x0 in candidate_starts:
        try:
            sol = least_squares(
                residuals, x0=np.clip(x0, lower, upper), bounds=(lower, upper),
                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000,
            )
        except Exception:  # noqa: BLE001 - a bad start must not kill the sweep
            continue
        if sol.cost < best_cost:
            best_x, best_cost = sol.x, sol.cost

    if best_x is None:
        raise RuntimeError("NSS calibration failed from every starting point")

    params = NSSParams.from_array(best_x)
    errors = _yield_residuals(best_x, instruments, compounding)
    diagnostics = {
        "errors_bp": pd.Series(errors, index=[i.label or f"inst{k}"
                                              for k, (i, _) in enumerate(instruments)]),
        "rmse_bp": float(np.sqrt(np.mean(errors ** 2))),
        "max_abs_bp": float(np.max(np.abs(errors))),
        "n_instruments": len(instruments),
        "tau_at_bound": bool(
            min(abs(best_x[4] - tau_bounds[0]), abs(best_x[4] - tau_bounds[1]),
                abs(best_x[5] - tau_bounds[0]), abs(best_x[5] - tau_bounds[1])) < 1e-3
        ),
    }
    return params, diagnostics


def bootstrap_par_curve(instruments, compounding: Compounding = "annual",
                        grid: np.ndarray | None = None) -> ZeroCurve:
    """Sequential bootstrap: solve instruments shortest-first for the zero at each node.

    Each instrument adds exactly one unknown — the zero rate at its own maturity —
    because every earlier cash flow is discounted on the nodes already solved. That
    is what makes the result estimator-free: it reprices every input exactly, by
    construction, at the cost of saying nothing about maturities between nodes
    beyond the interpolation rule.
    """
    ordered = sorted(instruments, key=lambda p: p[0].maturity)
    tenors: list[float] = []
    zeros: list[float] = []

    for inst, mkt_ytm in ordered:
        target = inst.price_from_ytm(mkt_ytm)
        maturity = float(inst.maturity)

        def price_given_zero(z: float) -> float:
            probe = ZeroCurve.from_nodes(
                np.array(tenors + [maturity]), np.array(zeros + [z]), compounding=compounding
            )
            return float(np.sum(inst.flows * probe.df(inst.times)))

        lo, hi = -5.0, 60.0
        for _ in range(200):  # bisection: monotone in z, so this always converges
            mid = 0.5 * (lo + hi)
            if price_given_zero(mid) > target:
                lo = mid
            else:
                hi = mid
        tenors.append(maturity)
        zeros.append(0.5 * (lo + hi))

    curve = ZeroCurve.from_nodes(np.array(tenors), np.array(zeros), compounding=compounding,
                                 label="bootstrap")
    curve.nodes = pd.Series(zeros, index=tenors, name="zero_pct")  # type: ignore[attr-defined]
    return curve
