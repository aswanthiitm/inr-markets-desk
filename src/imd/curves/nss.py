"""The Nelson-Siegel-Svensson zero-coupon yield curve.

CCIL estimates the official zero-coupon sovereign rupee yield curve with this same
functional form, which is why the project uses it: it lets us compare a curve we fit
ourselves against the published benchmark parameter-for-parameter.

The zero rate at maturity ``t`` (years) is

.. math::
    y(t) = \\beta_0
         + \\beta_1 \\frac{1 - e^{-t/\\tau_1}}{t/\\tau_1}
         + \\beta_2 \\left(\\frac{1 - e^{-t/\\tau_1}}{t/\\tau_1} - e^{-t/\\tau_1}\\right)
         + \\beta_3 \\left(\\frac{1 - e^{-t/\\tau_2}}{t/\\tau_2} - e^{-t/\\tau_2}\\right)

Read economically: :math:`\\beta_0` is the long-run level, :math:`\\beta_1` the slope
(short minus long), and :math:`\\beta_2`, :math:`\\beta_3` two humps whose locations are
set by :math:`\\tau_1` and :math:`\\tau_2`. The Svensson extension over plain
Nelson-Siegel is exactly that second hump, which the INR curve needs because RBI
liquidity operations regularly kink the 1-5 year segment independently of the 10-30
year segment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NSSParams:
    """Six Nelson-Siegel-Svensson parameters. Betas in percent, taus in years."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    def as_array(self) -> np.ndarray:
        return np.array([self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2])

    @classmethod
    def from_array(cls, x) -> "NSSParams":
        return cls(*[float(v) for v in x])

    def __str__(self) -> str:
        return (
            f"b0={self.beta0:.4f} b1={self.beta1:.4f} b2={self.beta2:.4f} "
            f"b3={self.beta3:.4f} t1={self.tau1:.4f} t2={self.tau2:.4f}"
        )


def _loading(t: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope and curvature loadings, with the removable singularity at t=0 handled."""
    tau = max(float(tau), 1e-6)
    x = np.maximum(np.asarray(t, dtype=float), 1e-8) / tau
    exp_neg = np.exp(-x)
    slope = np.where(x < 1e-6, 1.0 - x / 2.0, (1.0 - exp_neg) / x)
    curvature = slope - exp_neg
    return slope, curvature


def nss_zero(t, p: NSSParams) -> np.ndarray:
    """Annualised zero rate in percent at maturities `t` (years)."""
    scalar_in = np.ndim(t) == 0
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    s1, c1 = _loading(t_arr, p.tau1)
    _, c2 = _loading(t_arr, p.tau2)
    y = p.beta0 + p.beta1 * s1 + p.beta2 * c1 + p.beta3 * c2
    return float(y[0]) if scalar_in else y


def nss_from_ccil(row: "pd.Series | dict") -> NSSParams:
    """Build :class:`NSSParams` from a row of :func:`imd.data.fetch_ccil_zcyc_params`."""
    return NSSParams(
        beta0=float(row["beta0"]), beta1=float(row["beta1"]), beta2=float(row["beta2"]),
        beta3=float(row["beta3"]), tau1=float(row["tau1"]), tau2=float(row["tau2"]),
    )
