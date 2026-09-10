"""A zero-coupon curve object: discount factors, forwards, par swap rates, PV01."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from .nss import NSSParams, nss_zero

Compounding = Literal["annual", "continuous", "semiannual"]


@dataclass
class ZeroCurve:
    """Zero rates as a function of maturity, plus everything derived from them.

    Parameters
    ----------
    zero_fn
        ``t (years) -> zero rate in percent``.
    compounding
        How a zero rate converts to a discount factor. CCIL quotes its ZCYC on an
        annually-compounded basis, which is the default here so that a curve fitted
        in this project is directly comparable to the published benchmark.
    """

    zero_fn: Callable[[np.ndarray], np.ndarray]
    compounding: Compounding = "annual"
    label: str = "curve"
    params: NSSParams | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ core
    def zero(self, t) -> np.ndarray:
        """Zero rate(s) in percent. A scalar maturity returns a float."""
        scalar_in = np.ndim(t) == 0
        out = np.atleast_1d(
            np.asarray(self.zero_fn(np.atleast_1d(np.asarray(t, float))), float)
        )
        return float(out[0]) if scalar_in else out

    def df(self, t) -> np.ndarray:
        """Discount factor(s) for maturities `t` in years."""
        t_arr = np.atleast_1d(np.asarray(t, float))
        z = np.asarray(self.zero(t_arr), float) / 100.0
        if self.compounding == "continuous":
            return np.exp(-z * t_arr)
        if self.compounding == "semiannual":
            return (1.0 + z / 2.0) ** (-2.0 * t_arr)
        return (1.0 + z) ** (-t_arr)

    # -------------------------------------------------------------- forwards
    def forward(self, t1, t2) -> np.ndarray:
        """Annualised forward rate in percent between `t1` and `t2` years."""
        t1_a, t2_a = np.atleast_1d(np.asarray(t1, float)), np.atleast_1d(np.asarray(t2, float))
        df1, df2 = self.df(t1_a), self.df(t2_a)
        dt = t2_a - t1_a
        if self.compounding == "continuous":
            return 100.0 * np.log(df1 / df2) / dt
        return 100.0 * ((df1 / df2) ** (1.0 / dt) - 1.0)

    def instantaneous_forward(self, t, h: float = 1e-4) -> np.ndarray:
        return self.forward(np.asarray(t, float), np.asarray(t, float) + h)

    # ------------------------------------------------------------ swap maths
    def par_swap_rate(self, tenor: float, freq: int = 1) -> float:
        """Fair fixed rate (percent) of a par INR OIS/IRS of length `tenor` years.

        INR overnight-indexed swaps pay annually up to and including one year and
        annually thereafter on an Actual/365 basis, hence the default ``freq=1``.
        """
        times = np.arange(1, int(round(tenor * freq)) + 1, dtype=float) / freq
        if times.size == 0 or abs(times[-1] - tenor) > 1e-9:
            times = np.append(times, tenor)
        dfs = self.df(times)
        accruals = np.diff(np.concatenate([[0.0], times]))
        annuity = float(np.sum(accruals * dfs))
        return 100.0 * (1.0 - float(dfs[-1])) / annuity

    def annuity(self, tenor: float, freq: int = 1) -> float:
        times = np.arange(1, int(round(tenor * freq)) + 1, dtype=float) / freq
        accruals = np.diff(np.concatenate([[0.0], times]))
        return float(np.sum(accruals * self.df(times)))

    def pv01(self, tenor: float, notional: float = 1e7, freq: int = 1) -> float:
        """PV of a 1 bp move in the fixed rate of a `tenor`-year swap, in rupees."""
        return self.annuity(tenor, freq) * notional * 1e-4

    # ------------------------------------------------------------ bond maths
    def price_bond(self, instrument) -> float:
        return float(np.sum(instrument.flows * self.df(instrument.times)))

    def shifted(self, bp: float) -> "ZeroCurve":
        """A parallel copy of this curve shifted by `bp` basis points."""
        return ZeroCurve(
            zero_fn=lambda t, f=self.zero_fn: np.asarray(f(t), float) + bp / 100.0,
            compounding=self.compounding,
            label=f"{self.label}{bp:+.0f}bp",
        )

    # ---------------------------------------------------------- constructors
    @classmethod
    def from_nss(cls, params: NSSParams, compounding: Compounding = "annual",
                 label: str = "NSS") -> "ZeroCurve":
        return cls(zero_fn=lambda t: nss_zero(t, params), compounding=compounding,
                   label=label, params=params)

    @classmethod
    def from_nodes(cls, tenors, zeros, compounding: Compounding = "annual",
                   label: str = "bootstrapped") -> "ZeroCurve":
        """Log-linear interpolation on discount factors — the market-standard choice.

        Interpolating linearly on *zero rates* can produce negative or oscillating
        implied forwards; log-linear on discount factors keeps every implied forward
        piecewise-constant and positive.
        """
        tenors = np.asarray(tenors, float)
        zeros = np.asarray(zeros, float)
        order = np.argsort(tenors)
        tenors, zeros = tenors[order], zeros[order]

        if compounding == "continuous":
            log_df = -zeros / 100.0 * tenors
        elif compounding == "semiannual":
            log_df = -2.0 * tenors * np.log1p(zeros / 200.0)
        else:
            log_df = -tenors * np.log1p(zeros / 100.0)

        # Beyond the last node we extend the *forward* rate flat rather than the
        # discount factor, which is the market convention: clamping log-DF instead
        # would imply a zero forward rate and bend the long end downwards.
        slope_front = log_df[0] / tenors[0] if tenors[0] > 0 else 0.0
        slope_back = ((log_df[-1] - log_df[-2]) / (tenors[-1] - tenors[-2])
                      if len(tenors) > 1 else log_df[-1] / tenors[-1])

        def zero_fn(t: np.ndarray) -> np.ndarray:
            t = np.maximum(np.asarray(t, float), 1e-8)
            ldf = np.interp(t, tenors, log_df)
            ldf = np.where(t < tenors[0], slope_front * t, ldf)
            ldf = np.where(t > tenors[-1], log_df[-1] + slope_back * (t - tenors[-1]), ldf)
            if compounding == "continuous":
                return -ldf / t * 100.0
            if compounding == "semiannual":
                return (np.exp(-ldf / (2.0 * t)) - 1.0) * 200.0
            return (np.exp(-ldf / t) - 1.0) * 100.0

        return cls(zero_fn=zero_fn, compounding=compounding, label=label)
