"""G-Sec and T-Bill cash-flow conventions and yield/price conversion."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..config import GSEC_COUPON_FREQ, INR_DAYCOUNT


@dataclass(frozen=True)
class TBill:
    """A Treasury Bill: one redemption of 100 at `maturity` years.

    Indian T-Bills are quoted on a simple (money-market) yield: the implied yield is
    ``(100/P - 1) * 365/days``, *not* a compounded rate. Getting this wrong puts the
    short end of the curve out by several basis points.
    """

    maturity: float
    label: str = ""

    @property
    def times(self) -> np.ndarray:
        return np.array([self.maturity])

    @property
    def flows(self) -> np.ndarray:
        return np.array([100.0])

    def price_from_ytm(self, ytm: float) -> float:
        days = self.maturity * INR_DAYCOUNT
        return 100.0 / (1.0 + ytm / 100.0 * days / INR_DAYCOUNT)

    def ytm_from_price(self, price: float) -> float:
        return (100.0 / price - 1.0) * INR_DAYCOUNT / (self.maturity * INR_DAYCOUNT) * 100.0


@dataclass(frozen=True)
class CouponBond:
    """A dated G-Sec or SDL paying `coupon`% semi-annually until `maturity`.

    Cash-flow dates are stepped back from maturity in half-year intervals, which is
    the standard simplification when only the benchmark's tenor and coupon are
    published (CCIL's indicative-yield table gives exactly that). The residual
    stub-period error is under a day of accrued interest and is immaterial next to
    the 0.25 bp granularity of the quoted yields.
    """

    maturity: float
    coupon: float
    label: str = ""
    freq: int = GSEC_COUPON_FREQ

    @property
    def times(self) -> np.ndarray:
        step = 1.0 / self.freq
        n = max(int(round(self.maturity * self.freq)), 1)
        t = self.maturity - step * np.arange(n - 1, -1, -1)
        return t[t > 1e-8]

    @property
    def flows(self) -> np.ndarray:
        t = self.times
        cf = np.full(t.shape, self.coupon / self.freq)
        cf[-1] += 100.0
        return cf

    def price_from_ytm(self, ytm: float) -> float:
        t, cf = self.times, self.flows
        return float(np.sum(cf * (1.0 + ytm / 100.0 / self.freq) ** (-self.freq * t)))

    def ytm_from_price(self, price: float) -> float:
        return brentq(lambda y: self.price_from_ytm(y) - price, -5.0, 60.0, xtol=1e-10)

    def macaulay_duration(self, ytm: float) -> float:
        t, cf = self.times, self.flows
        pv = cf * (1.0 + ytm / 100.0 / self.freq) ** (-self.freq * t)
        return float(np.sum(t * pv) / np.sum(pv))

    def modified_duration(self, ytm: float) -> float:
        return self.macaulay_duration(ytm) / (1.0 + ytm / 100.0 / self.freq)


def price_from_ytm(instrument, ytm: float) -> float:
    return instrument.price_from_ytm(ytm)


def ytm_from_price(instrument, price: float) -> float:
    return instrument.ytm_from_price(price)
