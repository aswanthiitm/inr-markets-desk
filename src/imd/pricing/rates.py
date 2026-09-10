"""INR interest-rate swap valuation off a zero curve."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..curves.curve import ZeroCurve


@dataclass
class InterestRateSwap:
    """A single-currency fixed-for-floating INR swap (OIS or MIFOR-style IRS).

    Valued by the standard replication argument: the floating leg of a par swap is
    worth ``notional * (1 - DF(T))`` regardless of the projected fixings, so only the
    fixed leg needs the curve twice. ``pay_fixed=True`` is the client that has borrowed
    floating and wants certainty — the position that gains when rates rise.
    """

    tenor: float
    fixed_rate: float          # percent
    notional: float = 1e8      # INR
    freq: int = 1              # INR OIS convention: annual fixed payments
    pay_fixed: bool = True

    def _times(self) -> np.ndarray:
        times = np.arange(1, int(round(self.tenor * self.freq)) + 1, dtype=float) / self.freq
        if times.size == 0 or abs(times[-1] - self.tenor) > 1e-9:
            times = np.append(times, self.tenor)
        return times

    def annuity(self, curve: ZeroCurve) -> float:
        times = self._times()
        accruals = np.diff(np.concatenate([[0.0], times]))
        return float(np.sum(accruals * curve.df(times)))

    def par_rate(self, curve: ZeroCurve) -> float:
        return curve.par_swap_rate(self.tenor, self.freq)

    def npv(self, curve: ZeroCurve) -> float:
        """Mark-to-market in rupees, positive when the position is in the money."""
        edge = (self.par_rate(curve) - self.fixed_rate) / 100.0
        sign = 1.0 if self.pay_fixed else -1.0
        return sign * edge * self.annuity(curve) * self.notional

    def pv01(self, curve: ZeroCurve) -> float:
        """Rupee value of a 1 bp parallel move — the desk's unit of rate risk."""
        return (self.npv(curve.shifted(1.0)) - self.npv(curve.shifted(-1.0))) / 2.0
