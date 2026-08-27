"""Value-at-Risk and Expected Shortfall for a trading book.

VaR answers one narrow question — "on a normal day, how bad does it get?" — and is
reported as a positive loss number at a confidence level (99% and a 1-day horizon
under Basel's internal-models approach; 10-day for capital). Expected Shortfall asks
the better question, "given that it *is* one of those days, how bad on average?", and
is what FRTB replaced VaR with precisely because VaR is blind to tail shape.

Three estimators are implemented because they disagree in exactly the way that
matters. The parametric normal one is fast and wrong in the tail; the historical one
is assumption-free but cannot produce a loss larger than the worst day in its window;
Student-t sits in between and is usually the one that survives a backtest on FX.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class VaRResult:
    """A VaR/ES estimate, always reported as a positive loss in position currency."""

    var: float
    expected_shortfall: float
    confidence: float
    horizon_days: int
    method: str

    def __str__(self) -> str:
        return (
            f"{self.method:<16s} {self.confidence:.0%} / {self.horizon_days}d  "
            f"VaR {self.var:>14,.0f}   ES {self.expected_shortfall:>14,.0f}"
        )


def _scale(horizon_days: int) -> float:
    """Square-root-of-time scaling. Valid only under i.i.d. returns — see README."""
    return float(np.sqrt(horizon_days))


def historical_var(pnl: pd.Series | np.ndarray, confidence: float = 0.99,
                   horizon_days: int = 1) -> VaRResult:
    """Empirical quantile of the realised P&L distribution."""
    x = np.asarray(pnl, float)
    x = x[np.isfinite(x)]
    var = -np.quantile(x, 1.0 - confidence) * _scale(horizon_days)
    tail = x[x <= np.quantile(x, 1.0 - confidence)]
    es = -tail.mean() * _scale(horizon_days) if tail.size else var
    return VaRResult(float(var), float(es), confidence, horizon_days, "historical")


def parametric_var(pnl: pd.Series | np.ndarray, confidence: float = 0.99,
                   horizon_days: int = 1, vol: float | None = None) -> VaRResult:
    """Normal (variance-covariance) VaR. `vol` overrides the sample standard deviation."""
    x = np.asarray(pnl, float)
    x = x[np.isfinite(x)]
    mu = float(np.mean(x))
    sigma = float(vol if vol is not None else np.std(x, ddof=1))
    z = stats.norm.ppf(confidence)
    var = (z * sigma - mu) * _scale(horizon_days)
    es = (sigma * stats.norm.pdf(z) / (1 - confidence) - mu) * _scale(horizon_days)
    return VaRResult(float(var), float(es), confidence, horizon_days, "normal")


def student_t_var(pnl: pd.Series | np.ndarray, confidence: float = 0.99,
                  horizon_days: int = 1) -> VaRResult:
    """Student-t VaR with the degrees of freedom fitted by maximum likelihood.

    FX and rates returns are fat-tailed; a fitted nu of 3-5 is normal for USD/INR and
    pushes the 99% loss materially above the Gaussian number.
    """
    x = np.asarray(pnl, float)
    x = x[np.isfinite(x)]
    nu, loc, scale = stats.t.fit(x)
    q = stats.t.ppf(1 - confidence, nu, loc=loc, scale=scale)
    var = -q * _scale(horizon_days)
    t_q = stats.t.ppf(1 - confidence, nu)
    es_std = -(stats.t.pdf(t_q, nu) / (1 - confidence)) * ((nu + t_q ** 2) / (nu - 1))
    es = -(loc + scale * es_std) * _scale(horizon_days)
    result = VaRResult(float(var), float(es), confidence, horizon_days, f"student-t(nu={nu:.1f})")
    return result


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA volatility — the standard answer to volatility clustering.

    lambda = 0.94 is the RiskMetrics daily default: roughly a 33-day half-life, which
    lets the VaR react to a regime break within a fortnight instead of a quarter.
    """
    r = returns.dropna().astype(float)
    var = np.empty(len(r))
    var[0] = float(r.iloc[:20].var(ddof=1)) if len(r) > 20 else float(r.var(ddof=1))
    values = r.to_numpy()
    for i in range(1, len(r)):
        var[i] = lam * var[i - 1] + (1 - lam) * values[i - 1] ** 2
    return pd.Series(np.sqrt(var), index=r.index, name="ewma_vol")


def monte_carlo_var(pnl_fn, draws: np.ndarray, confidence: float = 0.99,
                    horizon_days: int = 1) -> VaRResult:
    """Full-revaluation VaR: apply `pnl_fn` to simulated risk-factor `draws`.

    Unlike the analytic estimators this stays correct for a book with optionality,
    where P&L is a curved function of the risk factor and a delta approximation
    understates the loss.
    """
    pnl = np.asarray([pnl_fn(d) for d in draws], float)
    var = -np.quantile(pnl, 1 - confidence) * _scale(horizon_days)
    tail = pnl[pnl <= np.quantile(pnl, 1 - confidence)]
    es = -tail.mean() * _scale(horizon_days) if tail.size else var
    return VaRResult(float(var), float(es), confidence, horizon_days, "monte-carlo")


def expected_shortfall(pnl: pd.Series | np.ndarray, confidence: float = 0.975) -> float:
    """FRTB-style ES at 97.5%, the regulatory replacement for 99% VaR."""
    x = np.asarray(pnl, float)
    x = x[np.isfinite(x)]
    cutoff = np.quantile(x, 1 - confidence)
    return float(-x[x <= cutoff].mean())
