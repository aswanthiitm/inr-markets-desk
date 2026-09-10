"""Regulatory VaR backtests.

A VaR number is a *forecast*, and a forecast that is never checked is decoration.
Basel requires a bank running an internal model to count exceptions — days where the
realised loss beat the 99% VaR — over a 250-day window, and escalates capital when
there are too many. The two likelihood-ratio tests below are the standard formal
version of that count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def kupiec_pof(exceptions: np.ndarray, confidence: float = 0.99) -> dict:
    """Kupiec proportion-of-failures test: is the *number* of exceptions right?

    Under a correct model exceptions are Bernoulli with p = 1 - confidence. The
    likelihood ratio below is chi-square with one degree of freedom; a p-value under
    0.05 rejects the model. Note what this test cannot see: ten exceptions spread
    evenly and ten exceptions on ten consecutive days score identically.
    """
    x = np.asarray(exceptions, bool)
    n, failures = x.size, int(x.sum())
    p = 1 - confidence
    if failures == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        pi = failures / n
        lr = -2 * (
            (n - failures) * np.log(1 - p) + failures * np.log(p)
            - (n - failures) * np.log(1 - pi) - failures * np.log(pi)
        )
    return {
        "n": n, "exceptions": failures, "expected": n * p,
        "rate": failures / n if n else np.nan,
        "LR_pof": float(lr), "p_value": float(1 - stats.chi2.cdf(lr, 1)),
        "reject_5pct": bool(1 - stats.chi2.cdf(lr, 1) < 0.05),
    }


def christoffersen_independence(exceptions: np.ndarray) -> dict:
    """Christoffersen test: are exceptions *independent*, or do they cluster?

    Clustering is the dangerous failure mode — it means the model is slow to react to
    a volatility regime change, so the bank is under-capitalised exactly during the
    week it matters. Tests the first-order Markov transition probabilities.
    """
    x = np.asarray(exceptions, int)
    n00 = int(np.sum((x[:-1] == 0) & (x[1:] == 0)))
    n01 = int(np.sum((x[:-1] == 0) & (x[1:] == 1)))
    n10 = int(np.sum((x[:-1] == 1) & (x[1:] == 0)))
    n11 = int(np.sum((x[:-1] == 1) & (x[1:] == 1)))

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def _ll(p0: float, p1: float) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = [
                n00 * np.log(1 - p0) if n00 and p0 < 1 else 0.0,
                n01 * np.log(p0) if n01 and p0 > 0 else 0.0,
                n10 * np.log(1 - p1) if n10 and p1 < 1 else 0.0,
                n11 * np.log(p1) if n11 and p1 > 0 else 0.0,
            ]
        return float(np.sum(terms))

    lr = -2 * (_ll(pi, pi) - _ll(pi01, pi11))
    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "LR_ind": float(lr), "p_value": float(1 - stats.chi2.cdf(lr, 1)),
        "reject_5pct": bool(1 - stats.chi2.cdf(lr, 1) < 0.05),
    }


def basel_traffic_light(exceptions_250d: int) -> str:
    """Basel's three-zone rule on a 250-day window at 99% (green / amber / red)."""
    if exceptions_250d <= 4:
        return "GREEN"
    if exceptions_250d <= 9:
        return "AMBER"
    return "RED"


def backtest_var(pnl: pd.Series, var_series: pd.Series, confidence: float = 0.99) -> dict:
    """Run both likelihood-ratio tests plus the Basel zone on a rolling VaR forecast."""
    aligned = pd.concat([pnl.rename("pnl"), var_series.rename("var")], axis=1).dropna()
    exceptions = (aligned["pnl"] < -aligned["var"]).to_numpy()
    pof = kupiec_pof(exceptions, confidence)
    ind = christoffersen_independence(exceptions)
    lr_cc = pof["LR_pof"] + ind["LR_ind"]
    return {
        **pof,
        "LR_ind": ind["LR_ind"], "p_value_ind": ind["p_value"],
        "LR_cc": float(lr_cc),
        "p_value_cc": float(1 - stats.chi2.cdf(lr_cc, 2)),
        "basel_zone_last250": basel_traffic_light(int(exceptions[-250:].sum())),
        "exception_dates": list(aligned.index[exceptions]),
    }
