"""Curve risk: key-rate DV01 and the level/slope/curvature decomposition.

A rates book is never exposed to "the yield" — it is exposed to a whole curve, and
the curve does not move in parallel. Principal components on daily curve changes
recover, in this order and with almost no exception across markets, a **level** shift
(~90% of variance), a **slope** twist, and a **curvature** butterfly. Hedging those
three factors is what a desk actually does; hedging a parallel shift alone leaves the
steepener risk entirely open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pca_curve_factors(curve_history: pd.DataFrame, n_factors: int = 3) -> dict:
    """PCA on daily changes of a curve panel (rows = dates, columns = tenor years)."""
    changes = curve_history.diff().dropna()
    x = changes.to_numpy(float)
    x = x - x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    # Sign convention: make the first loading of each factor positive so that
    # "level up" means rates up on every run.
    for k in range(vectors.shape[1]):
        if vectors[np.argmax(np.abs(vectors[:, k])), k] < 0:
            vectors[:, k] *= -1

    explained = values / values.sum()
    names = ["level", "slope", "curvature"][:n_factors] + [
        f"pc{k+1}" for k in range(3, n_factors)
    ]
    return {
        "loadings": pd.DataFrame(vectors[:, :n_factors], index=curve_history.columns,
                                 columns=names),
        "explained_variance_ratio": pd.Series(explained[:n_factors], index=names),
        "factor_vols_bp": pd.Series(np.sqrt(values[:n_factors]) * 100.0, index=names),
        "scores": pd.DataFrame(x @ vectors[:, :n_factors], index=changes.index, columns=names),
    }


def key_rate_dv01(price_fn, tenors, bump_bp: float = 1.0) -> pd.Series:
    """Rupee P&L of a 1 bp bump at each key tenor, holding the rest of the curve fixed.

    `price_fn` takes a dict ``{tenor: shift_in_bp}`` and returns the book's value.
    """
    base = price_fn({t: 0.0 for t in tenors})
    out = {}
    for t in tenors:
        up = price_fn({**{s: 0.0 for s in tenors}, t: bump_bp})
        out[t] = (up - base) / bump_bp
    return pd.Series(out, name="krd01")


def factor_shock_pnl(krd: pd.Series, loadings: pd.DataFrame, shocks_bp: dict) -> float:
    """P&L from shocking named PCA factors, given a key-rate DV01 profile."""
    total_bp = pd.Series(0.0, index=krd.index)
    for factor, size in shocks_bp.items():
        total_bp = total_bp.add(loadings[factor].reindex(krd.index).fillna(0.0) * size,
                                fill_value=0.0)
    return float((krd * total_bp).sum())
