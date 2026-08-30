"""Corporate FX hedging simulator — the sales side of a markets desk.

An importer with a dollar payable every month does not care about VaR; they care that
the rupee cost of a year of imports lands inside the budget rate they gave their board.
This module puts the standard hedging policies on the same footing so the trade-off can
be argued with numbers rather than adjectives:

* **unhedged** — the benchmark. Cheapest on average when the forward premium is high,
  and the reason treasurers get fired.
* **full forward** — certainty at the cost of the entire forward premium, and no
  participation if the rupee appreciates.
* **layered forward** — hedge a rising fraction as each exposure nears. What most
  Indian corporate FX policies actually mandate.
* **zero-cost collar** — pay nothing, cap the worst rate, give up the good tail.
* **forward + option blend** — hedge part with certainty, keep upside on the rest.

The scoring metric is **Cash-Flow-at-Risk**: the 95th-percentile *excess* rupee cost
over the budget rate. That is the number a CFO can act on, and it is what makes an
options structure look expensive or cheap in a way that a simple variance comparison
does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

from ..pricing.fx import garman_kohlhagen, zero_cost_collar


@dataclass
class ExposureSchedule:
    """A series of foreign-currency cash flows to be hedged.

    `amounts_usd` is positive for a payable (importer buying USD) and the simulator
    reports the INR *cost*; for an exporter pass `payable=False` and it reports INR
    *receipts*, with the sign of every risk measure flipped accordingly.
    """

    tenors_years: np.ndarray
    amounts_usd: np.ndarray
    payable: bool = True

    @classmethod
    def monthly(cls, months: int = 12, amount_usd: float = 1e6,
                payable: bool = True) -> "ExposureSchedule":
        return cls(np.arange(1, months + 1) / 12.0, np.full(months, float(amount_usd)), payable)

    @property
    def total_usd(self) -> float:
        return float(np.sum(self.amounts_usd))


@dataclass
class HedgePolicy:
    """A named hedging rule: what fraction is hedged with what instrument."""

    name: str
    kind: Literal["unhedged", "forward", "collar", "blend"] = "forward"
    hedge_ratio: float | np.ndarray = 1.0
    cap_over_forward: float = 0.02   # collar cap, as a fraction above the fair forward
    option_share: float = 0.5        # for "blend": share hedged with a bought call
    vol: float = 0.05


def simulate_paths(spot: float, tenors: np.ndarray, n_paths: int = 20_000,
                   *, method: Literal["bootstrap", "gbm"] = "bootstrap",
                   returns: pd.Series | None = None, drift_pct: float | None = None,
                   vol: float = 0.05, block: int = 10, seed: int = 7) -> np.ndarray:
    """Simulate USD/INR at each tenor. Returns an ``(n_paths, len(tenors))`` array.

    ``bootstrap`` resamples *blocks* of realised daily log returns, which preserves the
    volatility clustering and the sharp-depreciation/slow-appreciation asymmetry that
    USD/INR actually shows and that a lognormal model erases. ``gbm`` is provided as
    the analytic control.
    """
    rng = np.random.default_rng(seed)
    tenors = np.asarray(tenors, float)
    horizon_days = np.maximum((tenors * 252).round().astype(int), 1)
    max_days = int(horizon_days.max())

    if method == "gbm":
        drift = (0.0 if drift_pct is None else drift_pct / 100.0)
        z = rng.standard_normal((n_paths, max_days))
        steps = (drift - 0.5 * vol ** 2) / 252.0 + vol / np.sqrt(252.0) * z
    else:
        if returns is None:
            raise ValueError("bootstrap simulation needs a realised return series")
        r = np.asarray(returns.dropna(), float)
        n_blocks = int(np.ceil(max_days / block))
        starts = rng.integers(0, len(r) - block, size=(n_paths, n_blocks))
        idx = starts[:, :, None] + np.arange(block)[None, None, :]
        steps = r[idx].reshape(n_paths, -1)[:, :max_days]
        if drift_pct is not None:  # re-centre onto an explicit view, if one is held
            steps = steps - steps.mean() + drift_pct / 100.0 / 252.0

    log_paths = np.cumsum(steps, axis=1)
    return spot * np.exp(log_paths[:, horizon_days - 1])


def _forward_curve(spot: float, tenors: np.ndarray, df_inr, df_usd) -> np.ndarray:
    return spot * np.asarray(df_usd, float) / np.asarray(df_inr, float)


def run_policy(policy: HedgePolicy, exposure: ExposureSchedule, spot: float,
               forwards: np.ndarray, r_inr: float, r_usd: float,
               paths: np.ndarray) -> dict:
    """Total INR cost of the exposure under one policy, one row per simulated path."""
    tenors = exposure.tenors_years
    amounts = exposure.amounts_usd
    ratios = np.broadcast_to(np.asarray(policy.hedge_ratio, float), tenors.shape).astype(float)

    unhedged_rate = paths                                    # (paths, tenors)
    cost = np.zeros_like(paths)

    if policy.kind == "unhedged":
        cost = unhedged_rate * amounts

    elif policy.kind == "forward":
        cost = (ratios * forwards + (1 - ratios) * unhedged_rate) * amounts

    elif policy.kind == "collar":
        caps = forwards * (1.0 + policy.cap_over_forward)
        floors = np.array([
            zero_cost_collar(spot, f, r_inr, r_usd, policy.vol, t, cap)
            for f, t, cap in zip(forwards, tenors, caps)
        ])
        effective = np.clip(unhedged_rate, floors, caps)     # collared rate
        cost = (ratios * effective + (1 - ratios) * unhedged_rate) * amounts

    elif policy.kind == "blend":
        fwd_share = ratios * (1 - policy.option_share)
        opt_share = ratios * policy.option_share
        strikes = forwards
        premia = np.array([
            float(garman_kohlhagen(spot, k, r_inr, r_usd, policy.vol, t, "call"))
            for k, t in zip(strikes, tenors)
        ])
        capped = np.minimum(unhedged_rate, strikes) + premia  # bought call + premium paid
        cost = (fwd_share * forwards + opt_share * capped
                + (1 - fwd_share - opt_share) * unhedged_rate) * amounts
    else:
        raise ValueError(f"unknown policy kind {policy.kind!r}")

    total = cost.sum(axis=1)
    effective_rate = total / exposure.total_usd
    return {"total_inr": total, "effective_rate": effective_rate}


def compare_policies(policies: list[HedgePolicy], exposure: ExposureSchedule, spot: float,
                     forwards: np.ndarray, r_inr: float, r_usd: float, paths: np.ndarray,
                     budget_rate: float | None = None) -> pd.DataFrame:
    """Score every policy on the same simulated paths.

    Columns
    -------
    mean_rate, p95_rate
        Average and 95th-percentile realised rate paid per dollar.
    cfar_95_cr
        Cash-Flow-at-Risk: 95th-percentile cost *in excess of the budget rate*, in
        rupees crore. This is the headline number for a treasurer.
    vol_reduction
        Percentage cut in the standard deviation of total cost versus unhedged — the
        conventional measure of hedge effectiveness.
    cost_vs_unhedged_cr
        Mean cost of the policy minus mean cost of doing nothing. Positive means the
        certainty was paid for.
    """
    budget = budget_rate if budget_rate is not None else float(forwards[0])
    rows, base_std, base_mean = [], None, None

    for policy in policies:
        res = run_policy(policy, exposure, spot, forwards, r_inr, r_usd, paths)
        total, rate = res["total_inr"], res["effective_rate"]
        if policy.kind == "unhedged":
            base_std, base_mean = float(total.std(ddof=1)), float(total.mean())
        rows.append({
            "policy": policy.name,
            "mean_rate": float(rate.mean()),
            "p95_rate": float(np.quantile(rate, 0.95)),
            "worst_1pct_rate": float(np.quantile(rate, 0.99)),
            "mean_cost_cr": float(total.mean()) / 1e7,
            "cfar_95_cr": float(np.quantile(total - budget * exposure.total_usd, 0.95)) / 1e7,
            "std_cost_cr": float(total.std(ddof=1)) / 1e7,
        })

    frame = pd.DataFrame(rows)
    if base_std:
        frame["vol_reduction_pct"] = (1 - frame["std_cost_cr"] * 1e7 / base_std) * 100
        frame["cost_vs_unhedged_cr"] = frame["mean_cost_cr"] - base_mean / 1e7
    return frame
