#!/usr/bin/env python
"""Append today's CCIL curve inputs to the local history.

CCIL's site exposes only the last two business days, so a curve *history* has to be
accumulated. Run this daily (cron: ``30 19 * * 1-5``) and the panel in
``data/snapshots/history.csv`` grows into the time series that the PCA and the
rates-VaR modules want.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imd.config import SNAPSHOTS  # noqa: E402
from imd.curves import ZeroCurve, nss_from_ccil  # noqa: E402
from imd.data import fetch_ccil_tenorwise_yields, fetch_ccil_zcyc_params  # noqa: E402

GRID = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
HISTORY = SNAPSHOTS / "history.csv"


def main() -> None:
    zcyc = fetch_ccil_zcyc_params(cache=False)
    yields = fetch_ccil_tenorwise_yields(cache=False)

    rows = []
    for _, row in zcyc.iterrows():
        curve = ZeroCurve.from_nss(nss_from_ccil(row), "continuous")
        rows.append({"date": row["date"], **{f"z{t:g}y": round(float(z), 4)
                                             for t, z in zip(GRID, curve.zero(GRID))}})
    panel = pd.DataFrame(rows)

    if HISTORY.exists():
        panel = pd.concat([pd.read_csv(HISTORY, parse_dates=["date"]), panel])
    panel = panel.drop_duplicates("date", keep="last").sort_values("date")
    panel.to_csv(HISTORY, index=False)

    asof = yields["date"].max().date()
    yields[yields["date"] == yields["date"].max()].to_csv(
        SNAPSHOTS / f"ccil_yields_{asof}.csv", index=False
    )
    print(f"history now spans {panel['date'].min().date()} to {panel['date'].max().date()} "
          f"({len(panel)} business days)")


if __name__ == "__main__":
    main()
