"""US rates adapters — the USD leg of every covered-interest-parity calculation.

* **US Treasury** publishes its daily par yield curve as CSV, one file per year,
  with no key and no rate limit.
* **The New York Fed** publishes SOFR (the USD overnight risk-free rate) as JSON.
"""
from __future__ import annotations

import io
import json

import pandas as pd

from ._http import get

UST_CSV = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)
SOFR_JSON = "https://markets.newyorkfed.org/api/rates/secured/sofr/last/{n}.json"

# Treasury column label -> tenor in years
UST_TENORS = {
    "1 Mo": 1 / 12, "1.5 Month": 1.5 / 12, "2 Mo": 2 / 12, "3 Mo": 0.25,
    "4 Mo": 4 / 12, "6 Mo": 0.5, "1 Yr": 1.0, "2 Yr": 2.0, "3 Yr": 3.0,
    "5 Yr": 5.0, "7 Yr": 7.0, "10 Yr": 10.0, "20 Yr": 20.0, "30 Yr": 30.0,
}


def fetch_ust_curve(years: list[int] | int, cache: bool = True) -> pd.DataFrame:
    """Daily US Treasury par yield curve, indexed by date with tenor-year columns."""
    if isinstance(years, int):
        years = [years]
    frames = []
    for year in years:
        text = get(UST_CSV.format(year=year), cache=cache)
        frame = pd.read_csv(io.StringIO(text))
        frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
        frame = frame.set_index("Date")
        keep = [c for c in frame.columns if c in UST_TENORS]
        frame = frame[keep].rename(columns=UST_TENORS)
        frames.append(frame)
    out = pd.concat(frames).sort_index()
    return out.loc[:, sorted(out.columns)]


def fetch_sofr(n: int = 100, cache: bool = True) -> pd.Series:
    """Last `n` SOFR fixings in percent, indexed by effective date."""
    payload = json.loads(get(SOFR_JSON.format(n=n), cache=cache))
    rows = payload["refRates"]
    series = pd.Series(
        {pd.Timestamp(r["effectiveDate"]): float(r["percentRate"]) for r in rows},
        name="SOFR",
    )
    return series.sort_index()
