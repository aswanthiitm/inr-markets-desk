"""Adapters for CCIL (The Clearing Corporation of India Ltd) public market data.

CCIL is the central counterparty for the Indian G-Sec, money and forex markets and
publishes two series this project depends on:

* **ZCYC parameters** — the six Nelson-Siegel-Svensson parameters of the *official*
  zero-coupon sovereign rupee yield curve. We use these as the benchmark against
  which our own independently-fitted curve is validated.
* **Tenor-wise indicative yields** — end-of-day indicative YTMs for the most liquid
  T-Bill, central-government (G-Sec) and state-government (SDL) securities.

Both pages are Liferay portlets that render their table server-side, so a plain GET
plus an HTML table parse is sufficient and stable.
"""
from __future__ import annotations

import io
import re

import pandas as pd

from ._http import get

ZCYC_URL = "https://www.ccilindia.com/zcyc-parameters"
TENORWISE_URL = "https://www.ccilindia.com/tenorwise-indicative-yields"


def _first_data_table(html: str, expected_header: str) -> pd.DataFrame:
    """Return the first HTML <table> whose header row contains `expected_header`."""
    for match in re.finditer(r"<table.*?</table>", html, re.S):
        try:
            frames = pd.read_html(io.StringIO(match.group(0)))
        except ValueError:
            continue
        for frame in frames:
            cols = [str(c) for c in frame.columns]
            if any(expected_header.lower() in c.lower() for c in cols) and not frame.empty:
                return frame
    raise RuntimeError(f"no CCIL table containing column ~ {expected_header!r}")


def fetch_ccil_zcyc_params(cache: bool = True) -> pd.DataFrame:
    """CCIL's published Nelson-Siegel-Svensson parameters for the sovereign ZCYC.

    Returns columns ``[date, beta0, beta1, beta2, beta3, tau1, tau2]``, most recent
    first. Betas are in percent; taus are in years.
    """
    frame = _first_data_table(get(ZCYC_URL, cache=cache), "Date")
    frame = frame.rename(columns={c: str(c).strip() for c in frame.columns})
    cols = list(frame.columns)
    frame = frame.rename(
        columns=dict(zip(cols[:7], ["date", "beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]))
    )
    frame = frame[["date", "beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    for c in frame.columns[1:]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame.dropna().sort_values("date", ascending=False).reset_index(drop=True)


# Tenor buckets as published by CCIL -> representative maturity in years.
# T-Bill buckets are exact; G-Sec buckets are taken at the bucket mid-point and are
# later refined using the maturity year parsed out of the security description.
_BUCKET_YEARS = {
    "91D": 0.25, "182D": 0.50, "364D": 1.00,
    "1Y-2Y": 1.5, "2Y-3Y": 2.5, "3Y-4Y": 3.5, "4Y-5Y": 4.5, "5Y-6Y": 5.5,
    "6Y-7Y": 6.5, "7Y-8Y": 7.5, "8Y-9Y": 8.5, "9Y-10Y": 9.5, "10Y-11Y": 10.5,
    "11Y-12Y": 11.5, "12Y-13Y": 12.5, "13Y-15Y": 14.0, "15Y-20Y": 17.5,
    "20Y-25Y": 22.5, "25Y-28Y": 26.5, "28Y-30Y": 29.0, "30Y-40Y": 35.0,
    "5Y": 5.0, "10Y": 10.0, "15Y": 15.0,
}

_COUPON_RE = re.compile(r"^\s*(\d+\.?\d*)\s*%")
_MATYEAR_RE = re.compile(r"(20\d{2})\s*$")
_DTB_RE = re.compile(r"\((\d{2})/(\d{2})/(\d{4})\)")


def _classify(security: str) -> str:
    s = security.upper()
    if "DTB" in s or "TB" in s.split()[0:1]:
        return "TBILL"
    if "SGS" in s or "SDL" in s:
        return "SDL"
    return "GSEC"


def fetch_ccil_tenorwise_yields(cache: bool = True) -> pd.DataFrame:
    """CCIL end-of-day indicative YTMs by tenor bucket.

    Returns ``[date, bucket, security, ytm, segment, coupon, maturity, tenor_years]``
    where ``segment`` is one of ``TBILL``/``GSEC``/``SDL``, ``ytm`` and ``coupon`` are
    in percent, and ``tenor_years`` is the time to maturity used for curve fitting.
    """
    frame = _first_data_table(get(TENORWISE_URL, cache=cache), "Tenor")
    cols = list(frame.columns)
    frame = frame.rename(columns=dict(zip(cols[:4], ["date", "bucket", "security", "ytm"])))
    frame = frame[["date", "bucket", "security", "ytm"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["ytm"] = pd.to_numeric(frame["ytm"], errors="coerce")
    frame["bucket"] = frame["bucket"].astype(str).str.strip()
    frame["security"] = frame["security"].astype(str).str.strip()
    frame = frame.dropna(subset=["ytm"])

    frame["segment"] = frame["security"].map(_classify)
    frame["coupon"] = frame["security"].str.extract(_COUPON_RE)[0].astype(float)

    def _tenor(row: pd.Series) -> float:
        asof = row["date"]
        dtb = _DTB_RE.search(row["security"])
        if dtb:  # T-Bills carry an explicit dd/mm/yyyy redemption date
            mat = pd.Timestamp(int(dtb.group(3)), int(dtb.group(2)), int(dtb.group(1)))
            return max((mat - asof).days / 365.0, 1 / 365.0)
        year = _MATYEAR_RE.search(row["security"])
        if year:  # dated securities are quoted as "<coupon>% GS <maturity year>"
            mat = pd.Timestamp(int(year.group(1)), 6, 30)
            return max((mat - asof).days / 365.0, 0.01)
        return float(_BUCKET_YEARS.get(row["bucket"], "nan"))

    frame["tenor_years"] = frame.apply(_tenor, axis=1)
    frame["bucket_years"] = frame["bucket"].map(_BUCKET_YEARS)
    return frame.sort_values(["date", "segment", "tenor_years"]).reset_index(drop=True)
