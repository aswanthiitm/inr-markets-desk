"""Project-wide paths and market conventions."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
SNAPSHOTS = DATA / "snapshots"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _p in (RAW, SNAPSHOTS, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) inr-markets-desk/1.0"
HTTP_TIMEOUT = 45

# --- Indian market conventions -------------------------------------------------
# G-Sec / SDL coupons pay semi-annually on an Actual/365 (fixed) basis; T-bills are
# quoted on a 365-day simple-yield basis. INR OIS fixes off the overnight MIBOR and
# settles Actual/365 with annual compounding beyond one year.
GSEC_COUPON_FREQ = 2
INR_DAYCOUNT = 365.0
USD_DAYCOUNT = 360.0          # SOFR / USD money-market basis
FX_SPOT_LAG_DAYS = 2          # USD/INR spot is T+2
