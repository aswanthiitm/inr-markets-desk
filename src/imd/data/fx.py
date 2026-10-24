"""FX and cross-asset price history from the public Yahoo Finance chart endpoint."""
from __future__ import annotations

import json

import pandas as pd

from ..config import SNAPSHOTS
from ._http import get

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range}&interval=1d"

SYMBOLS = {
    "USDINR": "USDINR%3DX",
    "EURINR": "EURINR%3DX",
    "GBPINR": "GBPINR%3DX",
    "JPYINR": "JPYINR%3DX",
    "DXY": "DX-Y.NYB",
    "EURUSD": "EURUSD%3DX",
    "BRENT": "BZ%3DF",
    "NIFTY": "%5ENSEI",
}


def fetch_yahoo_series(name: str, range_: str = "10y", cache: bool = True) -> pd.Series:
    """Daily close series for a named instrument (see :data:`SYMBOLS`)."""
    symbol = SYMBOLS.get(name, name)
    snapshot = SNAPSHOTS / f"fx_{name}.csv"
    try:
        payload = json.loads(get(CHART.format(symbol=symbol, range=range_), cache=cache))
    except Exception:  # noqa: BLE001 - unreachable; fall back to the committed capture
        if snapshot.exists():
            series = pd.read_csv(snapshot, index_col=0, parse_dates=True).squeeze("columns")
            series.name = name
            series.attrs["source"] = "snapshot"
            return series
        raise
    result = payload["chart"]["result"][0]
    idx = pd.to_datetime(result["timestamp"], unit="s").normalize()
    close = result["indicators"]["quote"][0]["close"]
    series = pd.Series(close, index=idx, name=name).dropna()
    out = series[~series.index.duplicated(keep="last")].sort_index()
    out.attrs["source"] = "live"
    return out
