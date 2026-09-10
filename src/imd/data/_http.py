"""Thin HTTP helper with retries and an on-disk cache.

Every network call in this project goes through here so that (a) a run is
reproducible from the cached payload and (b) a flaky endpoint cannot break a
notebook halfway through.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests

from ..config import HTTP_TIMEOUT, RAW, USER_AGENT

CACHE = RAW / "_httpcache"
CACHE.mkdir(parents=True, exist_ok=True)


def get(url: str, *, cache: bool = True, max_age_s: int = 6 * 3600, retries: int = 3) -> str:
    """GET `url` as text, served from disk if a fresh copy exists."""
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    path = CACHE / f"{key}.txt"
    if cache and path.exists() and (time.time() - path.stat().st_mtime) < max_age_s:
        return path.read_text(encoding="utf-8")

    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            if cache:
                path.write_text(r.text, encoding="utf-8")
            return r.text
        except Exception as exc:  # noqa: BLE001 - retry on any transport failure
            last = exc
            time.sleep(1.5 * (attempt + 1))

    if path.exists():  # stale cache beats no data
        return path.read_text(encoding="utf-8")
    raise RuntimeError(f"GET failed for {url}: {last}")


def cached_path(url: str) -> Path:
    return CACHE / f"{hashlib.sha256(url.encode()).hexdigest()[:20]}.txt"
