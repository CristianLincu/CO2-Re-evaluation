"""Energinet data access: fetch, cache, resample, and derive the energy balance."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from pipeline.config import (
    CACHE_DIR,
    CROSS_BORDER,
    DECISION_COLUMNS,
    GENERATION,
    INTERNAL_LINK,
    RESOLUTION_MIN,
)

BASE_URL = "https://api.energidataservice.dk/dataset"
POWER_DATASET = "PowerSystemRightNow"
FORECAST_DATASET = "Forecasts_5Min"

RENEWABLE_COLUMNS = ["SolarPower", "OffshoreWindPower", "OnshoreWindPower"]
FORECAST_TYPES = ["Offshore Wind", "Onshore Wind", "Solar"]

_MAX_RETRIES = 6
_BACKOFF_BASE = 2.0


def _ca_bundle():
    """Respect an explicit CA bundle if the environment provides one.

    Local machines running TLS-inspecting antivirus need the OS trust store;
    CI does not and falls back to certifi via requests' own default.
    """
    return os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")


def _request(dataset: str, params: dict) -> list[dict]:
    """GET one page, retrying politely through rate limits."""
    url = f"{BASE_URL}/{dataset}"
    verify = _ca_bundle()
    kwargs = {"timeout": 180}
    if verify:
        kwargs["verify"] = verify

    for attempt in range(_MAX_RETRIES):
        response = requests.get(url, params=params, **kwargs)
        if response.status_code == 200:
            return response.json().get("records", [])
        if response.status_code in (429, 502, 503, 504):
            wait = _BACKOFF_BASE**attempt
            print(f"  {dataset}: HTTP {response.status_code}, retrying in {wait:.0f}s")
            time.sleep(wait)
            continue
        response.raise_for_status()

    raise RuntimeError(f"{dataset}: exhausted retries on {params}")


def _chunks(start: datetime, end: datetime, days: int):
    cursor = start
    while cursor < end:
        stop = min(cursor + timedelta(days=days), end)
        yield cursor, stop
        cursor = stop


def _fmt(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M")


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # pyarrow missing or similar
        print(f"  cache write skipped ({exc})")


# --- Raw fetches -------------------------------------------------------------


def fetch_power_system(start: datetime, end: datetime, chunk_days: int = 20) -> pd.DataFrame:
    """One-minute power system records over an arbitrary range."""
    frames = []
    for chunk_start, chunk_end in _chunks(start, end, chunk_days):
        cache = _cache_path(f"power_{_fmt(chunk_start)[:10]}_{_fmt(chunk_end)[:10]}.parquet")
        cached = _read_cache(cache)
        if cached is not None:
            frames.append(cached)
            continue

        print(f"  fetching power {_fmt(chunk_start)} -> {_fmt(chunk_end)}")
        records = _request(
            POWER_DATASET,
            {
                "start": _fmt(chunk_start),
                "end": _fmt(chunk_end),
                "sort": "Minutes1UTC",
                "limit": 100000,
            },
        )
        frame = pd.DataFrame(records)
        if not frame.empty:
            _write_cache(frame, cache)
            frames.append(frame)
        time.sleep(0.4)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_renewable_forecasts(start: datetime, end: datetime, chunk_days: int = 20) -> pd.DataFrame:
    """Energinet's own wind and solar forecasts, summed across areas.

    Returns one row per five-minute slot with the forecast vintages kept
    separate: ``ForecastCurrent`` is the latest revision (what a live system
    sees), while ``Forecast5Hour`` was issued roughly five hours ahead and is
    therefore safe to use as a leak-free feature when backtesting.
    """
    frames = []
    for chunk_start, chunk_end in _chunks(start, end, chunk_days):
        cache = _cache_path(f"fcst_{_fmt(chunk_start)[:10]}_{_fmt(chunk_end)[:10]}.parquet")
        cached = _read_cache(cache)
        if cached is not None:
            frames.append(cached)
            continue

        print(f"  fetching forecasts {_fmt(chunk_start)} -> {_fmt(chunk_end)}")
        records = _request(
            FORECAST_DATASET,
            {
                "start": _fmt(chunk_start),
                "end": _fmt(chunk_end),
                "sort": "Minutes5UTC",
                "limit": 300000,
            },
        )
        frame = pd.DataFrame(records)
        if not frame.empty:
            _write_cache(frame, cache)
            frames.append(frame)
        time.sleep(0.4)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["ForecastType"].isin(FORECAST_TYPES)]
    raw["Minutes5UTC"] = pd.to_datetime(raw["Minutes5UTC"])

    value_cols = [c for c in ("ForecastCurrent", "Forecast5Hour", "ForecastDayAhead") if c in raw]
    # Sum DK1 + DK2 and the three renewable technologies into a national total.
    totals = raw.groupby("Minutes5UTC")[value_cols].sum(min_count=1)
    return totals.reset_index()


# --- Derived frame -----------------------------------------------------------


def add_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Renewables, Demand and the corrected exchange total.

    Demand follows the national balance identity

        demand = generation + net cross-border imports

    The Great Belt link (DK1-DK2) is deliberately excluded: it moves power
    between two Danish bidding zones and cannot change national supply. The
    original pipeline included it, which shifted demand by as much as 600 MW
    against a 1% balance tolerance.
    """
    out = df.copy()

    for column in RENEWABLE_COLUMNS + DECISION_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    out["Renewables"] = out[RENEWABLE_COLUMNS].sum(axis=1)
    out["NetImports"] = out[CROSS_BORDER].sum(axis=1)
    out["Generation"] = out[GENERATION].sum(axis=1) + out["Renewables"]
    out["Demand"] = out["Generation"] + out["NetImports"]

    return out


def resample_15min(df: pd.DataFrame) -> pd.DataFrame:
    """Average one-minute records onto the 15-minute settlement grid."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["Minutes1UTC"], utc=True)
    out = out.set_index("timestamp").sort_index()

    numeric = out.select_dtypes(include="number")
    resampled = numeric.resample(f"{RESOLUTION_MIN}min").mean()
    return resampled.dropna(subset=["CO2Emission"]).reset_index()


def build_training_frame(years: float = 1.0, end: datetime | None = None) -> pd.DataFrame:
    """Assemble the modelling table: 15-minute observations plus forecasts."""
    end = end or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(365 * years))

    print(f"Building training frame {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
    power = fetch_power_system(start, end)
    if power.empty:
        raise RuntimeError("no power system records returned")

    power = add_balance(power)
    frame = resample_15min(power)

    forecasts = fetch_renewable_forecasts(start, end)
    if not forecasts.empty:
        forecasts = forecasts.set_index("Minutes5UTC").sort_index()
        if forecasts.index.tz is None:
            forecasts.index = forecasts.index.tz_localize("UTC")
        forecasts = forecasts.resample(f"{RESOLUTION_MIN}min").mean().reset_index()
        forecasts = forecasts.rename(columns={"Minutes5UTC": "timestamp"})
        frame = frame.merge(forecasts, on="timestamp", how="left")

    return frame


def get_live_frame(hours: int = 30) -> pd.DataFrame:
    """Recent observations for serving, on the same 15-minute grid."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    power = fetch_power_system(start, end, chunk_days=5)
    if power.empty:
        raise RuntimeError("no recent power system records returned")
    return resample_15min(add_balance(power))


def get_live_renewable_forecast(steps: int) -> pd.DataFrame:
    """Energinet's latest wind and solar forecast for the coming horizon."""
    now = datetime.now(timezone.utc)
    end = now + timedelta(minutes=RESOLUTION_MIN * (steps + 4))
    frame = fetch_renewable_forecasts(now - timedelta(hours=2), end, chunk_days=5)
    if frame.empty:
        return frame
    frame = frame.set_index("Minutes5UTC").sort_index()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    return frame.resample(f"{RESOLUTION_MIN}min").mean().reset_index().rename(
        columns={"Minutes5UTC": "timestamp"}
    )
