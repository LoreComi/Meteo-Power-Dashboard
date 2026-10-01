"""SWE from the internal Exolabs model → {schema}.swe_daily.

Hydro_Report/SWE_Exolabs/Scripts/SWE_main.py (run by 102_SWE_lorenzo.bat in the
snow_obs environment) downloads the day's Exolabs Alps SWE raster, cuts it into
countries, Axpo catchments and height bands and appends one row per day to the
CSVs in Output_files/CSVs. This module reads those CSVs and uploads them as one
long table — nothing is recomputed, the numbers are the model's.

CSV → rows (level, region, band):
  t-/m-SWE_timeseries.csv                    alps        Alps          total
  t-/m-SWE_countries_timeseries.csv          country     Austria|Italy|France|Switzerland   total | below 1800 | above 1800
  t-/m-SWE_catchments_timeseries.csv         catchment   <34 Axpo catchments>               total
  t-/m-<catchment>_heightbands_timeseries.csv heightband <catchment>                         1 … 10
`t-` files hold the pixel SUM of SWE (kg/m² per pixel, summed — a volume proxy),
`m-` files the pixel MEAN (kg/m² = mm of water equivalent). `_roll` columns are
the model's 7-day centred rolling mean. Pixel size: the Alps raster is resampled
to 300 m (download_SWE.py: 20 m product ÷ xdim/20) and the country rasters are
cut from it; catchments and height bands are cut from the 20 m product. So
volume[m³] ≈ swe_total × pixel_size_m² × 1e-3.

Table swe_daily (MERGE key: day, level, region, band):
  day DATE, level, region, band, swe_total, swe_mean_mm, swe_total_roll, swe_mean_roll_mm,
  pixel_size_m INT, winter_season ('2025/2026'), day_of_winter (days since 1 Oct, as SWE_main), loaded_at
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("pipeline.swe")

SWE_COLUMNS = [
    ("day", "DATE"), ("level", "STRING"), ("region", "STRING"), ("band", "STRING"),
    ("swe_total", "DOUBLE"), ("swe_mean_mm", "DOUBLE"), ("swe_total_roll", "DOUBLE"), ("swe_mean_roll_mm", "DOUBLE"),
    ("pixel_size_m", "INT"), ("winter_season", "STRING"), ("day_of_winter", "INT"), ("loaded_at", "TIMESTAMP"),
]
SWE_KEYS = ["day", "level", "region", "band"]
PIXEL_ALPS, PIXEL_PRODUCT = 300, 20          # download_SWE.py: xdim = 300 (resampled), 20 m source product
SEASON_MONTH_IN, SEASON_MONTH_OUT = 10, 7     # SWE_main.py: plot_range_in Oct, plot_range_out Jul


def winter_season(days: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """SWE_main's winter_season / day_of_winter for any dates."""
    yr, mo = days.year, days.month
    start_year = np.where(mo <= SEASON_MONTH_OUT, yr - 1, yr)
    season = pd.Series([f"{y}/{y + 1}" for y in start_year], index=days)
    start = pd.to_datetime([f"{y}-{SEASON_MONTH_IN:02d}-01" for y in start_year])
    dow = pd.Series((days.values - start.values).astype("timedelta64[D]").astype(int) + 1, index=days)
    return season, dow


def _read(path: Path, header=0) -> pd.DataFrame | None:
    if not path.exists():
        log.warning("missing %s", path.name)
        return None
    df = pd.read_csv(path, header=header, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _pair(total: pd.Series | None, mean: pd.Series | None, total_roll, mean_roll,
          level: str, region: str, band: str, pixel: int) -> pd.DataFrame:
    idx = None
    for s in (total, mean, total_roll, mean_roll):
        if s is not None:
            idx = s.index if idx is None else idx.union(s.index)
    if idx is None or len(idx) == 0:
        return pd.DataFrame()
    out = pd.DataFrame(index=idx)
    out["level"], out["region"], out["band"] = level, region, band
    out["swe_total"] = pd.to_numeric(total, errors="coerce") if total is not None else np.nan
    out["swe_mean_mm"] = pd.to_numeric(mean, errors="coerce") if mean is not None else np.nan
    out["swe_total_roll"] = pd.to_numeric(total_roll, errors="coerce") if total_roll is not None else np.nan
    out["swe_mean_roll_mm"] = pd.to_numeric(mean_roll, errors="coerce") if mean_roll is not None else np.nan
    out["pixel_size_m"] = pixel
    return out


def read_swe_csvs(csv_dir: Path, include_heightbands: bool = True) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    # Alps total
    t, m = _read(csv_dir / "t-SWE_timeseries.csv"), _read(csv_dir / "m-SWE_timeseries.csv")
    parts.append(_pair(t["SWE_raw"] if t is not None else None, m["SWE_raw"] if m is not None else None,
                       t["SWE_roll"] if t is not None else None, m["SWE_roll"] if m is not None else None,
                       "alps", "Alps", "total", PIXEL_ALPS))

    # Countries (two-row header: country, band)
    t, m = _read(csv_dir / "t-SWE_countries_timeseries.csv", header=[0, 1]), _read(csv_dir / "m-SWE_countries_timeseries.csv", header=[0, 1])
    if t is not None or m is not None:
        cols = (t if t is not None else m).columns
        for country, band in [(c, b) for c, b in cols if not str(b).endswith("_roll")]:
            def col(df, b):
                return df[(country, b)] if df is not None and (country, b) in df.columns else None
            parts.append(_pair(col(t, band), col(m, band), col(t, f"{band}_roll"), col(m, f"{band}_roll"),
                               "country", str(country), str(band), PIXEL_ALPS))

    # Catchments
    t, m = _read(csv_dir / "t-SWE_catchments_timeseries.csv"), _read(csv_dir / "m-SWE_catchments_timeseries.csv")
    if t is not None or m is not None:
        cols = (t if t is not None else m).columns
        for ct in [c for c in cols if not str(c).endswith("_roll")]:
            def col(df, c):
                return df[c] if df is not None and c in df.columns else None
            parts.append(_pair(col(t, ct), col(m, ct), col(t, f"{ct}_roll"), col(m, f"{ct}_roll"),
                               "catchment", str(ct), "total", PIXEL_PRODUCT))

    # Height bands, one file pair per catchment
    if include_heightbands:
        for tp in sorted(csv_dir.glob("t-*_heightbands_timeseries.csv")):
            ct = re.sub(r"^t-(.*)_heightbands_timeseries\.csv$", r"\1", tp.name)
            t, m = _read(tp), _read(csv_dir / f"m-{ct}_heightbands_timeseries.csv")
            cols = (t if t is not None else m).columns
            for band in [c for c in cols if not str(c).endswith("_roll")]:
                def col(df, b):
                    return df[b] if df is not None and b in df.columns else None
                parts.append(_pair(col(t, band), col(m, band), col(t, f"{band}_roll"), col(m, f"{band}_roll"),
                                   "heightband", ct, str(band), PIXEL_PRODUCT))

    df = pd.concat([p for p in parts if not p.empty])
    df.index.name = "day"
    df = df.reset_index()
    df = df.dropna(subset=["swe_total", "swe_mean_mm", "swe_total_roll", "swe_mean_roll_mm"], how="all")
    season, dow = winter_season(pd.DatetimeIndex(df["day"]))
    df["winter_season"], df["day_of_winter"] = season.values, dow.values
    df["day"] = pd.to_datetime(df["day"]).dt.date
    df["loaded_at"] = pd.Timestamp.utcnow().tz_localize(None)
    return df[[c for c, _ in SWE_COLUMNS]]


def run(settings, writer, backfill: bool = False, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    df = read_swe_csvs(settings.swe_csv_dir, settings.swe_include_heightbands)
    if df.empty:
        log.warning("no SWE rows read from %s", settings.swe_csv_dir)
        return {"rows": 0}
    last = max(df["day"])
    if (today - last).days > 2:
        log.warning("SWE CSVs end on %s — the model has not run for %d days", last, (today - last).days)
    if not backfill:
        cutoff = today - dt.timedelta(days=settings.swe_refresh_days)
        df = df[df["day"] >= cutoff]
    writer.ensure_table(settings.swe_table_fq, SWE_COLUMNS,
                        "Exolabs SWE (internal model, Hydro_Report/SWE_Exolabs): daily pixel sum / mean per Alps, "
                        "country, Axpo catchment and height band, with the model's 7-day rolling means. "
                        "Written by Power_dashboard/pipeline.")
    n = writer.merge(settings.swe_table_fq, df, SWE_KEYS)
    summary = {"rows": n, "last_day": str(last), "levels": df.groupby("level").size().to_dict()}
    log.info("SWE done: %s", summary)
    return summary
