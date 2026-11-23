"""Google WeatherNext 3 ensemble means — 2 m temperature and 100 m wind speed → {schema}.google_wn_daily.

The app shows one AI model from the silver layer — ECMWF's AIFS-ENS, read by the notebook
from the Meteomatics tables and reduced to weighted means (cells 2b and 7). This job puts
Google's AI ensemble next to it on the same method, from Google's own bucket, so the
Morning Call / Gas Demand "AI models" source and the Forecast section's model-mean
overlay carry both.

Source — the WeatherNext 3 *statistics* store on Google Cloud Storage (Zarr v3), one
store per initialisation:

  gs://weathernext3_statistics_spatial/weathernext_3_0_0_statistics/zarr/2026_to_present/
      <YYYYMMDD>_<HH>hr_<XX>_preds/predictions.zarr

  temperature_2m_mean, wind_speed_100m_mean (lead_time, lat_0p1, lon_0p1) — the mean of the
  64 members (Kelvin, m/s), 0.1° global grid (longitudes 0–360), hourly lead times 1 h … 360 h
  for the 00/06/12/18 UTC inits. One chunk per lead time holds the whole globe.

It is opened the way Google's starter notebook does (obstore + zarr + xarray). The
bucket is not anonymous: the reader authenticates with Google application-default
credentials — `gcloud auth application-default login` once, with an account that can
read the WeatherNext buckets — found at GOOGLE_WN_CREDENTIALS, else
GOOGLE_APPLICATION_CREDENTIALS, else gcloud's own file (%APPDATA%\\gcloud on Windows).
The statistics bucket is not Requester Pays; GOOGLE_WN_BILLING_PROJECT adds the
x-goog-user-project header for a bucket that is (the full-ensemble one).

What is computed, per run and per field (FIELDS):

  daily mean   the lead times every GOOGLE_WN_STEP_HOURS (3) are averaged into the UTC
               day they end in (valid 03 … 24 UTC = 8 fields a day; step 1 reads all 24).
               n_points is the day's coverage on the 96-per-day scale of the other morning
               tables (a 12z run's first and last day are 48), so the app drops partial
               days by the rule it already has.
  0.5° cells   the 0.1° field is averaged over the 5 × 5 points of each 0.5° cell of the
               field's weights file (cell centre ± 0.2°)
  means        SUM(value × weight) / SUM(weight) over a country's cells, and over ALL the
               cells of a region's countries, exactly as the notebook's joins do:
                 tt  temperature, °C — population (pop_weights_0p5.csv), the regions of the
                     notebook's MM_REGION_GROUPS and every country
                 ws  100 m wind speed, m/s — installed wind capacity (wind_weights_0p5.csv,
                     build_wind_weights.py), the regions of the notebook's MM_WIND_GROUPS:
                     the wind where the turbines stand, not a production figure
  normal       tt only — the Volue normal the Meteomatics rows carry: per region from
               morning_daily_volue, per country from fcst_daily (read from the sandbox;
               left NULL on a dry run). There is no 100 m wind-speed normal in Databricks.

Table google_wn_daily — one row per run / family / scope / key / day (MERGE; runs older
than GOOGLE_WN_RETENTION_DAYS pruned):
  provider 'Google', model_family, pattern 'google-weathernext', family ('tt' | 'ws'),
  curve_name (the Zarr variable), reference_date (the INIT time, UTC — not a load time),
  init_hour, scope ('region' = the Morning Call's regions, 'country' = the Forecast
  section's areas), key, day, lead_day, value, n_points, normal, loaded_at

    python run_daily.py --only google              the latest cycles not in the table yet
    python run_daily.py --only google --backfill   the last GOOGLE_WN_BACKFILL_RUNS cycles, re-written
    python run_daily.py --only google --dry-run    no Databricks; the frame goes to pipeline/out/

Needs zarr >= 3 and obstore in the environment (snow_obs: pip install "numpy==1.26.4"
zarr obstore — the pin keeps pip from upgrading numpy under the SWE model).
WeatherNext data older than one hour is CC BY 4.0 (attribution: Google); newer data is
under Google's experimental real-time terms — hence GOOGLE_WN_MIN_AGE_HOURS >= 1.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("pipeline.google")

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent / "weather-power-desk-app"


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


TABLE = os.environ.get("GOOGLE_WN_TABLE", "google_wn_daily")
BUCKET = os.environ.get("GOOGLE_WN_BUCKET", "weathernext3_statistics_spatial")
PREFIX = os.environ.get("GOOGLE_WN_PREFIX", "weathernext_3_0_0_statistics/zarr/2026_to_present").strip("/")
RUN_SUFFIX = os.environ.get("GOOGLE_WN_RUN_SUFFIX", "01")                  # the <XX> of <YYYYMMDD>_<HH>hr_<XX>_preds
CYCLES = tuple(int(h) for h in _env_list("GOOGLE_WN_CYCLES", "0,12"))       # 6 and 18 exist too (360 h each)
STEP_HOURS = int(os.environ.get("GOOGLE_WN_STEP_HOURS", "3"))              # lead times read: every N hours
MIN_AGE_HOURS = float(os.environ.get("GOOGLE_WN_MIN_AGE_HOURS", "2"))      # a cycle is tried once it is this old
RUNS_PER_PASS = int(os.environ.get("GOOGLE_WN_RUNS", "2"))                 # latest cycles checked per run
BACKFILL_RUNS = int(os.environ.get("GOOGLE_WN_BACKFILL_RUNS", "6"))
RETENTION_DAYS = int(os.environ.get("GOOGLE_WN_RETENTION_DAYS", "10"))     # the app looks 10 days back
CREDENTIALS = os.environ.get("GOOGLE_WN_CREDENTIALS", "")
BILLING_PROJECT = os.environ.get("GOOGLE_WN_BILLING_PROJECT", "")
# A folder holding <run>/predictions.zarr stores read instead of the bucket (tests, a downloaded run)
LOCAL_ROOT = os.environ.get("GOOGLE_WN_LOCAL_ROOT", "")

PROVIDER = "Google"
MODEL_FAMILY = "Google WeatherNext"
PATTERN = "google-weathernext"          # the app's run pattern (_config.GOOGLE_WN)
N_POINTS_FULL = 96                      # a whole day on the other morning tables' scale (15-min values)
FINE_PER_CELL = 5                       # 0.1° points per 0.5° cell, each way
RETRIES, BACKOFF_S = 3, 10

# The Morning Call's regions as groups of countries — power_desk_refresh.py's MM_REGION_GROUPS
# (cell 7); be / nl are the Gas Demand section's LDZ countries. Keep the two in sync.
REGION_GROUPS: dict[str, list[str]] = {
    "fr": ["FR"], "de": ["DE"], "uk": ["UK"], "it": ["IT"], "hu": ["HU"],
    "np": ["NO", "SE", "FI", "DK"], "ib": ["ES", "PT"], "see": ["SI", "HR", "SK", "HU"],
    "be": ["BE"], "nl": ["NL"],
}
# The Morning Call's wind regions — power_desk_refresh.py's MM_WIND_GROUPS (cell 7). SEE is every
# south-eastern country with wind farms in the weights (Romania and Greece carry most of it).
WIND_REGION_GROUPS: dict[str, list[str]] = {
    "de": ["DE"], "uk": ["UK"], "fr": ["FR"], "it": ["IT"],
    "np": ["NO", "SE", "FI", "DK"], "ib": ["ES", "PT"],
    "see": ["HU", "HR", "RO", "BG", "GR", "RS", "BA", "MK", "ME", "AL", "XK"],
}

# family → the Zarr variable, its weights and what is written. `kelvin`: converted to °C;
# `valid`: the range a whole, physical field stays in (a missing chunk reads as NaN or 0).
FIELDS: dict[str, dict] = {
    "tt": {"variable": os.environ.get("GOOGLE_WN_VARIABLE", "temperature_2m_mean"),
           "weights": Path(os.environ.get("GOOGLE_WN_WEIGHTS_CSV", str(APP_DIR / "pop_weights_0p5.csv"))),
           "weight_col": "population", "groups": REGION_GROUPS, "scopes": ("country", "region"),
           "kelvin": True, "valid": (180.0, 340.0)},
    "ws": {"variable": os.environ.get("GOOGLE_WN_WIND_VARIABLE", "wind_speed_100m_mean"),
           "weights": Path(os.environ.get("GOOGLE_WN_WIND_WEIGHTS_CSV", str(APP_DIR / "wind_weights_0p5.csv"))),
           "weight_col": "capacity_mw", "groups": WIND_REGION_GROUPS, "scopes": ("region",),
           "kelvin": False, "valid": (0.0, 90.0)},
}
FAMILIES = [f for f in _env_list("GOOGLE_WN_FAMILIES", "tt,ws") if f in FIELDS]

COLUMNS = [
    ("provider", "STRING"), ("model_family", "STRING"), ("pattern", "STRING"), ("family", "STRING"),
    ("curve_name", "STRING"), ("reference_date", "TIMESTAMP"), ("init_hour", "INT"), ("scope", "STRING"),
    ("key", "STRING"), ("day", "DATE"), ("lead_day", "INT"), ("value", "DOUBLE"), ("n_points", "INT"),
    ("normal", "DOUBLE"), ("loaded_at", "TIMESTAMP"),
]
KEYS = ["pattern", "family", "reference_date", "scope", "key", "day"]


class RunIncomplete(RuntimeError):
    """The run's store exists but a field read from it has gaps — still being written."""


# ─── weights on the 0.5° grid ───────────────────────────────────────────────────

class Grid:
    """A weights file as a 0.5° cell grid, the 0.1° points each cell averages, and per
    country / region the cells and normalised weights of its mean."""

    def __init__(self, weights: pd.DataFrame, region_groups: dict[str, list[str]] = REGION_GROUPS,
                 weight_col: str = "population", scopes: tuple = ("country", "region")):
        need = {"area", "latitude", "longitude", weight_col}
        if not need <= set(weights.columns):
            raise ValueError(f"the weights need the columns {sorted(need)}, got {list(weights.columns)}")
        w = weights[weights[weight_col] > 0]
        res = 1.0 / 2
        self.lat_c = np.arange(w["latitude"].min(), w["latitude"].max() + res / 2, res)      # cell centres, south → north
        self.lon_c = np.arange(w["longitude"].min(), w["longitude"].max() + res / 2, res)    # west → east, −180…180
        half = (FINE_PER_CELL - 1) / 2 * 0.1
        self.lat_f = np.round(self.lat_c[0] - half + 0.1 * np.arange(FINE_PER_CELL * len(self.lat_c)), 4)
        self.lon_f = np.round(self.lon_c[0] - half + 0.1 * np.arange(FINE_PER_CELL * len(self.lon_c)), 4)
        i = np.rint((w["latitude"].to_numpy() - self.lat_c[0]) / res).astype(int)
        j = np.rint((w["longitude"].to_numpy() - self.lon_c[0]) / res).astype(int)
        cell = i * len(self.lon_c) + j
        wt = w[weight_col].to_numpy(dtype=float)
        area = w["area"].to_numpy()
        self.targets: list[tuple[str, str, np.ndarray, np.ndarray]] = []      # (scope, key, cells, weights)
        if "country" in scopes:
            for a in sorted(set(area)):
                m = area == a
                self.targets.append(("country", a, cell[m], wt[m] / wt[m].sum()))
        if "region" in scopes:
            for region, countries in region_groups.items():
                m = np.isin(area, countries)
                missing = [c for c in countries if c not in set(area)]
                if missing:
                    log.warning("region %s: no %s weights for %s — left out", region, weight_col, ", ".join(missing))
                if m.any():
                    self.targets.append(("region", region, cell[m], wt[m] / wt[m].sum()))

    def means(self, field: np.ndarray) -> list[tuple[str, str, float]]:
        """Weighted means of a (lat_c × lon_c) field."""
        flat = field.ravel()
        return [(scope, key, float((flat[cells] * wts).sum())) for scope, key, cells, wts in self.targets]


def load_grid(family: str = "tt") -> Grid:
    f = FIELDS[family]
    if not f["weights"].exists():
        raise FileNotFoundError(f"{family} weights not found at {f['weights']}")
    return Grid(pd.read_csv(f["weights"]), f["groups"], f["weight_col"], f["scopes"])


# ─── the bucket ─────────────────────────────────────────────────────────────────

def credentials_path() -> str | None:
    """The Google credentials file: the setting, the standard variable, or gcloud's
    application-default file (obstore only looks under ~/.config, where Windows has none)."""
    cands = [CREDENTIALS, os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")]
    if os.environ.get("APPDATA"):
        cands.append(os.path.join(os.environ["APPDATA"], "gcloud", "application_default_credentials.json"))
    cands.append(os.path.join(os.path.expanduser("~"), ".config", "gcloud", "application_default_credentials.json"))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def _store(sub: str = ""):
    """An obstore store on PREFIX[/sub] of the bucket — or of GOOGLE_WN_LOCAL_ROOT."""
    from obstore.store import GCSStore, LocalStore
    if LOCAL_ROOT:
        return LocalStore(prefix=str(Path(LOCAL_ROOT) / sub) if sub else LOCAL_ROOT)
    kwargs: dict = {"prefix": f"{PREFIX}/{sub}" if sub else PREFIX}
    cred = credentials_path()
    if cred:
        kwargs["config"] = {"application_credentials": cred}
    else:
        log.warning("no Google credentials file found — the bucket will refuse an anonymous read "
                    "(run `gcloud auth application-default login`, or set GOOGLE_WN_CREDENTIALS)")
    if BILLING_PROJECT:
        kwargs["client_options"] = {"default_headers": {"x-goog-user-project": BILLING_PROJECT}}
    return GCSStore(BUCKET, **kwargs)


def _explain(e: Exception) -> str:
    """A bucket error in the desk's words."""
    s = str(e)
    if any(k in s for k in ("401", "403", "ermission", "nauthenticated", "AccessDenied", "invalid_grant")):
        return ("Google refused the read of gs://%s — sign in with an account that can read the WeatherNext "
                "buckets (`gcloud auth application-default login`) or point GOOGLE_WN_CREDENTIALS at a "
                "credentials file. Original error: %s" % (BUCKET, s[:300]))
    return s[:400]


class Runs:
    """Finds a cycle's store folder, <YYYYMMDD>_<HH>hr_<RUN_SUFFIX>_preds, by asking for its
    zarr.json — one small request. (No bucket listing: the folder holds every hourly init since
    January 2026, 7 834 on 2026-11-23, all with the suffix '01', and listing it took 50 s.)"""

    def __init__(self):
        self.root = _store()

    def find(self, init: dt.datetime) -> str | None:
        import obstore
        name = f"{init:%Y%m%d}_{init:%H}hr_{RUN_SUFFIX}_preds"
        try:
            obstore.head(self.root, f"{name}/predictions.zarr/zarr.json")
            return name
        except FileNotFoundError:
            return None


def open_run(name: str):
    """The run's statistics store as a lazily-indexed xarray Dataset (no dask)."""
    import warnings
    import xarray as xr
    import zarr
    zstore = zarr.storage.ObjectStore(_store(f"{name}/predictions.zarr"), read_only=True)
    with warnings.catch_warnings():        # xarray warns when a store has no consolidated metadata; either is fine
        warnings.simplefilter("ignore", RuntimeWarning)
        return xr.open_zarr(zstore, chunks=None, decode_timedelta=True)


# ─── one run → daily weighted means ─────────────────────────────────────────────

def recent_cycles(now: dt.datetime, n: int, min_age_hours: float = MIN_AGE_HOURS, cycles=CYCLES) -> list[dt.datetime]:
    """The n latest inits of `cycles` at least min_age_hours before now (UTC, naive), newest first."""
    cutoff = now - dt.timedelta(hours=min_age_hours)
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    span = n // max(1, len(cycles)) + 2
    cands = sorted((day0 - dt.timedelta(days=d) + dt.timedelta(hours=h) for d in range(span + 1) for h in cycles),
                   reverse=True)
    return [c for c in cands if c <= cutoff][:n]


def _axis_index(coord: np.ndarray, wanted: np.ndarray, circular: bool = False) -> np.ndarray:
    """Positions of `wanted` on a regular coordinate axis (either direction); raises when
    the axis does not carry those points — a different grid than the one documented."""
    coord = np.asarray(coord, dtype=float)
    step = (coord[-1] - coord[0]) / (len(coord) - 1)
    target = np.mod(wanted, 360.0) if circular else wanted
    idx = np.rint((target - coord[0]) / step).astype(int)
    if circular:
        idx = np.mod(idx, len(coord))
    if idx.min() < 0 or idx.max() >= len(coord):
        raise ValueError(f"grid axis {coord[0]:g}…{coord[-1]:g} does not cover {wanted.min():g}…{wanted.max():g}")
    off = np.abs(coord[idx] - target)
    if circular:
        off = np.minimum(off, 360.0 - off)
    if off.max() > 0.02:
        raise ValueError(f"grid axis (step {step:g}) is not the 0.1° grid expected — off by up to {off.max():.3f}°")
    return idx


def daily_means(ds, init: dt.datetime, grid: Grid, family: str = "tt", step_hours: int = STEP_HOURS) -> pd.DataFrame:
    """scope, key, day, value, n_points for every UTC day the run covers, for one field of FIELDS."""
    spec = FIELDS[family]
    variable = spec["variable"]
    if variable not in ds.variables:
        raise KeyError(f"{variable} is not in the store (it has {len(ds.data_vars)} variables, e.g. "
                       f"{', '.join(list(ds.data_vars)[:4])})")
    da = ds[variable]
    lat_dim = next(d for d in da.dims if d.startswith("lat"))
    lon_dim = next(d for d in da.dims if d.startswith("lon"))
    lead_dim = next(d for d in da.dims if d not in (lat_dim, lon_dim))
    da = da.transpose(lead_dim, lat_dim, lon_dim)
    lead = ds[lead_dim].values
    if lead.dtype.kind != "m":
        raise TypeError(f"{lead_dim} is {lead.dtype}, not a timedelta — the store's time encoding changed")
    hours = (lead / np.timedelta64(1, "h")).astype(float)
    if "init_time" in ds.coords:
        t0 = pd.Timestamp(ds["init_time"].values)
        if t0 != pd.Timestamp(init):
            log.warning("the store's init_time is %s, the folder says %s — using the store's", t0, init)
            init = t0.to_pydatetime()

    ilat = _axis_index(ds[lat_dim].values, grid.lat_f)
    ilon = _axis_index(ds[lon_dim].values, grid.lon_f, circular=True)
    lo, hi = int(ilat.min()), int(ilat.max()) + 1
    rows_in_band = ilat - lo
    n_lat, n_lon = len(grid.lat_c), len(grid.lon_c)

    keep = np.where((hours > 0) & (np.mod(hours, step_hours) == 0))[0]
    valid = pd.Timestamp(init) + pd.to_timedelta(hours[keep], unit="h")
    days = (valid - pd.Timedelta(minutes=1)).normalize()              # 24:00 belongs to the day it ends
    per_day = 24 // step_hours
    out = []
    t0 = time.time()
    lo_ok, hi_ok = spec["valid"]
    for day in days.unique():
        idx = keep[days == day]
        band = _read(da, {lead_dim: idx, lat_dim: slice(lo, hi)})     # (n, lat band, every longitude)
        fine = band[:, rows_in_band][:, :, ilon]                      # (n, 0.1° window south → north, west → east)
        # A chunk not written yet reads as the store's fill value (NaN or 0): every field must be
        # a whole, physical field before it is averaged into the day.
        if not np.isfinite(fine).all() or fine.min() < lo_ok or fine.max() > hi_ok:
            raise RunIncomplete(f"{variable} has gaps or fill values on {day:%Y-%m-%d} — the run is still being written")
        cells = fine.mean(axis=0, dtype=np.float64).reshape(n_lat, FINE_PER_CELL, n_lon, FINE_PER_CELL).mean(axis=(1, 3))
        if spec["kelvin"]:
            cells = cells - 273.15
        n_points = int(round(N_POINTS_FULL * len(idx) / per_day))
        out += [(scope, key, day.date(), value, n_points) for scope, key, value in grid.means(cells)]
    log.info("  %s: %d days, %d fields (%.0f s)", variable, len(days.unique()), len(keep), time.time() - t0)
    return pd.DataFrame(out, columns=["scope", "key", "day", "value", "n_points"])


def _read(da, sel: dict) -> np.ndarray:
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            return np.asarray(da.isel(sel).values)
        except Exception as e:      # a dropped connection mid-chunk; an auth failure will not heal by waiting
            last = e
            if any(k in str(e) for k in ("401", "403", "ermission", "nauthenticated")):
                break
            log.warning("read attempt %d/%d failed: %s", attempt, RETRIES, str(e)[:160])
            time.sleep(BACKOFF_S * attempt)
    raise RuntimeError(_explain(last))


def build_frame(means: pd.DataFrame, init: dt.datetime, loaded: pd.Timestamp, family: str = "tt") -> pd.DataFrame:
    df = means.copy()
    df["provider"], df["model_family"], df["pattern"] = PROVIDER, MODEL_FAMILY, PATTERN
    df["family"], df["curve_name"] = family, FIELDS[family]["variable"]
    df["reference_date"] = pd.Timestamp(init)
    df["init_hour"] = int(init.hour)
    df["lead_day"] = [(d - init.date()).days for d in df["day"]]
    df["normal"] = np.nan
    df["loaded_at"] = loaded
    return df[[c for c, _ in COLUMNS]]


# ─── sandbox reads: the Volue normals, the runs already loaded ──────────────────

def load_normals(writer, schema: str) -> pd.DataFrame:
    """family 'tt': scope, key, day, normal — the Volue temperature normal per Morning Call
    region (morning_daily_volue) and per country (the Volue rows of fcst_daily). Empty on a dry run."""
    cols = ["family", "scope", "key", "day", "normal"]
    query = getattr(writer, "query_df", None)
    if query is None:
        return pd.DataFrame(columns=cols)
    regions = ",".join(f"'{r}'" for r in REGION_GROUPS)
    parts = []
    for scope, sql in (
        ("region", f"SELECT region AS key, day, MAX(normal) AS normal FROM {schema}.morning_daily_volue "
                   f"WHERE family = 'tt' AND region IN ({regions}) AND normal IS NOT NULL "
                   f"AND day >= current_date() - INTERVAL 3 DAYS GROUP BY region, day"),
        ("country", f"SELECT area AS key, day, MAX(normal) AS normal FROM {schema}.fcst_daily "
                    f"WHERE provider = 'Volue' AND metric = 'Temperature' AND normal IS NOT NULL "
                    f"AND day >= current_date() - INTERVAL 3 DAYS GROUP BY area, day"),
    ):
        try:
            d = query(sql)
        except Exception as e:
            log.warning("%s normals not read (%s) — those rows carry no normal", scope, str(e)[:160])
            continue
        if not d.empty:
            d["scope"] = scope
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=cols)
    n = pd.concat(parts, ignore_index=True)
    n["family"] = "tt"
    n["day"] = pd.to_datetime(n["day"]).dt.date
    n["normal"] = pd.to_numeric(n["normal"], errors="coerce")
    return n[cols]


def runs_in_table(writer, table: str, families: list[str] = FAMILIES) -> set[pd.Timestamp]:
    """The runs the table holds with every family — a run loaded before a family was added is redone."""
    query = getattr(writer, "query_df", None)
    if query is None:
        return set()
    fam = ",".join(f"'{f}'" for f in families)
    try:
        d = query(f"SELECT reference_date FROM {table} WHERE pattern = '{PATTERN}' AND family IN ({fam}) "
                  f"GROUP BY reference_date HAVING COUNT(DISTINCT family) >= {len(families)}")
    except Exception as e:
        log.info("%s: cannot list its runs (%s) — treating as empty", table, str(e)[:120])
        return set()
    out = set()
    for v in d.iloc[:, 0]:
        t = pd.Timestamp(v)
        out.add(t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t)
    return out


# ─── the step ───────────────────────────────────────────────────────────────────

def run(settings, writer, backfill: bool = False, now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.utcnow()
    table = f"{settings.schema}.{TABLE}"
    writer.ensure_table(table, COLUMNS,
                        "Google WeatherNext 3 ensemble means (statistics Zarr store on Google Cloud Storage): daily UTC-day "
                        "means of 2 m temperature (family tt, population-weighted per country and Morning Call region, with the "
                        "Volue normal) and 100 m wind speed (family ws, weighted by installed wind capacity per region); "
                        "reference_date = the init time. Contains WeatherNext data from Google (CC BY 4.0). "
                        "Written by Power_dashboard/pipeline (google_weathernext.py).")
    grids = {f: load_grid(f) for f in FAMILIES}
    cycles = recent_cycles(now, BACKFILL_RUNS if backfill else RUNS_PER_PASS)
    have = set() if backfill else runs_in_table(writer, table)
    todo = [c for c in cycles if pd.Timestamp(c) not in have]
    if not todo:
        log.info("%s already holds %s — nothing to do (--backfill re-writes)", table,
                 ", ".join(f"{c:%d %b %Hz}" for c in cycles))
        return {"runs": [], "rows": 0, "skipped": [f"{c:%Y-%m-%d %Hz}" for c in cycles]}

    try:
        import obstore  # noqa: F401
        import zarr
        if int(zarr.__version__.split(".")[0]) < 3:
            raise ImportError(f"zarr {zarr.__version__} is installed, the store is Zarr v3")
    except ImportError as e:
        raise RuntimeError(f'the google step needs zarr >= 3 and obstore in this environment ({e}): '
                           f'pip install "numpy=={np.__version__}" zarr obstore — the pin keeps pip from '
                           f'upgrading numpy') from e
    try:
        runs = Runs()
    except Exception as e:
        raise RuntimeError(_explain(e)) from e
    loaded = pd.Timestamp.utcnow().tz_localize(None)
    frames, done, waiting = [], [], []
    t0 = time.time()
    for init in sorted(todo):                                   # oldest first
        try:
            name = runs.find(init)
        except Exception as e:
            raise RuntimeError(_explain(e)) from e
        if name is None:
            log.info("%s: not in the bucket yet", f"{init:%Y-%m-%d %Hz}")
            waiting.append(f"{init:%Y-%m-%d %Hz}")
            continue
        log.info("%s: reading %s every %d h from %s", f"{init:%Y-%m-%d %Hz}",
                 ", ".join(FIELDS[f]["variable"] for f in FAMILIES), STEP_HOURS, name)
        try:
            ds = open_run(name)
            parts = [build_frame(daily_means(ds, init, grids[f], f), init, loaded, f) for f in FAMILIES]
        except RunIncomplete as e:                              # a run is written whole or not at all
            log.info("%s: %s", f"{init:%Y-%m-%d %Hz}", e)
            waiting.append(f"{init:%Y-%m-%d %Hz}")
            continue
        except (KeyError, TypeError, ValueError):
            raise                                               # the store is not what the docs describe
        except Exception as e:
            raise RuntimeError(_explain(e)) from e
        frames += parts
        done.append(f"{init:%Y-%m-%d %Hz}")
    if not frames:
        return {"runs": [], "rows": 0, "waiting": waiting}

    df = pd.concat(frames, ignore_index=True)
    normals = load_normals(writer, settings.schema)
    if not normals.empty:
        df = df.drop(columns=["normal"]).merge(normals, on=["family", "scope", "key", "day"], how="left")[[c for c, _ in COLUMNS]]
    n = writer.merge(table, df, KEYS)
    writer.prune(table, f"reference_date < current_timestamp() - INTERVAL {RETENTION_DAYS} DAYS")
    full = df[df["n_points"] >= 0.9 * N_POINTS_FULL]
    summary = {"runs": done, "rows": n, "families": df.groupby("family").size().to_dict(),
               "days": int(full["day"].nunique()), "with_normal": int(df["normal"].notna().sum()),
               "waiting": waiting, "seconds": round(time.time() - t0)}
    log.info("google weathernext done: %s", summary)
    return summary
