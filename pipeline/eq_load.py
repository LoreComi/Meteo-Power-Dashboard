"""Power demand (load) from Energy Quantified → {schema}.power_load_daily and {schema}.power_load_fcst.

The Power Demand section (weather-power-desk-app/_power.py) has two views, and each
reads one of these tables:

  Load vs temperature      every day of actual load since 2015 against the day's
                           temperature, per country — the scatter, the fitted curves
                           (_power_demand_model.py) and the year × month slopes.
  Temperature sensitivity  the curve of each country evaluated on the current
                           temperature forecast: what one degree more or less does to
                           load over the coming days.

power_load_daily (MERGE key: area, day) — one row per country and day:
  load         MW, daily mean of '{a} Consumption MWh/h <res> Actual'. The resolution
               differs per country (15min, 30min, H — LOAD_ACTUAL_RESOLUTIONS, first
               that exists). Where the actual stopped (NL: January 2019) EQ's
               '15min Synthetic' consumption fills the days after it; load_source says
               which ('actual' / 'synthetic').
  temp         °C, daily mean of '{a} Consumption Temperature °C 15min Synthetic' —
               EQ's consumption-weighted temperature, gap-free since 2010, the series
               its temperature forecasts continue.
  load_normal, temp_normal   EQ's normals, loaded LOAD_NORMAL_AHEAD_DAYS past today so
               the forecast days have one (those rows have no load yet).
Daily values are EQ's (P1D AVERAGE, CET days, threshold_pct=10: a day missing more
than 10 % of its points is dropped), and load is never written for today.
The first run for a country — or --backfill — loads from LOAD_HISTORY_START; after
that the last LOAD_REFRESH_DAYS are re-sent every run (TSOs revise).

power_load_fcst (MERGE key: area, model, day) — the latest issue per model, per day:
  temp, load   the ensemble mean of '{a} Consumption Temperature °C 15min Forecast' and
               of EQ's own load forecast '{a} Consumption MWh/h 15min Forecast'
  temp_p10, temp_p90, n_members      the members' temperature range
  mem_slope, mem_se, mem_r2          load regressed on temperature ACROSS THE MEMBERS of
               that day (both curves come with the same scenarios e00…eNN): the MW per
               °C EQ's load model implies — the "reverse-engineered" sensitivity, kept
               as a cross-check next to the fitted curve. Only when both instances are
               the same issue.
Older issues of a model are pruned after the MERGE. ≈ 21 areas × 2 models × 15 days.

    python run_daily.py --only load
    python run_daily.py --only load --backfill
    python run_daily.py --only load --dry-run
"""
from __future__ import annotations

import datetime as dt
import logging
import time

import numpy as np
import pandas as pd

log = logging.getLogger("pipeline.eq_load")

LOAD_COLUMNS = [
    ("area", "STRING"), ("day", "DATE"), ("load", "DOUBLE"), ("temp", "DOUBLE"), ("load_normal", "DOUBLE"),
    ("temp_normal", "DOUBLE"), ("load_source", "STRING"), ("loaded_at", "TIMESTAMP"),
]
LOAD_KEYS = ["area", "day"]
FCST_COLUMNS = [
    ("area", "STRING"), ("model", "STRING"), ("reference_date", "TIMESTAMP"), ("day", "DATE"), ("lead_day", "INT"),
    ("temp", "DOUBLE"), ("load", "DOUBLE"), ("temp_p10", "DOUBLE"), ("temp_p90", "DOUBLE"), ("n_members", "INT"),
    ("mem_slope", "DOUBLE"), ("mem_se", "DOUBLE"), ("mem_r2", "DOUBLE"), ("loaded_at", "TIMESTAMP"),
]
FCST_KEYS = ["area", "model", "day"]

AREA_TO_EQ = {"UK": "GB"}
LOAD_ACTUAL = "{a} Consumption MWh/h {res} Actual"
LOAD_ACTUAL_RESOLUTIONS = ["15min", "30min", "H"]
LOAD_SYNTHETIC = "{a} Consumption MWh/h 15min Synthetic"
LOAD_NORMAL = "{a} Consumption MWh/h 15min Normal"
LOAD_FORECAST = "{a} Consumption MWh/h 15min Forecast"
TEMP_SYNTHETIC = "{a} Consumption Temperature °C 15min Synthetic"
TEMP_NORMAL = "{a} Consumption Temperature °C 15min Normal"
TEMP_FORECAST = "{a} Consumption Temperature °C 15min Forecast"
SYNTHETIC_AFTER_DAYS = 10            # an actual this stale is topped up with the synthetic consumption
MIN_MEMBERS = 10                     # fewer paired members → no member slope
RETRIES, BACKOFF_S = 3, 5


def _retry(fn, what: str):
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == RETRIES or "not found" in str(e).lower():
                raise
            log.warning("%s failed (%s) — retry %d/%d", what, str(e)[:120], attempt, RETRIES)
            time.sleep(BACKOFF_S * attempt)


def _naive_utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def _daily(eq, curve: str, begin: dt.date, end: dt.date) -> pd.Series:
    """One timeseries curve as a daily-mean series indexed by day (naive, CET days)."""
    from energyquantified.time import Frequency
    from energyquantified.metadata import Aggregation
    ts = _retry(lambda: eq.timeseries.load(curve, begin=begin, end=end, frequency=Frequency.P1D,
                                           aggregation=Aggregation.AVERAGE, threshold_pct=10), curve)
    df = ts.to_pandas_dataframe()
    if df.empty:
        return pd.Series(dtype=float)
    s = df.iloc[:, 0]
    idx = pd.to_datetime(s.index)
    idx = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
    return pd.Series(s.values.astype(float), index=idx.normalize()).dropna()


def _try_daily(eq, curve: str, begin: dt.date, end: dt.date) -> pd.Series | None:
    """_daily, or None when EQ has no such curve (anything else is raised)."""
    try:
        return _daily(eq, curve, begin, end)
    except Exception as e:
        if "not found" in str(e).lower():
            return None
        raise


# ─── history ───────────────────────────────────────────────────────────────────

def load_area_history(eq, area: str, begin: dt.date, today: dt.date, normal_ahead_days: int) -> pd.DataFrame:
    """Daily load / temperature / normals of one country from `begin`."""
    a = AREA_TO_EQ.get(area, area)
    end = today + dt.timedelta(days=1)
    yesterday = pd.Timestamp(today) - pd.Timedelta(days=1)
    actual = None
    for res in LOAD_ACTUAL_RESOLUTIONS:
        actual = _try_daily(eq, LOAD_ACTUAL.format(a=a, res=res), begin, end)
        if actual is not None:
            break
    if actual is None:
        log.warning("  %s: no actual consumption curve (%s)", area, ", ".join(LOAD_ACTUAL_RESOLUTIONS))
        actual = pd.Series(dtype=float)
    actual = actual.loc[:yesterday]                       # never today: the day is still filling
    source = pd.Series("actual", index=actual.index)
    if actual.empty or actual.index.max() < yesterday - pd.Timedelta(days=SYNTHETIC_AFTER_DAYS):
        syn = _try_daily(eq, LOAD_SYNTHETIC.format(a=a), begin, end)
        if syn is not None and not syn.empty:
            syn = syn.loc[:yesterday]
            syn = syn[~syn.index.isin(actual.index)]
            log.info("  %s: actual ends %s — %d days from the synthetic consumption", area,
                     "nowhere" if actual.empty else f"{actual.index.max():%Y-%m-%d}", len(syn))
            actual = pd.concat([actual, syn]).sort_index()
            source = pd.concat([source, pd.Series("synthetic", index=syn.index)]).sort_index()

    temp = _try_daily(eq, TEMP_SYNTHETIC.format(a=a), begin, end)
    normal_end = today + dt.timedelta(days=normal_ahead_days)
    load_n = _try_daily(eq, LOAD_NORMAL.format(a=a), begin, normal_end)
    temp_n = _try_daily(eq, TEMP_NORMAL.format(a=a), begin, normal_end)
    df = pd.DataFrame({"load": actual, "temp": temp if temp is not None else pd.Series(dtype=float),
                       "load_normal": load_n if load_n is not None else pd.Series(dtype=float),
                       "temp_normal": temp_n if temp_n is not None else pd.Series(dtype=float)})
    df = df.dropna(how="all")
    if df.empty:
        return df
    df["load_source"] = source.reindex(df.index)
    df.insert(0, "area", area)
    df.insert(1, "day", [d.date() for d in df.index])
    return df.reset_index(drop=True)


# ─── forecast ──────────────────────────────────────────────────────────────────

def _members(ts) -> tuple[pd.Series, pd.DataFrame]:
    """One ensemble instance → (ensemble mean per day, day × scenario frame)."""
    names = list(ts.scenario_names or [])
    mean, rows = {}, {}
    for v in ts.data:
        day = pd.Timestamp(v.date)
        day = (day.tz_localize(None) if day.tzinfo is not None else day).normalize()
        if v.value is not None:
            mean[day] = float(v.value)
        if names and v.has_scenarios():
            rows[day] = [np.nan if s is None else float(s) for s in v.scenarios]
    return pd.Series(mean, dtype=float).sort_index(), pd.DataFrame.from_dict(rows, orient="index", columns=names).sort_index()


def member_slope(temp: np.ndarray, load: np.ndarray) -> tuple[float, float, float]:
    """(MW per °C, its standard error, R²) of load on temperature across one day's members."""
    keep = ~(np.isnan(temp) | np.isnan(load))
    T, y = temp[keep], load[keep]
    n = len(y)
    if n < MIN_MEMBERS or np.std(T) < 1e-6:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(T, y, 1)
    resid = y - (a + b * T)
    sxx, ss_tot = float(np.sum((T - T.mean()) ** 2)), float(np.sum((y - y.mean()) ** 2))
    return float(b), float(np.sqrt(np.sum(resid ** 2) / (n - 2) / sxx)), (float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan)


def load_area_forecast(eq, area: str, model: str) -> pd.DataFrame:
    """The latest issue of `model` for one country: temperature and load per day, the
    members' temperature range and the member-implied sensitivity."""
    from energyquantified.time import Frequency
    from energyquantified.metadata import Aggregation
    a = AREA_TO_EQ.get(area, area)

    def latest(curve):
        lst = _retry(lambda: eq.instances.load(curve, tags=[model], limit=1, ensembles=True, frequency=Frequency.P1D,
                                               aggregation=Aggregation.AVERAGE, threshold_pct=10), f"ensembles {curve}")
        return lst[0] if lst else None

    t_ts = latest(TEMP_FORECAST.format(a=a))
    if t_ts is None:
        return pd.DataFrame()
    t_mean, t_mem = _members(t_ts)
    issued = _naive_utc(t_ts.instance.issued)
    out = pd.DataFrame({"temp": t_mean})
    for c in ("load", "temp_p10", "temp_p90", "n_members", "mem_slope", "mem_se", "mem_r2"):
        out[c] = np.nan
    if not t_mem.empty:
        out["temp_p10"], out["temp_p90"] = t_mem.quantile(0.1, axis=1), t_mem.quantile(0.9, axis=1)
        out["n_members"] = t_mem.notna().sum(axis=1)
    try:
        l_ts = latest(LOAD_FORECAST.format(a=a))
    except Exception as e:
        log.info("  %s [%s] load forecast: %s", area, model, str(e)[:120])
        l_ts = None
    if l_ts is not None and _naive_utc(l_ts.instance.issued) == issued:
        l_mean, l_mem = _members(l_ts)
        out["load"] = l_mean
        common = [c for c in t_mem.columns if c in l_mem.columns]
        for day in out.index.intersection(t_mem.index).intersection(l_mem.index):
            b, se, r2 = member_slope(t_mem.loc[day, common].to_numpy(float), l_mem.loc[day, common].to_numpy(float))
            out.loc[day, ["mem_slope", "mem_se", "mem_r2"]] = [b, se, r2]
    elif l_ts is not None:
        log.info("  %s [%s]: load forecast is the %s issue, temperature the %s — load left out", area, model,
                 f"{_naive_utc(l_ts.instance.issued):%d %b %Hz}", f"{issued:%d %b %Hz}")
    out = out.dropna(subset=["temp"]).reset_index(names="day")
    out["lead_day"] = [(d.date() - issued.date()).days for d in out["day"]]
    out["day"] = [d.date() for d in out["day"]]
    out["area"], out["model"], out["reference_date"] = area, model, issued
    return out


# ─── orchestration ──────────────────────────────────────────────────────────────

def run(settings, writer, backfill: bool = False, today: dt.date | None = None) -> dict:
    from eq_morning import eq_session
    today = today or dt.date.today()
    eq = eq_session(settings.eq_api_key, settings.eq_ssl_verify)
    now = pd.Timestamp.utcnow().tz_localize(None)
    t0 = time.time()

    # history
    table = settings.load_table_fq
    writer.ensure_table(table, LOAD_COLUMNS,
                        "Daily actual power demand (MW) and consumption-weighted temperature per country from Energy "
                        "Quantified, with the EQ normals. Written by Power_dashboard/pipeline/eq_load.py.")
    have = writer.table_max(table, "day", group_by="area", where="load IS NOT NULL")
    known = set() if have is None or have.empty else {str(a) for a, mx in zip(have.iloc[:, 0], have.iloc[:, 1]) if mx is not None}
    frames = []
    for area in settings.load_areas:
        begin = settings.load_history_start if (backfill or area not in known) \
            else today - dt.timedelta(days=settings.load_refresh_days)
        try:
            df = load_area_history(eq, area, begin, today, settings.load_normal_ahead_days)
        except Exception as e:
            log.error("  %s history: FAILED — %s", area, str(e)[:200])
            continue
        if df.empty:
            continue
        got = df["load"].notna()
        log.info("  %s: %d days of load %s → %s (%s)", area, int(got.sum()),
                 df.loc[got, "day"].min() if got.any() else "-", df.loc[got, "day"].max() if got.any() else "-",
                 "from the start" if begin == settings.load_history_start else f"last {settings.load_refresh_days} d")
        frames.append(df)
    n_hist = 0
    if frames:
        hist = pd.concat(frames, ignore_index=True)
        hist["loaded_at"] = now
        n_hist = writer.merge(table, hist[[c for c, _ in LOAD_COLUMNS]], LOAD_KEYS)
    log.info("load history: %d rows, %d areas, %.0f s", n_hist, len(frames), time.time() - t0)

    # forecast
    ftable = settings.load_fcst_table_fq
    writer.ensure_table(ftable, FCST_COLUMNS,
                        "Latest temperature and load forecast per country and model from Energy Quantified (ensemble "
                        "mean, member temperature range, member-implied MW per degC). Written by "
                        "Power_dashboard/pipeline/eq_load.py.")
    frames = []
    for model in settings.load_fcst_models:
        got = 0
        for area in settings.load_areas:
            try:
                df = load_area_forecast(eq, area, model)
            except Exception as e:
                log.info("  %s [%s] forecast: %s", area, model, str(e)[:140])
                continue
            if not df.empty:
                frames.append(df)
                got += 1
        log.info("load forecast %s: %d of %d areas, %.0f s", model, got, len(settings.load_areas), time.time() - t0)
    n_fcst, latest = 0, {}
    if frames:
        fc = pd.concat(frames, ignore_index=True)
        fc["loaded_at"] = now
        fc["n_members"] = pd.to_numeric(fc["n_members"]).round().astype("Int64")
        n_fcst = writer.merge(ftable, fc[[c for c, _ in FCST_COLUMNS]], FCST_KEYS)
        # one issue per model in the table: a shorter new horizon must not show the old run's tail
        latest = fc.groupby("model")["reference_date"].max().to_dict()
        for model, ref in latest.items():
            writer.prune(ftable, f"model = '{model}' AND reference_date < TIMESTAMP '{ref:%Y-%m-%d %H:%M:%S}'")
    summary = {"history_rows": n_hist, "forecast_rows": n_fcst,
               "issues": {m: f"{t:%d %b %Hz}" for m, t in latest.items()}, "seconds": round(time.time() - t0)}
    log.info("load done: %s", summary)
    return summary
