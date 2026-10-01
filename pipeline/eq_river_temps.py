"""River temperatures and flows from Energy Quantified → {schema}.river_temp_eq,
{schema}.river_flow_eq (+ river_stations_eq).

Same client and key as Lorenzo_Trainee/EQ_project/eq_fundamentals.py. Two curve
families, discovered from EQ's metadata so a new station or area appears by itself:

  River Temperature   "<AREA> @<River>-<Site> River Temperature °C H <Backcast|Normal|Forecast|Actual>"
                      FR 8, DE 14, HU 1 stations today (the French ones feed the nuclear
                      fleet's cooling constraints)                     → river_temp_eq
  River Flow          "<AREA> @<River>-<Site> River Flow m^3/s <H Actual | D Normal>"
                      the 8 French stations (Fessenheim without a normal); no backcast,
                      no forecast                                       → river_flow_eq

Per curve type:
  Backcast / Actual   TIMESERIES — daily mean (P1D, AVERAGE, the curve's CET days).
                      Incremental: the last RIVER_BACKCAST_DAYS; a curve not yet in
                      the table, or --backfill, loads from RIVER_HISTORY_START.
  Normal              TIMESERIES — the same window plus RIVER_NORMAL_AHEAD_DAYS ahead,
                      so the forecast can be read against its normal.
  Forecast            INSTANCE — the latest issue per tag (EQ offers 'ec-ens' and
                      'ec-ext'), daily mean, keyed on (issued, tag): every run adds
                      the new issue and keeps the old ones.

Both value tables share one layout (MERGE key: curve_name, data_type, day, issued, tag):
  curve_name, station_key, station, area, data_type ('backcast'|'normal'|'forecast'|'actual'),
  day DATE, value DOUBLE, issued TIMESTAMP (UTC, forecast only), tag STRING ('' when not a forecast),
  unit ('°C' | 'm³/s'), loaded_at
river_stations_eq (MERGE key: station_key): station, area, river, site, latitude, longitude,
  data_types ('temperature: backcast,forecast,normal; flow: actual,normal').
"""
from __future__ import annotations

import datetime as dt
import logging
import time
import warnings
from typing import Iterable

import pandas as pd

log = logging.getLogger("pipeline.rivers")

RIVER_COLUMNS = [
    ("curve_name", "STRING"), ("station_key", "STRING"), ("station", "STRING"), ("area", "STRING"),
    ("data_type", "STRING"), ("day", "DATE"), ("value", "DOUBLE"), ("issued", "TIMESTAMP"), ("tag", "STRING"),
    ("unit", "STRING"), ("loaded_at", "TIMESTAMP"),
]
RIVER_KEYS = ["curve_name", "data_type", "day", "issued", "tag"]
STATION_COLUMNS = [
    ("station_key", "STRING"), ("station", "STRING"), ("area", "STRING"), ("river", "STRING"), ("site", "STRING"),
    ("latitude", "DOUBLE"), ("longitude", "DOUBLE"), ("data_types", "STRING"), ("loaded_at", "TIMESTAMP"),
]
STATION_KEYS = ["station_key"]
WANTED_TYPES = {"BACKCAST": "backcast", "NORMAL": "normal", "FORECAST": "forecast", "ACTUAL": "actual"}
# variable -> (metadata search text, substring every curve name must contain, unit stored,
#              Settings attribute of the fully qualified target table)
RIVER_VARIABLES = {
    "temperature": {"q": "River Temperature", "match": "River Temperature", "unit": "°C", "table_attr": "river_table_fq",
                    "comment": "Energy Quantified river temperatures (°C, daily mean): backcast / normal / actual "
                               "timeseries and the latest forecast issues per tag. Written by Power_dashboard/pipeline."},
    "flow":        {"q": "River Flow", "match": "River Flow", "unit": "m³/s", "table_attr": "river_flow_table_fq",
                    "comment": "Energy Quantified river flows (m³/s, daily mean of the hourly actual) and the daily "
                               "normal. No backcast or forecast exists. Written by Power_dashboard/pipeline."},
}
RETRIES, BACKOFF_S = 3, 5


def _retry(fn, what: str):
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except Exception as e:  # network / 5xx / rate limit
            if attempt == RETRIES:
                raise
            log.warning("%s failed (%s) — retry %d/%d in %ds", what, str(e)[:120], attempt, RETRIES, BACKOFF_S * attempt)
            time.sleep(BACKOFF_S * attempt)


def eq_session(api_key: str, ssl_verify: bool):
    if not api_key:
        raise RuntimeError("No EQ_API_KEY — set it in pipeline/.env or keep eq_fundamentals.py reachable")
    from energyquantified import EnergyQuantified
    if not ssl_verify:
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    return EnergyQuantified(api_key=api_key, ssl_verify=ssl_verify)


def _station_key(curve) -> str:
    return curve.place.key if curve.place else curve.name.split(" River ")[0]


# ─── discovery ──────────────────────────────────────────────────────────────────

def discover_curves(eq, areas: Iterable[str], stations: Iterable[str] = (), q: str = "River Temperature",
                    match: str = "River Temperature") -> list:
    """Every curve of one river family EQ has (backcast, normal, forecast, actual) —
    for the given areas, or for every area when `areas` is empty — optionally
    restricted to stations whose place key or name contains one of the fragments."""
    frags = [s.lower() for s in stations]
    areas = [a for a in areas if a]
    found: dict[str, object] = {}
    for area in (areas or [None]):
        kw = {"area": area} if area else {}
        page = _retry(lambda: eq.metadata.curves(q=q, page_size=50, **kw), f"metadata {q} {area or 'all'}")
        curves = list(page)
        while page.has_next_page():
            page = page.get_next_page()
            curves += list(page)
        for c in curves:
            if match not in c.name or c.data_type is None or c.data_type.name not in WANTED_TYPES:
                continue
            if area and c.area is not None and c.area.tag != area:
                continue
            if frags:
                hay = (c.name + " " + (c.place.key if c.place else "")).lower()
                if not any(f in hay for f in frags):
                    continue
            found[c.name] = c
    out = sorted(found.values(), key=lambda c: c.name)
    log.info("EQ: %d '%s' curves in %s (%d stations)", len(out), match, ",".join(areas) or "all areas",
             len({_station_key(c) for c in out}))
    return out


def station_frame(curves_by_variable: dict[str, list]) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    now = pd.Timestamp.utcnow().tz_localize(None)
    for variable, curves in curves_by_variable.items():
        for c in curves:
            key = _station_key(c)
            name = c.place.name if c.place else key
            river, _, site = name.partition("-")
            loc = (c.place.location if c.place else None) or [None, None]
            r = rows.setdefault(key, {"station_key": key, "station": name, "area": c.area.tag if c.area else None,
                                      "river": river, "site": site, "latitude": loc[0], "longitude": loc[1],
                                      "data_types": {}, "loaded_at": now})
            r["data_types"].setdefault(variable, set()).add(WANTED_TYPES[c.data_type.name])
    for r in rows.values():
        r["data_types"] = "; ".join(f"{v}: {','.join(sorted(t))}" for v, t in sorted(r["data_types"].items()))
    return pd.DataFrame(list(rows.values()))


# ─── loading ────────────────────────────────────────────────────────────────────

def _series_rows(ts, curve, data_type: str, unit: str, issued=None, tag: str = "") -> list[dict]:
    """A Timeseries (daily) → rows. EQ returns the curve's CET days; the date is the day.
    Only the first (value) column is used, so the header shape — which differs
    between energyquantified 0.15 (snow_obs) and 0.17 — does not matter."""
    df = ts.to_pandas_dataframe()
    if df.empty:
        return []
    s = df.iloc[:, 0]
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s.dropna()
    now = pd.Timestamp.utcnow().tz_localize(None)
    key = _station_key(curve)
    name = curve.place.name if curve.place else key
    if issued is not None:
        issued = pd.Timestamp(issued)
        issued = issued.tz_convert("UTC").tz_localize(None) if issued.tzinfo is not None else issued
    return [{"curve_name": curve.name, "station_key": key, "station": name,
             "area": curve.area.tag if curve.area else None, "data_type": data_type,
             "day": d.date(), "value": float(v), "issued": issued, "tag": tag or "",
             "unit": unit, "loaded_at": now} for d, v in s.items()]


def load_timeseries(eq, curve, data_type: str, unit: str, begin: dt.date, end: dt.date) -> list[dict]:
    from energyquantified.metadata import Aggregation
    from energyquantified.time import Frequency
    if end <= begin:
        return []
    ts = _retry(lambda: eq.timeseries.load(curve, begin=begin, end=end, frequency=Frequency.P1D,
                                           aggregation=Aggregation.AVERAGE), f"timeseries {curve.name}")
    rows = _series_rows(ts, curve, data_type, unit)
    log.info("  %-58s %s %s→%s: %d days", curve.name, data_type, begin, end, len(rows))
    return rows


def load_forecasts(eq, curve, unit: str, tags: list[str], runs: int = 1) -> list[dict]:
    from energyquantified.metadata import Aggregation
    from energyquantified.time import Frequency
    available = _retry(lambda: eq.instances.tags(curve), f"tags {curve.name}")
    available = sorted(available) if available else []
    use = [t for t in tags if t in available] if tags else available
    if tags and not use:
        log.warning("  %s: none of the wanted tags %s exist (has %s)", curve.name, tags, available)
    rows: list[dict] = []
    if not use:  # a curve without tags: just the latest issue
        ts = _retry(lambda: eq.instances.latest(curve, frequency=Frequency.P1D, aggregation=Aggregation.AVERAGE),
                    f"latest {curve.name}")
        if ts is not None:
            rows += _series_rows(ts, curve, "forecast", unit, ts.instance.issued, ts.instance.tag or "")
    for tag in use:
        if runs <= 1:
            ts = _retry(lambda: eq.instances.latest(curve, tags=[tag], frequency=Frequency.P1D,
                                                    aggregation=Aggregation.AVERAGE), f"latest {curve.name} {tag}")
            series = [ts] if ts is not None else []
        else:
            series = list(_retry(lambda: eq.instances.load(curve, tags=[tag], limit=runs, frequency=Frequency.P1D,
                                                           aggregation=Aggregation.AVERAGE), f"instances {curve.name} {tag}"))
        for ts in series:
            rows += _series_rows(ts, curve, "forecast", unit, ts.instance.issued, ts.instance.tag or tag)
    issues = sorted({(r["issued"], r["tag"]) for r in rows})
    log.info("  %-58s forecast: %d rows, issues %s", curve.name, len(rows),
             ", ".join(f"{t} {i:%d %b %Hz}" for i, t in issues) if issues else "none")
    return rows


# ─── orchestration ──────────────────────────────────────────────────────────────

def _run_variable(eq, settings, writer, variable: str, cfg: dict, backfill: bool, today: dt.date) -> tuple[list, int, dict]:
    """Discover and MERGE one river family. Returns (curves, rows written, rows by type)."""
    curves = discover_curves(eq, settings.river_areas, settings.river_stations, q=cfg["q"], match=cfg["match"])
    if not curves:
        log.warning("no '%s' curves found for %s", cfg["match"], settings.river_areas or "all areas")
        return [], 0, {}
    table = getattr(settings, cfg["table_attr"])
    writer.ensure_table(table, RIVER_COLUMNS, cfg["comment"])

    # which timeseries curves already have data, and up to when
    have = writer.table_max(table, "day", group_by="curve_name", where="data_type IN ('backcast','normal','actual')")
    last_day: dict[str, dt.date] = {}
    if have is not None and not have.empty:
        for name, mx in zip(have.iloc[:, 0], have.iloc[:, 1]):
            if mx is not None and not pd.isna(mx):
                last_day[str(name)] = pd.Timestamp(mx).date()

    rows: list[dict] = []
    end_hist = today + dt.timedelta(days=1)
    for c in curves:
        kind = WANTED_TYPES[c.data_type.name]
        try:
            if kind == "forecast":
                rows += load_forecasts(eq, c, cfg["unit"], settings.river_forecast_tags, settings.river_forecast_runs)
                continue
            first_time = c.name not in last_day
            begin = settings.river_history_start if (backfill or first_time) \
                else today - dt.timedelta(days=settings.river_backcast_days)
            end = end_hist if kind != "normal" else today + dt.timedelta(days=settings.river_normal_ahead_days)
            rows += load_timeseries(eq, c, kind, cfg["unit"], begin, end)
        except Exception as e:
            log.error("  %s: FAILED — %s", c.name, str(e)[:200])

    df = pd.DataFrame(rows, columns=[c for c, _ in RIVER_COLUMNS])
    n = writer.merge(table, df, RIVER_KEYS)
    by_type = df.groupby("data_type").size().to_dict() if not df.empty else {}
    log.info("%s done: %d curves, %d rows %s", variable, len(curves), n, by_type)
    return curves, n, by_type


def run(settings, writer, backfill: bool = False, today: dt.date | None = None) -> dict:
    """Discover, load and MERGE every river family. Returns a small summary dict."""
    today = today or dt.date.today()
    eq = eq_session(settings.eq_api_key, settings.eq_ssl_verify)
    writer.ensure_table(settings.river_station_table_fq, STATION_COLUMNS,
                        "River stations (EQ places) with coordinates and the variables / curve types each has. "
                        "Written by Power_dashboard/pipeline.")
    summary: dict = {}
    curves_by_variable: dict[str, list] = {}
    for variable, cfg in RIVER_VARIABLES.items():
        try:
            curves, n, by_type = _run_variable(eq, settings, writer, variable, cfg, backfill, today)
        except Exception as e:
            log.exception("%s failed", variable)
            summary[variable] = f"FAILED: {str(e)[:160]}"
            continue
        curves_by_variable[variable] = curves
        summary[variable] = {"curves": len(curves), "rows": n, "by_type": by_type}
    if curves_by_variable:
        summary["stations"] = writer.merge(settings.river_station_table_fq, station_frame(curves_by_variable), STATION_KEYS)
    log.info("rivers done: %s", summary)
    if any(isinstance(v, str) and v.startswith("FAILED") for v in summary.values()):
        raise RuntimeError(f"a river family failed: {summary}")
    return summary
