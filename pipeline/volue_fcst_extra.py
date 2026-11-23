"""Forecast section — the countries and normals the Volue silver tables lack → {schema}.fcst_extra_daily / fcst_extra_normals.

The notebook (power_desk_refresh.py, cell 2) builds fcst_daily from dna_prod_silver.volue, which
carries only part of Volue's catalogue. Missing there, but in Volue (wapi):

  forecasts  Austria and Switzerland (temperature, wind, solar, precipitation energy), and the
             national precipitation energy of Norway, Sweden, Spain, Italy and Portugal — the
             silver table has only NO1-5 / SE1-4 / IB / IT-NORD
  normals    the national temperature normal of CZ ES PT SK SI HR NO SE DK (silver has none, or
             only the bidding zones), the UK solar normal, and every normal of the series above

Both tables are written in the notebook's own units and daily convention: Volue's raw unit
(°C, MWh/h, GWh per 15 min), the CET-day AVERAGE of the 15-min values — exactly what cell 2
does with AVG(value) before its scale — so the notebook applies the same scale to these rows.

  fcst_extra_daily    per metric / area / pattern / issue / day the ensemble statistics cell 2
                      computes from the members: ens_mean (the 'Avg' tag), p10…p90, spread_std
                      (sample std, as Spark's STDDEV), ens_min, ens_max, n_members. Statistics,
                      not members: EC-Extended alone has 101 members × 46 days per series and issue.
  fcst_extra_normals  per metric / area / day the 30-year normal, today −3 … +50 days.

reference_date is the issue DAY at midnight CET stored in UTC, as Volue's silver tables carry
it (cell 2 joins on fcst_runs with ±3 h). The notebook uses these rows only where the silver
table has nothing for that run / area (or day, for normals), so if Volue adds a country to the
silver layer it takes over by itself.
"""
from __future__ import annotations

import datetime as dt
import logging
import os

import numpy as np
import pandas as pd

from volue_morning import _credentials, _retry

log = logging.getLogger("pipeline.fcst_extra")

DAILY_TABLE = os.environ.get("FCST_EXTRA_DAILY_TABLE", "fcst_extra_daily")
NORMALS_TABLE = os.environ.get("FCST_EXTRA_NORMALS_TABLE", "fcst_extra_normals")
DAILY_COLUMNS = [
    ("provider", "STRING"), ("metric", "STRING"), ("area", "STRING"), ("pattern", "STRING"), ("curve_name", "STRING"),
    ("reference_date", "TIMESTAMP"), ("day", "DATE"), ("ens_mean", "DOUBLE"), ("p10", "DOUBLE"), ("p25", "DOUBLE"),
    ("p50", "DOUBLE"), ("p75", "DOUBLE"), ("p90", "DOUBLE"), ("spread_std", "DOUBLE"), ("ens_min", "DOUBLE"),
    ("ens_max", "DOUBLE"), ("n_members", "INT"), ("loaded_at", "TIMESTAMP"),
]
DAILY_KEYS = ["metric", "area", "pattern", "reference_date", "day"]
NORMALS_COLUMNS = [("metric", "STRING"), ("area", "STRING"), ("day", "DATE"), ("normal", "DOUBLE"),
                   ("curve_name", "STRING"), ("loaded_at", "TIMESTAMP")]
NORMALS_KEYS = ["metric", "area", "day"]

# the notebook's VOLUE_MODELS patterns (EC-ENS 00z / 12z, GFS-ENS 00z, EC-Extended)
PATTERNS = [p.strip() for p in os.environ.get("FCST_EXTRA_PATTERNS", "ec00ens,ec12ens,gfs00ens,ecmonthly").split(",")
            if p.strip()]
LOOKBACK_DAYS = int(os.environ.get("FCST_EXTRA_LOOKBACK_DAYS", "2"))     # issue days re-read per run
BACKFILL_DAYS = int(os.environ.get("FCST_EXTRA_BACKFILL_DAYS", "7"))     # the notebook keeps 6 runs per family
RETENTION_DAYS = int(os.environ.get("FCST_EXTRA_RETENTION_DAYS", "14"))  # fcst_runs looks 14 days back
NORMAL_BACK_DAYS, NORMAL_AHEAD_DAYS = 3, 50

# notebook metric name -> forecast / normal curve templates ({r} = lower-case area, {run} = pattern)
CURVES = {
    "Temperature":          {"fcst": "tt {r} con {run} °c cet min15 f",     "norm": "tt {r} con °c cet min15 n"},
    "Wind":                 {"fcst": "pro {r} wnd {run} mwh/h cet min15 f", "norm": "pro {r} wnd mwh/h cet min15 n"},
    "Solar":                {"fcst": "pro {r} spv {run} mwh/h cet min15 f", "norm": "pro {r} spv mwh/h cet min15 n"},
    # the national rre curves, net as the morning feed's: forecast and normal on the same definition
    "Precipitation energy": {"fcst": "rre {r} {run} gwh cet min15 f",       "norm": "rre {r} gwh cet min15 n"},
}
# forecasts (with their normals) the silver table lacks
FORECASTS: dict[str, list[str]] = {
    "Temperature": ["AT", "CH"],
    "Wind": ["AT", "CH"],
    "Solar": ["AT", "CH"],
    "Precipitation energy": ["NO", "SE", "ES", "IT", "PT", "AT", "CH"],
}
# normals only — the forecast is in silver, its normal is not
NORMALS_ONLY: dict[str, list[str]] = {
    "Temperature": ["CZ", "ES", "PT", "SK", "SI", "HR", "NO", "SE", "DK"],
    "Solar": ["UK"],
}


def _utc_naive(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return (t.tz_convert("UTC") if t.tzinfo is not None else t.tz_localize("CET").tz_convert("UTC")).tz_localize(None)


def _daily(ts) -> pd.Series:
    s = ts.to_pandas().dropna()
    if getattr(s.index, "tz", None) is not None:
        s = s.tz_localize(None)                     # CET wall-clock days, as the notebook's CET_DAY
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s.astype(float)


def _stats(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Per day the statistics cell 2 computes from the members (tags other than 'Avg')."""
    members = pd.DataFrame({t: s for t, s in series.items() if t != "Avg"})
    avg = series.get("Avg")
    if members.empty and avg is None:
        return pd.DataFrame()
    idx = members.index if not members.empty else avg.index
    if avg is not None:
        idx = idx.union(avg.index)
    out = pd.DataFrame(index=idx)
    out["ens_mean"] = avg.reindex(idx) if avg is not None else np.nan
    if not members.empty:
        m = members.reindex(idx)
        for q in (10, 25, 50, 75, 90):
            out[f"p{q}"] = m.quantile(q / 100, axis=1)
        out["spread_std"] = m.std(axis=1, ddof=1)
        out["ens_min"], out["ens_max"] = m.min(axis=1), m.max(axis=1)
        out["n_members"] = m.notna().sum(axis=1).astype(int)
    else:
        for c in ("p10", "p25", "p50", "p75", "p90", "spread_std", "ens_min", "ens_max"):
            out[c] = np.nan
        out["n_members"] = 0
    out["ens_mean"] = out["ens_mean"].fillna(out.get("p50"))
    return out


def load_forecast(session, metric: str, area: str, pattern: str, first: dt.date, last: dt.date,
                  have: frozenset = frozenset()) -> pd.DataFrame:
    """Every issue of one curve between `first` and `last` (issue days), as daily statistics;
    issues in `have` (reference_date, UTC naive) are skipped."""
    name = CURVES[metric]["fcst"].format(r=area.lower(), run=pattern)
    try:
        curve = session.get_curve(name=name)
    except Exception as e:
        log.info("  no curve %s: %s", name, str(e)[:80])
        return pd.DataFrame()
    found = _retry(lambda: curve.search_instances(issue_date_from=pd.Timestamp(first),
                                                  issue_date_to=pd.Timestamp(last + dt.timedelta(days=1)),
                                                  with_data=False), f"instances {name}") or []
    by_issue: dict[str, set] = {}
    for i in found:
        by_issue.setdefault(str(i.issue_date), set()).add(i.tag)
    frames = []
    for issue, tags in sorted(by_issue.items()):
        if _utc_naive(issue) in have:
            continue
        try:
            got = _retry(lambda: curve.get_instance(issue_date=pd.Timestamp(issue), tag=sorted(tags), function="AVERAGE",
                                                    frequency="D"), f"instance {name} {issue}")
        except Exception as e:
            log.warning("  %s %s: %s", name, issue, str(e)[:120])
            continue
        got = got if isinstance(got, list) else [got]
        series = {g.tag: _daily(g) for g in got if g is not None}
        st = _stats({t: s for t, s in series.items() if len(s)})
        if st.empty:
            continue
        st = st.reset_index(names="day")
        st.insert(0, "reference_date", _utc_naive(issue))
        frames.append(st)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "curve_name", name)
    df.insert(0, "pattern", pattern)
    df.insert(0, "area", area)
    df.insert(0, "metric", metric)
    df.insert(0, "provider", "Volue")
    return df


def load_normal(session, metric: str, area: str, first: dt.date, last: dt.date) -> pd.DataFrame:
    name = CURVES[metric]["norm"].format(r=area.lower())
    try:
        curve = session.get_curve(name=name)
        ts = _retry(lambda: curve.get_data(data_from=pd.Timestamp(first), data_to=pd.Timestamp(last),
                                           function="AVERAGE", frequency="D"), f"normal {name}")
    except Exception as e:
        log.warning("  normal %s: %s", name, str(e)[:120])
        return pd.DataFrame()
    s = _daily(ts)
    return pd.DataFrame({"metric": metric, "area": area, "day": s.index, "normal": s.values, "curve_name": name})


def _loaded_issues(writer, table: str, first: dt.date, today: dt.date) -> dict:
    """(metric, area, pattern) -> issues already in the table, so a run fetches only new ones (Volue
    publishes an instance complete); --backfill re-reads everything; a dry run (no query_df) reads all."""
    query = getattr(writer, "query_df", None)
    if query is None:
        return {}
    since = _utc_naive(pd.Timestamp(first).tz_localize("CET"))
    until = _utc_naive(pd.Timestamp(today + dt.timedelta(days=1)).tz_localize("CET"))
    try:
        df = query(f"SELECT DISTINCT metric, area, pattern, reference_date FROM {table} "
                   f"WHERE reference_date >= '{since}' AND reference_date < '{until}'")
    except Exception as e:
        log.info("no loaded issues read (%s) — fetching all", str(e)[:80])
        return {}
    out: dict = {}
    for r in df.itertuples(index=False):
        out.setdefault((r.metric, r.area, r.pattern), set()).add(_utc_naive(r.reference_date))
    return {k: frozenset(v) for k, v in out.items()}


def run(settings, writer, backfill: bool = False, today: dt.date | None = None) -> dict:
    import wapi
    today = today or dt.date.today()
    cid, sec = _credentials()
    session = wapi.Session(client_id=cid, client_secret=sec, timeout=60000)
    t_daily, t_norm = f"{settings.schema}.{DAILY_TABLE}", f"{settings.schema}.{NORMALS_TABLE}"
    writer.ensure_table(t_daily, DAILY_COLUMNS,
                        "Forecast section: Volue ensemble statistics per metric / area / run / day for the countries the "
                        "silver tables lack (AT, CH; national rre of NO SE ES IT PT). Raw Volue units, CET-day average. "
                        "Written by Power_dashboard/pipeline (volue_fcst_extra.py), read by power_desk_refresh.py cell 2.")
    writer.ensure_table(t_norm, NORMALS_COLUMNS,
                        "Forecast section: Volue 30-year normals the silver tables lack, per metric / area / CET day, raw "
                        "units. Written by Power_dashboard/pipeline (volue_fcst_extra.py).")
    first = today - dt.timedelta(days=BACKFILL_DAYS if backfill else LOOKBACK_DAYS)
    loaded = pd.Timestamp.utcnow().tz_localize(None)
    have = {} if backfill else _loaded_issues(writer, t_daily, first, today)

    fc = []
    for metric, areas in FORECASTS.items():
        for area in areas:
            for pattern in PATTERNS:
                d = load_forecast(session, metric, area, pattern, first, today,
                                  have.get((metric, area, pattern), frozenset()))
                if not d.empty:
                    fc.append(d)
        log.info("%s forecasts: %s", metric, ", ".join(areas))
    n_fc = 0
    if fc:
        df = pd.concat(fc, ignore_index=True)
        df["day"] = pd.to_datetime(df["day"]).dt.date
        df["loaded_at"] = loaded
        n_fc = writer.merge(t_daily, df[[c for c, _ in DAILY_COLUMNS]], DAILY_KEYS)
        writer.prune(t_daily, f"reference_date < current_timestamp() - INTERVAL {RETENTION_DAYS} DAYS")

    nm = []
    wanted = {(m, a) for m, areas in FORECASTS.items() for a in areas} | \
             {(m, a) for m, areas in NORMALS_ONLY.items() for a in areas}
    for metric, area in sorted(wanted):
        d = load_normal(session, metric, area, today - dt.timedelta(days=NORMAL_BACK_DAYS),
                        today + dt.timedelta(days=NORMAL_AHEAD_DAYS))
        if not d.empty:
            nm.append(d)
    n_nm = 0
    if nm:
        df = pd.concat(nm, ignore_index=True)
        df["day"] = pd.to_datetime(df["day"]).dt.date
        df["loaded_at"] = loaded
        n_nm = writer.merge(t_norm, df[[c for c, _ in NORMALS_COLUMNS]], NORMALS_KEYS)
        writer.prune(t_norm, f"day < current_date() - INTERVAL {NORMAL_BACK_DAYS + 7} DAYS")

    summary = {"forecast_rows": n_fc, "normal_rows": n_nm,
               "series": sorted({f"{d['metric'].iloc[0][:4]} {d['area'].iloc[0]}" for d in fc}) if fc else [],
               "normals": len(nm), "normals_missing": len(wanted) - len(nm)}
    log.info("fcst extra done: %s", summary)
    return summary
