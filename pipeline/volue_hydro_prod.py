"""Hydro production and precipitation energy from Volue (wapi) → {schema}.hydro_prod_daily.

The Production tab of the Hydro section: per Volue hydro area and day, in GWh/day,

  pro  'pro {a} hydro ror mwh/h cet h sa'   run-of-river production, synthetic actual (hourly SUM × 0.001)
       'pro {a} hydro tot mwh/h cet h sa'   total hydro production — reservoir = tot − ror, the split
                                            Hydro_Report/analysis_ror_reservoir_italy.py makes
       '… h n'                              the Volue normal of each
       '… h f'                              Volue's own production forecast: one issue a day at 00:00 CET,
                                            730 days long — the first FORECAST_DAYS are kept
  rre  'rre {a} gwh cet min15 sa' / '… n'   precipitation energy, actual and normal (daily SUM)
       'rre {a} {run} gwh cet min15 f'      the ensemble-mean 'Avg' of PRECIP_PATTERNS per issue day — the
                                            call volue_morning.py makes, here on the hydro areas

Table hydro_prod_daily (MERGE key: area, variable, data_type, pattern, reference_date, day):
  area            Volue area: fr ch at it it-nord cwe de es pt ib np no se fi see
  variable        'ror' | 'tot' | 'rre'
  data_type       'sa' actual | 'n' normal | 'f' forecast
  pattern         '' for sa / n; 'volue' for the production forecast; the run pattern for rre
  reference_date  issue day 00:00 CET stored in UTC (as the silver tables carry it), NULL for sa / n
  day, value (GWh/day), curve_name, loaded_at

Today's actual is a partial day and is dropped. Forecast issues older than RETENTION_DAYS are
pruned; actuals and normals are kept (from HISTORY_START on --backfill, the last ACTUAL_DAYS
otherwise). Credentials as volue_morning.py.
"""
from __future__ import annotations

import datetime as dt
import logging
import os

import pandas as pd

from volue_morning import _credentials, _retry, _days, load_issue

log = logging.getLogger("pipeline.hydro_prod")

COLUMNS = [
    ("area", "STRING"), ("variable", "STRING"), ("data_type", "STRING"), ("pattern", "STRING"),
    ("reference_date", "TIMESTAMP"), ("day", "DATE"), ("value", "DOUBLE"), ("curve_name", "STRING"),
    ("loaded_at", "TIMESTAMP"),
]
KEYS = ["area", "variable", "data_type", "pattern", "reference_date", "day"]


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


TABLE = os.environ.get("HYDRO_PROD_TABLE", "hydro_prod_daily")
AREAS = _env_list("HYDRO_PROD_AREAS", "fr,ch,at,it,it-nord,cwe,de,es,pt,ib,np,no,se,fi,see")
PRECIP_PATTERNS = _env_list("HYDRO_PROD_PRECIP_PATTERNS", "ec00ens,ec12ens,gfs00ens")
ACTUAL_DAYS = int(os.environ.get("HYDRO_PROD_ACTUAL_DAYS", "45"))           # actual days re-read per run
HISTORY_START = dt.date.fromisoformat(os.environ.get("HYDRO_PROD_HISTORY_START", "2023-01-01"))
NORMAL_AHEAD_DAYS = 60
FORECAST_DAYS = int(os.environ.get("HYDRO_PROD_FORECAST_DAYS", "45"))       # of the 730-day Volue issue
LOOKBACK_DAYS = int(os.environ.get("HYDRO_PROD_LOOKBACK_DAYS", "2"))        # issue days re-read per run
BACKFILL_DAYS = 10
RETENTION_DAYS = int(os.environ.get("HYDRO_PROD_RETENTION_DAYS", "21"))

# variable -> curve template ({a} = area, {k} = sa | n | f), scale to GWh/day after the daily SUM
VARIABLES = {
    "ror": ("pro {a} hydro ror mwh/h cet h {k}", 0.001),
    "tot": ("pro {a} hydro tot mwh/h cet h {k}", 0.001),
    "rre": ("rre {a} gwh cet min15 {k}", 1.0),
}
RRE_FORECAST = "rre {a} {run} gwh cet min15 f"


def _series(session, name: str, first: dt.date, last: dt.date, scale: float) -> pd.Series:
    curve = session.get_curve(name=name)
    ts = _retry(lambda: curve.get_data(data_from=pd.Timestamp(first), data_to=pd.Timestamp(last),
                                       function="SUM", frequency="D"), name)
    return _days(ts, False) * scale


def _issue_ref(issue_day: dt.date) -> pd.Timestamp:
    """Issue day 00:00 CET, in UTC — what the silver tables' reference_date holds."""
    return pd.Timestamp(issue_day, tz="CET").tz_convert("UTC").tz_localize(None)


def _frame(area: str, variable: str, data_type: str, pattern: str, ref, s: pd.Series, curve_name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "area": area, "variable": variable, "data_type": data_type, "pattern": pattern,
        "reference_date": pd.to_datetime([ref] * len(s)),       # typed even when all NaT, so concat keeps the dtype
        "day": [x.date() for x in s.index], "value": s.values.astype(float), "curve_name": curve_name,
    })


def load_production_issue(curve, issue_day: dt.date, scale: float) -> pd.Series | None:
    """The first FORECAST_DAYS of one issue of a Volue production forecast (an INSTANCES curve)."""
    try:
        ts = _retry(lambda: curve.get_instance(issue_date=pd.Timestamp(issue_day), function="SUM", frequency="D",
                                               data_to=pd.Timestamp(issue_day + dt.timedelta(days=FORECAST_DAYS))),
                    f"instance {curve.name} {issue_day}")
    except Exception:
        return None             # not issued (yet)
    if ts is None:
        return None
    s = _days(ts, False) * scale
    return s if len(s) else None


def run(settings, writer, backfill: bool = False, today: dt.date | None = None) -> dict:
    import wapi
    today = today or dt.date.today()
    cid, sec = _credentials()
    session = wapi.Session(client_id=cid, client_secret=sec, timeout=60000)
    table = f"{settings.schema}.{TABLE}"
    writer.ensure_table(table, COLUMNS,
                        "Hydro production (run-of-river, total; reservoir = tot - ror), Volue's production forecast and "
                        "precipitation energy (actual, normal, ensemble means) per Volue hydro area, GWh/day. "
                        "Written by Power_dashboard/pipeline (volue_hydro_prod.py).")
    first_actual = HISTORY_START if backfill else today - dt.timedelta(days=ACTUAL_DAYS)
    last_normal = today + dt.timedelta(days=NORMAL_AHEAD_DAYS)
    n_back = BACKFILL_DAYS if backfill else LOOKBACK_DAYS
    issue_days = [today - dt.timedelta(days=i) for i in range(n_back + 1)]
    frames, missing, errors = [], 0, 0
    for area in AREAS:
        for variable, (tmpl, scale) in VARIABLES.items():
            for kind in ("sa", "n"):
                name = tmpl.format(a=area, k=kind)
                try:
                    s = _series(session, name, first_actual, today if kind == "sa" else last_normal, scale)
                except Exception as e:
                    log.warning("  %s: %s", name, str(e)[:120])
                    errors += 1
                    continue
                if kind == "sa":
                    s = s[s.index < pd.Timestamp(today)]        # today is a partial day
                if len(s):
                    frames.append(_frame(area, variable, kind, "", pd.NaT, s, name))
            if variable == "rre":
                for run_name in PRECIP_PATTERNS:
                    name = RRE_FORECAST.format(a=area, run=run_name)
                    try:
                        curve = session.get_curve(name=name)
                    except Exception as e:
                        log.info("  no curve %s: %s", name, str(e)[:80])
                        continue
                    for d in issue_days:
                        s = load_issue(curve, d, "rre")
                        if s is None:
                            missing += 1
                            continue
                        frames.append(_frame(area, "rre", "f", run_name, _issue_ref(d), s, name))
            else:
                name = tmpl.format(a=area, k="f")
                try:
                    curve = session.get_curve(name=name)
                except Exception as e:
                    log.info("  no curve %s: %s", name, str(e)[:80])
                    continue
                for d in issue_days:
                    s = load_production_issue(curve, d, scale)
                    if s is None:
                        missing += 1
                        continue
                    frames.append(_frame(area, variable, "f", "volue", _issue_ref(d), s, name))
        log.info("%s: read", area)
    if not frames:
        return {"rows": 0, "missing_instances": missing, "errors": errors}
    df = pd.concat(frames, ignore_index=True)
    df["loaded_at"] = pd.Timestamp.utcnow().tz_localize(None)
    df = df[[c for c, _ in COLUMNS]]
    n = writer.merge(table, df, KEYS)
    writer.prune(table, f"data_type = 'f' AND reference_date < current_timestamp() - INTERVAL {RETENTION_DAYS} DAYS")
    last_obs = df[(df["data_type"] == "sa") & (df["variable"] == "tot")].groupby("area")["day"].max()
    summary = {"rows": n, "areas": sorted(df["area"].unique()), "missing_instances": missing, "errors": errors,
               "last_actual": {a: f"{d:%d %b}" for a, d in last_obs.items()}}
    log.info("hydro production done: %s", summary)
    return summary
