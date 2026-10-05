"""Morning Call / Gas Demand input from Volue (wapi) → {schema}.morning_daily_volue.

The same calls as Morning_Report/import_00z_add_solar_np_tot.py, made locally and
written to Databricks the way eq_morning.py writes the EQ table:

  forecast  session.get_curve('<tt|pro|rre> <region> [wnd|spv] <run> <unit> cet min15 f')
            .get_instance(issue_date=<issue day 00:00>, tag='Avg', function=AVERAGE|SUM, frequency='D')
            — the ensemble-mean 'Avg' series, daily; the report drops the last day of the
            temperature series (it holds 12 hours only) and so does this
  normal    '<…> cet min15 n'.get_data(function=…, frequency='D')
  curves    tt  'tt {r} con {run} °c cet min15 f'            regions fr de uk it hu np ib be nl
            wnd 'pro {r} wnd {run} mwh/h cet min15 f'        regions de uk fr it see np ib be nl
            spv 'pro {r} spv {run} mwh/h cet min15 f'        regions de fr it see np ib be nl
            rre 'rre {r} {run} gwh cet min15 f' (daily SUM)  regions cwe it-nord np see ib
  runs      ec00ens ec12ens gfs00ens gfs12ens (whichever Volue has issued)

be / nl are there for the Gas Demand section (its LDZ countries), as in rdl_forecast.py.

Table morning_daily_volue (MERGE key: family, region, pattern, reference_date, day), the
shape of morning_daily and morning_daily_eq: provider 'Volue', family, region, pattern,
curve_name, reference_date, day, value, n_points, normal, loaded_at. reference_date is
the issue DAY at midnight CET stored in UTC (22:00 / 23:00 the evening before), exactly
as Volue's silver tables carry it, so the app's volue_init_time gives the same init.
Units are Volue's own (°C, MWh/h, GWh); the app converts.

Credentials: VOLUE_CLIENT_ID / VOLUE_CLIENT_SECRET from the environment or pipeline/.env,
else read from the report script's wapi.Session(...) line, never copied.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("pipeline.volue")

VOLUE_COLUMNS = [
    ("provider", "STRING"), ("family", "STRING"), ("region", "STRING"), ("pattern", "STRING"), ("curve_name", "STRING"),
    ("reference_date", "TIMESTAMP"), ("day", "DATE"), ("value", "DOUBLE"), ("n_points", "INT"), ("normal", "DOUBLE"),
    ("loaded_at", "TIMESTAMP"),
]
VOLUE_KEYS = ["family", "region", "pattern", "reference_date", "day"]
N_POINTS = 96
REPORT_SCRIPT = Path(r"P:\QFA\TonyWeather\Lorenzo_Trainee\Morning_Report\import_00z_add_solar_np_tot.py")
TABLE = os.environ.get("VOLUE_MORNING_TABLE", "morning_daily_volue")
PATTERNS = [p.strip() for p in os.environ.get("VOLUE_MORNING_PATTERNS", "ec00ens,ec12ens,gfs00ens,gfs12ens").split(",") if p.strip()]
LOOKBACK_DAYS = int(os.environ.get("VOLUE_MORNING_LOOKBACK_DAYS", "2"))      # issue days re-read per run
BACKFILL_DAYS = int(os.environ.get("VOLUE_MORNING_BACKFILL_DAYS", "10"))
RETENTION_DAYS = int(os.environ.get("VOLUE_MORNING_RETENTION_DAYS", "21"))
NORMAL_AHEAD_DAYS = 50
RETRIES, BACKOFF_S = 3, 5

# family -> curve template, normal template, daily function, drop the last (partial) day?
FAMILIES = {
    "tt":  {"curve": "tt {r} con {run} °c cet min15 f", "normal": "tt {r} con °c cet min15 n", "fn": "AVERAGE", "drop_last": True,
            "regions": ["fr", "de", "uk", "it", "hu", "np", "ib", "be", "nl"]},
    "wnd": {"curve": "pro {r} wnd {run} mwh/h cet min15 f", "normal": "pro {r} wnd mwh/h cet min15 n", "fn": "AVERAGE", "drop_last": False,
            "regions": ["de", "uk", "fr", "it", "see", "np", "ib", "be", "nl"]},
    "spv": {"curve": "pro {r} spv {run} mwh/h cet min15 f", "normal": "pro {r} spv mwh/h cet min15 n", "fn": "AVERAGE", "drop_last": False,
            "regions": ["de", "fr", "it", "see", "np", "ib", "be", "nl"]},
    "rre": {"curve": "rre {r} {run} gwh cet min15 f", "normal": "rre {r} gwh cet min15 n", "fn": "SUM", "drop_last": False,
            "regions": ["cwe", "it-nord", "np", "see", "ib"]},
}


def _credentials() -> tuple[str, str]:
    cid, sec = os.environ.get("VOLUE_CLIENT_ID", ""), os.environ.get("VOLUE_CLIENT_SECRET", "")
    if cid and sec:
        return cid, sec
    try:
        src = REPORT_SCRIPT.read_text(encoding="utf-8")
        m = re.search(r"wapi\.Session\(\s*client_id\s*=\s*['\"]([^'\"]+)['\"]\s*,\s*client_secret\s*=\s*['\"]([^'\"]+)['\"]", src)
        if m:
            log.info("VOLUE_CLIENT_ID not set — using the credentials of the Morning Report script")
            return m.group(1), m.group(2)
    except OSError:
        pass
    raise RuntimeError("No Volue credentials — set VOLUE_CLIENT_ID / VOLUE_CLIENT_SECRET in pipeline/.env")


def _retry(fn, what: str):
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if attempt == RETRIES or any(k in msg for k in ("not found", "no data", "does not exist", "no instance", "404")):
                raise
            log.warning("%s failed (%s) — retry %d/%d", what, str(e)[:100], attempt, RETRIES)
            time.sleep(BACKOFF_S * attempt)


def _days(ts, drop_last: bool) -> pd.Series:
    s = ts.to_pandas()
    if getattr(s.index, "tz", None) is not None:
        s = s.tz_localize(None)
    s = s.dropna()
    if drop_last and len(s) > 1:
        s = s.iloc[:-1]
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s


def load_normal(session, family: str, region: str, first: dt.date, last: dt.date) -> pd.Series:
    cfg = FAMILIES[family]
    curve = session.get_curve(name=cfg["normal"].format(r=region))
    ts = _retry(lambda: curve.get_data(data_from=pd.Timestamp(first), data_to=pd.Timestamp(last),
                                       function=cfg["fn"], frequency="D"), f"normal {family}/{region}")
    return _days(ts, False)


def load_issue(curve, issue_day: dt.date, family: str) -> pd.Series | None:
    cfg = FAMILIES[family]
    try:
        ts = _retry(lambda: curve.get_instance(issue_date=pd.Timestamp(issue_day), tag="Avg", function=cfg["fn"],
                                               frequency="D"), f"instance {curve.name} {issue_day}")
    except Exception:
        return None              # not issued (yet) or no 'Avg' tag for this run
    if ts is None:
        return None
    s = _days(ts, cfg["drop_last"])
    return s if len(s) else None


def run(settings, writer, backfill: bool = False, today: dt.date | None = None) -> dict:
    import wapi
    today = today or dt.date.today()
    cid, sec = _credentials()
    session = wapi.Session(client_id=cid, client_secret=sec, timeout=60000)
    table = f"{settings.schema}.{TABLE}"
    writer.ensure_table(table, VOLUE_COLUMNS,
                        "Morning Call / Gas Demand input from Volue (wapi): daily 'Avg' ensemble means per family / region / run / "
                        "issue day with the Volue normal. Written by Power_dashboard/pipeline (volue_morning.py).")
    n_back = BACKFILL_DAYS if backfill else LOOKBACK_DAYS
    issue_days = [today - dt.timedelta(days=i) for i in range(n_back + 1)]
    frames, missing = [], 0
    for family, cfg in FAMILIES.items():
        for region in cfg["regions"]:
            try:
                normal = load_normal(session, family, region, today - dt.timedelta(days=n_back + 3),
                                     today + dt.timedelta(days=NORMAL_AHEAD_DAYS))
            except Exception as e:
                log.warning("  normal %s/%s: %s", family, region, str(e)[:120])
                normal = pd.Series(dtype=float)
            for run_name in PATTERNS:
                try:
                    curve = session.get_curve(name=cfg["curve"].format(r=region, run=run_name))
                except Exception as e:
                    log.info("  no curve %s/%s/%s: %s", family, region, run_name, str(e)[:80])
                    continue
                for d in issue_days:
                    s = load_issue(curve, d, family)
                    if s is None:
                        missing += 1
                        continue
                    ref = pd.Timestamp(d, tz="CET").tz_convert("UTC").tz_localize(None)    # issue day 00:00 CET, in UTC
                    frames.append(pd.DataFrame({
                        "provider": "Volue", "family": family, "region": region, "pattern": run_name,
                        "curve_name": cfg["curve"].format(r=region, run=run_name), "reference_date": ref,
                        "day": [x.date() for x in s.index], "value": s.values.astype(float), "n_points": N_POINTS,
                        "normal": normal.reindex(s.index).values}))
        log.info("%s: %d regions read", family, len(cfg["regions"]))
    if not frames:
        return {"rows": 0, "missing_instances": missing}
    df = pd.concat(frames, ignore_index=True)
    df["loaded_at"] = pd.Timestamp.utcnow().tz_localize(None)
    df = df[[c for c, _ in VOLUE_COLUMNS]]
    n = writer.merge(table, df, VOLUE_KEYS)
    writer.prune(table, f"reference_date < current_timestamp() - INTERVAL {RETENTION_DAYS} DAYS")
    latest = df.groupby("pattern")["reference_date"].max()
    summary = {"rows": n, "families": sorted(df["family"].unique()), "missing_instances": missing,
               "latest_issue": {p: f"{t:%d %b}" for p, t in latest.items()}}
    log.info("volue morning done: %s", summary)
    return summary
