"""EC-ENS 500 hPa geopotential height MEMBERS from Meteomatics via the DSL gateway → {schema}.fcst_member_spatial_ecens.

Why this runs locally. The Forecast → Scenarios tab clusters the ensemble members on
their weekly-mean Z500 field — the spatial pattern that drives a week's weather — and
reads temperature, wind and solar per scenario off the member IDs afterwards. The
silver layer carries per-member Z500 only for ecmwf-aifs-ens and ncep-gfs-ens (the
notebook's cell 4b reads the AIFS members from there); the standard ecmwf-ens — the 50
perturbations the Volue members are — has no Z500 in Databricks. This job fetches
exactly that field and writes it in the layout of the notebook's fcst_member_spatial,
so the app clusters both ensembles through one code path.

How Meteomatics is called here: NOT api.meteomatics.com but the desk's DSL gateway
(fs-gateway-prod.trd.axusr.net/dsl/v3/meteo), with an OAuth client-credential token
(msal; the client secret in the MDS_SECRET environment variable) — the way
CLIMATOLOGIES/update_climp.py downloads ERA5 and corso_model_wr's
download_forecast_dsl.py downloads the same EC-ENS Z500 members for the weather-regime
tool. One POST per group of members:

  {parameters: 'geopotential_height_500hPa:m', source: 'ecmwf-ens',
   ensembleMemberSelection: 'member:1-10', location: '74,-30_30,40:2,2',
   datetime: '<init>T00:00:00Z--<init+15d>T18:00:00Z:PT6H', outputFormat: 'json'}

on the app's spatial grid (SPATIAL_BBOX / SPATIAL_GRID_RES in _config.py: 2° over
35–72 N, 15 W–35 E = 19 × 26 points; the gateway interpolates, so no coarsening is
done here), for each field in Z500_METRICS — the Z500 the scenarios are clustered on
and the 2 m temperature (t_2m:C) the scenario maps are drawn from. The reply's `data` list holds one entry per member, in the order asked.
The values of a day (PT6H: 00/06/12/18 UTC) are averaged into a daily mean, the UTC
date, like the silver 'p1d' curves. The run is the latest 00z / 12z at least
Z500_MIN_AGE_HOURS old (the gateway serves the latest run; the v1 get_time_range
request is logged as a check). The ensemble mean per grid point and day is computed
here, and `anomaly` = value − ensemble mean, which is what the clustering uses (the
ensemble mean is a constant per feature, so it cannot change the clusters — it only
makes the centroid maps readable as departures from the mean).

Table fcst_member_spatial_ecens (snapshot: INSERT the whole run, then prune older runs):
  provider 'Meteomatics', model, reference_date (the INIT time, UTC — unlike the silver
  tables' created_at), metric 'Z500', member ('m1'…'m50', the silver naming), day,
  lead_day, latitude, longitude, value (m), ens_mean, anomaly, snapshot_ts

≈ 50 × 16 × 828 ≈ 660 k rows per run; five gateway calls of ~20 MB JSON each.

    python run_daily.py --only z500              fetch the latest run if it is not in the table yet
    python run_daily.py --only z500 --backfill   fetch it even if it is (re-write)
    python run_daily.py --only z500 --dry-run    no Databricks; the frame goes to pipeline/out/
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import time
import warnings

import pandas as pd

log = logging.getLogger("pipeline.z500")

PARAM = "geopotential_height_500hPa:m"
# The fields written per member: the Z500 the scenarios are clustered on, and the 2 m
# temperature the scenario maps are drawn from (6-hourly instantaneous, averaged into
# the UTC day like the Z500). The table's `metric` column tells them apart.
FIELDS = {"Z500": "geopotential_height_500hPa:m", "T2M": "t_2m:C"}
Z500_COLUMNS = [
    ("provider", "STRING"), ("model", "STRING"), ("reference_date", "TIMESTAMP"), ("metric", "STRING"),
    ("member", "STRING"), ("day", "DATE"), ("lead_day", "INT"), ("latitude", "DOUBLE"), ("longitude", "DOUBLE"),
    ("value", "DOUBLE"), ("ens_mean", "DOUBLE"), ("anomaly", "DOUBLE"), ("snapshot_ts", "TIMESTAMP"),
]
CYCLES = (0, 12)
RETRIES, BACKOFF_S, TIMEOUT_S = 3, 15, 600
DSL_TENANT = "8619c67c-945a-48ae-8e77-35b1b71c9b98"
DSL_SCOPE = "api://e504750d-ac05-4807-a2e7-8fcd77b247aa/.default"


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def latest_cycle(now: dt.datetime, min_age_hours: int, cycles=CYCLES) -> dt.datetime:
    """The latest 00z / 12z init at least min_age_hours before now (UTC, naive)."""
    cutoff = now - dt.timedelta(hours=min_age_hours)
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    cands = [day0 + dt.timedelta(days=d, hours=h) for d in (-2, -1, 0) for h in cycles]
    return max(c for c in cands if c <= cutoff)


# ─── DSL gateway ───────────────────────────────────────────────────────────────

def dsl_headers(settings) -> dict:
    """OAuth client-credential token for the DSL API (as update_climp.dsl_authentication)."""
    from msal import ConfidentialClientApplication
    secret = os.environ.get(settings.dsl_secret_env, "")
    if not secret:
        raise RuntimeError(f"No DSL client secret: set the {settings.dsl_secret_env} environment variable "
                           "(the one CLIMATOLOGIES/update_climp.py uses) for the account that runs the task")
    app = ConfidentialClientApplication(client_id=settings.dsl_app_id,
                                        authority=f"https://login.microsoftonline.com/{DSL_TENANT}",
                                        client_credential=secret)
    result = app.acquire_token_for_client(scopes=[DSL_SCOPE])
    if "access_token" not in result:
        raise RuntimeError(f"DSL token refused: {result.get('error_description', 'unknown error')}")
    return {"Authorization": f"Bearer {result['access_token']}",
            "User-Agent": os.environ.get("USERNAME", "axpo-user"),
            "Content-Type": "application/json", "Accept-Encoding": "gzip, deflate"}


def _session(settings, headers: dict):
    import requests
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    s = requests.Session()
    s.headers.update(headers)
    s.verify = settings.dsl_ssl_verify
    return s


def query_time_range(session, settings, model: str, param: str = PARAM) -> tuple[dt.datetime | None, dt.datetime | None]:
    """The gateway's v1 get_time_range for the model: (first, last) valid time it
    serves, or (None, None) — logged as a check on the cycle rule."""
    url = f"{settings.dsl_url_v1}/get_time_range?model={model}&parameters={param}"
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
        text = r.content.decode("utf-8").strip()
    except Exception as e:
        log.info("get_time_range not usable (%s)", str(e)[:120])
        return None, None
    stamps = []
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if "T" in part and "-" in part:
            try:
                t = pd.Timestamp(part)
                stamps.append((t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t).to_pydatetime())
            except Exception:
                continue
    return (min(stamps), max(stamps)) if len(stamps) >= 2 else (None, None)


def fetch_members(session, settings, members: list[int], init: dt.datetime,
                  t_end: dt.datetime | None = None, param: str = PARAM) -> dict[int, pd.DataFrame]:
    """One gateway call for a group of members: {member: DataFrame(day, latitude,
    longitude, value)}, the values of each UTC day averaged. The horizon runs from
    the init to init + Z500_DAYS (the EC-ENS range the gateway serves: exactly 15 days,
    so the last day holds its 00 UTC value only), capped at `t_end` when given — a
    request past the served range is refused with a 404."""
    lat0, lat1 = settings.z500_lat
    lon0, lon1 = settings.z500_lon
    res = settings.z500_grid_res
    step = settings.z500_time_step.upper()
    t0 = init                                             # P1D: one value a day at the init hour
    t1 = init + dt.timedelta(days=settings.z500_days)
    if t_end is not None:
        t1 = min(t1, t_end)
    if step == "P1D" and t1.hour != t0.hour:              # keep whole days of the step
        t1 = t1.replace(hour=t0.hour) - (dt.timedelta(days=1) if t1.hour < t0.hour else dt.timedelta(0))
    sel = f"member:{members[0]}-{members[-1]}" if len(members) > 1 else f"member:{members[0]}"
    payload = {
        "parameters": param, "source": settings.z500_model, "ensembleMemberSelection": sel,
        "location": f"{lat1:g},{lon0:g}_{lat0:g},{lon1:g}:{res:g},{res:g}",
        "datetime": f"{_iso(t0)}--{_iso(t1)}:{settings.z500_time_step}", "outputFormat": "json",
    }
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.post(settings.dsl_url, json=payload, timeout=TIMEOUT_S)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.content[:200].decode('utf-8', 'replace')}")
            js = r.json()
            break
        except Exception as e:
            last = e
            log.warning("members %s attempt %d/%d failed: %s", sel, attempt, RETRIES, str(e)[:160])
            time.sleep(BACKOFF_S * attempt)
    else:
        raise RuntimeError(f"members {sel}: {last}")
    data = js.get("data") if isinstance(js, dict) else None
    if isinstance(data, dict):
        data = [data]
    if not data:
        raise RuntimeError(f"members {sel}: empty reply")
    if len(data) != len(members):
        log.warning("members %s: asked %d, the reply holds %d entries — matched in order", sel, len(members), len(data))
    out = {}
    for m, entry in zip(members, data):
        rows = []
        for c in entry.get("coordinates", []):
            lat, lon = float(c["lat"]), float(c["lon"])
            for d in c.get("dates", []):
                v = d.get("value")
                if v is None:
                    continue
                rows.append((d["date"][:10], lat, lon, float(v)))
        if not rows:
            log.warning("member %d: no values", m)
            continue
        df = pd.DataFrame(rows, columns=["day", "latitude", "longitude", "value"])
        df = df.groupby(["day", "latitude", "longitude"], as_index=False)["value"].mean()
        df["day"] = pd.to_datetime(df["day"]).dt.date
        out[m] = df
    return out


def build_frame(members: dict[int, pd.DataFrame], init: dt.datetime, model: str, metric: str = "Z500") -> pd.DataFrame:
    parts = []
    for m, df in members.items():
        parts.append(df.assign(member=f"m{m}"))
    all_ = pd.concat(parts, ignore_index=True)
    ens = all_.groupby(["day", "latitude", "longitude"])["value"].mean().rename("ens_mean").reset_index()
    out = all_.merge(ens, on=["day", "latitude", "longitude"], how="left")
    out["anomaly"] = out["value"] - out["ens_mean"]
    out["provider"] = "Meteomatics"
    out["model"] = model
    out["reference_date"] = pd.Timestamp(init)
    out["metric"] = metric
    out["lead_day"] = [(d - init.date()).days for d in out["day"]]
    out["snapshot_ts"] = pd.Timestamp.utcnow().tz_localize(None)
    return out[[c for c, _ in Z500_COLUMNS]]


def run(settings, writer, backfill: bool = False, now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.utcnow()
    session = _session(settings, dsl_headers(settings))
    init = latest_cycle(now, settings.z500_min_age_hours)
    first, last = query_time_range(session, settings, settings.z500_model)
    if last is not None:
        # The gateway serves the latest run out to init + 15 days exactly (checked
        # 2026-11-03: max_date 18 Nov 00Z at 13 UTC on 3 Nov = the 00z run), so its
        # last valid time names the run better than the clock does.
        from_range = last - dt.timedelta(days=settings.z500_days)
        if from_range.hour in CYCLES and abs((from_range - init).total_seconds()) <= 36 * 3600:
            if from_range != init:
                log.info("gateway's range ends %s → the %s run (the cycle rule said %s)", _iso(last), _iso(from_range), _iso(init))
            init = from_range
        else:
            log.warning("gateway range %s–%s does not end %d days after a 00z/12z cycle — labelling the run %s by the clock",
                        _iso(first), _iso(last), settings.z500_days, _iso(init))
        log.info("gateway serves %s to %s; run %s, horizon capped there", settings.z500_model, _iso(last), _iso(init))
    table = settings.z500_table_fq
    writer.ensure_table(table, Z500_COLUMNS,
                        "EC-ENS 500 hPa geopotential height members on the scenario grid, from Meteomatics via the DSL "
                        "gateway (reference_date = init time). Written by Power_dashboard/pipeline/mm_z500_members.py.")
    have = writer.table_max(table, "reference_date", where=f"model = '{settings.z500_model}'")
    have_ts = pd.Timestamp(have["mx"].iloc[0]) if have is not None and not have.empty and pd.notna(have["mx"].iloc[0]) else None
    if have_ts is not None and have_ts.tzinfo is not None:
        have_ts = have_ts.tz_convert("UTC").tz_localize(None)
    if have_ts is not None and have_ts >= pd.Timestamp(init) and not backfill:
        log.info("%s already holds the %s run — nothing to do (--backfill re-writes it)", table, _iso(init))
        return {"run": _iso(init), "rows": 0, "skipped": True}

    t0 = time.time()
    wanted = list(range(1, settings.z500_members + 1))
    per = max(1, settings.z500_members_per_call)
    frames, n_members = [], {}
    for metric in settings.z500_metrics:
        param = FIELDS.get(metric)
        if not param:
            log.warning("unknown field %s — skipped (known: %s)", metric, ", ".join(FIELDS))
            continue
        members: dict[int, pd.DataFrame] = {}
        for i in range(0, len(wanted), per):
            group = wanted[i:i + per]
            try:
                members.update(fetch_members(session, settings, group, init, t_end=last, param=param))
            except Exception as e:
                log.warning("%s members %d-%d skipped: %s", metric, group[0], group[-1], str(e)[:200])
            log.info("  %s: %d/%d members fetched (%.0f s)", metric, len(members), settings.z500_members, time.time() - t0)
        if metric == "Z500" and len(members) < settings.z500_min_members:
            raise RuntimeError(f"only {len(members)} of {settings.z500_members} Z500 members fetched — run not written")
        if members:
            frames.append(build_frame(members, init, settings.z500_model, metric))
            n_members[metric] = len(members)
    df = pd.concat(frames, ignore_index=True)
    n_pts = df.groupby(["latitude", "longitude"]).ngroups
    log.info("run %s: %s members × %d days × %d grid points = %d rows", _iso(init), n_members,
             df["day"].nunique(), n_pts, len(df))
    if have_ts is not None and have_ts >= pd.Timestamp(init):
        writer.prune(table, f"model = '{settings.z500_model}' AND reference_date = TIMESTAMP '{init:%Y-%m-%d %H:%M:%S}'")
    n = writer.insert(table, df)
    keep_from = init - dt.timedelta(hours=12 * max(0, settings.z500_retention_runs - 1))
    writer.prune(table, f"model = '{settings.z500_model}' AND reference_date < TIMESTAMP '{keep_from:%Y-%m-%d %H:%M:%S}'")
    return {"run": _iso(init), "rows": n, "members": n_members, "grid_points": int(n_pts),
            "days": int(df["day"].nunique()), "seconds": round(time.time() - t0)}
