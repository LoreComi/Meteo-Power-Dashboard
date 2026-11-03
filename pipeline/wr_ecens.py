"""EC-ENS members projected onto the 7 European weather regimes → {schema}.wr_forecast_members_ecens.

The Forecast → Weather Regimes tab reads per run / member / lead day the 7 IWR values
and the assigned regime that notebook cell 7c computes in Spark from the silver
layer's Z500 members — which exist for ecmwf-aifs-ens and ncep-gfs-ens only. The
ECMWF IFS ensemble the original tool (corso_model_wr) runs on has no Z500 members in
Databricks, so this job fetches them from Meteomatics through the DSL gateway (as
mm_z500_members.py does for the scenario grid, and as the tool's own
download_forecast_dsl.py does) on the regime domain, runs the SAME projection locally
in numpy, and writes the ~820 rows per run the tab needs.

The projection (cell 7c, max_proj_members.py). Per member and forecast day, on the
fixed domain 30–90 N, 80 W–40 E at 0.5° (121 × 241):
    anomaly   = Z500 − ERA5 day-of-year mean (wr_gph_clim, built by the notebook; pulled
                once into pipeline/cache/wr_gph_clim.npz, 365 × 121 × 241 float32)
    raw_r     = Σ (anomaly / amplitude[month, day] × pattern_r × cos lat) / (Σ cos lat × n_lon)
    IWR_r     = (raw_r − proj_avg_r) / denom_r
    regime    = argmax IWR if max IWR ≥ 1 + lead × (−0.018774), else 'no'
with the 7 patterns, proj_avg, denom and the 366-entry amplitude from
weather-power-desk-app/wr_patterns.npz — the very file the notebook loaded into
wr_patterns / wr_norm / wr_amplitude. The 365-day day-of-year (29 Feb folds onto 28
Feb) and the leap handling are the notebook's.

Fields: members 0–50 (the control is member 0, as in the tool's 51), P1D at 00 UTC from
the init to init + 15 days (the gateway's range), WR_MEMBERS_PER_CALL members per call
(5 ≈ 115 MB of JSON, 6 s).

Table wr_forecast_members_ecens (MERGE key: model, reference_date, day, member) — the
columns of wr_forecast_members plus nothing: model 'ecmwf-ens', reference_date (the
init, UTC), day, lead_day, member ('m0'…'m50'), iwr_<regime> × 7, max_iwr, threshold,
regime, snapshot_ts. Runs older than WR_RETENTION_DAYS are pruned.

    python run_daily.py --only wr-ecens              the latest run, if not in the table yet
    python run_daily.py --only wr-ecens --backfill   re-write it
    python run_daily.py --only wr-ecens --dry-run    no Databricks (the climatology cache must exist)
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

import mm_z500_members as z5

log = logging.getLogger("pipeline.wr_ecens")

HERE = Path(__file__).resolve().parent
PATTERNS_NPZ = HERE.parent / "weather-power-desk-app" / "wr_patterns.npz"
WR_REGIMES = ["ScTr", "GL", "EuBl", "AR", "AT", "ScBl", "ZO"]
WR_THRESHOLD, WR_LEAD_FACTOR = 1.0, -0.018774217528098960
DOMAIN_LAT, DOMAIN_LON, DOMAIN_RES = (30.0, 90.0), (-80.0, 40.0), 0.5
WR_COLUMNS = ([("model", "STRING"), ("reference_date", "TIMESTAMP"), ("day", "DATE"), ("lead_day", "INT"),
               ("member", "STRING")] + [(f"iwr_{r}", "DOUBLE") for r in WR_REGIMES]
              + [("max_iwr", "DOUBLE"), ("threshold", "DOUBLE"), ("regime", "STRING"), ("snapshot_ts", "TIMESTAMP")])
WR_KEYS = ["model", "reference_date", "day", "member"]


# ─── the fixed inputs ───────────────────────────────────────────────────────────

def load_patterns() -> dict:
    z = np.load(PATTERNS_NPZ)
    pat, lats, lons = z["patterns"].astype(float), z["lats"].astype(float), z["lons"].astype(float)
    assert pat.shape == (7, len(lats), len(lons)), pat.shape
    amp = {(int(m), int(d)): float(v) for m, d, v in zip(z["amp_month"], z["amp_day"], z["amp_value"])}
    coslat = np.cos(np.deg2rad(lats))
    return {"patterns": pat, "lats": lats, "lons": lons, "proj_avg": z["proj_avg"].astype(float),
            "denom": z["denom"].astype(float), "amplitude": amp, "coslat": coslat,
            "weighted": pat * coslat[None, :, None],                 # cos-lat pre-multiplied, as wr_patterns
            "normalizer": float(coslat.sum()) * len(lons)}


def clim_cache_path(settings) -> Path:
    return Path(settings.wr_cache_dir) / "wr_gph_clim.npz"


def ensure_climatology(settings, reader, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """The ERA5 day-of-year mean Z500 on the domain, (365, n_lat, n_lon): from the local
    cache, else pulled from {schema}.wr_gph_clim one latitude at a time (121 queries,
    ~5 min) and cached. `reader` needs query_df (the Databricks writer); a dry run
    without the cache cannot build it."""
    path = clim_cache_path(settings)
    if path.exists():
        z = np.load(path)
        if z["clim"].shape == (365, len(lats), len(lons)) and np.allclose(z["lats"], lats) and np.allclose(z["lons"], lons):
            return z["clim"].astype(float)
        log.warning("climatology cache %s has another grid — rebuilding", path)
    if not hasattr(reader, "query_df"):
        raise RuntimeError(f"no climatology cache at {path} and no Databricks connection to build it "
                           "(run once without --dry-run)")
    log.info("building the climatology cache from %s.wr_gph_clim (%d latitudes) …", settings.schema, len(lats))
    clim = np.full((365, len(lats), len(lons)), np.nan, dtype=np.float32)
    lon_idx = {round(float(v), 3): i for i, v in enumerate(lons)}
    t0 = time.time()
    for i, lat in enumerate(lats):
        d = reader.query_df(f"SELECT longitude, doy, gph_normal FROM {settings.schema}.wr_gph_clim "
                            f"WHERE latitude = {float(lat)}")
        if d.empty:
            raise RuntimeError(f"wr_gph_clim has no rows at latitude {lat}")
        li = d["longitude"].astype(float).round(3).map(lon_idx)
        ok = li.notna() & d["doy"].between(1, 365)
        clim[d.loc[ok, "doy"].astype(int).values - 1, i, li[ok].astype(int).values] = d.loc[ok, "gph_normal"].astype(float).values
        if (i + 1) % 20 == 0:
            log.info("  %d/%d latitudes (%.0f s)", i + 1, len(lats), time.time() - t0)
    missing = int(np.isnan(clim).sum())
    if missing:
        raise RuntimeError(f"climatology cache incomplete: {missing} of {clim.size} values missing")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, clim=clim, lats=lats, lons=lons, built=str(dt.datetime.utcnow()))
    log.info("climatology cache written: %s (%.0f s)", path, time.time() - t0)
    return clim.astype(float)


# ─── the fields ─────────────────────────────────────────────────────────────────

def fetch_fields(session, settings, members: list[int], init: dt.datetime, t_end: dt.datetime | None,
                 lats: np.ndarray, lons: np.ndarray) -> tuple[dict[int, np.ndarray], list[dt.date]]:
    """One gateway call: {member: array(n_day, n_lat, n_lon)} and the days, P1D at the
    init hour from the init to init + WR_DAYS (capped at the gateway's last time)."""
    t0 = init
    t1 = init + dt.timedelta(days=settings.wr_days)
    if t_end is not None and t_end < t1:
        t1 = t_end
    sel = f"member:{members[0]}-{members[-1]}" if len(members) > 1 else f"member:{members[0]}"
    payload = {"parameters": z5.PARAM, "source": settings.wr_model, "ensembleMemberSelection": sel,
               "location": f"{DOMAIN_LAT[1]:g},{DOMAIN_LON[0]:g}_{DOMAIN_LAT[0]:g},{DOMAIN_LON[1]:g}:{DOMAIN_RES:g},{DOMAIN_RES:g}",
               "datetime": f"{z5._iso(t0)}--{z5._iso(t1)}:P1D", "outputFormat": "json"}
    last = None
    for attempt in range(1, z5.RETRIES + 1):
        try:
            r = session.post(settings.dsl_url, json=payload, timeout=z5.TIMEOUT_S)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.content[:200].decode('utf-8', 'replace')}")
            js = r.json()
            break
        except Exception as e:
            last = e
            log.warning("%s attempt %d/%d failed: %s", sel, attempt, z5.RETRIES, str(e)[:160])
            time.sleep(z5.BACKOFF_S * attempt)
    else:
        raise RuntimeError(f"{sel}: {last}")
    data = js.get("data") if isinstance(js, dict) else None
    if isinstance(data, dict):
        data = [data]
    if not data:
        raise RuntimeError(f"{sel}: empty reply")
    lat_idx = {round(float(v), 3): i for i, v in enumerate(lats)}
    lon_idx = {round(float(v), 3): i for i, v in enumerate(lons)}
    days = [(t0 + dt.timedelta(days=k)).date() for k in range((t1 - t0).days + 1)]
    day_idx = {d.isoformat(): i for i, d in enumerate(days)}
    out = {}
    for m, entry in zip(members, data):
        arr = np.full((len(days), len(lats), len(lons)), np.nan)
        for c in entry.get("coordinates", []):
            i, j = lat_idx.get(round(float(c["lat"]), 3)), lon_idx.get(round(float(c["lon"]), 3))
            if i is None or j is None:
                continue
            for d in c.get("dates", []):
                k = day_idx.get(d["date"][:10])
                v = d.get("value")
                if k is not None and v is not None:
                    arr[k, i, j] = float(v)
        if np.isnan(arr).all():
            log.warning("member %d: no values", m)
            continue
        out[m] = arr
    return out, days


# ─── the projection ─────────────────────────────────────────────────────────────

def doy365(d: dt.date) -> int:
    leap = d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0)
    return d.timetuple().tm_yday - (1 if (leap and d.month > 2) else 0)


def project_day(field: np.ndarray, day: dt.date, clim: np.ndarray, pat: dict) -> np.ndarray:
    """The 7 IWR values of one gridded field on one day."""
    anom = (field - clim[doy365(day) - 1]) / pat["amplitude"][(day.month, day.day)]
    raw = np.nansum(pat["weighted"] * anom[None, :, :], axis=(1, 2)) / pat["normalizer"]
    return (raw - pat["proj_avg"]) / pat["denom"]


def classify(iwr: np.ndarray, lead: int) -> tuple[float, float, str]:
    thr = WR_THRESHOLD + lead * WR_LEAD_FACTOR
    k = int(np.argmax(iwr))
    return float(iwr[k]), float(thr), (WR_REGIMES[k] if iwr[k] >= thr else "no")


def build_rows(fields: dict[int, np.ndarray], days: list[dt.date], init: dt.datetime, clim: np.ndarray,
               pat: dict, model: str) -> pd.DataFrame:
    rows = []
    snap = pd.Timestamp.utcnow().tz_localize(None)
    for m, arr in fields.items():
        for k, day in enumerate(days):
            if np.isnan(arr[k]).all():
                continue
            iwr = project_day(arr[k], day, clim, pat)
            mx, thr, reg = classify(iwr, k)
            rows.append({"model": model, "reference_date": pd.Timestamp(init), "day": day, "lead_day": k, "member": f"m{m}",
                         **{f"iwr_{r}": float(v) for r, v in zip(WR_REGIMES, iwr)},
                         "max_iwr": mx, "threshold": thr, "regime": reg, "snapshot_ts": snap})
    return pd.DataFrame(rows, columns=[c for c, _ in WR_COLUMNS])


# ─── orchestration ──────────────────────────────────────────────────────────────

def run(settings, writer, backfill: bool = False, now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.utcnow()
    pat = load_patterns()
    clim = ensure_climatology(settings, writer, pat["lats"], pat["lons"])
    session = z5._session(settings, z5.dsl_headers(settings))
    init = z5.latest_cycle(now, settings.z500_min_age_hours)
    first, last = z5.query_time_range(session, settings, settings.wr_model)
    if last is not None:
        from_range = last - dt.timedelta(days=settings.wr_days)
        if from_range.hour in z5.CYCLES and abs((from_range - init).total_seconds()) <= 36 * 3600:
            init = from_range
        log.info("gateway serves %s to %s; run %s", settings.wr_model, z5._iso(last), z5._iso(init))
    table = settings.wr_ecens_table_fq
    writer.ensure_table(table, WR_COLUMNS,
                        "EC-ENS members projected onto the 7 European weather regimes (the layout of wr_forecast_members), "
                        "Z500 from Meteomatics via the DSL gateway. Written by Power_dashboard/pipeline/wr_ecens.py.")
    have = writer.table_max(table, "reference_date", where=f"model = '{settings.wr_model}'")
    have_ts = pd.Timestamp(have["mx"].iloc[0]) if have is not None and not have.empty and pd.notna(have["mx"].iloc[0]) else None
    if have_ts is not None and have_ts.tzinfo is not None:
        have_ts = have_ts.tz_convert("UTC").tz_localize(None)
    if have_ts is not None and have_ts >= pd.Timestamp(init) and not backfill:
        log.info("%s already holds the %s run — nothing to do", table, z5._iso(init))
        return {"run": z5._iso(init), "rows": 0, "skipped": True}

    fields: dict[int, np.ndarray] = {}
    days: list[dt.date] = []
    wanted = list(range(0 if settings.wr_include_control else 1, settings.z500_members + 1))
    per = max(1, settings.wr_members_per_call)
    t0 = time.time()
    for i in range(0, len(wanted), per):
        group = wanted[i:i + per]
        try:
            got, days = fetch_fields(session, settings, group, init, last, pat["lats"], pat["lons"])
            fields.update(got)
        except Exception as e:
            log.warning("members %d-%d skipped: %s", group[0], group[-1], str(e)[:200])
        log.info("  %d/%d members fetched (%.0f s)", len(fields), len(wanted), time.time() - t0)
    if len(fields) < settings.z500_min_members:
        raise RuntimeError(f"only {len(fields)} of {len(wanted)} members fetched — run not written")
    df = build_rows(fields, days, init, clim, pat, settings.wr_model)
    shares = df[df["lead_day"].between(1, 7)]["regime"].value_counts(normalize=True).round(2).to_dict()
    log.info("run %s: %d members × %d days → %d rows; regime shares days 1–7: %s", z5._iso(init), len(fields), len(days), len(df), shares)
    n = writer.merge(table, df, WR_KEYS)
    writer.prune(table, f"reference_date < current_timestamp() - INTERVAL {settings.wr_retention_days} DAYS")
    return {"run": z5._iso(init), "rows": n, "members": len(fields), "days": len(days), "shares_d1_7": shares,
            "seconds": round(time.time() - t0)}
