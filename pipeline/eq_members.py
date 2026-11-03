"""Per-member country values of the AI ensemble (AIFS-ENS) from Energy Quantified → {schema}.fcst_members_eq.

The Forecast → Scenarios tab clusters the members of an ensemble on their Z500 field
and then reads, per scenario, temperature, wind and solar off the same member IDs.
For EC-ENS those land variables are the Volue members the notebook already writes
(fcst_members, provider 'Volue'). Volue has no AIFS-ENS members; EQ's ensembles
endpoint has them — every country curve EQ carries, as 50 scenarios per instance —
and this job writes them in the layout of fcst_members so the app reads both the same
way (provider 'EQ', model = the EQ tag).

Curves, per area (the app's country codes; EQ calls the UK 'GB'):
  Temperature  {a} Consumption Temperature °C 15min Forecast     daily mean, °C
  Wind         {a} Wind Power Production MWh/h 15min Forecast     daily mean, MWh/h × 0.001 → GW (as fcst_members)
  Solar        {a} Solar Photovoltaic Production MWh/h 15min Forecast             idem
with the EQ normal of each ('… 15min Normal') for the anomaly, as eq_morning.py.

Loaded with eq.instances.load(curve, tags=[model], ensembles=True, limit=1, P1D
AVERAGE, threshold_pct=10): the latest issue, its mean plus the scenarios e00…eNN.
Member naming — the scenarios are renamed to the silver tables' 'm<k>' so a cluster
found on the Meteomatics Z500 members is read on these rows through the app's
member-id normalisation: for aifs-ens EQ serves 50 scenarios (e00…e49) and the
silver AIFS Z500 has m1…m50, so e<i> → m<i+1>; for ec-ens EQ serves 51 (e00 = the
control run, e01…e50 the perturbations, as Meteomatics' m1…m50), so e<i> → m<i>.
EQ_MEMBER_OFFSET below holds that per tag — adjust it if a provider renumbers.

Table fcst_members_eq (MERGE key: provider, model, metric, area, day, member):
  provider 'EQ', model (EQ tag), reference_date (issue, UTC), metric, area, day,
  lead_day, member, value, normal, anomaly, snapshot_ts
Issues older than EQ_MEMBERS_RETENTION_DAYS are pruned; the app reads the latest
reference_date per (model, metric). ≈ 21 areas × 3 metrics × 50 members × 15 days ≈ 47 k
rows per model per run; ~130 EQ calls, a couple of minutes.

    python run_daily.py --only eq-members
    python run_daily.py --only eq-members --dry-run
"""
from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd

log = logging.getLogger("pipeline.eq_members")

MEMBER_COLUMNS = [
    ("provider", "STRING"), ("model", "STRING"), ("reference_date", "TIMESTAMP"), ("metric", "STRING"),
    ("area", "STRING"), ("day", "DATE"), ("lead_day", "INT"), ("member", "STRING"), ("value", "DOUBLE"),
    ("normal", "DOUBLE"), ("anomaly", "DOUBLE"), ("snapshot_ts", "TIMESTAMP"),
]
MEMBER_KEYS = ["provider", "model", "metric", "area", "day", "member"]
FAMILIES = {
    "Temperature": {"curve": "{a} Consumption Temperature °C 15min Forecast",
                    "normal": "{a} Consumption Temperature °C 15min Normal", "scale": 1.0},
    "Wind":        {"curve": "{a} Wind Power Production MWh/h 15min Forecast",
                    "normal": "{a} Wind Power Production MWh/h 15min Normal", "scale": 0.001},
    "Solar":       {"curve": "{a} Solar Photovoltaic Production MWh/h 15min Forecast",
                    "normal": "{a} Solar Photovoltaic Production MWh/h 15min Normal", "scale": 0.001},
}
AREA_TO_EQ = {"UK": "GB"}
EQ_MEMBER_OFFSET = {"aifs-ens": 1, "ec-ens": 0, "gfs-ens": 0}
NORMAL_AHEAD_DAYS = 25
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


def scenarios_to_frame(ts, model: str, metric: str, area: str, offset: int) -> pd.DataFrame:
    """One EQ ensemble instance → long rows (day, member, value)."""
    names = list(ts.scenario_names or [])
    rows = []
    for v in ts.data:
        if not v.has_scenarios():
            continue
        day = pd.Timestamp(v.date)
        day = (day.tz_localize(None) if day.tzinfo is not None else day).normalize().date()
        for name, s in zip(names, v.scenarios):
            if s is None:
                continue
            digits = "".join(ch for ch in str(name) if ch.isdigit())
            k = int(digits) + offset if digits else None
            if k is None:
                continue
            rows.append((day, f"m{k}", float(s)))
    if not rows:
        return pd.DataFrame(columns=["day", "member", "value"])
    df = pd.DataFrame(rows, columns=["day", "member", "value"])
    df["model"], df["metric"], df["area"] = model, metric, area
    df["reference_date"] = _naive_utc(ts.instance.issued)
    return df


def load_area(eq, model: str, metric: str, area: str, today: dt.date) -> tuple[pd.DataFrame, pd.DataFrame]:
    from energyquantified.time import Frequency
    from energyquantified.metadata import Aggregation
    cfg = FAMILIES[metric]
    eq_area = AREA_TO_EQ.get(area, area)
    curve = cfg["curve"].format(a=eq_area)
    offset = EQ_MEMBER_OFFSET.get(model, 0)
    try:
        lst = _retry(lambda: eq.instances.load(curve, tags=[model], limit=1, ensembles=True, frequency=Frequency.P1D,
                                               aggregation=Aggregation.AVERAGE, threshold_pct=10), f"ensembles {curve}")
    except Exception as e:
        log.info("  %s [%s]: %s", curve, model, str(e)[:140])
        return pd.DataFrame(), pd.DataFrame()
    frames = [scenarios_to_frame(ts, model, metric, area, offset) for ts in lst]
    fc = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if fc.empty:
        log.info("  %s [%s]: instance without scenarios", curve, model)
        return fc, pd.DataFrame()
    ncurve = cfg["normal"].format(a=eq_area)
    try:
        nts = _retry(lambda: eq.timeseries.load(ncurve, begin=today - dt.timedelta(days=2),
                                                end=today + dt.timedelta(days=NORMAL_AHEAD_DAYS),
                                                frequency=Frequency.P1D, aggregation=Aggregation.AVERAGE), f"normal {ncurve}")
        s = nts.to_pandas_dataframe().iloc[:, 0]
        idx = pd.to_datetime(s.index)
        idx = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
        nm = pd.DataFrame({"day": [d.date() for d in idx.normalize()], "normal": s.values.astype(float)}).dropna()
    except Exception as e:
        log.info("  %s: %s", ncurve, str(e)[:120])
        nm = pd.DataFrame(columns=["day", "normal"])
    return fc, nm


def run(settings, writer, backfill: bool = False, today: dt.date | None = None) -> dict:
    from eq_morning import eq_session
    today = today or dt.date.today()
    eq = eq_session(settings.eq_api_key, settings.eq_ssl_verify)
    table = settings.eq_members_table_fq
    writer.ensure_table(table, MEMBER_COLUMNS,
                        "Per-member country values (temperature, wind GW, solar GW) of the ensembles Energy Quantified "
                        "serves with scenarios — the land variables the AIFS-ENS scenarios are read on. Layout of "
                        "fcst_members. Written by Power_dashboard/pipeline/eq_members.py.")
    frames, t0 = [], time.time()
    for model in settings.eq_members_models:
        for metric, cfg in FAMILIES.items():
            got = 0
            for area in settings.eq_members_areas:
                fc, nm = load_area(eq, model, metric, area, today)
                if fc.empty:
                    continue
                fc = fc.merge(nm, on="day", how="left") if not nm.empty else fc.assign(normal=float("nan"))
                fc["value"] = fc["value"] * cfg["scale"]
                fc["normal"] = fc["normal"] * cfg["scale"]
                frames.append(fc)
                got += 1
            log.info("%s %s: %d of %d areas, %.0f s", model, metric, got, len(settings.eq_members_areas), time.time() - t0)
    if not frames:
        return {"rows": 0}
    df = pd.concat(frames, ignore_index=True)
    df["provider"] = "EQ"
    df["anomaly"] = df["value"] - df["normal"]
    df["lead_day"] = [(d - r.date()).days for d, r in zip(df["day"], df["reference_date"])]
    df["snapshot_ts"] = pd.Timestamp.utcnow().tz_localize(None)
    df = df[[c for c, _ in MEMBER_COLUMNS]]
    n = writer.merge(table, df, MEMBER_KEYS)
    # one run per (model, metric) in the table: rows of an older issue would otherwise linger
    # on the days the new horizon does not reach
    latest = df.groupby(["model", "metric"])["reference_date"].max()
    for (model, metric), ref in latest.items():
        writer.prune(table, f"model = '{model}' AND metric = '{metric}' AND reference_date < TIMESTAMP '{ref:%Y-%m-%d %H:%M:%S}'")
    writer.prune(table, f"reference_date < current_timestamp() - INTERVAL {settings.eq_members_retention_days} DAYS")
    summary = {"rows": n, "models": {m: f"{t:%d %b %Hz}" for (m, _), t in latest.items()},
               "members": int(df["member"].nunique()), "areas": int(df["area"].nunique()), "seconds": round(time.time() - t0)}
    log.info("eq-members done: %s", summary)
    return summary
