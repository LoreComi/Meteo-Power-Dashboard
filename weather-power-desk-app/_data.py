"""Data access layer — Power Desk Weather Dashboard.

Every query hits the pre-aggregated sandbox tables in SBX_SCHEMA written by
power_desk_refresh.py — the app never scans the raw Volue / Meteomatics tables.
Queries run through the Databricks SQL Statement API, authenticating as the
App's service principal (same pattern as the LPG desk app) with a fallback to
the user's forwarded token when the SP has no warehouse access.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

from _config import (
    SBX_SCHEMA, VOLUE_SCHEMA, METRICS, VOLUE_MODELS, HYDRO_COMPONENTS, HYDRO_AREA_CODES, WR_REGIMES,
    MORNING_CURVES, MORNING_REGION_CODES, MORNING_MODELS, MORNING_VOLUE_TABLES,
    GAS_VOLUE_FAMILIES, GAS_VOLUE_PATTERNS, GAS_VOLUE_AREAS, LIVE_HISTORY_DAYS, EXPECTED_HORIZON,
    DELTASHARE_PATTERN_MAP, hist_source, HIST_DEFAULT_SOURCE,
    MEMBER_TABLES, SCENARIO_SPATIAL_SOURCES, SPATIAL_METRIC, WR_MODELS, WR_MEMBERS_TABLE,
    HYDRO_START_YEAR, HYDRO_PROD_TABLE, RESERVOIR_OUTLOOK_TABLE, RESERVOIR_OUTLOOK_RUNS_TABLE,
    MORNING_SOURCES, GAS_SOURCES, enabled_sources,
)

log = logging.getLogger("power_desk.data")

# ─── Connection config ───────────────────────────────────────────────────────────
_raw_host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_HOST = _raw_host if _raw_host.startswith("https://") else f"https://{_raw_host}"
WAREHOUSE_ID = os.environ.get("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH", "").split("/")[-1]

# The big loaders are cached until their table is rewritten instead of on a short
# timer: their cache key carries data_version(table), the table's last write time,
# which load_snapshot_times reads for every table at once — one small query, cached
# SNAPSHOT_CHECK_S, so new data shows up on the first interaction after that much
# time. VERSIONED_TTL is only the safety net. Notebook tables stamp every row with
# snapshot_ts, the pipeline's tables with loaded_at.
VERSIONED_TTL = "12h"
SNAPSHOT_CHECK_S = 300


@st.cache_resource(show_spinner=False)
def _workspace_client():
    """One WorkspaceClient per app process — it caches the service principal's OAuth
    token and refreshes it before expiry. Not cached when it raises (no SP credentials)."""
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


@st.cache_resource(show_spinner=False)
def _http() -> requests.Session:
    """One HTTP session for every Statement API call (submit, poll, chunks): keeps the
    TLS connection to the workspace open instead of a new handshake per request."""
    return requests.Session()


def _headers() -> dict:
    """Auth headers: app service principal first, forwarded user token second."""
    h = {"Content-Type": "application/json"}
    try:
        tok = _workspace_client().config.authenticate()
        if isinstance(tok, dict):
            h.update(tok)
            return h
        if isinstance(tok, str):
            h["Authorization"] = f"Bearer {tok}"
            return h
    except Exception:
        pass
    try:
        user_token = st.context.headers.get("x-forwarded-access-token")
    except Exception:
        user_token = None
    if user_token:
        h["Authorization"] = f"Bearer {user_token}"
        return h
    raise RuntimeError("No Databricks credentials available (SP auth failed, no forwarded user token).")


def run_query(sql: str, params: dict | None = None, wait_timeout: str = "50s",
              poll_timeout: int = 180) -> pd.DataFrame:
    """Execute SQL via the Statement API and return a DataFrame (strings; cast downstream).

    `params` fills the statement's named markers (:name) — every value a caller
    passes in (metric, areas, station keys, …) goes through them rather than into
    the SQL text; table names and integers still do (markers cannot hold an
    identifier). If the query does not finish within *wait_timeout* (max 50 s per
    API rules), it is polled every 5 s for up to *poll_timeout* seconds.
    """
    t0 = time.monotonic()
    http = _http()
    body = {"warehouse_id": WAREHOUSE_ID, "statement": sql, "wait_timeout": wait_timeout,
            "disposition": "INLINE", "format": "JSON_ARRAY"}
    if params:
        body["parameters"] = [{"name": k, "value": None if v is None else str(v)} for k, v in params.items()]
    resp = http.post(f"{DATABRICKS_HOST}/api/2.0/sql/statements/", headers=_headers(), json=body, timeout=90)
    if resp.status_code != 200:
        log.warning("query HTTP %s: %s | %s", resp.status_code, resp.text[:300], _sql_head(sql))
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    state = data.get("status", {}).get("state")
    # --- poll until the statement finishes or we time out ---
    stmt_id = data.get("statement_id")
    elapsed = 0
    while state in ("PENDING", "RUNNING") and elapsed < poll_timeout and stmt_id:
        time.sleep(5)
        elapsed += 5
        poll = http.get(f"{DATABRICKS_HOST}/api/2.0/sql/statements/{stmt_id}", headers=_headers(), timeout=30)
        if poll.status_code != 200:
            raise RuntimeError(f"Poll HTTP {poll.status_code}: {poll.text[:400]}")
        data = poll.json()
        state = data.get("status", {}).get("state")
    if state == "FAILED":
        msg = data["status"]["error"]["message"]
        log.warning("query failed: %s | %s", msg[:300], _sql_head(sql))
        raise RuntimeError(msg)
    if state in ("PENDING", "RUNNING"):
        log.warning("query still running after %ss | %s", elapsed, _sql_head(sql))
        raise RuntimeError(f"Query still running after {elapsed}s — narrow the selection or check the warehouse.")
    cols = [c["name"] for c in data.get("manifest", {}).get("schema", {}).get("columns", [])]
    rows = data.get("result", {}).get("data_array", [])
    df = pd.DataFrame(rows, columns=cols)
    # follow chunked results if any
    next_link = data.get("result", {}).get("next_chunk_internal_link")
    while next_link:
        r = http.get(f"{DATABRICKS_HOST}{next_link}", headers=_headers(), timeout=90)
        r.raise_for_status()
        chunk = r.json()
        df = pd.concat([df, pd.DataFrame(chunk.get("data_array", []), columns=cols)], ignore_index=True)
        next_link = chunk.get("next_chunk_internal_link")
    log.info("query %.1fs, %d rows | %s", time.monotonic() - t0, len(df), _sql_head(sql))
    return df


def _sql_head(sql: str) -> str:
    """The statement on one line, cut short — enough to tell queries apart in the log."""
    return " ".join(sql.split())[:160]


def _in(name: str, values, params: dict) -> str:
    """An IN list as named markers (:name_0, :name_1, …), the values added to `params`."""
    marks = []
    for i, v in enumerate(values):
        params[f"{name}_{i}"] = v
        marks.append(f":{name}_{i}")
    return ", ".join(marks) or "NULL"


def prefetch(*calls) -> None:
    """Run independent loaders at the same time so a cold page waits for the slowest
    query, not the sum of them. Each call is a zero-argument callable (a cached
    loader with its arguments bound); the results land in the loaders' caches, so
    the page then reads them as usual. Failures are left for the page's own call to
    raise and report."""
    if len(calls) < 2:
        for fn in calls:
            _quiet(fn)
        return
    from concurrent.futures import ThreadPoolExecutor
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:
        add_script_run_ctx, ctx = None, None

    def run(fn):
        if add_script_run_ctx is not None and ctx is not None:
            add_script_run_ctx(threading.current_thread(), ctx)
        _quiet(fn)

    with ThreadPoolExecutor(max_workers=min(len(calls), 6)) as pool:
        list(pool.map(run, calls))


def _quiet(fn) -> None:
    try:
        fn()
    except Exception as e:
        log.info("prefetch skipped: %s", str(e)[:200])


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _dt(df: pd.DataFrame, cols, utc: bool = False) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=utc)
            if utc:
                df[c] = df[c].dt.tz_convert(None)
    return df


def _sql_list(values) -> str:
    return ",".join(f"'{v}'" for v in values)


MM_MIN_AVAILABILITY_H = 5     # a cycle cannot be in the silver layer less than this many hours after its init


def snap_to_init_time(ref_dt: pd.Timestamp, init_hours: list[int]) -> pd.Timestamp:
    """The cycle a Meteomatics created_at belongs to: the latest 00z / 12z at
    least MM_MIN_AVAILABILITY_H hours before it. The silver layer is ingested
    twice a day, about 07:45 and 19:45 UTC, and both loads of a day cover the
    same forecast days with different values — the morning one is that day's
    00z, the evening one its 12z (checked 2026-10-05). Snapping to the *nearest*
    cycle put the evening load on the next day's 00z, 12 hours early. Not for
    Volue: see volue_init_time."""
    latest_ok = ref_dt - pd.Timedelta(hours=MM_MIN_AVAILABILITY_H)
    cands = [ref_dt.normalize() + pd.Timedelta(days=d, hours=h) for d in (-1, 0) for h in init_hours]
    cands = [c for c in cands if c <= latest_ok]
    return max(cands)


def volue_init_time(ref_dt: pd.Timestamp, init_hour: int) -> pd.Timestamp:
    """Init time of a Volue run from its reference_date and the pattern's cycle.

    Volue's reference_date is the run's ISSUE DAY at midnight CET, stored in
    UTC — 22:00 in summer, 23:00 in winter, the evening before — and it is the
    same for the 00z and the 12z run of that day. The init is therefore that
    CET calendar day at init_hour UTC: ref 22:00 UTC 20 Sep → 21 Sep 00z for
    ec00ens and 21 Sep 12z for ec12ens. (Snapping to the nearest cycle, or
    adding a day only for 00z, dated every 12z run one day early and made the
    run order disagree with the dates shown.) The refresh notebook ranks runs
    on the same CET day + hour (INIT_DAY in cell 1), so labels and ranks agree.
    """
    day = pd.Timestamp(ref_dt).tz_localize("UTC").tz_convert("CET").normalize().tz_localize(None)
    return day + pd.Timedelta(hours=int(init_hour))


_VOLUE_INIT_HOUR = {"ec00ens": 0, "ec12ens": 12, "gfs00ens": 0, "gfs12ens": 12, "ecmonthly": 0,
                    "ec00": 0, "gfs00": 0}


def init_time_for(pattern: str, ref_dt: pd.Timestamp) -> pd.Timestamp:
    """Init time for any run pattern in the sandbox tables: Volue patterns from the
    issue day + cycle, Meteomatics models from their creation timestamp."""
    if pattern in _VOLUE_INIT_HOUR:
        return volue_init_time(ref_dt, _VOLUE_INIT_HOUR[pattern])
    return snap_to_init_time(ref_dt, [0, 12])


def format_run(init_dt: pd.Timestamp) -> str:
    return init_dt.strftime("%a %d %b") + f" {init_dt.hour:02d}z"


def pick_run(df: pd.DataFrame, pattern: str, target_date: pd.Timestamp
             ) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Rows of `pattern`'s run initialised on target_date; else its latest run before it.

    Works on any frame carrying `pattern` and `init_date` (morning_daily,
    gas_demand_daily). Falling back to the newest earlier run is what keeps the
    page usable before today's cycle has landed — the caller compares the
    returned init date with what it asked for and says so.
    """
    p = df[df["pattern"] == pattern]
    if p.empty:
        return p, None
    exact = p[p["init_date"] == target_date]
    if not exact.empty:
        return exact, target_date
    earlier = p[p["init_date"] < target_date]
    if earlier.empty:
        return earlier, None
    latest = earlier["init_date"].max()
    return p[p["init_date"] == latest], latest


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def load_runs() -> pd.DataFrame:
    return _load_runs(data_version("fcst_daily"))


@st.cache_data(ttl=VERSIONED_TTL, max_entries=4, show_spinner=False)
def _load_runs(version: str) -> pd.DataFrame:
    df = run_query(f"SELECT * FROM {SBX_SCHEMA}.fcst_runs ORDER BY model_family, run_rank")
    if df.empty:
        return df
    df = _dt(df, ["reference_date", "snapshot_ts"], utc=True)
    df = _num(df, ["run_rank", "init_hour"])
    if "init_hour" not in df.columns or df["init_hour"].isna().any():
        # old sandbox schema without init_hour: the pattern name still carries it
        df["init_hour"] = [12 if "12" in str(p) else 0 for p in df["pattern"]]
    if "init_day" in df.columns:
        # the notebook stores the CET issue day it ranked on — use exactly that
        df = _dt(df, ["init_day"])
        df["init_time"] = df["init_day"] + pd.to_timedelta(df["init_hour"].astype(int), unit="h")
    else:
        df["init_time"] = [volue_init_time(r, int(h)) for r, h in zip(df["reference_date"], df["init_hour"])]
    # the runs must read in the order they are ranked; if an older sandbox ranked
    # them differently, the dates decide and the labels are rebuilt
    df = df.sort_values(["model_family", "init_time"], ascending=[True, False]).reset_index(drop=True)
    df["run_rank"] = df.groupby("model_family").cumcount() + 1
    df["run_label"] = ["Latest" if r == 1 else f"Latest -{r - 1}" for r in df["run_rank"]]
    df["run_display"] = [f"{lbl} — {format_run(t)}" for lbl, t in zip(df["run_label"], df["init_time"])]
    return df


def load_fcst_daily(metric: str, areas: tuple[str, ...], model_families: tuple[str, ...],
                    max_run_rank: int = 6) -> pd.DataFrame:
    """Daily ensemble statistics for the selected metric / areas / model families."""
    if not areas or not model_families:
        return pd.DataFrame()
    return _load_fcst_daily(metric, tuple(areas), tuple(model_families), int(max_run_rank), data_version("fcst_daily"))


@st.cache_data(ttl=VERSIONED_TTL, max_entries=64, show_spinner=False)
def _load_fcst_daily(metric: str, areas: tuple[str, ...], model_families: tuple[str, ...], max_run_rank: int,
                     version: str) -> pd.DataFrame:
    p = {"metric": metric}
    df = run_query(f"""
        SELECT provider, model_family, pattern, init_hour, reference_date, run_rank, run_label, metric, area, day,
               lead_day, ens_mean, p10, p25, p50, p75, p90, spread_std, ens_min, ens_max, n_members,
               normal, anomaly, anomaly_pct
        FROM {SBX_SCHEMA}.fcst_daily
        WHERE metric = :metric AND area IN ({_in("area", areas, p)})
          AND model_family IN ({_in("family", model_families, p)}) AND run_rank <= {int(max_run_rank)}
        ORDER BY model_family, run_rank, area, day
    """, p)
    if df.empty:
        return df
    df = _dt(df, ["reference_date"], utc=True)
    df = _dt(df, ["day"])
    return _num(df, ["run_rank", "lead_day", "ens_mean", "p10", "p25", "p50", "p75", "p90", "spread_std",
                     "ens_min", "ens_max", "n_members", "normal", "anomaly", "anomaly_pct"])


@st.cache_data(ttl=3600, max_entries=32, show_spinner=False)
def load_spread_clim(metric: str, areas: tuple[str, ...]) -> pd.DataFrame:
    """Normal ensemble spread per lead day (run_month 0 = all months)."""
    if not areas:
        return pd.DataFrame()
    p = {"metric": metric}
    df = run_query(f"""
        SELECT metric, area, lead_day, run_month, spread_std_mean, spread_std_p25, spread_std_p75,
               spread_p90_p10_mean, n_runs
        FROM {SBX_SCHEMA}.fcst_spread_clim
        WHERE metric = :metric AND area IN ({_in("area", areas, p)})
        ORDER BY area, run_month, lead_day
    """, p)
    return _num(df, ["lead_day", "run_month", "spread_std_mean", "spread_std_p25", "spread_std_p75",
                     "spread_p90_p10_mean", "n_runs"])


def _member_table(provider: str) -> str:
    """fcst_members for the notebook's providers (Volue, Meteomatics); the pipeline's
    table for the providers it writes (EQ → fcst_members_eq)."""
    return f"{SBX_SCHEMA}.{MEMBER_TABLES.get(provider, 'fcst_members')}"


@st.cache_data(ttl=900, show_spinner=False)
def load_member_sources() -> pd.DataFrame:
    """Which (provider, model) member sets exist, with member counts — fcst_members
    plus every pipeline member table (a table not written yet is simply absent)."""
    frames = []
    for table in dict.fromkeys([f"{SBX_SCHEMA}.fcst_members"] + [f"{SBX_SCHEMA}.{t}" for t in MEMBER_TABLES.values()]):
        try:
            df = run_query(f"""
                SELECT provider, model, MAX(reference_date) AS reference_date,
                       COUNT(DISTINCT member) AS n_members, COUNT(DISTINCT area) AS n_areas,
                       MIN(day) AS first_day, MAX(day) AS last_day
                FROM {table}
                WHERE metric = 'Temperature'
                GROUP BY provider, model
            """)
        except Exception:
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = _dt(df, ["reference_date"], utc=True)
    df = _dt(df, ["first_day", "last_day"])
    return _num(df, ["n_members", "n_areas"])


@st.cache_data(ttl=900, max_entries=64, show_spinner=False)
def load_members(provider: str, model: str, metric: str, areas: tuple[str, ...], init=None) -> pd.DataFrame:
    """Per-member daily values (and anomaly) for one member source: the run whose init is
    `init` (± 1 h — the Volue rows are stamped with the init since the notebook keeps
    two runs, the EQ rows with the issue time), else its latest run. Empty when that
    run is not in the table: member IDs only correspond within one run, so the
    Scenarios tab never reads a field's clusters on another run's members."""
    if not areas:
        return pd.DataFrame()
    table = _member_table(provider)
    p = {"provider": provider, "model": model, "metric": metric}
    if init is not None:
        t = pd.Timestamp(init)
        t = t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t
        run_where = (f"reference_date BETWEEN TIMESTAMP '{t - pd.Timedelta(hours=1):%Y-%m-%d %H:%M:%S}' "
                     f"AND TIMESTAMP '{t + pd.Timedelta(hours=1):%Y-%m-%d %H:%M:%S}'")
    else:
        run_where = (f"reference_date = (SELECT MAX(reference_date) FROM {table} "
                     f"WHERE provider = :provider AND model = :model AND metric = :metric)")
    df = run_query(f"""
        SELECT provider, model, reference_date, metric, area, day, lead_day, member, value, normal, anomaly
        FROM {table}
        WHERE provider = :provider AND model = :model AND metric = :metric
          AND area IN ({_in("area", areas, p)})
          AND {run_where}
        ORDER BY area, day, member
    """, p)
    if df.empty:
        return df
    df = _dt(df, ["reference_date"], utc=True)
    df = _dt(df, ["day"])
    return _num(df, ["lead_day", "value", "normal", "anomaly"])


@st.cache_data(ttl=900, show_spinner=False)
def load_spatial_sources() -> pd.DataFrame:
    """The Z500 member fields available for scenario clustering: one row per
    SCENARIO_SPATIAL_SOURCES key whose table holds the field — its latest run,
    member count, grid size and the days it covers (the trading weeks on offer).
    A table that is missing, empty or still carrying another metric is left out."""
    rows = []
    for key, cfg in SCENARIO_SPATIAL_SOURCES.items():
        table = f"{SBX_SCHEMA}.{cfg['table']}"
        try:
            df = run_query(f"""
                SELECT model, MAX(reference_date) AS reference_date, COUNT(DISTINCT member) AS n_members,
                       COUNT(DISTINCT CONCAT(latitude, '_', longitude)) AS n_points,
                       MIN(day) AS first_day, MAX(day) AS last_day
                FROM {table}
                WHERE model = :model AND metric = :metric
                  AND reference_date = (SELECT MAX(reference_date) FROM {table}
                                        WHERE model = :model AND metric = :metric)
                GROUP BY model
            """, {"model": cfg["model"], "metric": SPATIAL_METRIC})
        except Exception:
            continue
        if df.empty or pd.isna(df["reference_date"].iloc[0]):
            continue
        df = _dt(df, ["reference_date"], utc=True)
        df = _dt(df, ["first_day", "last_day"])
        df = _num(df, ["n_members", "n_points"])
        r = df.iloc[0]
        if int(r["n_members"]) < 5:
            continue
        rows.append({"source": key, **{c: r[c] for c in df.columns}})
    return pd.DataFrame(rows)


@st.cache_data(ttl=900, max_entries=32, show_spinner=False)
def load_member_spatial_days(source: str, day_min, day_max, metric: str = SPATIAL_METRIC) -> pd.DataFrame:
    """One member field of one source (Z500 to cluster on, T2M to map) averaged over
    the days of one trading week (one call per week): DataFrame(member, latitude,
    longitude, n_days, value, ens_mean, anomaly) on the run that holds the Z500 field
    — ~50 × 500 rows, aggregated in SQL so the full table (× 16 days) never crosses
    the Statement API's inline-result limit."""
    cfg = SCENARIO_SPATIAL_SOURCES[source]
    table = f"{SBX_SCHEMA}.{cfg['table']}"
    d0, d1 = pd.Timestamp(day_min).date(), pd.Timestamp(day_max).date()
    df = run_query(f"""
        SELECT provider, model, MAX(reference_date) AS reference_date, metric,
               member, latitude, longitude, COUNT(DISTINCT day) AS n_days,
               AVG(anomaly) AS anomaly, AVG(value) AS value, AVG(ens_mean) AS ens_mean
        FROM {table}
        WHERE model = :model AND metric = :metric
          AND reference_date = (SELECT MAX(reference_date) FROM {table}
                                WHERE model = :model AND metric = :run_metric)
          AND day BETWEEN DATE '{d0}' AND DATE '{d1}'
        GROUP BY provider, model, metric, member, latitude, longitude
    """, {"model": cfg["model"], "metric": metric, "run_metric": SPATIAL_METRIC})
    if df.empty:
        return df
    df = _dt(df, ["reference_date"], utc=True)
    return _num(df, ["latitude", "longitude", "n_days", "value", "ens_mean", "anomaly"])


@st.cache_data(ttl=900, max_entries=16, show_spinner=False)
def load_meteologica_members(metric: str, areas: tuple[str, ...]) -> pd.DataFrame:
    """Meteologica members if ingested; empty DataFrame otherwise (never raises)."""
    if not areas:
        return pd.DataFrame()
    p = {"metric": metric}
    try:
        df = run_query(f"""
            SELECT provider, model, reference_date, metric, area, day, lead_day, member, value
            FROM {SBX_SCHEMA}.meteologica_members
            WHERE metric = :metric AND area IN ({_in("area", areas, p)})
        """, p)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = _dt(df, ["reference_date"], utc=True)
    df = _dt(df, ["day"])
    return _num(df, ["lead_day", "value"])


# ══════════════════════════════════════════════════════════════════════════════
# MORNING CALL
# ══════════════════════════════════════════════════════════════════════════════

def load_morning_daily(source: str = "EQ") -> pd.DataFrame:
    """Daily values per run for the Morning Call, from the chosen source's table
    (MORNING_SOURCES): 'EQ' = morning_daily_eq written by Power_dashboard/pipeline
    (every model and cycle, the issue time is the real one); 'Volue' = the
    notebook's morning_daily (00z/12z 'Avg' curves, the issue DAY in reference_date).
    Columns: provider, family (tt/wnd/spv/rre), region, pattern, reference_date,
    day, value, n_points, normal, init_time, init_date. Only the last
    lookback_days of issues are read. Cached until the table is rewritten
    (data_version), not on a timer.
    """
    return _load_morning_daily(source, data_version(MORNING_SOURCES[source]["table"]))


@st.cache_data(ttl=VERSIONED_TTL, max_entries=8, show_spinner=False)
def _load_morning_daily(source: str, version: str) -> pd.DataFrame:
    cfg = MORNING_SOURCES[source]
    p = {"provider": cfg["provider"]} if cfg.get("provider") else {}
    df = run_query(f"""
        SELECT provider, family, region, pattern, reference_date, day,
               value, n_points, normal
        FROM {SBX_SCHEMA}.{cfg['table']}
        WHERE reference_date >= current_timestamp() - INTERVAL {int(cfg.get('lookback_days', 10))} DAYS
          {"AND provider = :provider" if p else ""}
        ORDER BY family, region, pattern, day
    """, p)
    if df.empty:
        return df
    df = _dt(df, ["reference_date"], utc=True)
    df = _dt(df, ["day"])
    df = _num(df, ["value", "n_points", "normal"])
    if source == "EQ":
        # EQ's reference_date is the issue time itself (00/06/12/18 UTC)
        df["init_time"] = df["reference_date"].dt.floor("h")
    else:
        # Volue patterns: CET issue day + cycle; Meteomatics: nearest cycle to created_at
        df["init_time"] = [init_time_for(p, r) for r, p in zip(df["reference_date"], df["pattern"])]
    df["init_date"] = df["init_time"].dt.normalize()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER REGIMES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def load_wr_forecast_members() -> pd.DataFrame:
    """Per run / member / lead day: the 7 IWR values and the assigned regime.

    Small by construction — the refresh notebook does the projection over the
    22 M gridded values in Spark and writes only ~900 rows per run, so the whole
    8-day run history for both models is a few tens of thousands of rows.
    """
    cols = ", ".join(f"iwr_{r}" for r in WR_REGIMES)
    tables = [WR_MEMBERS_TABLE] + sorted({v["table"] for v in WR_MODELS.values() if v.get("table")})
    frames = []
    for t in tables:
        try:
            part = run_query(f"""
                SELECT model, reference_date, day, lead_day, member, {cols},
                       max_iwr, threshold, regime
                FROM {SBX_SCHEMA}.{t}
                ORDER BY model, reference_date, member, lead_day
            """)
        except Exception:
            if t == WR_MEMBERS_TABLE:
                raise
            continue                      # a pipeline table not written yet
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = _dt(df, ["reference_date"], utc=True)
    df = _dt(df, ["day"])
    df = _num(df, ["lead_day", "max_iwr", "threshold"] + [f"iwr_{r}" for r in WR_REGIMES])
    df["init_date"] = df["reference_date"].dt.normalize()
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def load_wr_reanalysis() -> pd.DataFrame:
    """ERA5 classified into regimes, one row per day — the climatology.

    Cached for a day: the notebook only appends one or two rows per refresh, and
    every climatological statistic downstream is an average over decades.
    """
    cols = ", ".join(f"iwr_{r}" for r in WR_REGIMES)
    try:
        df = run_query(f"""
            SELECT day, {cols}, max_iwr, regime
            FROM {SBX_SCHEMA}.wr_reanalysis_daily ORDER BY day
        """)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = _dt(df, ["day"])
    # the notebook's daily append can write a day twice when the ERA5 table has not
    # moved on between refreshes (seen 2026-11-03: three identical rows for 28 Oct)
    df = df.drop_duplicates(subset=["day"], keep="last").reset_index(drop=True)
    return _num(df, ["max_iwr"] + [f"iwr_{r}" for r in WR_REGIMES])


# ══════════════════════════════════════════════════════════════════════════════
# GAS DEMAND
# ══════════════════════════════════════════════════════════════════════════════

def load_gas_demand_daily(source: str = "Volue") -> pd.DataFrame:
    """Daily ens-mean temperature / wind / solar per run, from the chosen source
    (GAS_SOURCES): 'Volue' = the notebook's gas_demand_daily (per country, GW);
    'EQ' = the Morning Call's morning_daily_eq — the pipeline's Energy Quantified
    feed, every model and cycle — with its regions mapped onto the gas areas
    (de → DE … ib → IB) and wind / solar scaled MWh/h → GW, so the two sources
    come out in one layout.

    Columns: provider, family (tt/wnd/spv), pattern, area, reference_date, day,
    value, n_points, normal, init_time (the real cycle for EQ, the CET issue
    day + cycle for Volue), init_date. Cached until the table is rewritten.
    """
    return _load_gas_demand_daily(source, data_version(GAS_SOURCES[source]["table"]))


@st.cache_data(ttl=VERSIONED_TTL, max_entries=8, show_spinner=False)
def _load_gas_demand_daily(source: str, version: str) -> pd.DataFrame:
    cfg = GAS_SOURCES[source]
    lookback = int(cfg.get("lookback_days") or 10)
    p = {"provider": cfg["provider"]} if cfg.get("provider") else {}
    if cfg.get("region_to_area"):
        r2a = cfg["region_to_area"]
        df = run_query(f"""
            SELECT provider, family, pattern, region, reference_date, day,
                   value, n_points, normal
            FROM {SBX_SCHEMA}.{cfg['table']}
            WHERE family IN ('tt', 'wnd', 'spv') AND region IN ({_in("region", r2a, p)})
              AND reference_date >= current_timestamp() - INTERVAL {lookback} DAYS
              {"AND provider = :provider" if cfg.get('provider') else ""}
            ORDER BY family, pattern, region, day
        """, p)
        if df.empty:
            return df
        df["area"] = df["region"].map(r2a)
        df = df.drop(columns=["region"])
    else:
        df = run_query(f"""
            SELECT provider, family, pattern, area, reference_date, day,
                   value, n_points, normal
            FROM {SBX_SCHEMA}.{cfg['table']}
            WHERE reference_date >= current_timestamp() - INTERVAL {lookback} DAYS
              {"AND provider = :provider" if cfg.get('provider') else ""}
            ORDER BY family, pattern, area, day
        """, p)
        if df.empty:
            return df
    df = _dt(df, ["reference_date"], utc=True)
    df = _dt(df, ["day"])
    df = _num(df, ["value", "n_points", "normal"])
    scale = float(cfg.get("prod_scale") or 1.0)
    if scale != 1.0:
        prod = df["family"].isin(["wnd", "spv"])
        df.loc[prod, ["value", "normal"]] = df.loc[prod, ["value", "normal"]] * scale
    if cfg.get("time_mode") == "eq":
        df["init_time"] = df["reference_date"].dt.floor("h")           # EQ: the issue time is the cycle
    else:
        df["init_time"] = [init_time_for(p, r) for r, p in zip(df["reference_date"], df["pattern"])]
    df["init_date"] = df["init_time"].dt.normalize()
    return df


def pick_run_at(df: pd.DataFrame, pattern: str, target_init: pd.Timestamp
                ) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Rows of `pattern`'s run initialised at target_init (a full cycle time);
    else its latest run before it. Like pick_run, but on init_time, so a table
    with several cycles a day (the EQ feed) never mixes two runs of one day."""
    p = df[df["pattern"] == pattern]
    if p.empty or "init_time" not in p.columns:
        return p.iloc[0:0], None
    exact = p[p["init_time"] == target_init]
    if not exact.empty:
        return exact, pd.Timestamp(target_init)
    earlier = p[p["init_time"] < target_init]
    if earlier.empty:
        return earlier, None
    latest = earlier["init_time"].max()
    return p[p["init_time"] == latest], pd.Timestamp(latest)


@st.cache_data(ttl=3600, max_entries=16, show_spinner=False)
def load_recent_actual_temp(areas: tuple[str, ...], days: int = 30) -> pd.DataFrame:
    """Trailing daily actual temperature per area — seeds the LDZ curves' multi-day
    effective temperature so the first forecast day is not cold-started.

    ldz_forecast.py approximated actuals with the 1-day-ahead deterministic
    forecast (`get_relative(data_offset='P1D')`) because it could not find a
    confirmed actuals curve on wapi. hist_daily already holds Volue's actual
    ('AF') temperature per area per day, so the app uses that instead — the
    same quantity the script was approximating, without the proxy.
    """
    if not areas:
        return pd.DataFrame()
    # Always the Volue series: the curves were fitted on Volue temperatures and
    # the forecast leg is Volue, so the seed must be the same series — never the
    # Meteomatics history the Historical section shows. On a sandbox written
    # before the `source` column existed every row is Volue's anyway.
    p: dict = {}
    df = run_query(f"""
        SELECT area, day, actual, normal
        FROM {SBX_SCHEMA}.hist_daily
        WHERE metric = 'Temperature' AND area IN ({_in("area", areas, p)})
          AND day >= current_date() - INTERVAL {int(days)} DAYS
          {_hist_where(HIST_DEFAULT_SOURCE if hist_sources() else None, p)}
        ORDER BY area, day
    """, p)
    if df.empty:
        return df
    df = _dt(df, ["day"])
    return _num(df, ["actual", "normal"])


# ══════════════════════════════════════════════════════════════════════════════
# RUN LISTING & COMPLETENESS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def list_available_runs(df: pd.DataFrame, pattern: str) -> list[pd.Timestamp]:
    """All unique init_date values for a pattern, newest first."""
    if df.empty or "pattern" not in df.columns:
        return []
    sub = df[df["pattern"] == pattern]
    if sub.empty:
        return []
    dates = sorted(sub["init_date"].dropna().unique(), reverse=True)
    return [pd.Timestamp(d) for d in dates]


def run_completeness(df: pd.DataFrame, pattern: str, init_date: pd.Timestamp) -> dict:
    """Check whether a run’s forecast looks complete or is still loading.

    Compares the number of distinct forecast days against the expected horizon
    for that pattern. Returns {complete: bool, n_days, expected, pct}.
    """
    expected = EXPECTED_HORIZON.get(pattern, 15)
    sub = df[(df["pattern"] == pattern) & (df["init_date"] == init_date.normalize())]
    if sub.empty:
        return {"complete": False, "n_days": 0, "expected": expected, "pct": 0.0}

    # Pick any one family/region combo to count distinct days
    for fam in sub["family"].unique():
        fam_sub = sub[sub["family"] == fam]
        # Use the area/region column that exists
        area_col = "region" if "region" in fam_sub.columns else "area"
        for area in fam_sub[area_col].unique():
            section = fam_sub[fam_sub[area_col] == area]
            n_days = int(section["day"].nunique())
            if n_days > 0:
                return {
                    "complete": n_days >= expected * 0.9,
                    "n_days": n_days,
                    "expected": expected,
                    "pct": round(n_days / expected * 100, 0),
                }
    return {"complete": False, "n_days": 0, "expected": expected, "pct": 0.0}


def list_family_runs(df: pd.DataFrame, patterns: list[str]) -> list[tuple[pd.Timestamp, str]]:
    """All (init_date, pattern) pairs for a list of patterns, newest first."""
    if df.empty or "pattern" not in df.columns:
        return []
    mask = df["pattern"].isin(patterns) & df["init_date"].notna()
    sub = df.loc[mask, ["init_date", "pattern"]].drop_duplicates()
    pairs = [(pd.Timestamp(dt), pat) for dt, pat in zip(sub["init_date"], sub["pattern"])]
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs


def list_runs_full(df: pd.DataFrame, patterns: list[str]) -> list[tuple[pd.Timestamp, str]]:
    """All (init_time, pattern) pairs for the patterns, newest first — the cycle
    included, so a 06z and a 12z of the same model and day are two runs."""
    if df.empty or "pattern" not in df.columns or "init_time" not in df.columns:
        return []
    mask = df["pattern"].isin(patterns) & df["init_time"].notna()
    sub = df.loc[mask, ["init_time", "pattern"]].drop_duplicates()
    order = {p: i for i, p in enumerate(patterns)}
    pairs = [(pd.Timestamp(t), str(p)) for t, p in zip(sub["init_time"], sub["pattern"])]
    pairs.sort(key=lambda x: (-x[0].value, order.get(x[1], 99)))
    return pairs


def format_run_label(init_dt: pd.Timestamp, pattern: str) -> str:
    """'EC-ENS 12z · Thu 02 Oct' — the Morning Call's label, from the model name
    per pattern and the cycle hour of the init time (works for Volue patterns,
    whose init_time already carries the cycle, and for EQ tags)."""
    from _config import MORNING_MODEL_LABELS
    return f"{MORNING_MODEL_LABELS.get(pattern, pattern)} {init_dt:%H}z · {init_dt:%a %d %b}"


def format_family_run(init_dt: pd.Timestamp, pattern: str) -> str:
    """Format a run for display: 'Mon 22 Sep 00z (EC)' or 'Mon 22 Sep 00z (EC Op)'."""
    hour = 12 if "12" in pattern else 0
    if "monthly" in pattern:
        tag = "EC-Ext"
    elif pattern.startswith("ec"):
        tag = "EC Op" if "ens" not in pattern else "EC"
    elif "gfs" in pattern:
        tag = "GFS Op" if "ens" not in pattern else "GFS"
    else:
        tag = pattern
    return f"{init_dt.strftime('%a %d %b')} {hour:02d}z ({tag})"


def average_multi_runs(df: pd.DataFrame, runs: list[tuple]) -> pd.DataFrame:
    """Average daily values across multiple selected (init_date, pattern) runs."""
    if not runs:
        return pd.DataFrame()
    pieces = []
    for init_dt, pat in runs:
        piece = df[(df["pattern"] == pat) & (df["init_date"] == init_dt.normalize())]
        pieces.append(piece)
    combined = pd.concat(pieces, ignore_index=True)
    if combined.empty:
        return combined
    group_cols = [c for c in ["family", "region", "area", "day"] if c in combined.columns]
    agg = {c: "mean" for c in ["value", "normal", "n_points"] if c in combined.columns}
    if not group_cols or not agg:
        return combined
    return combined.groupby(group_cols, as_index=False).agg(agg)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HISTORICAL
# ══════════════════════════════════════════════════════════════════════════════

# hist_daily carries a `source` column since the Meteomatics temperature history
# was added: 'Volue' rows for every metric, 'Meteomatics' rows for temperature.
# The two are different normals on different grids and must never be averaged
# together, so every history query filters on exactly one source. A sandbox
# written before the column existed holds Volue rows only — the helpers below
# notice and leave the filter out, so the app keeps working until the notebook
# has been re-run.

@st.cache_data(ttl=3600, show_spinner=False)
def hist_sources() -> dict[str, list[str]]:
    """metric -> sources present in hist_daily; {} when the table has no `source` column."""
    try:
        df = run_query(f"SELECT DISTINCT metric, source FROM {SBX_SCHEMA}.hist_daily")
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    for m, s in zip(df["metric"], df["source"]):
        out.setdefault(str(m), []).append(str(s))
    return out


def hist_source_options(metric: str) -> list[str]:
    """Sources the user can pick for `metric` (HIST_SOURCE_OPTIONS) that have rows in hist_daily."""
    from _config import HIST_SOURCE_OPTIONS
    have = hist_sources().get(metric, [])
    return [s for s in HIST_SOURCE_OPTIONS.get(metric, []) if s in have]


def resolve_hist_source(metric: str, choice: str | None = None) -> tuple[str | None, str | None]:
    """(source to read for `metric`, note for the UI when it is not the configured one).

    None as the source means "do not filter" — the sandbox predates the column
    and its rows are Volue's. `choice` is the user's pick from hist_source_options.
    """
    avail = hist_sources()
    if choice and choice in avail.get(metric, []):
        return choice, None
    if not avail:
        return None, ("hist_daily has no `source` column yet, so temperature is still Volue's — re-run "
                      "power_desk_refresh.py for the Meteomatics / ERA5 population-weighted history.")
    want = hist_source(metric)
    have = avail.get(metric, [])
    if want in have or not have:
        return want, None
    return have[0], (f"hist_daily has no {want} rows for {metric} — showing {have[0]} instead. The notebook "
                     "could not read the gold climatology on its last run (it must run with dna_prod_gold access).")


def _hist_where(source: str | None, params: dict) -> str:
    if not source:
        return ""
    params["source"] = source
    return "AND source = :source"


@st.cache_data(ttl=3600, max_entries=16, show_spinner=False)
def load_hist_years(metric: str, source: str | None = None) -> list[int]:
    p = {"metric": metric}
    df = run_query(f"SELECT DISTINCT year FROM {SBX_SCHEMA}.hist_daily "
                   f"WHERE metric = :metric {_hist_where(source, p)} ORDER BY year", p)
    return [int(y) for y in pd.to_numeric(df["year"], errors="coerce").dropna()] if not df.empty else []


@st.cache_data(ttl=3600, max_entries=32, show_spinner=False)
def load_hist_daily(metric: str, areas: tuple[str, ...], years: tuple[int, ...],
                    months: tuple[int, ...], source: str | None = None) -> pd.DataFrame:
    """Daily actual / normal / anomaly for the selected areas, years and months, from one source."""
    if not areas or not years or not months:
        return pd.DataFrame()
    p = {"metric": metric}
    df = run_query(f"""
        SELECT metric, area, day, year, month, iso_week, week_start, actual, normal, anomaly, anomaly_pct
        FROM {SBX_SCHEMA}.hist_daily
        WHERE metric = :metric AND area IN ({_in("area", areas, p)})
          AND year IN ({",".join(str(int(y)) for y in years)})
          AND month IN ({",".join(str(int(m)) for m in months)})
          {_hist_where(source, p)}
        ORDER BY area, day
    """, p)
    if df.empty:
        return df
    df = _dt(df, ["day", "week_start"])
    return _num(df, ["year", "month", "iso_week", "actual", "normal", "anomaly", "anomaly_pct"])


@st.cache_data(ttl=3600, max_entries=32, show_spinner=False)
def load_hist_monthly_all(metric: str, areas: tuple[str, ...], source: str | None = None) -> pd.DataFrame:
    """Monthly mean actual / normal / anomaly for every year — powers the year×month heatmap."""
    if not areas:
        return pd.DataFrame()
    p = {"metric": metric}
    df = run_query(f"""
        SELECT metric, area, year, month, AVG(actual) AS actual, AVG(normal) AS normal,
               AVG(anomaly) AS anomaly,
               CASE WHEN AVG(normal) != 0 THEN (AVG(actual) / AVG(normal) - 1) * 100 END AS anomaly_pct,
               COUNT(*) AS n_days
        FROM {SBX_SCHEMA}.hist_daily
        WHERE metric = :metric AND area IN ({_in("area", areas, p)}) {_hist_where(source, p)}
        GROUP BY metric, area, year, month ORDER BY area, year, month
    """, p)
    return _num(df, ["year", "month", "actual", "normal", "anomaly", "anomaly_pct", "n_days"])


def _table_missing(e: Exception) -> bool:
    """True when `e` says the table is not there (benign: the loader job has not run yet) — as
    opposed to a timeout, a permission or a warehouse error, which must not look like 'no data'."""
    m = str(e).upper()
    return any(k in m for k in ("TABLE_OR_VIEW_NOT_FOUND", "CANNOT BE FOUND", "DOES NOT EXIST", "NOT FOUND"))


@st.cache_data(ttl=3600, show_spinner=False)
def load_index_monthly() -> pd.DataFrame:
    """Monthly mean of every teleconnection index with the number of days behind it
    (index_name, year, month, value, n_days), from the daily weather_indexes table that
    pipeline/indices_upload.py loads. Aggregated in SQL — the daily table is ~580 k rows.
    Empty only when the table does not exist yet. Any other failure (warehouse cold start,
    timeout, permission) is RAISED, which st.cache_data does not cache — swallowing it would
    cache an empty frame for an hour and show 'not loaded yet' over data that is there."""
    try:
        df = run_query(f"""
            SELECT index_name, YEAR(date) AS year, MONTH(date) AS month, AVG(value) AS value, COUNT(value) AS n_days
            FROM {SBX_SCHEMA}.weather_indexes WHERE value IS NOT NULL
            GROUP BY index_name, YEAR(date), MONTH(date) ORDER BY index_name, year, month
        """)
    except Exception as e:
        if not _table_missing(e):
            raise
        df = pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=["index_name", "year", "month", "value", "n_days"])
    return _num(df, ["year", "month", "value", "n_days"])


@st.cache_data(ttl=1800, show_spinner=False)
def load_index_recent(days: int = 45) -> pd.DataFrame:
    """The last `days` days of every index, daily (index_name, date, value) — the 'where are
    the indices now' strip, which the monthly aggregate cannot give for a running month."""
    try:
        df = run_query(f"""
            SELECT index_name, date, value FROM {SBX_SCHEMA}.weather_indexes
            WHERE value IS NOT NULL AND date >= current_date() - INTERVAL {int(days)} DAYS
            ORDER BY index_name, date
        """)
    except Exception as e:
        if not _table_missing(e):
            raise
        df = pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=["index_name", "date", "value"])
    return _num(_dt(df, ["date"]), ["value"])


@st.cache_data(ttl=3600, max_entries=32, show_spinner=False)
def load_reanalysis_monthly(metric: str, areas: tuple[str, ...]) -> pd.DataFrame:
    """Meteologica ERA5 reanalysis generation (reanalysis_daily, built by the notebook's cell
    5b from the latest vintage of dna_prod_silver.meteologica.reanalysis) as monthly means
    per country: area, year, month, value_mw, capacity_mw, n_days. Wind and Solar only.
    Empty only until the notebook has built the table; other failures are raised (not cached)."""
    from _config import TELE_REANALYSIS_TABLE
    cols = ["area", "year", "month", "value_mw", "capacity_mw", "n_days"]
    if metric not in ("Wind", "Solar") or not areas:
        return pd.DataFrame(columns=cols)
    p = {"metric": metric}
    try:
        df = run_query(f"""
            SELECT area, year, month, AVG(value_mw) AS value_mw, MAX(capacity_mw) AS capacity_mw, COUNT(*) AS n_days
            FROM {SBX_SCHEMA}.{TELE_REANALYSIS_TABLE}
            WHERE metric = :metric AND area IN ({_in("area", areas, p)})
            GROUP BY area, year, month ORDER BY area, year, month
        """, p)
    except Exception as e:
        if not _table_missing(e):
            raise
        df = pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=cols)
    return _num(df, ["year", "month", "value_mw", "capacity_mw", "n_days"])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2b — ANOMALY MAPS (gold layer)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, max_entries=32, show_spinner=False)
def load_anomaly_map(metric: str, years: tuple[int, ...],
                     months: tuple[int, ...]) -> pd.DataFrame:
    """Average anomaly per grid point for the selected metric/months/years.

    Reads the pre-aggregated sandbox table (written by power_desk_refresh from
    the gold layer). The notebook runs as the user who has gold access; the app
    SP only needs sandbox access.
    """
    if not years or not months:
        return pd.DataFrame()
    df = run_query(f"""
        SELECT latitude, longitude,
               FIRST(city) AS city,
               AVG(anomaly) AS anomaly,
               AVG(value) AS value,
               AVG(normal) AS normal
        FROM {SBX_SCHEMA}.anomaly_map
        WHERE metric = :metric
          AND year IN ({",".join(str(int(y)) for y in years)})
          AND month IN ({",".join(str(int(m)) for m in months)})
        GROUP BY latitude, longitude
    """, {"metric": metric})
    return _num(df, ["latitude", "longitude", "anomaly", "value", "normal"])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HYDRO
# ══════════════════════════════════════════════════════════════════════════════

def load_hydro_available() -> pd.DataFrame:
    """(area, component, data_type, n, first_day, last_day) per series in hydro_daily —
    counted on the component frames the overview reads anyway, not a query of its own."""
    parts = []
    for comp in HYDRO_COMPONENTS.values():
        df = load_hydro_component(comp)
        if df.empty:
            continue
        g = df.groupby(["area", "data_type"])["day"].agg(n="size", first_day="min", last_day="max").reset_index()
        g.insert(1, "component", comp)
        parts.append(g)
    if not parts:
        return pd.DataFrame(columns=["area", "component", "data_type", "n", "first_day", "last_day"])
    return pd.concat(parts, ignore_index=True)


def load_hydro_component(component: str) -> pd.DataFrame:
    """Every area's actual (SA) and normal (N) daily series for one component, in
    one query — the overview map quantifies all areas at once. ~11 areas × 2
    curves × 13 years of days: around 100 k rows, well inside the inline limit.
    Read from HYDRO_START_YEAR on (the report's own history), cached until the
    notebook rewrites hydro_daily."""
    return _load_hydro_component(component, data_version("hydro_daily"))


@st.cache_data(ttl=VERSIONED_TTL, max_entries=8, show_spinner=False)
def _load_hydro_component(component: str, version: str) -> pd.DataFrame:
    df = run_query(f"""
        SELECT area, day, data_type, value FROM {SBX_SCHEMA}.hydro_daily
        WHERE component = :component AND day >= DATE '{int(HYDRO_START_YEAR)}-01-01'
        ORDER BY area, day
    """, {"component": component})
    if df.empty:
        return df
    df = _dt(df, ["day"])
    return _num(df, ["value"])


def load_hydro_prod(areas: tuple[str, ...]) -> pd.DataFrame:
    """hydro_prod_daily (pipeline/volue_hydro_prod.py) for the areas: actual (sa)
    and normal (n) rows of the last HYDRO_PROD_HISTORY_DAYS and every forecast
    issue of the last HYDRO_PROD_LOOKBACK_DAYS. Columns: area, variable (ror /
    tot / rre), data_type, pattern, reference_date, day, value (GWh/day),
    init_time. Empty (never raises on an empty table) until the pipeline has run."""
    if not areas:
        return pd.DataFrame()
    return _load_hydro_prod(tuple(areas), data_version(HYDRO_PROD_TABLE))


@st.cache_data(ttl=VERSIONED_TTL, max_entries=32, show_spinner=False)
def _load_hydro_prod(areas: tuple[str, ...], version: str) -> pd.DataFrame:
    from _config import HYDRO_PROD_LOOKBACK_DAYS, HYDRO_PROD_HISTORY_DAYS
    p: dict = {}
    df = run_query(f"""
        SELECT area, variable, data_type, pattern, reference_date, day, value
        FROM {SBX_SCHEMA}.{HYDRO_PROD_TABLE}
        WHERE area IN ({_in("area", areas, p)})
          AND ((data_type IN ('sa', 'n') AND day >= current_date() - INTERVAL {int(HYDRO_PROD_HISTORY_DAYS)} DAYS)
               OR (data_type = 'f' AND reference_date >= current_timestamp() - INTERVAL {int(HYDRO_PROD_LOOKBACK_DAYS)} DAYS))
        ORDER BY area, variable, data_type, pattern, reference_date, day
    """, p)
    if df.empty:
        return df
    df = _dt(df, ["reference_date"], utc=True)
    df = _dt(df, ["day"])
    df = _num(df, ["value"])
    df["pattern"] = df["pattern"].fillna("")
    # the production forecast ('volue') is issued at 00:00 CET; the rre ensembles at their cycle
    df["init_time"] = [pd.NaT if pd.isna(r) else (volue_init_time(r, 0) if p == "volue" else init_time_for(p, r))
                       for r, p in zip(df["reference_date"], df["pattern"])]
    return df


def load_reservoir_outlook(area: str, n_issues: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The last n issues of the reservoir outlook (pipeline/volue_reservoir_outlook.py):
    (paths, runs). paths: issue_date, series, day, value — every series of those issues
    (normal, baseline, weather, state, model, model_p10..p90, volue, analogue_<year>);
    runs: the inputs of each issue. Both empty until the pipeline has run."""
    return _load_reservoir_outlook(area, int(n_issues), data_version(RESERVOIR_OUTLOOK_TABLE))


@st.cache_data(ttl=VERSIONED_TTL, max_entries=8, show_spinner=False)
def _load_reservoir_outlook(area: str, n_issues: int, version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = run_query(f"""
        WITH issues AS (
            SELECT DISTINCT issue_date FROM {SBX_SCHEMA}.{RESERVOIR_OUTLOOK_TABLE}
            WHERE area = :area ORDER BY issue_date DESC LIMIT {int(n_issues)}
        )
        SELECT o.issue_date, o.series, o.day, o.value
        FROM {SBX_SCHEMA}.{RESERVOIR_OUTLOOK_TABLE} o JOIN issues i ON i.issue_date = o.issue_date
        WHERE o.area = :area ORDER BY o.issue_date, o.series, o.day
    """, {"area": area})
    runs = run_query(f"""
        SELECT * FROM {SBX_SCHEMA}.{RESERVOIR_OUTLOOK_RUNS_TABLE}
        WHERE area = :area ORDER BY issue_date DESC LIMIT {int(n_issues)}
    """, {"area": area})
    if not paths.empty:
        paths = _dt(paths, ["issue_date", "day"])
        paths = _num(paths, ["value"])
    if not runs.empty:
        runs = _dt(runs, ["issue_date", "s0_day", "swe_day"])
        runs = _num(runs, ["s0", "s_a0", "swe_a0", "n_members", "precip_anom_16d", "precip_anom_46d", "volue_offset"])
    return paths, runs


@st.cache_data(ttl=3600, show_spinner=False)
def load_reservoir_model() -> dict:
    """reservoir_model_fr.json shipped with the app — the fitted coefficients and their
    leave-one-year-out scores (for the skill panel). {} when the file is missing."""
    import json
    from pathlib import Path
    from _config import RESERVOIR_MODEL_FILE
    p = Path(__file__).resolve().parent / RESERVOIR_MODEL_FILE
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_swe_country_daily() -> pd.DataFrame:
    """The internal Exolabs SWE model per Alpine country (mean mm, daily) from
    swe_daily — written by Power_dashboard/pipeline, not by the notebook. Empty
    (never raises) until the pipeline has run. The whole record is read: it is the
    history the SWE layer's percentile is taken against."""
    return _load_swe_country_daily(data_version("swe_daily"))


@st.cache_data(ttl=VERSIONED_TTL, max_entries=4, show_spinner=False)
def _load_swe_country_daily(version: str) -> pd.DataFrame:
    try:
        df = run_query(f"""
            SELECT day, region, swe_mean_mm, swe_total, swe_mean_roll_mm
            FROM {SBX_SCHEMA}.swe_daily
            WHERE level = 'country' AND band = 'total' ORDER BY region, day
        """)
    except Exception:
        return pd.DataFrame(columns=["day", "region", "swe_mean_mm", "swe_total", "swe_mean_roll_mm"])
    if df.empty:
        return df
    df = _dt(df, ["day"])
    return _num(df, ["swe_mean_mm", "swe_total", "swe_mean_roll_mm"])


def _river_latest_sql(table: str, stations: str, lead_tags: tuple[str, ...]) -> str:
    """Latest observation (backcast preferred over actual on the same day), the
    normal for that day, and the newest forecast issue's 7-day range."""
    return f"""
        WITH obs AS (
          SELECT station_key, station, area, day, value, data_type,
                 ROW_NUMBER() OVER (PARTITION BY station_key
                                    ORDER BY day DESC, CASE WHEN data_type = 'backcast' THEN 0 ELSE 1 END) AS rn
          FROM {SBX_SCHEMA}.{table} WHERE data_type IN ('backcast', 'actual')
        ),
        latest AS (SELECT station_key, station, area, day, value, data_type FROM obs WHERE rn = 1),
        nm AS (SELECT station_key, day, value AS normal FROM {SBX_SCHEMA}.{table} WHERE data_type = 'normal'),
        fc AS (
          SELECT station_key, tag, issued, day, value, MAX(issued) OVER (PARTITION BY station_key) AS max_issued
          FROM {SBX_SCHEMA}.{table} WHERE data_type = 'forecast' AND tag IN ({_sql_list(lead_tags)})
        ),
        fc7 AS (
          SELECT f.station_key, f.tag, f.issued, MAX(f.value) AS fc_max7, MIN(f.value) AS fc_min7, MAX(f.day) AS fc_last_day
          FROM fc f JOIN latest l ON l.station_key = f.station_key
          WHERE f.issued = f.max_issued AND f.day > l.day AND f.day <= l.day + INTERVAL 7 DAYS
          GROUP BY f.station_key, f.tag, f.issued
        ),
        fcn AS (
          SELECT l.station_key, AVG(n.normal) AS fc_normal7
          FROM latest l JOIN nm n ON n.station_key = l.station_key AND n.day > l.day AND n.day <= l.day + INTERVAL 7 DAYS
          GROUP BY l.station_key
        )
        SELECT l.station_key, l.station, l.area, l.day, l.value, l.data_type, n.normal, l.value - n.normal AS anomaly,
               f.tag AS fc_tag, f.issued AS fc_issued, f.fc_max7, f.fc_min7, f.fc_last_day, fcn.fc_normal7,
               s.latitude, s.longitude, s.river, s.site
        FROM latest l
        LEFT JOIN nm n ON n.station_key = l.station_key AND n.day = l.day
        LEFT JOIN fc7 f ON f.station_key = l.station_key
        LEFT JOIN fcn ON fcn.station_key = l.station_key
        LEFT JOIN {SBX_SCHEMA}.{stations} s ON s.station_key = l.station_key
        ORDER BY l.station
    """


def _river_flow_latest_sql(table: str) -> str:
    """Latest flow observation per station with the normal of that day."""
    return f"""
        WITH obs AS (
          SELECT station_key, day, value,
                 ROW_NUMBER() OVER (PARTITION BY station_key ORDER BY day DESC) AS rn
          FROM {SBX_SCHEMA}.{table} WHERE data_type IN ('actual', 'backcast')
        ),
        latest AS (SELECT station_key, day, value FROM obs WHERE rn = 1),
        nm AS (SELECT station_key, day, value AS normal FROM {SBX_SCHEMA}.{table} WHERE data_type = 'normal')
        SELECT l.station_key, l.day AS flow_day, l.value AS flow, n.normal AS flow_normal,
               CASE WHEN n.normal > 0 THEN l.value / n.normal * 100 END AS flow_pct
        FROM latest l LEFT JOIN nm n ON n.station_key = l.station_key AND n.day = l.day
    """


@st.cache_data(ttl=1800, show_spinner=False)
def load_river_latest() -> pd.DataFrame:
    """One row per river station across every source in HYDRO_RIVER_SOURCES (EQ via
    the local pipeline, Volue via the notebook): the latest temperature, its
    normal, the newest forecast issue's 7-day range, the latest flow with its
    normal and % of normal (where the source has a flow table) and the station
    coordinates, with a `source` column. A source whose tables do not exist yet
    is skipped; never raises."""
    from _config import HYDRO_RIVER_SOURCES
    parts = []
    for source, cfg in HYDRO_RIVER_SOURCES.items():
        try:
            df = run_query(_river_latest_sql(cfg["table"], cfg["stations"], tuple(cfg["lead_tags"])))
        except Exception:
            continue
        if df.empty:
            continue
        df["source"] = source
        flow_cols = ["flow_day", "flow", "flow_normal", "flow_pct"]
        if cfg.get("flow_table"):
            try:
                fl = run_query(_river_flow_latest_sql(cfg["flow_table"]))
            except Exception:
                fl = pd.DataFrame()
            if not fl.empty:
                df = df.merge(fl, on="station_key", how="left")
        for c in flow_cols:
            if c not in df.columns:
                df[c] = None
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = _dt(df, ["day", "fc_last_day", "flow_day"])
    df = _dt(df, ["fc_issued"], utc=True)
    return _num(df, ["value", "normal", "anomaly", "fc_max7", "fc_min7", "fc_normal7", "latitude", "longitude",
                     "flow", "flow_normal", "flow_pct"])


@st.cache_data(ttl=1800, max_entries=32, show_spinner=False)
def load_river_series(source: str, station_keys: tuple[str, ...], months: int = 14) -> pd.DataFrame:
    """Daily series for the deep dive, one source at a time: observed (backcast /
    actual) and normal over the last `months` months (normal also 60 days
    ahead), plus the latest issue of every forecast tag — for the temperature
    table and, where the source has one, the flow table. Long frame:
    station_key, station, variable ('temperature' | 'flow'), data_type, tag,
    issued, day, value."""
    from _config import HYDRO_RIVER_SOURCES
    cfg = HYDRO_RIVER_SOURCES.get(source)
    if not cfg or not station_keys:
        return pd.DataFrame()
    parts = []
    for variable, table in (("temperature", cfg["table"]), ("flow", cfg.get("flow_table"))):
        if not table:
            continue
        t = f"{SBX_SCHEMA}.{table}"
        p: dict = {}
        keys = _in("key", station_keys, p)
        try:
            df = run_query(f"""
                WITH obs AS (
                  SELECT station_key, station, data_type, '' AS tag, CAST(NULL AS TIMESTAMP) AS issued, day, value
                  FROM {t}
                  WHERE station_key IN ({keys}) AND data_type IN ('backcast', 'actual', 'normal')
                    AND day >= current_date() - INTERVAL {int(months) * 31} DAYS
                    AND day <= current_date() + INTERVAL 60 DAYS
                ),
                fc AS (
                  SELECT station_key, station, data_type, tag, issued, day, value,
                         MAX(issued) OVER (PARTITION BY station_key, tag) AS max_issued
                  FROM {t}
                  WHERE station_key IN ({keys}) AND data_type = 'forecast' AND tag IN ({_sql_list(cfg['forecast_tags'])})
                )
                SELECT station_key, station, data_type, tag, issued, day, value FROM obs
                UNION ALL
                SELECT station_key, station, data_type, tag, issued, day, value FROM fc WHERE issued = max_issued
            """, p)
        except Exception:
            continue
        if df.empty:
            continue
        df["variable"] = variable
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = _dt(df, ["day"])
    df = _dt(df, ["issued"], utc=True)
    df["source"] = source
    return _num(df, ["value"])


def load_hydro_series(country: str, family: str) -> tuple[pd.Series, pd.Series | None]:
    """(actual, normal) daily series for one country and hydro family — cut out of
    the component frame (load_hydro_component), so a country costs no query.

    Normal is None where Volue has no normal curve (hydro balance), matching
    quantify_hydro_balance.py which then uses the mean across years.
    """
    area = HYDRO_AREA_CODES[country]
    df = load_hydro_component(HYDRO_COMPONENTS[family])
    df = df[df["area"] == area] if not df.empty else df
    if df.empty:
        return pd.Series(dtype=float), None
    actual = df[df["data_type"] == "SA"].set_index("day")["value"].sort_index()
    normal = df[df["data_type"] == "N"].set_index("day")["value"].sort_index()
    return actual, (normal if not normal.empty else None)


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

def _versioned_tables() -> dict[str, str]:
    """table -> its write-time column, for every table a versioned loader reads (the
    Morning Call / Gas sources switched off are left out: their table may not exist)."""
    tables = {"fcst_daily": "snapshot_ts", "fcst_members": "snapshot_ts", "hydro_daily": "snapshot_ts",
              "swe_daily": "loaded_at", HYDRO_PROD_TABLE: "loaded_at", RESERVOIR_OUTLOOK_TABLE: "loaded_at"}
    for sources in (MORNING_SOURCES, GAS_SOURCES):
        for name in enabled_sources(sources):
            t = sources[name]["table"]
            tables[t] = "snapshot_ts" if t in ("morning_daily", "gas_demand_daily") else "loaded_at"
    return tables


def _ts(v) -> pd.Timestamp | None:
    t = pd.to_datetime(v, utc=True, errors="coerce")
    return None if pd.isna(t) else t.tz_convert(None)


@st.cache_data(ttl=SNAPSHOT_CHECK_S, show_spinner=False)
def load_snapshot_times() -> dict[str, pd.Timestamp | None]:
    """table -> when it was last written (None when unknown). One UNION ALL query; if
    it fails (a table not created yet), each table on its own so the others still count."""
    tables = _versioned_tables()
    sql = " UNION ALL ".join(f"SELECT '{t}' AS t, MAX({col}) AS ts FROM {SBX_SCHEMA}.{t}" for t, col in tables.items())
    try:
        df = run_query(sql)
        return {t: _ts(ts) for t, ts in zip(df["t"], df["ts"])}
    except Exception as e:
        log.info("snapshot check: union failed (%s), checking table by table", str(e)[:200])
    out = {}
    for t, col in tables.items():
        try:
            df = run_query(f"SELECT MAX({col}) AS ts FROM {SBX_SCHEMA}.{t}")
            out[t] = _ts(df["ts"].iloc[0]) if not df.empty else None
        except Exception:
            out[t] = None
    return out


def data_version(table: str) -> str:
    """The cache-key part for loaders of `table`: its last write time. When that is
    unknown (the check failed, the table is missing) a SNAPSHOT_CHECK_S time bucket
    instead, so the loader falls back to refreshing on a short timer rather than
    holding its data for the whole VERSIONED_TTL."""
    try:
        ts = load_snapshot_times().get(table)
    except Exception:
        ts = None
    return ts.isoformat() if ts is not None else f"t{int(time.time() // SNAPSHOT_CHECK_S)}"
