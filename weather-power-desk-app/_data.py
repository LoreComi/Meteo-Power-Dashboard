"""Data access layer — Power Desk Weather Dashboard.

Every query hits the pre-aggregated sandbox tables in SBX_SCHEMA written by
power_desk_refresh.py — the app never scans the raw Volue / Meteomatics tables.
Queries run through the Databricks SQL Statement API, authenticating as the
App's service principal (same pattern as the LPG desk app) with a fallback to
the user's forwarded token when the SP has no warehouse access.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

from _config import (
    SBX_SCHEMA, VOLUE_SCHEMA, METRICS, VOLUE_MODELS, HYDRO_COMPONENTS, HYDRO_AREA_CODES, WR_REGIMES,
    MORNING_CURVES, MORNING_REGION_CODES, MORNING_MODELS, MORNING_VOLUE_TABLES,
    GAS_VOLUE_FAMILIES, GAS_VOLUE_PATTERNS, GAS_VOLUE_AREAS, LIVE_HISTORY_DAYS, EXPECTED_HORIZON,
    DELTASHARE_PATTERN_MAP,
)

# ─── Connection config ───────────────────────────────────────────────────────────
_raw_host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_HOST = _raw_host if _raw_host.startswith("https://") else f"https://{_raw_host}"
WAREHOUSE_ID = os.environ.get("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH", "").split("/")[-1]


def _headers() -> dict:
    """Auth headers: app service principal first, forwarded user token second."""
    h = {"Content-Type": "application/json"}
    try:
        from databricks.sdk import WorkspaceClient
        tok = WorkspaceClient().config.authenticate()
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


def run_query(sql: str, wait_timeout: str = "50s", poll_timeout: int = 180) -> pd.DataFrame:
    """Execute SQL via the Statement API and return a DataFrame (strings; cast downstream).

    If the query does not finish within *wait_timeout* (max 50 s per API rules),
    it is polled every 5 s for up to *poll_timeout* seconds before giving up.
    """
    import time as _time
    resp = requests.post(
        f"{DATABRICKS_HOST}/api/2.0/sql/statements/",
        headers=_headers(),
        json={"warehouse_id": WAREHOUSE_ID, "statement": sql, "wait_timeout": wait_timeout,
              "disposition": "INLINE", "format": "JSON_ARRAY"},
        timeout=90,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    state = data.get("status", {}).get("state")
    # --- poll until the statement finishes or we time out ---
    stmt_id = data.get("statement_id")
    elapsed = 0
    while state in ("PENDING", "RUNNING") and elapsed < poll_timeout and stmt_id:
        _time.sleep(5)
        elapsed += 5
        poll = requests.get(
            f"{DATABRICKS_HOST}/api/2.0/sql/statements/{stmt_id}",
            headers=_headers(), timeout=30,
        )
        if poll.status_code != 200:
            raise RuntimeError(f"Poll HTTP {poll.status_code}: {poll.text[:400]}")
        data = poll.json()
        state = data.get("status", {}).get("state")
    if state == "FAILED":
        raise RuntimeError(data["status"]["error"]["message"])
    if state in ("PENDING", "RUNNING"):
        raise RuntimeError(f"Query still running after {elapsed}s — narrow the selection or check the warehouse.")
    cols = [c["name"] for c in data.get("manifest", {}).get("schema", {}).get("columns", [])]
    rows = data.get("result", {}).get("data_array", [])
    df = pd.DataFrame(rows, columns=cols)
    # follow chunked results if any
    next_link = data.get("result", {}).get("next_chunk_internal_link")
    while next_link:
        r = requests.get(f"{DATABRICKS_HOST}{next_link}", headers=_headers(), timeout=90)
        r.raise_for_status()
        chunk = r.json()
        df = pd.concat([df, pd.DataFrame(chunk.get("data_array", []), columns=cols)], ignore_index=True)
        next_link = chunk.get("next_chunk_internal_link")
    return df


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


def snap_to_init_time(ref_dt: pd.Timestamp, init_hours: list[int]) -> pd.Timestamp:
    """Volue reference_date is data-arrival (~2 h after init); snap to nearest 00z/12z."""
    cands = [ref_dt.normalize() + pd.Timedelta(days=d, hours=h) for d in (-1, 0, 1) for h in init_hours]
    return min(cands, key=lambda c: abs((ref_dt - c).total_seconds()))


def init_time_from_hour(ref_dt: pd.Timestamp, init_hour: int) -> pd.Timestamp:
    """Build init time from reference_date and the explicit init_hour stored in
    fcst_runs.  Volue publishes 00z and 12z with the same reference_date
    (~22:00 UTC), so init_hour (derived from the pattern name) is the only
    reliable way to tell them apart."""
    base = ref_dt.normalize()  # midnight of the reference_date day
    cand = base + pd.Timedelta(hours=init_hour)
    # If reference_date is late evening and init_hour is 0, the init belongs
    # to the next calendar day (e.g. ref 22:00 Sep 12 → 00z Sep 13).
    if init_hour == 0 and ref_dt.hour >= 18:
        cand += pd.Timedelta(days=1)
    return cand


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

@st.cache_data(ttl=900, show_spinner=False)
def load_runs() -> pd.DataFrame:
    df = run_query(f"""
        SELECT model_family, pattern, init_hour, reference_date, run_rank, run_label, snapshot_ts
        FROM {SBX_SCHEMA}.fcst_runs ORDER BY model_family, run_rank
    """)
    if df.empty:
        return df
    df = _dt(df, ["reference_date", "snapshot_ts"], utc=True)
    df = _num(df, ["run_rank", "init_hour"])
    # Use the explicit init_hour when available (new schema); fall back to
    # snap_to_init_time for backward compatibility with old sandbox data.
    if "init_hour" in df.columns and df["init_hour"].notna().any():
        df["init_time"] = [
            init_time_from_hour(r, int(h)) if pd.notna(h)
            else snap_to_init_time(r, VOLUE_MODELS.get(m, {"init_hours": [0]})["init_hours"])
            for r, m, h in zip(df["reference_date"], df["model_family"], df["init_hour"])
        ]
    else:
        df["init_time"] = [
            snap_to_init_time(r, VOLUE_MODELS.get(m, {"init_hours": [0]})["init_hours"])
            for r, m in zip(df["reference_date"], df["model_family"])
        ]
    df["run_display"] = [f"{lbl} — {format_run(t)}" for lbl, t in zip(df["run_label"], df["init_time"])]
    return df


@st.cache_data(ttl=900, show_spinner=False)
def load_fcst_daily(metric: str, areas: tuple[str, ...], model_families: tuple[str, ...],
                    max_run_rank: int = 6) -> pd.DataFrame:
    """Daily ensemble statistics for the selected metric / areas / model families."""
    if not areas or not model_families:
        return pd.DataFrame()
    df = run_query(f"""
        SELECT provider, model_family, pattern, init_hour, reference_date, run_rank, run_label, metric, area, day,
               lead_day, ens_mean, p10, p25, p50, p75, p90, spread_std, ens_min, ens_max, n_members,
               normal, anomaly, anomaly_pct
        FROM {SBX_SCHEMA}.fcst_daily
        WHERE metric = '{metric}' AND area IN ({_sql_list(areas)})
          AND model_family IN ({_sql_list(model_families)}) AND run_rank <= {int(max_run_rank)}
        ORDER BY model_family, run_rank, area, day
    """)
    if df.empty:
        return df
    df = _dt(df, ["reference_date"], utc=True)
    df = _dt(df, ["day"])
    return _num(df, ["run_rank", "lead_day", "ens_mean", "p10", "p25", "p50", "p75", "p90", "spread_std",
                     "ens_min", "ens_max", "n_members", "normal", "anomaly", "anomaly_pct"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_spread_clim(metric: str, areas: tuple[str, ...]) -> pd.DataFrame:
    """Normal ensemble spread per lead day (run_month 0 = all months)."""
    if not areas:
        return pd.DataFrame()
    df = run_query(f"""
        SELECT metric, area, lead_day, run_month, spread_std_mean, spread_std_p25, spread_std_p75,
               spread_p90_p10_mean, n_runs
        FROM {SBX_SCHEMA}.fcst_spread_clim
        WHERE metric = '{metric}' AND area IN ({_sql_list(areas)})
        ORDER BY area, run_month, lead_day
    """)
    return _num(df, ["lead_day", "run_month", "spread_std_mean", "spread_std_p25", "spread_std_p75",
                     "spread_p90_p10_mean", "n_runs"])


@st.cache_data(ttl=900, show_spinner=False)
def load_member_sources() -> pd.DataFrame:
    """Which (provider, model) member sets exist in fcst_members, with member counts."""
    df = run_query(f"""
        SELECT provider, model, MAX(reference_date) AS reference_date,
               COUNT(DISTINCT member) AS n_members, COUNT(DISTINCT area) AS n_areas,
               MIN(day) AS first_day, MAX(day) AS last_day
        FROM {SBX_SCHEMA}.fcst_members
        WHERE metric = 'Temperature'
        GROUP BY provider, model
    """)
    if df.empty:
        return df
    df = _dt(df, ["reference_date"], utc=True)
    df = _dt(df, ["first_day", "last_day"])
    return _num(df, ["n_members", "n_areas"])


@st.cache_data(ttl=900, show_spinner=False)
def load_members(provider: str, model: str, metric: str, areas: tuple[str, ...]) -> pd.DataFrame:
    """Per-member daily values (and anomaly) for one member source."""
    if not areas:
        return pd.DataFrame()
    df = run_query(f"""
        SELECT provider, model, reference_date, metric, area, day, lead_day, member, value, normal, anomaly
        FROM {SBX_SCHEMA}.fcst_members
        WHERE provider = '{provider}' AND model = '{model}' AND metric = '{metric}'
          AND area IN ({_sql_list(areas)})
        ORDER BY area, day, member
    """)
    if df.empty:
        return df
    df = _dt(df, ["reference_date"], utc=True)
    df = _dt(df, ["day"])
    return _num(df, ["lead_day", "value", "normal", "anomaly"])


@st.cache_data(ttl=900, show_spinner=False)
def load_member_spatial(lead_day_min: int = 1, lead_day_max: int = 15) -> pd.DataFrame:
    """Gridded per-member temperature anomaly for spatial k-means clustering.

    Reads fcst_member_spatial (created by the refresh notebook from Meteomatics
    ecmwf-ens temperature members, coarsened to 1° over Europe).

    The full table is ~1.4 M rows (50 members × 15 days × 1 824 grid points)
    which exceeds the 25 MB inline-result limit.  We aggregate across days
    here in SQL so only ~91 k rows (50 × 1 824) come back.
    """
    df = run_query(f"""
        SELECT provider, model, MAX(reference_date) AS reference_date, metric,
               member, latitude, longitude,
               AVG(anomaly) AS anomaly, AVG(value) AS value, AVG(ens_mean) AS ens_mean
        FROM {SBX_SCHEMA}.fcst_member_spatial
        WHERE lead_day BETWEEN {int(lead_day_min)} AND {int(lead_day_max)}
        GROUP BY provider, model, metric, member, latitude, longitude
    """)
    if df.empty:
        return df
    df = _dt(df, ["reference_date"], utc=True)
    return _num(df, ["latitude", "longitude", "value", "ens_mean", "anomaly"])


@st.cache_data(ttl=900, show_spinner=False)
def load_member_spatial_days(day_min, day_max) -> pd.DataFrame:
    """Same as load_member_spatial but selected by delivery day — one call per trading
    week gives the weekly-mean anomaly map per member (~91 k rows per week)."""
    d0, d1 = pd.Timestamp(day_min).date(), pd.Timestamp(day_max).date()
    df = run_query(f"""
        SELECT provider, model, MAX(reference_date) AS reference_date, metric,
               member, latitude, longitude, COUNT(DISTINCT day) AS n_days,
               AVG(anomaly) AS anomaly, AVG(value) AS value, AVG(ens_mean) AS ens_mean
        FROM {SBX_SCHEMA}.fcst_member_spatial
        WHERE day BETWEEN DATE '{d0}' AND DATE '{d1}'
        GROUP BY provider, model, metric, member, latitude, longitude
    """)
    if df.empty:
        return df
    df = _dt(df, ["reference_date"], utc=True)
    return _num(df, ["latitude", "longitude", "n_days", "value", "ens_mean", "anomaly"])


@st.cache_data(ttl=900, show_spinner=False)
def load_meteologica_members(metric: str, areas: tuple[str, ...]) -> pd.DataFrame:
    """Meteologica members if ingested; empty DataFrame otherwise (never raises)."""
    if not areas:
        return pd.DataFrame()
    try:
        df = run_query(f"""
            SELECT provider, model, reference_date, metric, area, day, lead_day, member, value
            FROM {SBX_SCHEMA}.meteologica_members
            WHERE metric = '{metric}' AND area IN ({_sql_list(areas)})
        """)
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

@st.cache_data(ttl=300, show_spinner=False)
def load_morning_daily() -> pd.DataFrame:
    """Daily 'Avg' values per run from the sandbox morning_daily table.

    Written by power_desk_refresh.py from Volue ensemble curves (ec00ens, etc.).
    Columns: provider, family (tt/wnd/spv/rre), region, pattern, reference_date,
    day, value, n_points, normal.
    """
    df = run_query(f"""
        SELECT provider, family, region, pattern, reference_date, day,
               value, n_points, normal
        FROM {SBX_SCHEMA}.morning_daily
        ORDER BY family, region, pattern, day
    """)
    if df.empty:
        return df
    df = _dt(df, ["reference_date"], utc=True)
    df = _dt(df, ["day"])
    df = _num(df, ["value", "n_points", "normal"])
    init_hours = {"ec00ens": [0], "ec12ens": [12], "gfs00ens": [0], "ecmonthly": [0],
                  "ecmwf-ens": [0, 12], "ecmwf-aifs-ens": [0, 12]}
    df["init_time"] = [snap_to_init_time(r, init_hours.get(p, [0, 12]))
                       for r, p in zip(df["reference_date"], df["pattern"])]
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
    df = run_query(f"""
        SELECT model, reference_date, day, lead_day, member, {cols},
               max_iwr, threshold, regime
        FROM {SBX_SCHEMA}.wr_forecast_members
        ORDER BY model, reference_date, member, lead_day
    """)
    if df.empty:
        return df
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
    return _num(df, ["max_iwr"] + [f"iwr_{r}" for r in WR_REGIMES])


# ══════════════════════════════════════════════════════════════════════════════
# GAS DEMAND
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_gas_demand_daily() -> pd.DataFrame:
    """Daily ens-mean temperature / wind / solar per run from the sandbox.

    Written by power_desk_refresh.py from Volue ensemble curves.
    Columns: provider, family (tt/wnd/spv), pattern, area, reference_date,
    day, value, n_points, normal.
    """
    df = run_query(f"""
        SELECT provider, family, pattern, area, reference_date, day,
               value, n_points, normal
        FROM {SBX_SCHEMA}.gas_demand_daily
        ORDER BY family, pattern, area, day
    """)
    if df.empty:
        return df
    df = _dt(df, ["reference_date"], utc=True)
    df = _dt(df, ["day"])
    df = _num(df, ["value", "n_points", "normal"])
    init_hours = {"ec00ens": [0], "ec12ens": [12], "gfs00ens": [0], "gfs12ens": [12]}
    df["init_time"] = [snap_to_init_time(r, init_hours.get(p, [0, 12]))
                       for r, p in zip(df["reference_date"], df["pattern"])]
    df["init_date"] = df["init_time"].dt.normalize()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
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
    df = run_query(f"""
        SELECT area, day, actual, normal
        FROM {SBX_SCHEMA}.hist_daily
        WHERE metric = 'Temperature' AND area IN ({_sql_list(areas)})
          AND day >= current_date() - INTERVAL {int(days)} DAYS
        ORDER BY area, day
    """)
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


def format_family_run(init_dt: pd.Timestamp, pattern: str) -> str:
    """Format a run for display: 'Mon 22 Sep 00z (EC)'."""
    hour = 12 if "12" in pattern else 0
    tag = ("EC" if pattern.startswith("ec") and "monthly" not in pattern
           else "GFS" if "gfs" in pattern else "EC-Ext")
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

@st.cache_data(ttl=3600, show_spinner=False)
def load_hist_years(metric: str) -> list[int]:
    df = run_query(f"SELECT DISTINCT year FROM {SBX_SCHEMA}.hist_daily WHERE metric = '{metric}' ORDER BY year")
    return [int(y) for y in pd.to_numeric(df["year"], errors="coerce").dropna()] if not df.empty else []


@st.cache_data(ttl=3600, show_spinner=False)
def load_hist_daily(metric: str, areas: tuple[str, ...], years: tuple[int, ...],
                    months: tuple[int, ...]) -> pd.DataFrame:
    """Daily actual / normal / anomaly for the selected areas, years and months."""
    if not areas or not years or not months:
        return pd.DataFrame()
    df = run_query(f"""
        SELECT metric, area, day, year, month, iso_week, week_start, actual, normal, anomaly, anomaly_pct
        FROM {SBX_SCHEMA}.hist_daily
        WHERE metric = '{metric}' AND area IN ({_sql_list(areas)})
          AND year IN ({",".join(str(int(y)) for y in years)})
          AND month IN ({",".join(str(int(m)) for m in months)})
        ORDER BY area, day
    """)
    if df.empty:
        return df
    df = _dt(df, ["day", "week_start"])
    return _num(df, ["year", "month", "iso_week", "actual", "normal", "anomaly", "anomaly_pct"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_hist_monthly_all(metric: str, areas: tuple[str, ...]) -> pd.DataFrame:
    """Monthly mean actual / normal / anomaly for every year — powers the year×month heatmap."""
    if not areas:
        return pd.DataFrame()
    df = run_query(f"""
        SELECT metric, area, year, month, AVG(actual) AS actual, AVG(normal) AS normal,
               AVG(anomaly) AS anomaly,
               CASE WHEN AVG(normal) != 0 THEN (AVG(actual) / AVG(normal) - 1) * 100 END AS anomaly_pct,
               COUNT(*) AS n_days
        FROM {SBX_SCHEMA}.hist_daily
        WHERE metric = '{metric}' AND area IN ({_sql_list(areas)})
        GROUP BY metric, area, year, month ORDER BY area, year, month
    """)
    return _num(df, ["year", "month", "actual", "normal", "anomaly", "anomaly_pct", "n_days"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_weather_indexes() -> pd.DataFrame:
    """Teleconnection indexes; empty until the weather_indexes table is loaded."""
    try:
        df = run_query(f"SELECT index_name, date, value, source FROM {SBX_SCHEMA}.weather_indexes ORDER BY index_name, date")
    except Exception:
        return pd.DataFrame(columns=["index_name", "date", "value", "source"])
    if df.empty:
        return df
    df = _dt(df, ["date"])
    return _num(df, ["value"])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2b — ANOMALY MAPS (gold layer)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
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
        WHERE metric = '{metric}'
          AND year IN ({",".join(str(int(y)) for y in years)})
          AND month IN ({",".join(str(int(m)) for m in months)})
        GROUP BY latitude, longitude
    """)
    return _num(df, ["latitude", "longitude", "anomaly", "value", "normal"])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HYDRO
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_hydro_available() -> pd.DataFrame:
    df = run_query(f"""
        SELECT area, component, data_type, COUNT(*) AS n, MIN(day) AS first_day, MAX(day) AS last_day
        FROM {SBX_SCHEMA}.hydro_daily GROUP BY area, component, data_type
    """)
    if df.empty:
        return df
    df = _dt(df, ["first_day", "last_day"])
    return _num(df, ["n"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_hydro_series(country: str, family: str) -> tuple[pd.Series, pd.Series | None]:
    """(actual, normal) daily series for one country and hydro family.

    Normal is None where Volue has no normal curve (hydro balance), matching
    quantify_hydro_balance.py which then uses the mean across years.
    """
    area = HYDRO_AREA_CODES[country]
    comp = HYDRO_COMPONENTS[family]
    df = run_query(f"""
        SELECT day, data_type, value FROM {SBX_SCHEMA}.hydro_daily
        WHERE area = '{area}' AND component = '{comp}' ORDER BY day
    """)
    if df.empty:
        return pd.Series(dtype=float), None
    df = _dt(df, ["day"])
    df = _num(df, ["value"])
    actual = df[df["data_type"] == "SA"].set_index("day")["value"].sort_index()
    normal = df[df["data_type"] == "N"].set_index("day")["value"].sort_index()
    return actual, (normal if not normal.empty else None)


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_snapshot_times() -> dict[str, pd.Timestamp | None]:
    out = {}
    for t in ("fcst_daily", "fcst_members", "hydro_daily"):
        try:
            df = run_query(f"SELECT MAX(snapshot_ts) AS ts FROM {SBX_SCHEMA}.{t}")
            out[t] = pd.to_datetime(df["ts"].iloc[0], utc=True).tz_convert(None) if not df.empty else None
        except Exception:
            out[t] = None
    return out
