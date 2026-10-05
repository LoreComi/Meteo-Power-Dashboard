"""Gas Demand — Section 4.

Port of the two EU-gas-demand scripts, live off the sandbox tables instead of
wapi:

  ldz_forecast.py   temperature -> LDZ (heating) gas demand through the fitted
                    hinge curves, then the run-over-run cumulative delta and
                    the trade signal per country.
  rdl_forecast.py   wind + solar -> the gas-fired generation they displace,
                    same run-over-run delta, opposite sign convention.

Both scripts ask the same question — "how much did today's run move the next
two weeks of gas demand versus the run we compared against yesterday" — and
both answer it the same way:

    1. pull the ensemble-MEAN daily series of today's run and of the run it is
       paired with (the same pattern's previous 00z, Friday's on a Monday; for
       a later cycle of the day, that day's 00z),
    2. convert weather to gas demand (fitted curve for LDZ, efficiency
       arithmetic for wind/solar),
    3. sum the difference over a fixed window that starts one day after the run
       (the same day on a Monday) and runs to the end of the older run's
       horizon,
    4. call it LONG / SHORT / NO TRADE, per country and cumulatively.

The sign conventions are opposite and both are kept as the scripts have them:
a colder run lifts LDZ demand (positive delta = bullish gas), a windier/sunnier
run displaces more gas burn (positive delta = bearish gas).

Data (GAS_SOURCES): by default the Morning Call's Energy Quantified table,
{SBX_SCHEMA}.morning_daily_eq — every model EQ has and every cycle, refreshed
every couple of hours by Power_dashboard/pipeline/run_morning.bat, with BE and
NL loaded on purpose for this section — so the GWh here and the grid there are
built on the same runs; or the notebook's Volue {SBX_SCHEMA}.gas_demand_daily.
{SBX_SCHEMA}.hist_daily gives the trailing actual temperature that seeds the
curves' multi-day effective temperature (Volue actuals, the series the curves
were fitted on). Curves: curve_models.json, fitted offline by
EU-gas-demand/fit_demand_curves.py.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import _gas_demand_model as gdm
from _config import (
    GAS_CURVE_MODELS_FILE, GAS_LDZ_AREAS, GAS_LDZ_DEFAULT_AREAS,
    GAS_RDL_DEFAULT_REGIONS, GAS_EFFICIENCY, GAS_LOWER_LOAD, GAS_SOURCES, GAS_DEFAULT_SOURCE, enabled_sources,
    GAS_FORECAST_DAYS, GAS_HIST_LOOKBACK_DAYS, GAS_TOTAL_SIGNAL_GWH, EXPECTED_HORIZON,
)
from _charts import make_ldz_panel, make_rdl_chart, make_gas_delta_bars
from _data import (
    load_gas_demand_daily, load_recent_actual_temp, pick_run_at, list_runs_full, format_run_label,
)
from _ui import kpi_card, kpi_row, status_banner


@st.cache_resource(show_spinner=False)
def _models() -> dict:
    return gdm.load_models(Path(__file__).parent / GAS_CURVE_MODELS_FILE)


# ══════════════════════════════════════════════════════════════════════════════
# RUN PAIRING AND DELTA WINDOW (the scripts' rule, generalised to any cycle)
# ══════════════════════════════════════════════════════════════════════════════

def gas_run_pair(pattern: str, run_init: pd.Timestamp) -> dict:
    """Which two runs get compared, and over which days. `run_init` is the init
    TIME of the run being scored — day and cycle, what the run selector shows.

    From rdl_fcst_change() in both scripts, where `date` is the morning the
    report is made: a 00z run is that day's run (`date_min = date`), a 12z run
    is the previous evening's (`date_min = date - 1`). In both cases date_min
    is the run's own init day, so it is taken as given here and the weekday
    rule uses the report morning: the init day for a 00z or 06z run, the day
    after for a 12z or 18z run. Then
      - a 00z run is compared with the same model's previous 00z: one day
        back, three on a Monday (Friday's run),
      - any later cycle is compared with the 00z of its own day — or, when it
        was issued on the eve of a Monday report (a Sunday 12z / 18z), with
        Friday's 00z, which is the script's `date_min - 2` on a Monday,
      - the delta window opens the day after date_min and closes 13 days after
        it; on a Monday it opens on date_min itself and closes after 11 days.
        Either way it ends exactly on the last day of the older run's
        15-day horizon, which is why the Monday window is shorter.
    Volue's 12z pattern is a different pattern name (ec12ens) from its 00z
    (ec00ens); EQ's cycles share one tag. Both are handled.
    """
    run_init = pd.Timestamp(run_init)
    date_min = run_init.normalize()
    hour = int(run_init.hour) if run_init.hour else (12 if pattern.endswith("12ens") else 0)
    is_00z = hour == 0
    report_date = date_min if hour < 12 else date_min + pd.Timedelta(days=1)
    wd = int(report_date.weekday())

    if is_00z:
        prev_pattern = pattern
        prev_init = date_min - pd.Timedelta(days=3 if wd == 0 else 1)
    else:
        prev_pattern = pattern.replace("12ens", "00ens") if pattern.endswith("12ens") else pattern
        eve_of_monday = wd == 0 and report_date != date_min
        prev_init = date_min - pd.Timedelta(days=2 if eve_of_monday else 0)

    if wd == 0:
        d_in, d_out = date_min, date_min + pd.Timedelta(days=11)
    else:
        d_in, d_out = date_min + pd.Timedelta(days=1), date_min + pd.Timedelta(days=13)

    return {"pattern": pattern, "run_init": run_init, "date_min": date_min, "prev_pattern": prev_pattern,
            "prev_init": prev_init, "d_in": d_in, "d_out": d_out, "is_monday": wd == 0}


def _horizon(s: pd.Series, run_init: pd.Timestamp) -> pd.Series:
    """The script's `wailer[init_date : init_date + 14 days]` — 15 days inclusive, from the init day."""
    if s.empty:
        return s
    day0 = pd.Timestamp(run_init).normalize()
    return s.loc[day0:day0 + pd.Timedelta(days=GAS_FORECAST_DAYS)]


def _window_delta(cur: pd.Series, prev: pd.Series, d_in, d_out) -> tuple[float, float, int]:
    """(cumulative delta, current-run total, days contributing) over [d_in, d_out]."""
    c = cur.loc[d_in:d_out] if not cur.empty else cur
    p = prev.loc[d_in:d_out] if not prev.empty else prev
    if c.empty:
        return np.nan, np.nan, 0
    d = (c - p).dropna() if not p.empty else pd.Series(dtype=float)
    return (float(d.sum()) if len(d) else np.nan), float(c.sum()), int(len(d))


# ══════════════════════════════════════════════════════════════════════════════
# SERIES EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _family_series(run_df: pd.DataFrame, family: str, areas: list[str]
                   ) -> tuple[pd.Series, pd.Series, list[str]]:
    """(value, normal, areas that contributed) daily series for one family.

    Temperature is read per country, so `areas` is a single code. Wind and solar
    are summed across the codes making up a region (Volue's Iberia = ES + PT;
    EQ's is one code, IB). A code with no rows is left out rather than blanking
    the region — the caller shows which ones contributed, because a missing
    grid changes the level but barely the run-over-run delta this page is about.
    """
    empty = (pd.Series(dtype=float), pd.Series(dtype=float), [])
    if run_df is None or run_df.empty or "family" not in run_df.columns:
        return empty
    r = run_df[(run_df["family"] == family) & (run_df["area"].isin(areas))]
    if r.empty:
        return empty
    # One row per area and day: two reference_dates can snap to the same run
    # time (a re-issued cycle), and summing a region would then double-count.
    r = r.sort_values("reference_date").drop_duplicates(subset=["area", "day"], keep="last")
    present = sorted(set(r["area"]))
    agg = "mean" if len(areas) == 1 else "sum"
    v = r.groupby("day")["value"].agg(agg).sort_index()
    n = r.groupby("day")["normal"].agg(agg).sort_index()
    return v, n, present


def _continuous(s: pd.Series) -> tuple[pd.Series, int]:
    """Gap-free daily series — smooth_temperature() shifts by position, so a
    missing day would silently misalign the effective-temperature weights.
    Returns the filled series and how many days had to be interpolated."""
    if s.empty:
        return s, 0
    idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    out = s.reindex(idx)
    n_filled = int(out.isna().sum())
    if n_filled:
        out = out.interpolate(limit_direction="both")
    return out, n_filled


# ══════════════════════════════════════════════════════════════════════════════
# LDZ — temperature through the fitted curves
# ══════════════════════════════════════════════════════════════════════════════

def ldz_from_temp(country: str, fcst_temp: pd.Series, actual_hist: pd.Series
                  ) -> tuple[pd.Series, pd.Series, int]:
    """temp_to_ldz() from ldz_forecast.py.

    `actual_hist` must cover at least gdm.MAX_LAG_DAYS days immediately before
    `fcst_temp` starts, so the curve's multi-day effective temperature is not
    cold-started at the first forecast day. Where the two overlap the actual
    wins — that is what the script's `combine_first` does, and it is why the
    previous run's first days are evaluated on what actually happened.

    Returns (effective temperature over the forecast index, LDZ demand,
    interpolated days).
    """
    model = _models()[("ldz", GAS_LDZ_AREAS[country])]
    full = actual_hist.combine_first(fcst_temp).sort_index()
    full, n_filled = _continuous(full)
    eff_full = gdm.smooth_temperature(full, model.half_life_days, max_lag=gdm.MAX_LAG_DAYS)
    eff = eff_full.reindex(fcst_temp.index)
    daytype = gdm.assign_daytype(pd.Series(fcst_temp.index))
    ldz = pd.Series(model.predict(eff.to_numpy(), daytype), index=fcst_temp.index)
    return eff, ldz, n_filled


def _signal(delta: float, bullish_positive: bool, threshold: float = 0.0) -> tuple[str, str]:
    """(trade word, market read) for a demand delta.

    LDZ: a positive delta is more heating demand — long gas. Wind and solar:
    a positive delta is more gas displaced — short gas. `threshold` is 0 per
    country (the scripts trade any sign) and GAS_TOTAL_SIGNAL_GWH cumulatively.
    """
    if delta is None or np.isnan(delta) or abs(delta) <= threshold:
        return "NO TRADE", "Neutral"
    bullish = (delta > 0) if bullish_positive else (delta < 0)
    return ("LONG", "Bullish") if bullish else ("SHORT", "Bearish")


def _pair_runs(gas_df: pd.DataFrame, pattern: str, run_init: pd.Timestamp,
               prev_override: pd.Timestamp | None) -> tuple[dict, pd.DataFrame, pd.Timestamp | None,
                                                            pd.DataFrame, pd.Timestamp | None]:
    """The pairing plus the rows of both runs. `prev_override` replaces the
    script's automatic pairing — the comparison run is the same pattern
    initialised at that time."""
    pair = gas_run_pair(pattern, run_init)
    if prev_override is not None:
        pair["prev_init"] = pd.Timestamp(prev_override)
        pair["prev_pattern"] = pattern
    cur_df, cur_init = pick_run_at(gas_df, pair["pattern"], pair["run_init"])
    prev_df, prev_init = pick_run_at(gas_df, pair["prev_pattern"], pair["prev_init"])
    if prev_init is None and cur_init is not None and prev_override is None:
        # Nothing on or before the wanted cycle (a thin table — the EQ feed keeps 30 days,
        # but on its first day it only holds yesterday's later cycles): compare with the
        # latest earlier run of the model instead, and let the notice say which one.
        prev_df, prev_init = pick_run_at(gas_df, pair["prev_pattern"], cur_init - pd.Timedelta(seconds=1))
    return pair, cur_df, cur_init, prev_df, prev_init


def compute_ldz(gas_df: pd.DataFrame, actual_df: pd.DataFrame, areas: list[str],
                pattern: str, run_init: pd.Timestamp,
                prev_override: pd.Timestamp | None = None) -> dict:
    """One run's LDZ leg: per-country deltas, curves and the cumulative signal.
    `run_init` is the init time of the run being scored (see gas_run_pair)."""
    pair, cur_df, cur_init, prev_df, prev_init = _pair_runs(gas_df, pattern, run_init, prev_override)

    rows, curves, filled = [], {}, 0
    for area in areas:
        f_cur, n_cur, _ = _family_series(cur_df, "tt", [area])
        f_prev, _, _ = _family_series(prev_df, "tt", [area])
        f_cur = _horizon(f_cur, cur_init) if cur_init is not None else f_cur
        f_prev = _horizon(f_prev, prev_init) if prev_init is not None else f_prev

        hist = pd.Series(dtype=float)
        if actual_df is not None and not actual_df.empty and "area" in actual_df.columns:
            hist = actual_df[actual_df["area"] == area].set_index("day")["actual"].dropna().sort_index()
            if cur_init is not None:
                hist = hist.loc[:cur_init.normalize() - pd.Timedelta(days=1)].tail(GAS_HIST_LOOKBACK_DAYS)

        eff_cur = ldz_cur = eff_prev = ldz_prev = pd.Series(dtype=float)
        if not f_cur.empty and not hist.empty:
            eff_cur, ldz_cur, nf = ldz_from_temp(area, f_cur, hist)
            filled = max(filled, nf)
        if not f_prev.empty and not hist.empty:
            eff_prev, ldz_prev, nf = ldz_from_temp(area, f_prev, hist)
            filled = max(filled, nf)

        delta, total, n_days = _window_delta(ldz_cur, ldz_prev, pair["d_in"], pair["d_out"])
        trade, read = _signal(delta, bullish_positive=True)
        rows.append({"area": area, "delta": delta, "total": total, "n_days": n_days,
                     "trade": trade, "read": read,
                     "temp_delta": _window_delta(f_cur, f_prev, pair["d_in"], pair["d_out"])[0]})
        curves[area] = {"eff_prev": eff_prev, "eff_cur": eff_cur, "ldz_prev": ldz_prev,
                        "ldz_cur": ldz_cur, "norm_temp": n_cur}

    deltas = [r["delta"] for r in rows if not np.isnan(r["delta"])]
    total_delta = float(np.sum(deltas)) if deltas else np.nan
    trade, read = _signal(total_delta, True, GAS_TOTAL_SIGNAL_GWH)
    return {"leg": "LDZ", "pair": pair, "cur_init": cur_init, "prev_init": prev_init,
            "rows": rows, "curves": curves, "total_delta": total_delta,
            "total_trade": trade, "total_read": read, "interpolated_days": filled,
            "bullish_positive": True, "unit": "GWh"}


# ══════════════════════════════════════════════════════════════════════════════
# RDL — wind + solar as displaced gas
# ══════════════════════════════════════════════════════════════════════════════

def wind_solar_to_gas(wnd: pd.Series, spv: pd.Series) -> pd.Series:
    """GW of wind + solar -> GWh/day of gas-fired generation displaced.

    rdl_forecast.py: the MWh/h ensemble mean becomes GW, GW × 24 becomes the
    day's electricity, dividing by the CCGT efficiency turns electricity into
    the gas that would have produced it, and `lower_load` keeps only the share
    of the renewable swing that actually displaces gas rather than coal, hydro
    or exports. Both source tables arrive here in GW (the loader scales EQ's
    MWh/h), so the /1000 is done.
    """
    if wnd.empty and spv.empty:
        return pd.Series(dtype=float)
    total_gw = wnd.add(spv, fill_value=0.0) if not (wnd.empty or spv.empty) else (wnd if spv.empty else spv)
    return total_gw * 24.0 / GAS_EFFICIENCY * GAS_LOWER_LOAD


def _rdl_series(run_df: pd.DataFrame, areas: list[str]) -> tuple[pd.Series, pd.Series, list[str]]:
    """(gas displaced, normal gas displaced, contributing areas) for one region."""
    wnd_v, wnd_n, a1 = _family_series(run_df, "wnd", areas)
    spv_v, spv_n, a2 = _family_series(run_df, "spv", areas)
    return (wind_solar_to_gas(wnd_v, spv_v), wind_solar_to_gas(wnd_n, spv_n),
            sorted(set(a1) | set(a2)))


def compute_rdl(gas_df: pd.DataFrame, rdl_regions: dict[str, list[str]], regions: list[str], pattern: str,
                run_init: pd.Timestamp, prev_override: pd.Timestamp | None = None) -> dict:
    """One run's wind + solar leg: per-region deltas, curves and the cumulative signal.
    `rdl_regions` maps a region label to the area codes summed for it (per source);
    `run_init` is the init time of the run being scored (see gas_run_pair)."""
    pair, cur_df, cur_init, prev_df, prev_init = _pair_runs(gas_df, pattern, run_init, prev_override)

    rows, curves = [], {}
    for region in regions:
        areas = rdl_regions[region]
        g_cur, g_norm, present = _rdl_series(cur_df, areas)
        g_prev, _, _ = _rdl_series(prev_df, areas)
        g_cur = _horizon(g_cur, cur_init) if cur_init is not None else g_cur
        g_prev = _horizon(g_prev, prev_init) if prev_init is not None else g_prev

        delta, total, n_days = _window_delta(g_cur, g_prev, pair["d_in"], pair["d_out"])
        trade, read = _signal(delta, bullish_positive=False)
        rows.append({"area": region, "delta": delta, "total": total, "n_days": n_days,
                     "trade": trade, "read": read, "areas": present,
                     "partial": sorted(set(areas) - set(present))})
        curves[region] = {"prev": g_prev, "cur": g_cur, "norm": g_norm}

    deltas = [r["delta"] for r in rows if not np.isnan(r["delta"])]
    total_delta = float(np.sum(deltas)) if deltas else np.nan
    trade, read = _signal(total_delta, False, GAS_TOTAL_SIGNAL_GWH)
    return {"leg": "Wind & Solar", "pair": pair, "cur_init": cur_init, "prev_init": prev_init,
            "rows": rows, "curves": curves, "total_delta": total_delta,
            "total_trade": trade, "total_read": read, "bullish_positive": False, "unit": "GWh"}


# ══════════════════════════════════════════════════════════════════════════════
# HEADLESS SNAPSHOT (consumed by the Morning Call gas agents)
# ══════════════════════════════════════════════════════════════════════════════

def default_runs(gas_df: pd.DataFrame, source: str) -> list[tuple[pd.Timestamp, str]]:
    """The latest run of each of the source's default patterns (EC-ENS and GFS-ENS)."""
    cfg = GAS_SOURCES[source]
    out = []
    for pat in cfg["default_patterns"]:
        for init, p in list_runs_full(gas_df, [pat]):
            out.append((init, p))
            break
    return out


@st.cache_data(ttl=900, show_spinner=False)
def gas_demand_snapshot(source: str = GAS_DEFAULT_SOURCE,
                        runs: tuple[tuple[str, str], ...] | None = None) -> dict:
    """Both legs for the given runs, without rendering anything.

    The Morning Call's gas agent family calls this with the runs its grid shows
    — `runs` = ((pattern, init time ISO), …) — so its commentary is backed by the
    same numbers this section shows for the same runs, whether or not the user
    has opened the section. Without `runs`, the latest EC-ENS and GFS-ENS of the
    source. Returns {} when the table is missing or empty rather than raising —
    the brief then falls back to the Morning Call table alone.
    """
    if source not in GAS_SOURCES:
        return {}
    try:
        gas_df = load_gas_demand_daily(source)
    except Exception:
        return {}
    if gas_df.empty:
        return {}
    cfg = GAS_SOURCES[source]
    try:
        actual_df = load_recent_actual_temp(tuple(GAS_LDZ_DEFAULT_AREAS), days=40)
    except Exception:
        actual_df = pd.DataFrame(columns=["area", "day", "actual", "normal"])

    wanted: list[tuple[pd.Timestamp, str]]
    if runs:
        wanted = [(pd.Timestamp(t), p) for p, t in runs if p in cfg["patterns"]]
    else:
        wanted = default_runs(gas_df, source)

    legs = cfg.get("legs", ["ldz", "rdl"])
    out: dict[str, dict] = {}
    for init, pattern in wanted:
        try:
            ldz = compute_ldz(gas_df, actual_df, GAS_LDZ_DEFAULT_AREAS, pattern, init)
            rdl = (compute_rdl(gas_df, cfg["rdl_regions"], GAS_RDL_DEFAULT_REGIONS, pattern, init)
                   if "rdl" in legs else None)
        except Exception:
            continue
        if ldz["cur_init"] is None and (rdl is None or rdl["cur_init"] is None):
            continue
        out[format_run_label(init, pattern)] = {"ldz": _strip(ldz), "rdl": _strip(rdl) if rdl else None}
    return out


def _strip(res: dict) -> dict:
    """Result without the plotting series — small enough to put in a prompt."""
    pair = res["pair"]
    return {
        "rows": res["rows"],
        "total_delta": res["total_delta"],
        "total_trade": res["total_trade"],
        "total_read": res["total_read"],
        "window": (pair["d_in"].strftime("%d %b"), pair["d_out"].strftime("%d %b")),
        "cur_init": None if res["cur_init"] is None else res["cur_init"].strftime("%d %b %Hz"),
        "prev_init": None if res["prev_init"] is None else res["prev_init"].strftime("%d %b %Hz"),
        "bullish_positive": res["bullish_positive"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# RENDERING
# ══════════════════════════════════════════════════════════════════════════════

_TRADE_CLASS = {"LONG": "kpi-card-warm", "SHORT": "kpi-card-cool", "NO TRADE": "kpi-card-neutral"}


def _fmt(v, fmt="{:+,.0f}") -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def _delta_cell(v: float, bullish_positive: bool) -> str:
    """Signed delta coloured by its price read, not its sign."""
    if v is None or np.isnan(v):
        return '<span class="mc-muted">n/a</span>'
    if round(v, 1) == 0:
        return f'<span class="mc-zero">{v:+,.0f}</span>'
    bullish = (v > 0) if bullish_positive else (v < 0)
    return f'<span class="{"mc-neg" if bullish else "mc-pos"}">{v:+,.0f}</span>'


def _delta_table(results: dict[str, dict], label_col: str) -> str:
    """The scripts' `deltas_df`: countries down, runs across, cumulative column."""
    runs = list(results)
    if not runs:
        return ""
    bp = results[runs[0]]["bullish_positive"]
    areas = [r["area"] for r in results[runs[0]]["rows"]]
    head = (f'<tr><th>{label_col}</th>' + "".join(f"<th>Δ {r}</th>" for r in runs)
            + (f'<th>Δ cumulative</th>' if len(runs) > 1 else "") + "</tr>")
    body = ""
    for i, area in enumerate(areas):
        body += f'<tr><td class="mc-region">{area}</td>'
        per_run = []
        for r in runs:
            row = results[r]["rows"][i]
            per_run.append(row["delta"])
            body += f"<td>{_delta_cell(row['delta'], bp)}</td>"
        if len(runs) > 1:
            vals = [v for v in per_run if not np.isnan(v)]
            body += f"<td>{_delta_cell(float(np.sum(vals)) if vals else np.nan, bp)}</td>"
        body += "</tr>"
    # TOTAL row
    body += '<tr><td class="mc-region"><b>TOTAL</b></td>'
    totals = []
    for r in runs:
        totals.append(results[r]["total_delta"])
        body += f"<td><b>{_delta_cell(results[r]['total_delta'], bp)}</b></td>"
    if len(runs) > 1:
        vals = [v for v in totals if not np.isnan(v)]
        body += f"<td><b>{_delta_cell(float(np.sum(vals)) if vals else np.nan, bp)}</b></td>"
    body += "</tr>"
    return f'<div class="mc-block"><table class="mc-table">{head}{body}</table></div>'


def _signal_cards(results: dict[str, dict]) -> None:
    cards = []
    for label, res in results.items():
        delta = res["total_delta"]
        cards.append(kpi_card(
            f"{label} · {res['leg']}",
            _fmt(delta) + " GWh",
            _TRADE_CLASS.get(res["total_trade"], "kpi-card-neutral"),
            f'<div class="kpi-delta">{res["total_trade"]} · {res["total_read"]} gas</div>',
            f'<div class="kpi-rank">window {res["pair"]["d_in"]:%d %b} – {res["pair"]["d_out"]:%d %b}</div>',
        ))
    kpi_row(cards)


def _run_notices(results: dict[str, dict]) -> None:
    for label, res in results.items():
        pair, cur, prev = res["pair"], res["cur_init"], res["prev_init"]
        if cur is None:
            status_banner(f"{label}: no run found on or before {pair['run_init']:%d %b %H}z — leg skipped.", "critical")
            continue
        if cur != pair["run_init"]:
            status_banner(f"{label}: the run of {pair['run_init']:%d %b %H}z has not landed yet — "
                          f"using the run initialised {cur:%a %d %b %H}z.", "warning")
        if prev is None:
            status_banner(f"{label}: no comparison run near {pair['prev_init']:%d %b %H}z — deltas show n/a.", "warning")
        elif prev != pair["prev_init"]:
            status_banner(f"{label}: comparing against the run of {prev:%a %d %b %H}z, not "
                          f"{pair['prev_init']:%a %d %b %H}z — that run is not in the table.", "warning")


def _leg_csv(results: dict[str, dict]) -> bytes:
    recs = []
    for label, res in results.items():
        for row in res["rows"]:
            recs.append({"run": label, "leg": res["leg"], "area": row["area"],
                         "delta_gwh": row["delta"], "window_total_gwh": row["total"],
                         "days_in_window": row["n_days"], "trade": row["trade"], "read": row["read"],
                         "window_from": res["pair"]["d_in"].date(), "window_to": res["pair"]["d_out"].date(),
                         "run_init": res["cur_init"], "prev_init": res["prev_init"]})
        recs.append({"run": label, "leg": res["leg"], "area": "TOTAL",
                     "delta_gwh": res["total_delta"], "window_total_gwh": None, "days_in_window": None,
                     "trade": res["total_trade"], "read": res["total_read"],
                     "window_from": res["pair"]["d_in"].date(), "window_to": res["pair"]["d_out"].date(),
                     "run_init": res["cur_init"], "prev_init": res["prev_init"]})
    return pd.DataFrame(recs).to_csv(index=False).encode()


def _run_labels(res: dict) -> tuple[str, str]:
    prev = "previous run" if res["prev_init"] is None else f"{res['prev_init']:%a %d %b %H}z"
    cur = "this run" if res["cur_init"] is None else f"{res['cur_init']:%a %d %b %H}z"
    return prev, cur


def _render_ldz(results: dict[str, dict], areas: list[str]) -> None:
    st.caption("Heating gas demand implied by each run's ensemble-mean temperature, through the fitted "
               "hinge curves. A colder run lifts demand — positive delta, bullish gas. The trailing "
               "actual temperature seeds the curves' multi-day effective temperature, so the first "
               "forecast day is not cold-started.")
    if not results:
        return
    _signal_cards(results)
    st.markdown(_delta_table(results, "Country"), unsafe_allow_html=True)

    first = next(iter(results.values()))
    if first["interpolated_days"]:
        status_banner(f"{first['interpolated_days']} day(s) of the temperature path had to be interpolated "
                      "to keep the effective-temperature window gap-free.", "warning")

    st.plotly_chart(make_gas_delta_bars([r["area"] for r in first["rows"]],
                                        [r["delta"] for r in first["rows"]],
                                        f"LDZ delta per country — {next(iter(results))}",
                                        bullish_positive=True), use_container_width=True)

    run_pick = st.selectbox("Run to chart", list(results), key="gas_ldz_chart_run")
    res = results[run_pick]
    prev_lbl, cur_lbl = _run_labels(res)
    for area in areas:
        c = res["curves"].get(area)
        if not c or c["ldz_cur"].empty:
            continue
        row = next(r for r in res["rows"] if r["area"] == area)
        title = (f"{area} — Δ {_fmt(row['delta'])} GWh over {res['pair']['d_in']:%d %b}–"
                 f"{res['pair']['d_out']:%d %b} · {row['trade']} · {row['read']} gas")
        st.plotly_chart(make_ldz_panel(c["eff_prev"], c["eff_cur"], c["ldz_prev"], c["ldz_cur"],
                                       res["pair"]["d_in"], res["pair"]["d_out"], prev_lbl, cur_lbl,
                                       title, norm=c["norm_temp"]), use_container_width=True)

    with st.expander("Fitted curves — shape, thermal inertia and backtest"):
        recs = []
        for area in areas:
            m = _models()[("ldz", GAS_LDZ_AREAS[area])]
            recs.append({"Country": area, "Shape": gdm.describe_shape(m),
                         "Backtest MAE (GWh/d)": round(m.backtest_mae, 1),
                         "Naive MAE (GWh/d)": round(m.backtest_mae_naive, 1),
                         "Weighted R²": round(m.in_sample_r2, 3),
                         "Obs": int(m.n_obs)})
        st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)
        st.caption("Fitted by EU-gas-demand/fit_demand_curves.py: structure and temperature half-life "
                   "chosen on a held-out last-365-days backtest, shape fit with a 1.5-year recency "
                   "half-life, level re-anchored to the last year's mean. The naive column is a plain "
                   "weekday + linear-in-raw-temperature model — the gap is what the hinge and the "
                   "thermal inertia earn. Re-run that script and copy curve_models.json to refresh.")


def _render_rdl(results: dict[str, dict], regions: list[str]) -> None:
    st.caption(f"Wind + solar converted to the gas-fired generation they displace: GW × 24 ÷ "
               f"{GAS_EFFICIENCY:.0%} efficiency × {GAS_LOWER_LOAD:.0%} displacement share. A windier or "
               "sunnier run displaces more gas — positive delta, bearish gas, the opposite sign to LDZ.")
    if not results:
        return
    _signal_cards(results)
    st.markdown(_delta_table(results, "Region"), unsafe_allow_html=True)

    first = next(iter(results.values()))
    partial = {r["area"]: r["partial"] for r in first["rows"] if r["partial"]}
    if partial:
        status_banner("Missing grids in the source table, so these regions are a partial sum: "
                      + " · ".join(f"{k} without {', '.join(v)}" for k, v in partial.items())
                      + ". Levels are understated; the run-over-run delta is not, both runs use the "
                        "same grids.", "warning")

    st.plotly_chart(make_gas_delta_bars([r["area"] for r in first["rows"]],
                                        [r["delta"] for r in first["rows"]],
                                        f"Displaced-gas delta per region — {next(iter(results))}",
                                        bullish_positive=False), use_container_width=True)

    run_pick = st.selectbox("Run to chart", list(results), key="gas_rdl_chart_run")
    res = results[run_pick]
    prev_lbl, cur_lbl = _run_labels(res)
    cols = st.columns(2)
    for i, region in enumerate(regions):
        c = res["curves"].get(region)
        if not c or c["cur"].empty:
            continue
        row = next(r for r in res["rows"] if r["area"] == region)
        title = (f"{region} — Δ {_fmt(row['delta'])} GWh · {row['trade']} · {row['read']} gas")
        with cols[i % 2]:
            st.plotly_chart(make_rdl_chart(c["prev"], c["cur"], c["norm"], res["pair"]["d_in"],
                                           res["pair"]["d_out"], prev_lbl, cur_lbl, title),
                            use_container_width=True)


def _render_combined(ldz: dict[str, dict], rdl: dict[str, dict]) -> None:
    st.caption("Both legs on one screen. They are separate trades in the scripts and stay separate here; "
               "the net line is the arithmetic sum of the two, which is only a fair read when the "
               "displaced-gas leg really does clear against gas rather than coal or hydro.")
    rows = []
    for label in ldz:
        l_d = ldz[label]["total_delta"]
        r_d = rdl.get(label, {}).get("total_delta", np.nan)
        # LDZ bullish when positive, displaced gas bullish when negative
        net = (0 if np.isnan(l_d) else l_d) - (0 if np.isnan(r_d) else r_d)
        trade, read = _signal(net, True, GAS_TOTAL_SIGNAL_GWH)
        rows.append({"Run": label, "LDZ Δ (GWh)": l_d, "LDZ read": ldz[label]["total_read"],
                     "Wind+Solar Δ (GWh)": r_d, "Wind+Solar read": rdl.get(label, {}).get("total_read", "n/a"),
                     "Net demand Δ (GWh)": net, "Net read": read})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"LDZ Δ (GWh)": "{:+,.0f}", "Wind+Solar Δ (GWh)": "{:+,.0f}",
                                  "Net demand Δ (GWh)": "{:+,.0f}"}, na_rep="n/a"),
                 use_container_width=True, hide_index=True)
    st.caption(f"Net = LDZ delta − displaced-gas delta. |net| above {GAS_TOTAL_SIGNAL_GWH:,} GWh is the "
               "scripts' cumulative trade threshold.")


def _source_switch() -> str:
    """Data source for the section — follows the Morning Call's choice by default,
    so the gas GWh and the grid are read off the same runs."""
    if "gas_source" not in st.session_state:
        mc = st.session_state.get("mc_source")
        names = enabled_sources(GAS_SOURCES)
        st.session_state["gas_source"] = mc if mc in names else (GAS_DEFAULT_SOURCE if GAS_DEFAULT_SOURCE in names else names[0])
    names = enabled_sources(GAS_SOURCES)
    if st.session_state.get("gas_source") not in names:
        st.session_state["gas_source"] = names[0]
    if len(names) == 1:
        return names[0]
    return st.radio("Source", names, horizontal=True, key="gas_source",
                    help="\n".join(f"{k}: {GAS_SOURCES[k]['desc']}" for k in names))


def render_gas_demand():
    st.markdown("#### GAS DEMAND")
    source = _source_switch()
    cfg = GAS_SOURCES[source]
    st.caption(f"EU gas demand from the weather — {cfg['desc']} — through the same fitted curves as "
               "ldz_forecast.py and rdl_forecast.py. EC and GFS runs are shown side by side; pick runs below. "
               "Every run is scored against the run the scripts pair it with: a 00z against the same model's "
               "previous 00z (Friday's on a Monday), a later cycle against that day's 00z.")

    try:
        gas_df = load_gas_demand_daily(source)
    except Exception as e:
        st.error(f"Cannot read from the sandbox table {cfg['table']}: {e}")
        return
    if gas_df.empty:
        status_banner(f"No rows in {cfg['table']} yet — it is written by Power_dashboard/pipeline "
                      "(run_morning.bat, run_daily.py --only volue).", "warning")
        return
    if st.session_state.get("gas_source_last") != source:
        for k in ("gas_sel_runs", "gas_cmp_sel", "gas_ldz_chart_run", "gas_rdl_chart_run"):
            st.session_state.pop(k, None)
        st.session_state["gas_source_last"] = source

    # --- runs: every cycle of every model in the table, newest first ---
    avail = list_runs_full(gas_df, cfg["patterns"])
    run_map = {format_run_label(init, pat): (init, pat) for init, pat in avail}
    run_keys = list(run_map)
    defaults = [format_run_label(init, pat) for init, pat in default_runs(gas_df, source)]
    defaults = [k for k in defaults if k in run_map] or run_keys[:2]

    c1, c2, c3 = st.columns([2.6, 2.0, 3.4])
    with c1:
        if run_keys:
            sel = st.multiselect("Run(s)", run_keys, default=defaults, key="gas_sel_runs",
                                 help="Select one or more runs to compare — the latest EC-ENS and GFS-ENS "
                                      "by default; every model and cycle in the table is offered.")
            sel_runs = [run_map[k] for k in sel]
        else:
            sel_runs = []
            status_banner("No runs found.", "warning")

    with c2:
        cmp_mode = st.radio("Comparison", ["Auto (script logic)", "Manual"],
                            key="gas_cmp_mode", horizontal=True,
                            help="Auto follows the original script pairing (00z vs the previous 00z, a later "
                                 "cycle vs that day's 00z); Manual lets you pick any run as the comparison.")
        manual_cmp_init = None
        if cmp_mode == "Manual" and run_keys:
            default_cmp = min(1, len(run_keys) - 1)
            cmp_sel = st.selectbox("Compare with", run_keys, index=default_cmp, key="gas_cmp_sel")
            manual_cmp_init, _cmp_pat = run_map[cmp_sel]

    legs = cfg.get("legs", ["ldz", "rdl"])
    with c3:
        areas = st.multiselect("LDZ countries", list(GAS_LDZ_AREAS), GAS_LDZ_DEFAULT_AREAS, key="gas_areas")
        if "rdl" in legs:
            regions = st.multiselect("Wind & solar regions", list(cfg["rdl_regions"]), GAS_RDL_DEFAULT_REGIONS,
                                     key="gas_regions")
        else:
            regions = []
    if cfg.get("note"):
        st.caption(cfg["note"])

    if not sel_runs:
        status_banner("Pick at least one run.", "warning")
        return

    # Completeness banners — on the run's own rows (its cycle), not its day
    for init_dt, pattern in sel_runs:
        rows = gas_df[(gas_df["pattern"] == pattern) & (gas_df["init_time"] == init_dt)]
        n_days, expected = int(rows["day"].nunique()), EXPECTED_HORIZON.get(pattern, 15)
        if n_days < 0.9 * expected:
            status_banner(f"Run loading — {format_run_label(init_dt, pattern)} has {n_days}/{expected} forecast "
                          f"days. Shown data is partial.", "warning")

    try:
        actual_df = load_recent_actual_temp(tuple(areas) or tuple(GAS_LDZ_DEFAULT_AREAS), days=40)
    except Exception as e:
        st.error(f"Cannot read the trailing actual temperature from hist_daily ({e}).")
        return
    if areas and actual_df.empty:
        status_banner("No actual temperature in hist_daily for the selected countries — the LDZ curves "
                      "cannot be seeded, so the LDZ leg is empty.", "critical")

    ldz_res, rdl_res = {}, {}
    with st.spinner("Building the demand curves…"):
        for init_dt, pattern in sel_runs:
            label = format_run_label(init_dt, pattern)
            prev_ov = manual_cmp_init if (cmp_mode == "Manual" and manual_cmp_init is not None) else None
            if areas:
                ldz_res[label] = compute_ldz(gas_df, actual_df, areas, pattern, init_dt, prev_override=prev_ov)
            if regions:
                rdl_res[label] = compute_rdl(gas_df, cfg["rdl_regions"], regions, pattern, init_dt,
                                             prev_override=prev_ov)

    _run_notices(ldz_res or rdl_res)
    if source == "EQ" and areas:
        st.caption("The LDZ curves were fitted on Volue consumption temperatures and are seeded with Volue "
                   "actuals; EQ's consumption temperature is the same kind of population-weighted mean, so "
                   "a small level offset is possible — the run-over-run delta, which is what is traded "
                   "here, is unaffected because both runs come from the same source.")

    if "rdl" not in legs:
        if areas:
            _render_ldz(ldz_res, areas)
        else:
            status_banner("No LDZ country selected.", "warning")
        st.divider()
        if ldz_res:
            st.download_button("Download deltas (CSV)", _leg_csv({f"LDZ {k}": v for k, v in ldz_res.items()}),
                               f"gas_demand_{source.lower().replace(' ', '_')}_{max(i for i, _ in sel_runs):%Y%m%d_%H}z.csv",
                               "text/csv", key="gas_dl")
        return

    tabs = st.tabs(["LDZ — heating demand", "Wind & Solar — displaced gas", "Both legs"])
    with tabs[0]:
        if areas:
            _render_ldz(ldz_res, areas)
        else:
            status_banner("No LDZ country selected.", "warning")
    with tabs[1]:
        if regions:
            _render_rdl(rdl_res, regions)
        else:
            status_banner("No wind & solar region selected.", "warning")
    with tabs[2]:
        if ldz_res and rdl_res:
            _render_combined(ldz_res, rdl_res)
        else:
            status_banner("Both legs need at least one country and one region selected.", "warning")

    st.divider()
    dl = {**{f"LDZ {k}": v for k, v in ldz_res.items()}, **{f"W&S {k}": v for k, v in rdl_res.items()}}
    if dl:
        newest = max(init for init, _ in sel_runs)
        st.download_button("Download deltas (CSV)", _leg_csv(dl),
                           f"gas_demand_{source.lower().replace(' ', '_')}_{newest:%Y%m%d_%H}z.csv", "text/csv", key="gas_dl")
