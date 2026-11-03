"""Section 1 — Forecast.

Three tabs:
  Values        every Volue ensemble value by country (mean, median, P10-P90),
                normal, earlier runs, Meteomatics EC-ENS / AIFS-ENS means
  Uncertainty   spread of the ensemble distribution and how it compares with
                the normal spread for the same lead day — is this forecast
                more or less certain than usual?
  Scenarios     weather scenarios per TRADING WEEK: for each Mon–Sun week the
                members are clustered (k-means) on their weekly-mean 500 hPa
                geopotential height field of that week over Europe and the
                near Atlantic — the scenarios of week 45, the scenarios of
                week 46 — for EC-ENS (Z500
                members from the Meteomatics API, via the pipeline) or EC-AI
                (AIFS-ENS Z500 members from the silver layer, via the
                notebook); each scenario named from its centroid map; then,
                per country, the scenarios side by side for temperature,
                wind and solar read off the same member IDs — the Volue
                EC-ENS members for EC-ENS, the AIFS-ENS member rows the
                pipeline writes for EC-AI
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    AREAS, DEFAULT_AREAS, ANOMALY_VIEW_DEFAULT_AREAS, METRICS, DEFAULT_METRIC, VOLUE_MODELS, DEFAULT_MODEL,
    METEOMATICS_MODELS, SPREAD_RATIO_HIGH, SPREAD_RATIO_LOW,
    SCENARIO_SPATIAL_SOURCES, SCENARIO_DEFAULT_SOURCE, SCENARIO_OUTPUT_AREAS, SCENARIO_K_DEFAULT, SCENARIO_K_MAX,
    SCENARIO_NWE_AREAS, SPATIAL_GRID_RES, SPATIAL_BBOX,
    area_label,
)
from _data import (
    load_runs, load_fcst_daily, load_spread_clim, load_members, load_spatial_sources, load_member_spatial_days,
    format_run, volue_init_time, snap_to_init_time,
)
from _charts import (
    AREA_COLORS, make_fan_chart, make_anomaly_heatmap,
    make_spread_vs_normal_chart, make_spread_ratio_heatmap, make_member_distribution, make_member_strips,
    make_country_scenario_panel, scenario_color,
)
from _scenarios import (
    available_weeks, default_weeks, spatial_member_matrix, cluster_members, apply_scenarios, centroid_maps,
    name_cluster, short_name, nwe_read, scenario_daily_summary, scenario_weekly_values, scenario_name,
    describe_scenario, silhouette_for_k,
)
from _regimes import render_regimes
from _ui import (anomaly_kpi, kpi_card, kpi_row, status_banner, page_header, sec_header, tile, tiles, note,
                 keyed)
from _style import PAGE_CSS


# ─── shared controls ─────────────────────────────────────────────────────────────

def _area_picker(key: str, default: list[str]) -> list[str]:
    return st.multiselect("Countries", list(AREAS.keys()), default, key=key,
                          format_func=area_label, label_visibility="collapsed")


def _runs_for(runs: pd.DataFrame, family: str) -> pd.DataFrame:
    return runs[runs["model_family"] == family].sort_values("run_rank") if not runs.empty else runs


def _horizon_mean(df: pd.DataFrame, col: str, max_lead: int) -> pd.Series:
    d = df[(df["lead_day"] >= 1) & (df["lead_day"] <= max_lead)]
    return d.groupby("area")[col].mean()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VALUES
# ══════════════════════════════════════════════════════════════════════════════

_METRIC_ICON = {"Temperature": "tt", "Wind": "wnd", "Solar": "spv", "Precipitation energy": "rre"}
_GAUGE_FULL = {"diff": 3.0, "pct": 40.0}        # the anomaly that fills a tile's gauge (°C, or % of normal)


def _value_tile(a: str, metric: str, m: dict, mean, anom, sd, a_unit: str) -> str:
    """Days 1–14 mean of one country: value, anomaly on a gauge (red = warmer / less wind,
    solar, precipitation; blue = colder / more) and the mean member spread."""
    def num(x):
        try:
            x = float(x)
            return None if np.isnan(x) else x
        except (TypeError, ValueError):
            return None
    mean, anom, sd = num(mean), num(anom), num(sd)
    if mean is None:
        return tile(_METRIC_ICON.get(metric, "forecast"), area_label(a), "days 1–14 · ensemble mean", "n/a", "",
                    None, "flat", "no data")
    tone, pill, gauge = "flat", "no normal", None
    if anom is not None:
        tone = "flat" if round(anom, 1) == 0 else ("warm" if (anom > 0) == m["warm_is_positive"] else "cool")
        pill = (f"{anom:+.1f} {m['unit']} vs normal" if a_unit != "%" else f"{anom:+.0f}% vs normal")
        gauge = anom / _GAUGE_FULL[m["anomaly_kind"]]
    foot = f"σ {sd:.1f} {m['unit']}" if sd is not None else ""
    return tile(_METRIC_ICON.get(metric, "forecast"), area_label(a), "days 1–14 · ensemble mean",
                f"{mean:.1f}", m["unit"], gauge, tone, pill, foot)


def _render_values(runs: pd.DataFrame):
    with keyed("tb_fv"):
        c1, c2, c3, c4 = st.columns([1.2, 1.1, 3, 1.4])
        with c1:
            metric = st.selectbox("Metric", list(METRICS.keys()), index=list(METRICS).index(DEFAULT_METRIC), key="fv_metric")
        with c2:
            family = st.selectbox("Model", list(VOLUE_MODELS.keys()), index=list(VOLUE_MODELS).index(DEFAULT_MODEL), key="fv_model")
        with c3:
            areas = _area_picker("fv_areas", [a for a in ANOMALY_VIEW_DEFAULT_AREAS if a in AREAS])
        fam_runs = _runs_for(runs, family)
        with c4:
            run_opts = fam_runs["run_display"].tolist() if not fam_runs.empty else ["Latest"]
            run_pick = st.selectbox("Run", run_opts, key="fv_run")
        run_rank = int(fam_runs.loc[fam_runs["run_display"] == run_pick, "run_rank"].iloc[0]) if not fam_runs.empty else 1

        o1, o2, _ = st.columns([1.2, 1.6, 4])
        with o1:
            show_prev = st.checkbox("Earlier runs", value=True, key="fv_prev", help="Faded ensemble-mean lines of the previous runs")
        with o2:
            show_mm = st.checkbox("Meteomatics EC / AIFS means", value=(metric == "Temperature"), key="fv_mm",
                                  disabled=(metric != "Temperature"),
                                  help="Population-weighted country mean of the Meteomatics ensemble MEAN (silver tables; "
                                       "weights from GHS-POP, see pop_weights_0p5.csv)")
    if not areas:
        st.info("Select at least one country.")
        return

    m = METRICS[metric]
    try:
        df = load_fcst_daily(metric, tuple(areas), (family,), max_run_rank=6)
        mm_df = (load_fcst_daily(metric, tuple(areas), tuple(METEOMATICS_MODELS.keys()), 1)
                 if show_mm and metric == "Temperature" else pd.DataFrame())
    except Exception as e:
        st.error(f"Failed to load forecast data: {e}")
        return
    if df.empty:
        status_banner("No forecast rows for this selection — run power_desk_refresh.py or check the run list.", "warning")
        return

    cur = df[df["run_rank"] == run_rank]
    prev = df[df["run_rank"] > run_rank] if show_prev else pd.DataFrame()
    ref = cur["reference_date"].max()
    init_hour = int(cur["init_hour"].dropna().iloc[0]) if cur["init_hour"].notna().any() else \
        (12 if "12" in str(cur["pattern"].iloc[0]) else 0)
    run_txt = (f"Volue · {family} · run {format_run(volue_init_time(ref, init_hour))} · "
               f"{int(cur['n_members'].max() or 0)} members · normal = Volue 30-yr normal")

    anom_col = "anomaly" if m["anomaly_kind"] == "diff" else "anomaly_pct"
    a_unit = m["unit"] if m["anomaly_kind"] == "diff" else "%"
    h_mean = _horizon_mean(cur, "ens_mean", 14)
    h_anom = _horizon_mean(cur, anom_col, 14)
    h_sd = _horizon_mean(cur, "spread_std", 14)
    sec_header(_METRIC_ICON.get(metric, "forecast"), f"{metric} · days 1–14", run_txt)
    tiles([_value_tile(a, metric, m, h_mean.get(a), h_anom.get(a), h_sd.get(a), a_unit) for a in areas])
    st.plotly_chart(make_anomaly_heatmap(cur, anom_col, a_unit, title=f"{metric} — daily anomaly vs normal ({a_unit})"),
                    use_container_width=True)

    sec_header("spread", "Ensemble distribution by country",
               "P10–P90 and P25–P75 bands · ensemble mean, median and normal · earlier runs faded")
    cols = st.columns(2)
    for i, a in enumerate(areas):
        with cols[i % 2]:
            fig = make_fan_chart(cur[cur["area"] == a], a, m["unit"], color=AREA_COLORS.get(a, "#2a78d6"),
                                 prev_runs=prev[prev["area"] == a] if not prev.empty else None,
                                 other_means=mm_df[mm_df["area"] == a] if not mm_df.empty else None)
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("All values — table"):
        show = cur[["area", "day", "lead_day", "ens_mean", "p10", "p25", "p50", "p75", "p90", "spread_std",
                    "ens_min", "ens_max", "n_members", "normal", "anomaly", "anomaly_pct"]].copy()
        show["area"] = show["area"].map(area_label)
        show = show.sort_values(["area", "day"])
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", show.to_csv(index=False).encode(), f"forecast_{metric}_{family}.csv",
                           "text/csv", key="fv_dl")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UNCERTAINTY
# ══════════════════════════════════════════════════════════════════════════════

def _spread_ratio(cur: pd.DataFrame, clim: pd.DataFrame, month: int) -> pd.DataFrame:
    """This run's spread vs the climatological spread at the same lead day; same-month
    baseline when it has enough runs, else all months."""
    if cur.empty or clim.empty:
        return pd.DataFrame()
    cm = clim[(clim["run_month"] == month) & (clim["n_runs"] >= 10)]
    base = cm if not cm.empty else clim[clim["run_month"] == 0]
    base = base[["area", "lead_day", "spread_std_mean", "spread_std_p25", "spread_std_p75", "n_runs"]]
    d = cur[["area", "lead_day", "day", "spread_std"]].merge(base, on=["area", "lead_day"], how="inner")
    d["ratio"] = d["spread_std"] / d["spread_std_mean"]
    d["baseline"] = "same month" if not cm.empty else "all months"
    return d


def _ratio_verdict(r: float) -> tuple[str, str]:
    if np.isnan(r):
        return "N/A", "kpi-card-neutral"
    if r >= SPREAD_RATIO_HIGH:
        return f"{r:.2f}× · high uncertainty", "kpi-card-warning"
    if r <= SPREAD_RATIO_LOW:
        return f"{r:.2f}× · unusually confident", "kpi-card-good"
    return f"{r:.2f}× · normal", "kpi-card-neutral"


def _ratio_tile(a: str, metric: str, r: float, max_lead: int) -> str:
    """Spread ratio of one country: × the normal spread on a gauge centred on 1 (orange = wider
    than usual, blue = tighter)."""
    icon = _METRIC_ICON.get(metric, "spread")
    if np.isnan(r):
        return tile(icon, area_label(a), f"spread, days 1–{max_lead}", "n/a", "", None, "flat", "no baseline")
    if r >= SPREAD_RATIO_HIGH:
        tone, pill = "warm", "more uncertain than usual"
    elif r <= SPREAD_RATIO_LOW:
        tone, pill = "cool", "more confident than usual"
    else:
        tone, pill = "flat", "a typical spread"
    return tile(icon, area_label(a), f"spread, days 1–{max_lead}", f"{r:.2f}", "× normal", (r - 1) / 0.5, tone, pill)


def _render_uncertainty(runs: pd.DataFrame):
    with keyed("tb_fu"):
        c1, c2, c3 = st.columns([1.2, 3, 1.4])
        with c1:
            metric = st.selectbox("Metric", [k for k in METRICS if k != "Precipitation energy"], key="fu_metric")
        with c2:
            areas = _area_picker("fu_areas", DEFAULT_AREAS)
        with c3:
            max_lead = st.slider("Lead days for the verdict", 3, 15, 10, key="fu_lead")
    if not areas:
        st.info("Select at least one country.")
        return
    m = METRICS[metric]
    try:
        df = load_fcst_daily(metric, tuple(areas), ("EC-ENS",), max_run_rank=1)
        clim = load_spread_clim(metric, tuple(areas))
    except Exception as e:
        st.error(f"Failed to load spread data: {e}")
        return
    if df.empty:
        status_banner("No EC-ENS rows for this selection.", "warning")
        return
    if df["n_members"].fillna(0).max() < 5:
        status_banner("Fewer than 5 member tags in the Volue ensemble tables — spread statistics need "
                      "per-member data (tag != 'Avg'). Check the tag convention in the delta share.", "critical")

    ratio = _spread_ratio(df, clim, int(pd.Timestamp.today().month))
    if ratio.empty:
        status_banner("No spread climatology yet — fcst_spread_clim is built by the refresh job from the "
                      "last 365 days of EC-ENS 00z runs.", "warning")
    else:
        sec_header("spread", f"Spread vs normal · days 1–{max_lead}",
                   f"this run's across-member σ against the last year of EC-ENS 00z runs "
                   f"({ratio['baseline'].iloc[0]} baseline) · above {SPREAD_RATIO_HIGH}× = more uncertain than usual, "
                   f"below {SPREAD_RATIO_LOW}× = more confident")
        rr = ratio[(ratio["lead_day"] >= 1) & (ratio["lead_day"] <= max_lead)].groupby("area")["ratio"].mean()
        tiles([_ratio_tile(a, metric, float(rr.get(a, np.nan)), max_lead) for a in areas])
        high = [area_label(a) for a in areas if rr.get(a, 0) >= SPREAD_RATIO_HIGH]
        low = [area_label(a) for a in areas if 0 < rr.get(a, 9) <= SPREAD_RATIO_LOW]
        if high:
            status_banner(f"Ensemble spread is unusually wide for {', '.join(high)} — the forecast carries more "
                          f"uncertainty than a typical run at these lead times.", "warning")
        if low:
            status_banner(f"Ensemble spread is unusually tight for {', '.join(low)} — higher-than-usual confidence.", "good")
        st.plotly_chart(make_spread_ratio_heatmap(ratio), use_container_width=True)

    sec_header("members", "Spread of the distribution · country detail",
               "this run's spread by lead day against the normal spread, then every member day by day")
    d1, _ = st.columns([1.4, 3])
    with d1:
        focus = st.selectbox("Country", areas, format_func=area_label, key="fu_focus")
    cur_a = df[df["area"] == focus]
    clim_a = clim[(clim["area"] == focus) & (clim["run_month"] == 0)] if not clim.empty else pd.DataFrame()
    st.plotly_chart(make_spread_vs_normal_chart(cur_a, clim_a, focus, m["unit"]), use_container_width=True)

    try:
        mem = load_members("Volue", "EC-ENS", metric, (focus,))
    except Exception:
        mem = pd.DataFrame()
    if mem.empty:
        st.caption("No per-member rows for this country (fcst_members).")
        return
    e1, e2 = st.columns([2, 1.2])
    with e1:
        st.plotly_chart(make_member_strips(mem, focus, m["unit"]), use_container_width=True)
    with e2:
        days = sorted(mem["day"].dropna().unique())
        pick = st.select_slider("Day", options=days, value=days[min(4, len(days) - 1)],
                                format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b"), key="fu_day")
        dd = mem[mem["day"] == pick]
        nrm = float(dd["normal"].mean()) if dd["normal"].notna().any() else None
        st.plotly_chart(make_member_distribution(dd, focus, m["unit"], nrm, pd.Timestamp(pick).strftime("%a %d %b")),
                        use_container_width=True)
        if not dd.empty:
            st.caption(f"{len(dd)} members · σ {dd['value'].std():.2f} {m['unit']} · P10–P90 "
                       f"{dd['value'].quantile(0.1):.1f} to {dd['value'].quantile(0.9):.1f} {m['unit']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SCENARIOS (Z500 field clustering, two scenarios, side by side per country)
# ══════════════════════════════════════════════════════════════════════════════
# The engine below is shared with the Morning Call's country deep dive
# (_deep_dive.py), which runs it with the tab's defaults.

_UNITS = {"Temperature": "°C", "Wind": "GW", "Solar": "GW"}


def spatial_source_options() -> dict[str, dict]:
    """The sources of SCENARIO_SPATIAL_SOURCES whose Z500 member field is in the
    sandbox, with the run they hold: key -> cfg + n_members, n_points,
    reference_date, init (the cycle), first_day, last_day."""
    src = load_spatial_sources()
    out: dict[str, dict] = {}
    for key, cfg in SCENARIO_SPATIAL_SOURCES.items():
        hit = src[src["source"] == key] if not src.empty else pd.DataFrame()
        if hit.empty:
            continue
        r = hit.iloc[0]
        ref = pd.Timestamp(r["reference_date"])
        ref_naive = ref.tz_convert("UTC").tz_localize(None) if ref.tzinfo is not None else ref
        init = ref_naive if cfg["run_is_init"] else snap_to_init_time(ref_naive, cfg["init_hours"])
        out[key] = {**cfg, "key": key, "n_members": int(r["n_members"]), "n_points": int(r["n_points"]),
                    "reference_date": ref_naive, "init": init,
                    "first_day": pd.Timestamp(r["first_day"]), "last_day": pd.Timestamp(r["last_day"])}
    return out


def spatial_weeks(info: dict) -> pd.DataFrame:
    """The trading weeks a source's run covers (available_weeks over its day range)."""
    days = pd.DataFrame({"day": pd.date_range(info["first_day"], info["last_day"], freq="D")})
    return available_weeks(days)


def land_members(info: dict, metric: str, areas: tuple[str, ...]) -> pd.DataFrame:
    """The member rows the scenarios are read on for this source (its `land` entry)."""
    try:
        return load_members(info["land"]["provider"], info["land"]["model"], metric, tuple(areas))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def cluster_spatial(source: str, weeks: tuple[str, ...], k: int, ref_iso: str) -> dict | None:
    """Cluster one source's members on their weekly-mean Z500 field over the given
    trading weeks (ISO dates of their Mondays), k fixed. `ref_iso` only keys the cache
    on the run. Returns the clustering, the PCA diagnostics, the centroid maps and the
    cluster names, or None when the field is not there."""
    grid_by_week = {}
    for w in weeks:
        ws = pd.Timestamp(w).normalize()
        g = load_member_spatial_days(source, ws, ws + pd.Timedelta(days=6))
        if not g.empty:
            grid_by_week[ws] = g
    if not grid_by_week:
        return None
    features, grid = spatial_member_matrix(grid_by_week)
    if features.empty or features.shape[0] < 6:
        return None
    res = cluster_members(features, int(k), scale=False)
    if not res:
        return None
    cmaps = centroid_maps(grid, res["assignments"])
    names = {int(s): name_cluster(cmaps[cmaps["scenario"] == s]) for s in res["sizes"] if s != 0}
    return {
        "assign": res["assignments"], "sizes": res["sizes"], "n_members": res["n_members"], "k": res["k"],
        "silhouette": res["silhouette"], "features": features, "grid_points": int(features.attrs.get("n_grid_points", 0)),
        "explained_pct": float(features.attrs.get("explained_variance_pct", 0.0)), "n_pcs": int(features.shape[1]),
        "weeks": [pd.Timestamp(w).normalize() for w in grid_by_week], "cmaps": cmaps, "names": names,
    }


def default_scenarios(source_key: str = SCENARIO_DEFAULT_SOURCE):
    """The Scenarios tab with its defaults — the default source (or the first one
    available), the first two usable trading weeks (this week and the next), each
    clustered on its own, k = SCENARIO_K_DEFAULT.
    Returns (results: {week_start: result}, source info, weeks selected, all weeks);
    results is empty when nothing can be clustered."""
    opts = spatial_source_options()
    info = opts.get(source_key) or (next(iter(opts.values())) if opts else None)
    if not info:
        return {}, None, pd.DataFrame(), pd.DataFrame()
    weeks_all = spatial_weeks(info)
    usable = weeks_all[weeks_all["usable"]]
    if usable.empty:
        return {}, info, pd.DataFrame(), weeks_all
    weeks = default_weeks(usable)
    results = {}
    for ws in weeks:
        res = cluster_spatial(info["key"], (str(pd.Timestamp(ws).date()),), SCENARIO_K_DEFAULT, str(info["reference_date"]))
        if res:
            results[pd.Timestamp(ws)] = res
    return results, info, usable[usable["week_start"].isin(weeks)].sort_values("week_start"), weeks_all


def scenario_ids(res: dict) -> list[int]:
    return sorted(s for s in res["sizes"] if s != 0) + ([0] if 0 in res["sizes"] else [])


def scenario_cards(res: dict, info: dict, weeks: list[pd.Timestamp]) -> list[str]:
    """One card per scenario of one week: 'Scenario 1 · 55 % probability · Scandi Block'
    and, under it, the one-line read of North-West Europe against the ensemble ('Cold
    and low wind in NWE', the opposite, or flat) from the land temperature and wind
    members of that week."""
    temp_m = land_members(info, "Temperature", tuple(SCENARIO_NWE_AREAS))
    wind_m = land_members(info, "Wind", tuple(SCENARIO_NWE_AREAS))
    tagged_t = apply_scenarios(temp_m, res["assign"]) if not temp_m.empty else pd.DataFrame()
    tagged_w = apply_scenarios(wind_m, res["assign"]) if not wind_m.empty else pd.DataFrame()
    cards = []
    n_tot = res["n_members"]
    for s in scenario_ids(res):
        share = res["sizes"][s] / n_tot
        if s == 0:
            title = f"Other · {share:.0%} · clusters too small to keep"
            desc = ""
        else:
            label = short_name(res["names"].get(s))
            title = f"Scenario {s} · {share:.0%} probability" + (f" · {label}" if label else "")
            desc = describe_scenario(nwe_read(tagged_t, tagged_w, s, weeks))
        cards.append(
            f'<div class="kpi-card" style="border-top:3px solid {scenario_color(s)}; text-align:left;">'
            f'<div class="kpi-label" style="font-size:0.95rem;color:var(--text-primary) !important;font-weight:700;'
            f'text-transform:none;letter-spacing:-0.01em;">{title}</div>'
            f'<div class="kpi-delta kpi-delta-flat">{desc or "&nbsp;"}</div></div>')
    return cards


def week_title(wk: pd.Series) -> str:
    ws = pd.Timestamp(wk["week_start"])
    t = f"{wk['week_label']} · {ws:%d %b}–{ws + pd.Timedelta(days=6):%d %b}"
    return t if wk.get("full", True) else t + f" · {int(wk['n_days'])} forecast days"


def week_caption(info: dict, res: dict) -> str:
    return (f"{info['label']} · run {pd.Timestamp(info['init']):%a %d %b %Hz} · {res['n_members']} members clustered on "
            f"their weekly-mean Z500 field of this week ({res['grid_points']} grid points, {SPATIAL_GRID_RES:g}°, PCA "
            f"{res['n_pcs']} PCs, {res['explained_pct']:.0f}% var) · k={res['k']} fixed · silhouette {res['silhouette']:.2f} "
            f"(diagnostic) · temperature, wind and solar read on the {info['land']['label']} of the same member IDs")


def _fmt(v, fmt="{:.1f}") -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def _fmt_d(v, fmt="{:+.1f}") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<span class="mc-muted">n/a</span>'
    cls = "mc-pos" if round(v, 1) > 0 else ("mc-neg" if round(v, 1) < 0 else "mc-zero")
    return f'<span class="{cls}">{fmt.format(v)}</span>'


def _week_table_html(title: str, weekly: dict[str, pd.DataFrame], scen_ids: list[int], week_start: pd.Timestamp,
                     sizes: dict[int, int], n_tot: int, names: dict | None, areas: list[str],
                     area_names: dict[str, str] | None = None) -> str:
    """One week, every country: rows = country × metric, per scenario three cells
    (value, Δ vs ensemble mean, Δ vs normal) — the scenarios side by side."""
    head1 = "<tr><th></th>"
    head2 = "<tr><th>country · metric</th>"
    for s in scen_ids:
        col = scenario_color(s)
        head1 += (f'<th class="sc-scen" colspan="3" style="border-color:{col}; text-align:center;">'
                  f'{scenario_name(s, names)} <span class="mc-muted">({sizes.get(s, 0)}/{n_tot})</span></th>')
        head2 += "<th>value</th><th>Δ ens</th><th>Δ norm</th>"
    head1 += "</tr>"; head2 += "</tr>"
    ws = pd.Timestamp(week_start).normalize()
    body = ""
    for a in areas:
        first = True
        for metric in ("Temperature", "Wind", "Solar"):
            w = weekly.get(metric)
            if w is None or w.empty:
                continue
            row = w[(w["area"] == a) & (w["week_start"] == ws)]
            if row.empty:
                continue
            label = ((area_names or {}).get(a, area_label(a)) if first else "") + \
                f" <span class='mc-muted'>{metric} ({_UNITS[metric]})</span>"
            first = False
            body += f"<tr><td>{label}</td>"
            for s in scen_ids:
                r = row[row["scenario"] == s]
                if r.empty:
                    body += "<td colspan='3' class='mc-muted'>–</td>"
                    continue
                r = r.iloc[0]
                body += f"<td>{_fmt(r['mean'])}</td><td>{_fmt_d(r['d_ens'])}</td><td>{_fmt_d(r['d_norm'])}</td>"
            body += "</tr>"
    return f'<div class="mc-block"><div class="mc-title">{title}</div><table class="sc-table">{head1}{head2}{body}</table></div>'


def _side_by_side_html(area: str, weekly: dict[str, pd.DataFrame], scen_ids: list[int],
                       weeks: pd.DataFrame, sizes: dict[int, int], n_tot: int, names: dict | None = None) -> str:
    """One country, one week's scenarios: rows = metric, per scenario value, Δ vs
    ensemble mean, Δ vs normal (the deep dive's table)."""
    wk = weeks.iloc[0]
    return _week_table_html(f"{area_label(area)} · {week_title(wk)}", weekly, scen_ids, wk["week_start"], sizes, n_tot, names,
                            [area], {area: area_label(area)})


def _week_days(tagged: dict[str, pd.DataFrame], ws: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """The daily member rows of one trading week per metric."""
    ws = pd.Timestamp(ws).normalize()
    return {m: t[t["week_start"] == ws] for m, t in tagged.items() if not t.empty}


def _render_week(info: dict, res: dict, wk: pd.Series, tagged: dict[str, pd.DataFrame], show_areas: list[str],
                 key: str) -> None:
    """One trading week's scenarios: the cards, every country side by side, the daily
    paths per country in an expander, the diagnostics in another."""
    ws = pd.Timestamp(wk["week_start"]).normalize()
    scen_ids = scenario_ids(res)
    sec_header("scen", f"Scenarios · {week_title(wk)}", week_caption(info, res))
    kpi_row(scenario_cards(res, info, [ws]), max_cols=4)
    tagged_w = {m: apply_scenarios(t, res["assign"]) for m, t in tagged.items() if not t.empty}
    if not tagged_w or all(t.empty for t in tagged_w.values()):
        status_banner(f"No {info['land']['label']} rows for temperature, wind and solar yet — the scenarios are shown "
                      "on the field alone." + (" Run the pipeline's eq-members step (run_scenarios.bat)."
                                               if info["land"]["provider"] == "EQ" else " Run power_desk_refresh.py."),
                      "warning")
        return
    weekly = {m: scenario_weekly_values(t, [ws], "value") for m, t in tagged_w.items()}
    if show_areas:
        st.markdown(_week_table_html(f"Countries · {wk['week_label']}", weekly, scen_ids, ws, res["sizes"], res["n_members"],
                                     res["names"], show_areas), unsafe_allow_html=True)
        with st.expander(f"Daily paths by country · {wk['week_label']}"):
            daily = {m: scenario_daily_summary(t, "value") for m, t in _week_days(tagged_w, ws).items() if not t.empty}
            for a in show_areas:
                st.plotly_chart(make_country_scenario_panel(daily, a, _UNITS, [ws]), use_container_width=True,
                                key=f"{key}_panel_{a}")
    with st.expander(f"Member → scenario map · cluster names · separability · {wk['week_label']}"):
        d1, d2, d3 = st.columns([1, 1.6, 1.2])
        with d1:
            st.dataframe(res["assign"].sort_values(["scenario", "member_id"]).rename(
                columns={"member_id": "member", "scenario": "scenario id"})[["member", "scenario id"]],
                use_container_width=True, hide_index=True, height=260, key=f"{key}_map")
        with d2:
            rows = []
            for s, nm in res["names"].items():
                rows.append({"scenario": s, "label": short_name(nm), "centroid": nm["name"],
                             "ridge": f"{nm['ridge']['region']} ({nm['ridge']['value']:+.0f} m)" if nm.get("ridge") else "",
                             "trough": f"{nm['trough']['region']} ({nm['trough']['value']:+.0f} m)" if nm.get("trough") else "",
                             "closest regime": (f"{nm['regime_label']} (r {nm['r']:.2f})" if nm.get("regime")
                                                else (f"none above r {nm.get('r', np.nan):.2f}" if nm.get("similarity")
                                                      else "patterns not available"))})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, key=f"{key}_names")
            st.caption("Labels: the Weather Regimes tab's pattern the centroid correlates with best, when that correlation "
                       "is high enough; else where the centroid's highest and lowest Z500 departure sit.")
        with d3:
            sil = silhouette_for_k(res["features"], ks=range(2, SCENARIO_K_MAX + 1))
            if sil:
                st.dataframe(pd.DataFrame({"k": list(sil.keys()), "silhouette": [round(v, 3) for v in sil.values()]}),
                             use_container_width=True, hide_index=True, key=f"{key}_sil")
                st.caption("Diagnostic only: k is fixed by the desk (2 = scenario + alternative), not chosen by silhouette.")


def _render_scenarios(runs: pd.DataFrame):
    try:
        opts = spatial_source_options()
    except Exception as e:
        st.error(f"Failed to read the Z500 member tables: {e}")
        return
    if not opts:
        status_banner("No Z500 member field in the sandbox yet. EC-ENS: run the pipeline's z500 step "
                      "(pipeline/run_scenarios.bat → fcst_member_spatial_ecens); EC-AI: run power_desk_refresh.py "
                      "(cell 4b → fcst_member_spatial).", "critical")
        return
    missing = [k for k in SCENARIO_SPATIAL_SOURCES if k not in opts]
    if missing:
        status_banner(f"{', '.join(SCENARIO_SPATIAL_SOURCES[k]['label'] for k in missing)}: no Z500 member field yet "
                      f"({', '.join(SCENARIO_SPATIAL_SOURCES[k]['table'] for k in missing)} is empty or carries "
                      f"another field).", "warning")

    keys = list(opts.keys())
    with keyed("tb_fs"):
        c1, c2, c3, c4 = st.columns([1.6, 3.0, 1.0, 2.2])
        with c1:
            source = st.selectbox("Ensemble", keys, index=keys.index(SCENARIO_DEFAULT_SOURCE) if SCENARIO_DEFAULT_SOURCE in keys else 0,
                                  format_func=lambda k: opts[k]["label"], key="fs_source",
                                  help="Each trading week's members are clustered on their weekly-mean 500 hPa geopotential "
                                       "height field of that week; temperature, wind and solar are then read per scenario on "
                                       "the same member IDs.")
        info = opts[source]
        weeks_all = spatial_weeks(info)
        usable = weeks_all[weeks_all["usable"]]
        if usable.empty:
            status_banner("The run does not cover enough days of any trading week to form weekly fields.", "warning")
            return
        wk_idx = usable.set_index("week_start")
        with c2:
            weeks_pick = st.multiselect(
                "Trading weeks (Mon–Sun), each clustered on its own", list(usable["week_start"]), default_weeks(usable),
                key=f"fs_weeks_{source}",
                format_func=lambda w: f"{wk_idx.loc[w, 'week_label']}  {pd.Timestamp(w):%d %b}–{pd.Timestamp(w) + pd.Timedelta(days=6):%d %b}"
                                      f"{'' if wk_idx.loc[w, 'full'] else '  (partial, %d d)' % wk_idx.loc[w, 'n_days']}")
        with c3:
            k = st.number_input("Scenarios", 2, SCENARIO_K_MAX, SCENARIO_K_DEFAULT, key="fs_k")
        with c4:
            out_areas = st.multiselect("Countries shown", list(AREAS.keys()),
                                       [a for a in SCENARIO_OUTPUT_AREAS if a in AREAS], format_func=area_label, key="fs_out")
    if not weeks_pick:
        st.info("Pick at least one week.")
        return
    weeks_sel = usable[usable["week_start"].isin(weeks_pick)].sort_values("week_start")
    show_areas = [a for a in out_areas if a in AREAS]

    # the land members once, for every week (cached loads)
    tagged_src: dict[str, pd.DataFrame] = {}
    for metric in ("Temperature", "Wind", "Solar"):
        pm = land_members(info, metric, tuple(show_areas))
        if not pm.empty:
            tagged_src[metric] = pm

    for i, (_, wk) in enumerate(weeks_sel.iterrows()):
        try:
            res = cluster_spatial(source, (str(pd.Timestamp(wk["week_start"]).date()),), int(k), str(info["reference_date"]))
        except Exception as e:
            st.error(f"{wk['week_label']}: failed to load the Z500 member field: {e}")
            continue
        if not res:
            status_banner(f"{wk['week_label']}: not enough complete members in the Z500 field.", "warning")
            continue
        _render_week(info, res, wk, tagged_src, show_areas, key=f"fs_{source}_{wk['week_label']}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def render_forecast():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    with keyed("pg_forecast"):
        page_header("forecast", "Section 01", "Forecast",
                    "Volue ensembles by country · spread against the normal spread · scenarios from the Z500 field · "
                    "European weather regimes")
        # Weather Regimes reads its own table, so a failure of the Volue run list
        # must not take it down with the other three tabs.
        runs, runs_error = None, None
        try:
            runs = load_runs()
        except Exception as e:
            runs_error = f"Cannot read the run list ({e}). Has power_desk_refresh.py run?"

        tabs = st.tabs(["Values", "Uncertainty", "Scenarios", "Weather Regimes"])
        for tab, render in zip(tabs[:3], (_render_values, _render_uncertainty, _render_scenarios)):
            with tab:
                if runs_error:
                    st.error(runs_error)
                else:
                    render(runs)
        with tabs[3]:
            render_regimes()
