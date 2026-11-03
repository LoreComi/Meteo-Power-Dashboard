"""Section 1 — Forecast.

Three tabs:
  Values        every Volue ensemble value by country (mean, median, P10-P90),
                normal, earlier runs, Meteomatics EC-ENS / AIFS-ENS means
  Uncertainty   spread of the ensemble distribution and how it compares with
                the normal spread for the same lead day — is this forecast
                more or less certain than usual?
  Scenarios     weather scenarios from k-means on the members' WEEKLY-MEAN
                500 hPa geopotential height field over Europe and the near
                Atlantic, over real Mon–Sun trading weeks, for EC-ENS (Z500
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
    SPATIAL_N_PCA, SPATIAL_GRID_RES, SPATIAL_BBOX,
    area_label,
)
from _data import (
    load_runs, load_fcst_daily, load_spread_clim, load_members, load_spatial_sources, load_member_spatial_days,
    load_meteologica_members, format_run, volue_init_time, snap_to_init_time,
)
from _charts import (
    AREA_COLORS, make_fan_chart, make_anomaly_heatmap,
    make_spread_vs_normal_chart, make_spread_ratio_heatmap, make_member_distribution, make_member_strips,
    make_scenario_table_heatmap, make_scenario_centroid_maps, make_country_scenario_panel, scenario_color,
)
from _scenarios import (
    available_weeks, default_weeks, spatial_member_matrix, cluster_members, apply_scenarios, centroid_maps,
    name_cluster, scenario_daily_summary, scenario_weekly_values, scenario_table, scenario_name, describe_scenario,
    temperature_read, silhouette_for_k,
)
from _regimes import render_regimes
from _ui import anomaly_kpi, kpi_card, kpi_row, status_banner


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

def _render_values(runs: pd.DataFrame):
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
    st.caption(f"Volue · {family} · run {format_run(volue_init_time(ref, init_hour))} · "
               f"{int(cur['n_members'].max() or 0)} members · normal = Volue 30-yr normal")

    anom_col = "anomaly" if m["anomaly_kind"] == "diff" else "anomaly_pct"
    a_unit = m["unit"] if m["anomaly_kind"] == "diff" else "%"
    h_mean = _horizon_mean(cur, "ens_mean", 14)
    h_anom = _horizon_mean(cur, anom_col, 14)
    a_suffix = "%" if a_unit == "%" else f" {m['unit']}"
    kpi_row([anomaly_kpi(f"{area_label(a)} · d1–14 mean", h_mean.get(a), m["unit"], h_anom.get(a),
                         anomaly_unit=a_suffix, warm_is_positive=m["warm_is_positive"]) for a in areas])

    st.divider()
    st.plotly_chart(make_anomaly_heatmap(cur, anom_col, a_unit, title=f"{metric} — daily anomaly vs normal ({a_unit})"),
                    use_container_width=True)

    st.markdown("##### Ensemble distribution by country")
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


def _render_uncertainty(runs: pd.DataFrame):
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
        st.caption(f"Baseline: mean across-member σ per lead day from the last year of EC-ENS 00z runs "
                   f"({ratio['baseline'].iloc[0]} baseline). Ratio > {SPREAD_RATIO_HIGH} = more uncertain than "
                   f"usual; < {SPREAD_RATIO_LOW} = more confident than usual.")
        rr = ratio[(ratio["lead_day"] >= 1) & (ratio["lead_day"] <= max_lead)].groupby("area")["ratio"].mean()
        cards = []
        for a in areas:
            txt, cls = _ratio_verdict(float(rr.get(a, np.nan)))
            cards.append(kpi_card(f"{area_label(a)} · spread d1–{max_lead}", txt, cls))
        kpi_row(cards)
        high = [area_label(a) for a in areas if rr.get(a, 0) >= SPREAD_RATIO_HIGH]
        low = [area_label(a) for a in areas if 0 < rr.get(a, 9) <= SPREAD_RATIO_LOW]
        if high:
            status_banner(f"Ensemble spread is unusually wide for {', '.join(high)} — the forecast carries more "
                          f"uncertainty than a typical run at these lead times.", "warning")
        if low:
            status_banner(f"Ensemble spread is unusually tight for {', '.join(low)} — higher-than-usual confidence.", "good")
        st.divider()
        st.plotly_chart(make_spread_ratio_heatmap(ratio), use_container_width=True)

    st.markdown("##### Spread of the distribution — country detail")
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
    available), the next two usable trading weeks, k = SCENARIO_K_DEFAULT.
    Returns (result, source info, weeks selected, all weeks); result None when nothing
    can be clustered."""
    opts = spatial_source_options()
    info = opts.get(source_key) or (next(iter(opts.values())) if opts else None)
    if not info:
        return None, None, pd.DataFrame(), pd.DataFrame()
    weeks_all = spatial_weeks(info)
    usable = weeks_all[weeks_all["usable"]]
    if usable.empty:
        return None, info, pd.DataFrame(), weeks_all
    weeks = default_weeks(usable)
    res = cluster_spatial(info["key"], tuple(str(pd.Timestamp(w).date()) for w in weeks), SCENARIO_K_DEFAULT,
                          str(info["reference_date"]))
    return res, info, usable[usable["week_start"].isin(weeks)].sort_values("week_start"), weeks_all


def scenario_ids(res: dict) -> list[int]:
    return sorted(s for s in res["sizes"] if s != 0) + ([0] if 0 in res["sizes"] else [])


def scenario_cards(res: dict, tagged_t: pd.DataFrame, weeks: list[pd.Timestamp]) -> list[str]:
    """One KPI card per scenario: its name, its member count, the regime it resembles
    and how it reads in temperature."""
    cards = []
    n_tot = res["n_members"]
    for s in scenario_ids(res):
        share = res["sizes"][s] / n_tot
        if s == 0:
            desc = f"{share:.0%} of members in clusters too small to keep"
        else:
            desc = describe_scenario(share, temperature_read(tagged_t, s, weeks), res["names"].get(s))
        cards.append(
            f'<div class="kpi-card" style="border-top:3px solid {scenario_color(s)}; text-align:left;">'
            f'<div class="kpi-label">{scenario_name(s, res["names"])}</div>'
            f'<div class="kpi-value">{res["sizes"][s]} <span style="font-size:0.9rem;color:var(--text-muted)">members</span></div>'
            f'<div class="kpi-delta kpi-delta-flat">{desc}</div></div>')
    return cards


def scenario_run_caption(info: dict, res: dict, weeks_sel: pd.DataFrame) -> str:
    return (f"{info['label']} · run {pd.Timestamp(info['init']):%a %d %b %Hz} · {res['n_members']} members · "
            f"weekly-mean Z500 field, {res['grid_points']} grid points ({SPATIAL_GRID_RES:g}°, "
            f"{SPATIAL_BBOX['lat_min']}–{SPATIAL_BBOX['lat_max']} N, {abs(SPATIAL_BBOX['lon_min'])} W–{SPATIAL_BBOX['lon_max']} E) × "
            f"{len(weeks_sel)} week(s) ({', '.join(weeks_sel['week_label'])}) · PCA {res['n_pcs']} PCs "
            f"({res['explained_pct']:.0f}% var) · k={res['k']} fixed · silhouette {res['silhouette']:.2f} (diagnostic) · "
            f"temperature, wind and solar read on the {info['land']['label']} of the same member IDs")


def centroid_titles(res: dict) -> dict[int, str]:
    out = {}
    for s, nm in res["names"].items():
        t = f"Scenario {s} ({res['sizes'][s]}/{res['n_members']}) — {nm['name']}"
        if nm.get("regime"):
            t += f"\n≈ {nm['regime_label']} (r {nm['r']:.2f})"
        out[s] = t
    return out


def _fmt(v, fmt="{:.1f}") -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def _fmt_d(v, fmt="{:+.1f}") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<span class="mc-muted">n/a</span>'
    cls = "mc-pos" if round(v, 1) > 0 else ("mc-neg" if round(v, 1) < 0 else "mc-zero")
    return f'<span class="{cls}">{fmt.format(v)}</span>'


def _side_by_side_html(area: str, weekly: dict[str, pd.DataFrame], scen_ids: list[int],
                       weeks: pd.DataFrame, sizes: dict[int, int], n_tot: int, names: dict | None = None) -> str:
    """Rows = metric × week; per scenario three cells: value, Δ vs ensemble mean, Δ vs normal."""
    head1 = "<tr><th></th>"
    head2 = "<tr><th>metric · week</th>"
    for s in scen_ids:
        col = scenario_color(s)
        head1 += (f'<th class="sc-scen" colspan="3" style="border-color:{col}; text-align:center;">'
                  f'{scenario_name(s, names)} <span class="mc-muted">({sizes.get(s, 0)}/{n_tot})</span></th>')
        head2 += "<th>value</th><th>Δ ens</th><th>Δ norm</th>"
    head1 += "</tr>"; head2 += "</tr>"
    body = ""
    for metric in ("Temperature", "Wind", "Solar"):
        w = weekly.get(metric)
        if w is None or w.empty:
            continue
        wa = w[w["area"] == area]
        for _, wk in weeks.iterrows():
            ws = wk["week_start"]
            row = wa[wa["week_start"] == ws]
            if row.empty:
                continue
            label = f"{metric} · {wk['week_label']} <span class='mc-muted'>({_UNITS[metric]})</span>"
            body += f"<tr><td>{label}</td>"
            for s in scen_ids:
                r = row[row["scenario"] == s]
                if r.empty:
                    body += "<td colspan='3' class='mc-muted'>–</td>"
                    continue
                r = r.iloc[0]
                body += f"<td>{_fmt(r['mean'])}</td><td>{_fmt_d(r['d_ens'])}</td><td>{_fmt_d(r['d_norm'])}</td>"
            body += "</tr>"
    return f'<div class="mc-block"><div class="mc-title">{area_label(area)}</div><table class="sc-table">{head1}{head2}{body}</table></div>'


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
    c1, c2, c3, c4 = st.columns([1.6, 3.0, 1.0, 2.2])
    with c1:
        source = st.selectbox("Ensemble", keys, index=keys.index(SCENARIO_DEFAULT_SOURCE) if SCENARIO_DEFAULT_SOURCE in keys else 0,
                              format_func=lambda k: opts[k]["label"], key="fs_source",
                              help="Clustered on the members' weekly-mean 500 hPa geopotential height field; temperature, "
                                   "wind and solar are then read per scenario on the same member IDs.")
    info = opts[source]
    weeks_all = spatial_weeks(info)
    usable = weeks_all[weeks_all["usable"]]
    if usable.empty:
        status_banner("The run does not cover enough days of any trading week to form weekly fields.", "warning")
        return
    wk_idx = usable.set_index("week_start")
    with c2:
        weeks_pick = st.multiselect(
            "Trading weeks to cluster on (Mon–Sun)", list(usable["week_start"]), default_weeks(usable), key=f"fs_weeks_{source}",
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

    try:
        res = cluster_spatial(source, tuple(str(pd.Timestamp(w).date()) for w in weeks_sel["week_start"]), int(k),
                              str(info["reference_date"]))
    except Exception as e:
        st.error(f"Failed to load the Z500 member field: {e}")
        return
    if not res:
        status_banner("Not enough complete members in the Z500 field for these weeks.", "warning")
        return
    weeks = list(weeks_sel["week_start"])
    scen_ids = scenario_ids(res)
    st.caption(scenario_run_caption(info, res, weeks_sel))

    show_areas = [a for a in out_areas if a in AREAS]
    temp_m = land_members(info, "Temperature", tuple(set(show_areas) | set(SCENARIO_OUTPUT_AREAS)))
    tagged_t = apply_scenarios(temp_m, res["assign"]) if not temp_m.empty else pd.DataFrame()
    if temp_m.empty:
        status_banner(f"No {info['land']['label']} rows for temperature, wind and solar yet — the scenarios are shown "
                      "on the field alone." + (" Run the pipeline's eq-members step (run_scenarios.bat)."
                                               if info["land"]["provider"] == "EQ" else " Run power_desk_refresh.py."),
                      "warning")
    kpi_row(scenario_cards(res, tagged_t, weeks), max_cols=4)

    # ── the scenarios side by side: the centroid maps ──
    st.markdown("##### The scenarios — mean Z500 departure from the ensemble mean over the clustered weeks")
    st.caption("Red = higher 500 hPa heights than the ensemble mean (a ridge), blue = lower (a trough). The names "
               "read the ridge and the trough off these maps; ≈ a regime when the map correlates with one of the "
               "seven North-Atlantic regimes of the Weather Regimes tab.")
    try:
        st.pyplot(make_scenario_centroid_maps(res["cmaps"], centroid_titles(res), scen_ids, bbox=SPATIAL_BBOX),
                  use_container_width=True)
    except Exception as e:
        st.warning(f"Centroid maps could not be drawn: {e}")

    if not tagged_t.empty:
        st.plotly_chart(make_scenario_table_heatmap(
            scenario_table(tagged_t, weeks, "anomaly", res["names"]), "°C",
            f"Temperature anomaly by scenario — mean over {', '.join(weeks_sel['week_label'])}"), use_container_width=True)

    # ── per country: the scenarios side by side for temperature, wind, solar ──
    st.divider()
    st.markdown("##### Per country — scenarios side by side: temperature, wind, solar")
    st.caption(f"Weekly means over the clustered trading weeks, read on the {info['land']['label']} with the same "
               "member IDs as the Z500 clusters. Δ ens = vs full-ensemble mean · Δ norm = vs the normal.")
    if not show_areas:
        st.info("Select at least one country to show.")
        return
    tagged: dict[str, pd.DataFrame] = {"Temperature": tagged_t[tagged_t["area"].isin(show_areas)] if not tagged_t.empty else pd.DataFrame()}
    for metric in ("Wind", "Solar"):
        pm = land_members(info, metric, tuple(show_areas))
        tagged[metric] = apply_scenarios(pm, res["assign"]) if not pm.empty else pd.DataFrame()
    if all(t.empty for t in tagged.values()):
        return
    weekly = {m: scenario_weekly_values(t, weeks, "value") for m, t in tagged.items() if not t.empty}
    daily = {m: scenario_daily_summary(t, "value") for m, t in tagged.items() if not t.empty}
    monday_lines = list(weeks_all["week_start"])

    for a in show_areas:
        st.markdown(_side_by_side_html(a, weekly, scen_ids, weeks_sel, res["sizes"], res["n_members"], res["names"]),
                    unsafe_allow_html=True)
        st.plotly_chart(make_country_scenario_panel(daily, a, _UNITS, monday_lines), use_container_width=True)

    with st.expander("Meteologica members (same scenario IDs)"):
        found = False
        for metric in ("Wind", "Solar", "Hydro", "Temperature"):
            mlm = load_meteologica_members(metric, tuple(show_areas))
            if mlm.empty:
                continue
            found = True
            w = scenario_weekly_values(apply_scenarios(mlm, res["assign"]), weeks, "value")
            st.markdown(f"**{metric}**")
            st.dataframe(w[["area", "week_label", "scenario", "mean", "ens_mean", "d_ens", "n"]].round(2),
                         use_container_width=True, hide_index=True)
        if not found:
            st.caption("Meteologica is not ingested into Databricks yet — populate meteologica_members and this fills in.")

    with st.expander("Member → scenario map · cluster names · separability diagnostic"):
        d1, d2, d3 = st.columns([1, 1.6, 1.2])
        with d1:
            st.dataframe(res["assign"].sort_values(["scenario", "member_id"]).rename(
                columns={"member_id": "member", "scenario": "scenario id"})[["member", "scenario id"]],
                use_container_width=True, hide_index=True, height=260)
        with d2:
            rows = []
            for s, nm in res["names"].items():
                rows.append({"scenario": s, "name": nm["name"],
                             "ridge": f"{nm['ridge']['region']} ({nm['ridge']['value']:+.0f} m)" if nm.get("ridge") else "",
                             "trough": f"{nm['trough']['region']} ({nm['trough']['value']:+.0f} m)" if nm.get("trough") else "",
                             "closest regime": (f"{nm['regime_label']} (r {nm['r']:.2f})" if nm.get("regime")
                                                else (f"none above r {res['names'][s].get('r', np.nan):.2f}"
                                                      if nm.get("similarity") else "patterns not available"))})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Names: where the centroid's highest and lowest Z500 departure sit (named regions), and the "
                       "Weather Regimes tab's pattern it correlates with best, when that correlation is high enough.")
        with d3:
            sil = silhouette_for_k(res["features"], ks=range(2, SCENARIO_K_MAX + 1))
            if sil:
                st.dataframe(pd.DataFrame({"k": list(sil.keys()), "silhouette": [round(v, 3) for v in sil.values()]}),
                             use_container_width=True, hide_index=True)
                st.caption("Diagnostic only: k is fixed by the desk (2 = scenario + alternative), not chosen by silhouette.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def render_forecast():
    st.markdown("#### FORECAST")
    st.caption("Volue ensemble forecasts by country · distribution spread vs normal · member scenarios from the "
               "Z500 field · European weather regimes. "
               "Source: Volue delta share, Meteomatics silver layer and API, via power_desk_refresh.py and the pipeline.")
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
