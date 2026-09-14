"""Section 1 — Forecast.

Three tabs:
  Values        every Volue ensemble value by country (mean, median, P10-P90),
                normal, earlier runs, Meteomatics EC-ENS / AIFS-ENS means
  Uncertainty   spread of the ensemble distribution and how it compares with
                the normal spread for the same lead day — is this forecast
                more or less certain than usual?
  Scenarios     k-means clustering of ensemble members (Meteomatics EC-ENS /
                AIFS-ENS members where available, else Volue EC-ENS members),
                then the same member IDs read off Volue and Meteologica to
                give each scenario its temperature / wind / solar values
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    AREAS, DEFAULT_AREAS, METRICS, DEFAULT_METRIC, VOLUE_MODELS, DEFAULT_MODEL,
    METEOMATICS_MODELS, SPREAD_RATIO_HIGH, SPREAD_RATIO_LOW, SCENARIO_SOURCES,
    SCENARIO_DEFAULT_AREAS, SCENARIO_DEFAULT_HORIZON, SCENARIO_K_RANGE, METEOMATICS_MEMBER_TABLE,
    area_label,
)
from _data import (
    load_runs, load_fcst_daily, load_spread_clim, load_member_sources, load_members,
    load_meteologica_members, format_run,
)
from _charts import (
    AREA_COLORS, make_fan_chart, make_anomaly_heatmap, make_multi_area_lines,
    make_spread_vs_normal_chart, make_spread_ratio_heatmap, make_member_distribution, make_member_strips,
    make_scenario_lines, make_scenario_table_heatmap, make_silhouette_chart, scenario_color,
)
from _scenarios import (
    member_matrix, cluster_members, apply_scenarios, scenario_daily_summary, scenario_table,
    scenario_name, describe_scenario,
)
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
        areas = _area_picker("fv_areas", DEFAULT_AREAS)
    fam_runs = _runs_for(runs, family)
    with c4:
        run_opts = fam_runs["run_display"].tolist() if not fam_runs.empty else ["Latest"]
        run_pick = st.selectbox("Run", run_opts, key="fv_run")
    run_rank = int(fam_runs.loc[fam_runs["run_display"] == run_pick, "run_rank"].iloc[0]) if not fam_runs.empty else 1

    o1, o2, o3 = st.columns([1.2, 1.6, 4])
    with o1:
        show_prev = st.checkbox("Earlier runs", value=True, key="fv_prev", help="Faded ensemble-mean lines of the previous runs")
    with o2:
        show_mm = st.checkbox("Meteomatics EC / AIFS means", value=(metric == "Temperature"), key="fv_mm",
                              disabled=(metric != "Temperature"),
                              help="Country-mean ensemble MEAN from the Meteomatics silver tables (no members stored there)")
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
    init_hours = VOLUE_MODELS[family]["init_hours"]
    from _data import snap_to_init_time
    st.caption(f"Volue · {family} · run {format_run(snap_to_init_time(ref, init_hours))} · "
               f"{int(cur['n_members'].max() or 0)} members · normal = Volue 30-yr normal")

    # KPI row — horizon mean anomaly (days 1–14)
    anom_col = "anomaly" if m["anomaly_kind"] == "diff" else "anomaly_pct"
    a_unit = m["unit"] if m["anomaly_kind"] == "diff" else "%"
    h_mean = _horizon_mean(cur, "ens_mean", 14)
    h_anom = _horizon_mean(cur, anom_col, 14)
    a_suffix = "%" if a_unit == "%" else f" {m['unit']}"
    cards = [anomaly_kpi(f"{area_label(a)} · d1–14 mean", h_mean.get(a), m["unit"], h_anom.get(a),
                         anomaly_unit=a_suffix, warm_is_positive=m["warm_is_positive"]) for a in areas]
    kpi_row(cards)

    st.divider()
    st.plotly_chart(make_anomaly_heatmap(cur, anom_col, a_unit,
                                         title=f"{metric} — daily anomaly vs normal ({a_unit})"),
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
    """Merge this run's spread with the climatological spread at the same lead day.
    Prefer the same-calendar-month baseline when it has enough runs, else all-months."""
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

    month = int(pd.Timestamp.today().month)
    ratio = _spread_ratio(df, clim, month)
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
    d1, d2 = st.columns([1.4, 3])
    with d1:
        focus = st.selectbox("Country", areas, format_func=area_label, key="fu_focus")
    cur_a = df[df["area"] == focus]
    clim_a = clim[(clim["area"] == focus) & (clim["run_month"] == 0)] if not clim.empty else pd.DataFrame()
    st.plotly_chart(make_spread_vs_normal_chart(cur_a, clim_a, focus, m["unit"]), use_container_width=True)

    # Member-level view (Volue EC-ENS members)
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
            spread = float(dd["value"].std())
            st.caption(f"{len(dd)} members · σ {spread:.2f} {m['unit']} · P10–P90 "
                       f"{dd['value'].quantile(0.1):.1f} to {dd['value'].quantile(0.9):.1f} {m['unit']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

def _available_sources(src: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for name, cfg in SCENARIO_SOURCES.items():
        hit = src[(src["provider"] == cfg["provider"]) & (src["model"] == cfg["model"])] if not src.empty else pd.DataFrame()
        if not hit.empty and int(hit["n_members"].iloc[0]) >= 5:
            out[name] = {**cfg, "n_members": int(hit["n_members"].iloc[0]),
                         "reference_date": hit["reference_date"].iloc[0]}
    return out


def _render_scenarios(runs: pd.DataFrame):
    try:
        src = load_member_sources()
    except Exception as e:
        st.error(f"Failed to read member sources: {e}")
        return
    avail = _available_sources(src)
    if not avail:
        status_banner("No per-member forecast rows found in fcst_members. Run power_desk_refresh.py; "
                      "if the Volue tags are percentiles rather than members, clustering is not possible.", "critical")
        return
    if not any(v["provider"] == "Meteomatics" for v in avail.values()):
        status_banner("Meteomatics EC-ENS / AIFS-ENS members are not in the silver tables (ensemble mean only). "
                      "Clustering runs on Volue EC-ENS members — the same 50 ECMWF perturbations. To cluster on "
                      "Meteomatics or AIFS members, set METEOMATICS_MEMBER_TABLE in power_desk_refresh.py "
                      "and the app config.", "warning")

    c1, c2, c3 = st.columns([1.6, 3, 1.6])
    with c1:
        source = st.selectbox("Cluster on", list(avail.keys()), key="fs_source")
    with c2:
        areas = st.multiselect("Feature countries", list(AREAS.keys()),
                               [a for a in SCENARIO_DEFAULT_AREAS if a in AREAS], format_func=area_label, key="fs_areas")
    with c3:
        horizon = st.slider("Lead days", 1, 15, SCENARIO_DEFAULT_HORIZON, key="fs_horizon")
    o1, o2, o3 = st.columns([1.4, 1.4, 3])
    with o1:
        k_mode = st.radio("Number of scenarios", ["Auto (silhouette)", "Fixed"], horizontal=True, key="fs_kmode")
    with o2:
        k_fixed = st.slider("k", SCENARIO_K_RANGE[0], SCENARIO_K_RANGE[1], 3, key="fs_k",
                            disabled=(k_mode != "Fixed"))
    with o3:
        use_wind = st.checkbox("Also cluster on wind anomaly (Volue members)", value=False, key="fs_wind",
                               help="Adds Volue EC-ENS wind-production anomaly to the feature space")
    if not areas:
        st.info("Select at least one feature country.")
        return

    cfg = avail[source]
    try:
        temp_m = load_members(cfg["provider"], cfg["model"], "Temperature", tuple(areas))
        wind_m = load_members("Volue", "EC-ENS", "Wind", tuple(areas)) if use_wind else pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load members: {e}")
        return

    feat_t = member_matrix(temp_m, areas, (horizon[0], horizon[1]), "anomaly")
    parts = {"Temperature": feat_t}
    if use_wind and not wind_m.empty:
        feat_w = member_matrix(wind_m, areas, (horizon[0], horizon[1]), "anomaly")
        common = feat_t.index.intersection(feat_w.index)
        parts = {"Temperature": feat_t.loc[common], "Wind": feat_w.loc[common]}
    features = pd.concat(parts, axis=1)
    if features.empty or features.shape[0] < 6:
        status_banner("Not enough complete members to cluster for this selection.", "warning")
        return

    res = cluster_members(features, None if k_mode.startswith("Auto") else int(k_fixed))
    if not res:
        status_banner("Clustering failed — too few members.", "warning")
        return
    assign = res["assignments"]
    ref = cfg["reference_date"]
    st.caption(f"{source} · run {pd.Timestamp(ref):%a %d %b %H:%M} UTC · {res['n_members']} members · "
               f"features: {', '.join(parts.keys())} anomaly × {len(areas)} countries × lead days "
               f"{horizon[0]}–{horizon[1]} · k={res['k']} · silhouette {res['silhouette']:.2f}")

    # Scenario cards
    n_tot = res["n_members"]
    scen_ids = sorted(s for s in res["sizes"] if s != 0) + ([0] if 0 in res["sizes"] else [])
    cards = []
    for s in scen_ids:
        share = res["sizes"][s] / n_tot
        row = res["centroids"].loc[s]
        desc = describe_scenario(row.xs("Temperature", level=0), share) if s != 0 else f"{share:.0%} of members in clusters too small to keep"
        color = scenario_color(s)
        cards.append(
            f'<div class="kpi-card" style="border-top:3px solid {color}; text-align:left;">'
            f'<div class="kpi-label">{scenario_name(s)}</div>'
            f'<div class="kpi-value">{res["sizes"][s]} <span style="font-size:0.9rem;color:var(--text-muted)">members</span></div>'
            f'<div class="kpi-delta kpi-delta-flat">{desc}</div></div>'
        )
    kpi_row(cards, max_cols=4)

    tagged_t = apply_scenarios(temp_m, assign)
    g1, g2 = st.columns([2.2, 1])
    with g1:
        st.plotly_chart(make_scenario_table_heatmap(
            scenario_table(tagged_t[tagged_t["lead_day"].between(*horizon)], "anomaly", "T anomaly"),
            "°C", f"Temperature anomaly by scenario — mean over lead days {horizon[0]}–{horizon[1]}"),
            use_container_width=True)
    with g2:
        st.plotly_chart(make_silhouette_chart(res["silhouette_by_k"], res["k"]), use_container_width=True)
        with st.expander("Member → scenario map"):
            st.dataframe(assign.sort_values(["scenario", "member_id"]).rename(
                columns={"member_id": "member", "scenario": "scenario id"})[["member", "scenario id"]],
                use_container_width=True, hide_index=True, height=240)

    st.markdown("##### Scenario paths — temperature")
    summ_t = scenario_daily_summary(tagged_t, "value")
    cols = st.columns(2)
    for i, a in enumerate(areas):
        with cols[i % 2]:
            st.plotly_chart(make_scenario_lines(summ_t, a, "°C", "temperature", tagged_t), use_container_width=True)

    # ── The same members read off Volue (and Meteologica) power curves ──
    st.divider()
    st.markdown("##### What each scenario means for power — Volue members with the same IDs")
    st.caption("Member IDs are matched across providers (all are the 50 ECMWF-ENS perturbations); each table "
               "is the mean over the clustered lead days, with the change vs the full-ensemble mean.")
    power_tabs = st.tabs(["Wind", "Solar", "Temperature", "Meteologica"])
    for tab, metric in zip(power_tabs[:3], ["Wind", "Solar", "Temperature"]):
        with tab:
            try:
                pm = load_members("Volue", "EC-ENS", metric, tuple(areas))
            except Exception:
                pm = pd.DataFrame()
            _render_power_by_scenario(pm, assign, areas, horizon, METRICS[metric]["unit"], metric)
    with power_tabs[3]:
        found_any = False
        for metric in ["Wind", "Solar", "Hydro", "Temperature"]:
            mlm = load_meteologica_members(metric, tuple(areas))
            if mlm.empty:
                continue
            found_any = True
            st.markdown(f"**{metric}**")
            _render_power_by_scenario(mlm, assign, areas, horizon, "", metric, key_suffix="mtlg")
        if not found_any:
            status_banner("Meteologica is not ingested into Databricks yet. When it is, populate "
                          "meteologica_members (same layout as fcst_members) and this tab fills in.", "warning")


def _render_power_by_scenario(pm: pd.DataFrame, assign: pd.DataFrame, areas: list[str],
                              horizon: tuple[int, int], unit: str, metric: str, key_suffix: str = "volue"):
    if pm.empty:
        st.caption(f"No {metric} member rows available.")
        return
    tagged = apply_scenarios(pm, assign)
    tagged = tagged[tagged["lead_day"].between(*horizon)]
    if tagged.empty:
        st.caption("No overlap between scenario members and this provider's member IDs.")
        return
    ens = tagged.groupby("area")["value"].mean()
    tbl = tagged.groupby(["scenario", "area"])["value"].mean().unstack("area")
    tbl = tbl.reindex(columns=[a for a in areas if a in tbl.columns])
    delta = tbl.sub(ens[tbl.columns], axis=1)
    pct = delta.div(ens[tbl.columns], axis=1) * 100
    out = pd.DataFrame(index=[scenario_name(s) for s in tbl.index])
    for a in tbl.columns:
        if metric == "Temperature":
            out[f"{area_label(a)} ({unit})"] = [f"{v:.1f} ({d:+.1f})" for v, d in zip(tbl[a], delta[a])]
        else:
            out[f"{area_label(a)} ({unit})"] = [f"{v:.2f} ({p:+.0f}%)" for v, p in zip(tbl[a], pct[a])]
    out.insert(0, "members", [int(assign[assign["scenario"] == s].shape[0]) for s in tbl.index])
    st.dataframe(out, use_container_width=True)
    summ = scenario_daily_summary(tagged, "value")
    pick = st.selectbox("Country path", areas, format_func=area_label, key=f"fs_pick_{metric}_{key_suffix}")
    st.plotly_chart(make_scenario_lines(summ, pick, unit, metric.lower(), tagged), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def render_forecast():
    st.markdown("#### FORECAST")
    st.caption("Volue ensemble forecasts by country · distribution spread vs normal · member-clustered scenarios. "
               "Source: Volue delta share, Meteomatics silver layer, via power_desk_refresh.py.")
    try:
        runs = load_runs()
    except Exception as e:
        st.error(f"Cannot read the run list ({e}). Has power_desk_refresh.py run?")
        return
    tabs = st.tabs(["Values", "Uncertainty", "Scenarios"])
    with tabs[0]:
        _render_values(runs)
    with tabs[1]:
        _render_uncertainty(runs)
    with tabs[2]:
        _render_scenarios(runs)
