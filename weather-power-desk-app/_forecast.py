"""Section 1 — Forecast.

Three tabs:
  Values        every Volue ensemble value by country (mean, median, P10-P90),
                normal, earlier runs, Meteomatics EC-ENS / AIFS-ENS means
  Uncertainty   spread of the ensemble distribution and how it compares with
                the normal spread for the same lead day — is this forecast
                more or less certain than usual?
  Scenarios     two weather scenarios from k-means on members' WEEKLY-MEAN
                temperature anomaly over real Mon–Sun weeks (Meteomatics
                EC-ENS / AIFS-ENS members where available, else Volue EC-ENS
                members), then, per country, the two scenarios side by side
                for temperature, wind and solar read off the same member IDs
                in Volue (and Meteologica when ingested)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    AREAS, DEFAULT_AREAS, METRICS, DEFAULT_METRIC, VOLUE_MODELS, DEFAULT_MODEL,
    METEOMATICS_MODELS, SPREAD_RATIO_HIGH, SPREAD_RATIO_LOW, SCENARIO_SOURCES,
    SCENARIO_DEFAULT_AREAS, SCENARIO_OUTPUT_AREAS, SCENARIO_K_DEFAULT, SCENARIO_K_MAX,
    SCENARIO_USE_GEOPOTENTIAL, SPATIAL_N_PCA, SPATIAL_GRID_RES,
    area_label,
)
from _data import (
    load_runs, load_fcst_daily, load_spread_clim, load_member_sources, load_members,
    load_meteologica_members, load_member_spatial_days, format_run, snap_to_init_time,
)
from _charts import (
    AREA_COLORS, make_fan_chart, make_anomaly_heatmap,
    make_spread_vs_normal_chart, make_spread_ratio_heatmap, make_member_distribution, make_member_strips,
    make_scenario_table_heatmap, make_country_scenario_panel, scenario_color,
)
from _scenarios import (
    available_weeks, weekly_member_matrix, spatial_member_matrix, cluster_members, apply_scenarios,
    scenario_daily_summary, scenario_weekly_values, scenario_table, scenario_name, describe_scenario,
    silhouette_for_k,
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
        areas = _area_picker("fv_areas", DEFAULT_AREAS)
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
    st.caption(f"Volue · {family} · run {format_run(snap_to_init_time(ref, VOLUE_MODELS[family]['init_hours']))} · "
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
# TAB 3 — SCENARIOS (weekly-mean clustering, two scenarios, side by side per country)
# ══════════════════════════════════════════════════════════════════════════════

def _available_sources(src: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for name, cfg in SCENARIO_SOURCES.items():
        hit = src[(src["provider"] == cfg["provider"]) & (src["model"] == cfg["model"])] if not src.empty else pd.DataFrame()
        if not hit.empty and int(hit["n_members"].iloc[0]) >= 5:
            out[name] = {**cfg, "n_members": int(hit["n_members"].iloc[0]), "reference_date": hit["reference_date"].iloc[0]}
    return out


def _fmt(v, fmt="{:.1f}") -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def _fmt_d(v, fmt="{:+.1f}") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<span class="mc-muted">n/a</span>'
    cls = "mc-pos" if round(v, 1) > 0 else ("mc-neg" if round(v, 1) < 0 else "mc-zero")
    return f'<span class="{cls}">{fmt.format(v)}</span>'


def _side_by_side_html(area: str, weekly: dict[str, pd.DataFrame], scen_ids: list[int],
                       weeks: pd.DataFrame, sizes: dict[int, int], n_tot: int) -> str:
    """Rows = metric × week; per scenario three cells: value, Δ vs ensemble mean, Δ vs normal."""
    units = {"Temperature": "°C", "Wind": "GW", "Solar": "GW"}
    head1 = "<tr><th></th>"
    head2 = "<tr><th>metric · week</th>"
    for s in scen_ids:
        col = scenario_color(s)
        head1 += (f'<th class="sc-scen" colspan="3" style="border-color:{col}; text-align:center;">'
                  f'{scenario_name(s)} <span class="mc-muted">({sizes.get(s, 0)}/{n_tot})</span></th>')
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
            label = f"{metric} · {wk['week_label']} <span class='mc-muted'>({units[metric]})</span>"
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
        status_banner("No Meteomatics EC-ENS / AIFS-ENS member rows in fcst_members yet (the refresh job reads them "
                      "from curve_member in the silver temperature table). Country-mean clustering runs on Volue "
                      "EC-ENS member temperatures — the same 50 ECMWF perturbations.", "warning")

    c1, c2, c3 = st.columns([1.6, 3.2, 1.6])
    with c1:
        source = st.selectbox("Cluster on", list(avail.keys()), key="fs_source")
    with c2:
        areas = st.multiselect("Feature countries (weekly-mean temperature anomaly)", list(AREAS.keys()),
                               [a for a in SCENARIO_DEFAULT_AREAS if a in AREAS], format_func=area_label, key="fs_areas")
    with c3:
        use_spatial = st.checkbox(
            "Spatial grid clustering", value=False, key="fs_spatial",
            help=f"Cluster on the full European weekly-mean anomaly FIELD (Meteomatics ecmwf-ens members, "
                 f"{SPATIAL_GRID_RES:g}° grid, PCA to {SPATIAL_N_PCA} components) instead of country averages. "
                 + ("Field: Z500 geopotential." if SCENARIO_USE_GEOPOTENTIAL else
                    "Field: temperature as a proxy — Z500 members exist only for ecmwf-aifs-ens today."))
    if not areas:
        st.info("Select at least one feature country.")
        return

    cfg = avail[source]
    try:
        temp_m = load_members(cfg["provider"], cfg["model"], "Temperature", tuple(set(areas) | set(SCENARIO_OUTPUT_AREAS)))
    except Exception as e:
        st.error(f"Failed to load members: {e}")
        return
    if temp_m.empty:
        status_banner("No temperature members for the selected countries.", "warning")
        return

    weeks_all = available_weeks(temp_m)
    usable = weeks_all[weeks_all["usable"]]
    if usable.empty:
        status_banner("The forecast does not cover enough days of any week to form weekly means.", "warning")
        return
    today = pd.Timestamp.today().normalize()
    this_week = today - pd.Timedelta(days=today.weekday())
    default_weeks = [w for w in usable["week_start"] if w > this_week][:2] or list(usable["week_start"][:2])

    o1, o2, o3 = st.columns([3, 1.2, 1.6])
    with o1:
        weeks_pick = st.multiselect(
            "Trading weeks to cluster on (Mon–Sun)", list(usable["week_start"]), default_weeks, key="fs_weeks",
            format_func=lambda w: f"{usable.set_index('week_start').loc[w, 'week_label']}  "
                                  f"{pd.Timestamp(w):%d %b}–{pd.Timestamp(w) + pd.Timedelta(days=6):%d %b}"
                                  f"{'' if usable.set_index('week_start').loc[w, 'full'] else '  (partial, %d d)' % usable.set_index('week_start').loc[w, 'n_days']}")
    with o2:
        k = st.number_input("Scenarios", 2, SCENARIO_K_MAX, SCENARIO_K_DEFAULT, key="fs_k")
    with o3:
        out_areas = st.multiselect("Countries shown", list(AREAS.keys()),
                                   [a for a in SCENARIO_OUTPUT_AREAS if a in AREAS], format_func=area_label, key="fs_out")
    if not weeks_pick:
        st.info("Pick at least one week.")
        return
    weeks_sel = usable[usable["week_start"].isin(weeks_pick)].sort_values("week_start")

    spatial_ref = None
    if use_spatial:
        grid_by_week = {}
        try:
            for _, wk in weeks_sel.iterrows():
                g = load_member_spatial_days(wk["first_day"], wk["last_day"])
                if not g.empty:
                    grid_by_week[wk["week_start"]] = g
                    spatial_ref = g["reference_date"].max()
        except Exception as e:
            st.error(f"Failed to load spatial member data: {e}")
            return
        if not grid_by_week:
            status_banner("fcst_member_spatial has no rows for these weeks — run power_desk_refresh.py.", "warning")
            return
        features = spatial_member_matrix(grid_by_week)
    else:
        features = weekly_member_matrix(temp_m, areas, list(weeks_sel["week_start"]), "anomaly")
    if features.empty or features.shape[0] < 6:
        status_banner("Not enough complete members to cluster for this selection.", "warning")
        return
    res = cluster_members(features, int(k), scale=not use_spatial)   # PCA scores keep their variance ordering
    if not res:
        status_banner("Clustering failed — too few members.", "warning")
        return
    assign = res["assignments"]
    n_tot = res["n_members"]
    if use_spatial:
        field = "Z500 geopotential" if SCENARIO_USE_GEOPOTENTIAL else "temperature (proxy for Z500)"
        st.caption(f"Spatial clustering · Meteomatics ecmwf-ens · run {pd.Timestamp(spatial_ref):%a %d %b %H:%M} UTC · "
                   f"{n_tot} members · {features.attrs.get('n_grid_points', 0)} grid points ({SPATIAL_GRID_RES:g}°) × "
                   f"{len(weeks_sel)} week(s) ({', '.join(weeks_sel['week_label'])}) · PCA {features.shape[1]} PCs "
                   f"({features.attrs.get('explained_variance_pct', 0):.0f}% var) · {field} · k={res['k']} fixed · "
                   f"silhouette {res['silhouette']:.2f} (diagnostic)")
        if not SCENARIO_USE_GEOPOTENTIAL:
            status_banner("Spatial clustering uses weekly temperature maps as a proxy for Z500 — switch "
                          "SCENARIO_USE_GEOPOTENTIAL on when ecmwf-ens geopotential members are ingested.", "warning")
    else:
        st.caption(f"{source} · run {pd.Timestamp(cfg['reference_date']):%a %d %b %H:%M} UTC · {n_tot} members · "
                   f"features: weekly-mean T anomaly × {len(areas)} countries × {len(weeks_sel)} week(s) "
                   f"({', '.join(weeks_sel['week_label'])}) · k={res['k']} fixed · silhouette {res['silhouette']:.2f} (diagnostic)")

    tagged_t = apply_scenarios(temp_m, assign)
    scen_ids = sorted(s for s in res["sizes"] if s != 0) + ([0] if 0 in res["sizes"] else [])
    cards = []
    for s in scen_ids:
        share = res["sizes"][s] / n_tot
        if s == 0:
            desc = f"{share:.0%} of members in clusters too small to keep"
        elif use_spatial:
            # centroids are PC scores — describe the cluster through its members' country temperatures
            st_ = tagged_t[(tagged_t["scenario"] == s) & tagged_t["week_start"].isin(weeks_sel["week_start"])]
            if st_.empty or st_["anomaly"].isna().all():
                desc = f"{share:.0%} of members"
            else:
                by_area = st_.groupby("area")["anomaly"].mean().sort_values()
                ma = float(st_["anomaly"].mean())
                tone = "warm" if ma > 0.75 else ("cold" if ma < -0.75 else "near-normal")
                desc = (f"{share:.0%} of members · {tone} on average ({ma:+.1f}°C) · coldest {by_area.index[0]} "
                        f"({by_area.iloc[0]:+.1f}), warmest {by_area.index[-1]} ({by_area.iloc[-1]:+.1f})")
        else:
            desc = describe_scenario(res["centroids"].loc[s], share)
        cards.append(
            f'<div class="kpi-card" style="border-top:3px solid {scenario_color(s)}; text-align:left;">'
            f'<div class="kpi-label">{scenario_name(s)}</div>'
            f'<div class="kpi-value">{res["sizes"][s]} <span style="font-size:0.9rem;color:var(--text-muted)">members</span></div>'
            f'<div class="kpi-delta kpi-delta-flat">{desc}</div></div>')
    kpi_row(cards, max_cols=4)

    tagged_t = apply_scenarios(temp_m, assign)
    st.plotly_chart(make_scenario_table_heatmap(
        scenario_table(tagged_t, list(weeks_sel["week_start"]), "anomaly"), "°C",
        f"Temperature anomaly by scenario — mean over {', '.join(weeks_sel['week_label'])}"), use_container_width=True)

    # ── per country: the scenarios side by side for temperature, wind, solar ──
    st.divider()
    st.markdown("##### Per country — scenarios side by side: temperature, wind, solar")
    st.caption("Weekly means over the clustered trading weeks. Wind and solar are the Volue EC-ENS members with the "
               "same member IDs as the temperature clusters (all providers share the 50 ECMWF perturbations). "
               "Δ ens = vs full-ensemble mean · Δ norm = vs Volue normal.")
    show_areas = [a for a in out_areas if a in AREAS]
    if not show_areas:
        st.info("Select at least one country to show.")
        return
    tagged: dict[str, pd.DataFrame] = {"Temperature": tagged_t[tagged_t["area"].isin(show_areas)]}
    for metric in ("Wind", "Solar"):
        try:
            pm = load_members("Volue", "EC-ENS", metric, tuple(show_areas))
        except Exception:
            pm = pd.DataFrame()
        tagged[metric] = apply_scenarios(pm, assign) if not pm.empty else pd.DataFrame()
    weekly = {m: scenario_weekly_values(t, list(weeks_sel["week_start"]), "value") for m, t in tagged.items()}
    daily = {m: scenario_daily_summary(t, "value") for m, t in tagged.items() if not t.empty}
    units = {"Temperature": "°C", "Wind": "GW", "Solar": "GW"}
    monday_lines = list(weeks_all["week_start"])

    for a in show_areas:
        st.markdown(_side_by_side_html(a, weekly, scen_ids, weeks_sel, res["sizes"], n_tot), unsafe_allow_html=True)
        st.plotly_chart(make_country_scenario_panel(daily, a, units, monday_lines), use_container_width=True)

    with st.expander("Meteologica members (same scenario IDs)"):
        found = False
        for metric in ("Wind", "Solar", "Hydro", "Temperature"):
            mlm = load_meteologica_members(metric, tuple(show_areas))
            if mlm.empty:
                continue
            found = True
            w = scenario_weekly_values(apply_scenarios(mlm, assign), list(weeks_sel["week_start"]), "value")
            st.markdown(f"**{metric}**")
            st.dataframe(w[["area", "week_label", "scenario", "mean", "ens_mean", "d_ens", "n"]].round(2),
                         use_container_width=True, hide_index=True)
        if not found:
            st.caption("Meteologica is not ingested into Databricks yet — populate meteologica_members and this fills in.")

    with st.expander("Member → scenario map · separability diagnostic"):
        d1, d2 = st.columns([1, 1.4])
        with d1:
            st.dataframe(assign.sort_values(["scenario", "member_id"]).rename(
                columns={"member_id": "member", "scenario": "scenario id"})[["member", "scenario id"]],
                use_container_width=True, hide_index=True, height=260)
        with d2:
            sil = silhouette_for_k(features, ks=range(2, SCENARIO_K_MAX + 1))
            if sil:
                st.dataframe(pd.DataFrame({"k": list(sil.keys()), "silhouette": [round(v, 3) for v in sil.values()]}),
                             use_container_width=True, hide_index=True)
                st.caption("Diagnostic only: k is fixed by the desk (2 = scenario + alternative), not chosen by silhouette. "
                           "Proper regime clustering on 500 hPa geopotential members replaces this when that field is loaded.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def render_forecast():
    st.markdown("#### FORECAST")
    st.caption("Volue ensemble forecasts by country · distribution spread vs normal · weekly-mean member "
               "scenarios · European weather regimes. "
               "Source: Volue delta share, Meteomatics silver layer, via power_desk_refresh.py.")
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
