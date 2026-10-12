"""Section 2 — Historical & Analysis.

Tab 1  History by country: monthly or weekly values, any combination of years
       and months, shown as anomaly vs normal (default) or actual vs normal,
       plus a full year × month anomaly heatmap per country. Temperature is the
       Meteomatics / ERA5 gold climatology reduced to population-weighted
       country means (hist_daily, source 'Meteomatics' — the same rows as the
       Anomaly Maps tab); wind, solar and precipitation energy are Volue
       actuals vs the Volue normal (source 'Volue').
Tab 2  Anomaly Maps: gridded ERA5 anomalies from the gold layer (via anomaly_map).
Tab 3  Teleconnections (_teleconnections.py): the weather indices against temperature,
       wind, solar and precipitation energy by country — correlations, anomalies by index
       phase, blends and joint regimes of several indices, regression, analogue years —
       per month or calendar quarter, with the Meteologica ERA5 reanalysis (since 1979)
       next to Volue for wind and solar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    AREAS, DEFAULT_AREAS, METRICS, DEFAULT_METRIC, MONTH_NAMES,
    SBX_SCHEMA, area_label, MAP_METRICS, HIST_SOURCE_NOTES, HIST_DEFAULT_SOURCE,
)
from _data import (
    load_hist_years, load_hist_daily, load_hist_monthly_all, load_anomaly_map,
    resolve_hist_source, hist_source_options,
)
from _charts import (
    make_year_month_heatmap, make_period_bars, make_period_lines, make_anomaly_heatmap,
    make_geo_anomaly_map,
)
from _teleconnections import render_teleconnections
from _ui import anomaly_kpi, kpi_row, status_banner


def _aggregate(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if df.empty:
        return df
    if granularity == "Monthly":
        g = df.groupby(["area", "year", "month"]).agg(actual=("actual", "mean"), normal=("normal", "mean"),
                                                        anomaly=("anomaly", "mean"), n_days=("actual", "count")).reset_index()
        g["period"] = [f"{int(y)}-{MONTH_NAMES[int(m) - 1]}" for y, m in zip(g["year"], g["month"])]
        g["sort"] = g["year"] * 100 + g["month"]
    else:
        g = df.groupby(["area", "week_start"]).agg(actual=("actual", "mean"), normal=("normal", "mean"),
                                                    anomaly=("anomaly", "mean"), n_days=("actual", "count")).reset_index()
        g["period"] = g["week_start"].dt.strftime("%Y-W%V")
        g["sort"] = g["week_start"]
    g["anomaly_pct"] = np.where(g["normal"].abs() > 0, (g["actual"] / g["normal"] - 1) * 100, np.nan)
    return g.sort_values(["sort", "area"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def _render_history():
    c1, c2 = st.columns([1.2, 4])
    with c1:
        metric = st.selectbox("Metric", list(METRICS.keys()), index=list(METRICS).index(DEFAULT_METRIC), key="h_metric")
    with c2:
        areas = st.multiselect("Countries", list(AREAS.keys()), DEFAULT_AREAS, format_func=area_label, key="h_areas")
    # Temperature reads the Meteomatics / ERA5 rows of hist_daily, the other
    # metrics Volue's; an older sandbox (no `source` column) or a notebook run
    # without gold access falls back and says so.
    opts = hist_source_options(metric)
    choice = st.radio("Source", opts, horizontal=True, key=f"h_src_{metric}",
                      format_func=lambda s: {"Volue": "Volue actuals (2013+)", "Reanalysis": "Meteologica reanalysis (1979+)"}.get(s, s)
                      ) if len(opts) > 1 else None
    source, note = resolve_hist_source(metric, choice)
    if note:
        status_banner(note, "warning")
    st.caption(f"Source — {HIST_SOURCE_NOTES.get(source or HIST_DEFAULT_SOURCE, source)}.")
    try:
        years_avail = load_hist_years(metric, source)
    except Exception as e:
        st.error(f"Failed to read history: {e}")
        return
    if not years_avail:
        status_banner("hist_daily is empty — run power_desk_refresh.py.", "warning")
        return

    y1, y2, y3, y4 = st.columns([2.4, 2.4, 1.2, 1.4])
    with y1:
        years = st.multiselect("Years (multiple allowed)", years_avail, years_avail[-3:], key="h_years")
    with y2:
        months = st.multiselect("Months (multiple allowed)", list(range(1, 13)), list(range(1, 13)),
                                format_func=lambda m: MONTH_NAMES[m - 1], key="h_months")
    with y3:
        gran = st.radio("Granularity", ["Monthly", "Weekly"], key="h_gran")
    with y4:
        view = st.radio("View", ["Anomaly", "Actual vs normal"], key="h_view")
    if not (areas and years and months):
        st.info("Select at least one country, year and month.")
        return

    m = METRICS[metric]
    try:
        df = load_hist_daily(metric, tuple(areas), tuple(years), tuple(months), source)
        monthly_all = load_hist_monthly_all(metric, tuple(areas), source)
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return
    if df.empty:
        status_banner("No rows for that selection.", "warning")
        return

    agg = _aggregate(df, gran)
    a_col = "anomaly" if m["anomaly_kind"] == "diff" else "anomaly_pct"
    a_unit = m["unit"] if m["anomaly_kind"] == "diff" else "%"

    # KPI — mean over the whole selection per country
    sel_mean = df.groupby("area").agg(actual=("actual", "mean"), normal=("normal", "mean"))
    sel_mean["anom"] = sel_mean["actual"] - sel_mean["normal"] if a_col == "anomaly" else \
        (sel_mean["actual"] / sel_mean["normal"] - 1) * 100
    label = f"{len(years)} yr × {len(months)} mo mean"
    kpi_row([anomaly_kpi(f"{area_label(a)} · {label}", sel_mean.loc[a, "actual"] if a in sel_mean.index else None,
                         m["unit"], sel_mean.loc[a, "anom"] if a in sel_mean.index else None,
                         anomaly_unit=(f" {m['unit']}" if a_col == "anomaly" else "%"),
                         warm_is_positive=m["warm_is_positive"]) for a in areas])

    st.divider()
    title = f"{metric} — {gran.lower()} {'anomaly vs normal' if view == 'Anomaly' else 'actual and normal'}"
    if view == "Anomaly":
        st.plotly_chart(make_period_bars(agg, a_unit, title, "period", a_col, m["warm_is_positive"]),
                        use_container_width=True)
    else:
        st.plotly_chart(make_period_lines(agg, m["unit"], title, "period", "actual"), use_container_width=True)

    if gran == "Weekly":
        st.plotly_chart(make_anomaly_heatmap(
            agg.rename(columns={"week_start": "day"}), a_col, a_unit,
            title=f"{metric} — weekly anomaly by country ({a_unit})"), use_container_width=True)

    st.markdown("##### Year × month anomaly — full history")
    cols = st.columns(2)
    for i, a in enumerate(areas):
        with cols[i % 2]:
            st.plotly_chart(make_year_month_heatmap(monthly_all, a, a_unit, a_col), use_container_width=True)

    with st.expander("Values — table"):
        piv = agg.pivot_table(index="period", columns="area", values=a_col if view == "Anomaly" else "actual",
                              aggfunc="mean", sort=False)
        piv.columns = [area_label(c) for c in piv.columns]
        st.dataframe(piv.round(2), use_container_width=True)
        st.download_button("Download CSV", agg.drop(columns=["sort"]).to_csv(index=False).encode(),
                           f"history_{metric}_{gran}.csv", "text/csv", key="h_dl")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANOMALY MAPS (gold layer)
# ══════════════════════════════════════════════════════════════════════════════

def _render_maps():
    """Gridded ERA5 anomaly maps from dna_prod_gold.weather for temperature,
    wind and precipitation. User selects metric, year(s) and month(s); the
    query averages the anomaly per grid point over the selection."""
    from datetime import datetime as _dt
    cur_year = _dt.now().year
    c1, c2, c3 = st.columns([1.4, 2.2, 3])
    with c1:
        metric = st.selectbox("Variable", list(MAP_METRICS.keys()), key="map_metric")
    with c2:
        years = st.multiselect("Years", list(range(cur_year, 1977, -1)), [cur_year], key="map_years")
    with c3:
        months = st.multiselect("Months", list(range(1, 13)),
                                [_dt.now().month],
                                format_func=lambda m: MONTH_NAMES[m - 1], key="map_months")
    if not (years and months):
        st.info("Select at least one year and one month.")
        return

    m = MAP_METRICS[metric]
    with st.spinner("Loading anomaly map..."):
        try:
            df = load_anomaly_map(metric, tuple(sorted(years)), tuple(sorted(months)))
        except Exception as e:
            st.error(f"Failed to load anomaly map: {e}")
            return
    if df.empty:
        st.warning("No data for that selection.")
        return

    period_label = ", ".join(MONTH_NAMES[mm - 1] for mm in sorted(months))
    year_label = ", ".join(str(y) for y in sorted(years))
    title = f"{metric} anomaly \u2014 {period_label} {year_label}"

    fig_mpl = make_geo_anomaly_map(df, m["unit"], title, m["symmetric"], m["warm_is_positive"])
    st.pyplot(fig_mpl, use_container_width=True)
    import matplotlib.pyplot as plt
    plt.close(fig_mpl)

    # Summary stats
    mean_anom = df["anomaly"].mean()
    st.caption(f"{len(df):,} grid points \u00b7 mean anomaly {mean_anom:+.2f} {m['unit']} "
               f"\u00b7 source: {SBX_SCHEMA}.anomaly_map (ERA5 via gold layer)")


def render_historical():
    st.markdown("#### HISTORICAL & ANALYSIS")
    st.caption("History by country — temperature from the Meteomatics / ERA5 climatology since 1979 "
               "(population-weighted country means, the Anomaly Maps' own value / normal / anomaly), wind, "
               "solar and precipitation energy from Volue since 2013 · monthly and weekly · multi-year and "
               "multi-month selections · ERA5 anomaly maps · teleconnections: weather indices against temperature, wind "
               "and solar (reanalysis since 1979) — correlations, phases, blends, regression, analogues.")
    tabs = st.tabs(["History by country", "Anomaly Maps", "Teleconnections"])
    with tabs[0]:
        _render_history()
    with tabs[1]:
        _render_maps()
    with tabs[2]:
        render_teleconnections()
