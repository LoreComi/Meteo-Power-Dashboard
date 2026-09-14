"""Section 2 — Historical & Analysis.

Tab 1  History by country: monthly or weekly values, any combination of years
       and months, shown as anomaly vs normal (default) or actual vs normal,
       plus a full year × month anomaly heatmap per country.
Tab 2  Analogues with weather indexes: pick an index and a target period, find
       the historical years whose index was closest to the current value, and
       composite the countries' anomalies in those years. Data-driven from
       {SBX_SCHEMA}.weather_indexes — until that table is loaded the tab shows
       the expected layout and stays otherwise inert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    AREAS, DEFAULT_AREAS, METRICS, DEFAULT_METRIC, MONTH_NAMES, WEATHER_INDEXES, ANALOG_N_YEARS,
    SBX_SCHEMA, area_label,
)
from _data import load_hist_years, load_hist_daily, load_hist_monthly_all, load_weather_indexes
from _charts import (
    make_year_month_heatmap, make_period_bars, make_period_lines, make_index_chart, make_anomaly_heatmap,
)
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
    try:
        years_avail = load_hist_years(metric)
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
        df = load_hist_daily(metric, tuple(areas), tuple(years), tuple(months))
        monthly_all = load_hist_monthly_all(metric, tuple(areas))
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
# TAB 2 — ANALOGUES WITH WEATHER INDEXES
# ══════════════════════════════════════════════════════════════════════════════

def _render_analogues():
    idx = load_weather_indexes()
    if idx.empty:
        status_banner(f"Weather indexes are not loaded yet. The tab reads {SBX_SCHEMA}.weather_indexes "
                      "(index_name, date, value, source); it comes alive as soon as rows land there.", "warning")
        st.markdown("**Expected indexes**")
        st.dataframe(pd.DataFrame({"index_name": list(WEATHER_INDEXES.keys()),
                                   "description": list(WEATHER_INDEXES.values())}),
                     use_container_width=True, hide_index=True)
        st.markdown(
            "**Method once loaded** — pick an index and a target period; the index is averaged over that "
            "period for every historical year and the closest years to the current (or assumed) value become "
            "the analogues. Their country anomalies (from hist_daily) are composited as a mean and spread, "
            "next to the same period's ensemble forecast for a sanity check.")
        return

    names = sorted(idx["index_name"].unique())
    c1, c2, c3, c4 = st.columns([1.3, 2.4, 2.4, 1.2])
    with c1:
        index_name = st.selectbox("Index", names, format_func=lambda n: f"{n} — {WEATHER_INDEXES.get(n, '')}", key="an_idx")
    with c2:
        months = st.multiselect("Target months", list(range(1, 13)), [pd.Timestamp.today().month],
                                format_func=lambda m: MONTH_NAMES[m - 1], key="an_months")
    with c3:
        areas = st.multiselect("Countries", list(AREAS.keys()), DEFAULT_AREAS, format_func=area_label, key="an_areas")
    with c4:
        n_analog = st.slider("Analog years", 3, 10, ANALOG_N_YEARS, key="an_n")
    metric = st.selectbox("Metric to composite", list(METRICS.keys()), key="an_metric")
    if not (months and areas):
        st.info("Pick months and countries.")
        return

    s = idx[idx["index_name"] == index_name].dropna(subset=["value"]).sort_values("date")
    s["year"] = s["date"].dt.year
    s["month"] = s["date"].dt.month
    per_year = s[s["month"].isin(months)].groupby("year")["value"].mean()
    latest_val = float(s["value"].iloc[-1])
    target = st.slider(f"Assumed {index_name} value", float(np.floor(s["value"].min())), float(np.ceil(s["value"].max())),
                       round(latest_val, 2), 0.05, key="an_target",
                       help=f"Defaults to the latest observed value ({latest_val:+.2f}, {s['date'].iloc[-1]:%b %Y})")
    this_year = pd.Timestamp.today().year
    cand = per_year[per_year.index < this_year]
    analogs = (cand - target).abs().sort_values().head(n_analog)
    st.caption("Analog years: " + ", ".join(f"{y} ({cand[y]:+.2f})" for y in analogs.index))
    st.plotly_chart(make_index_chart(s, index_name, list(analogs.index)), use_container_width=True)

    try:
        monthly = load_hist_monthly_all(metric, tuple(areas))
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return
    m = METRICS[metric]
    comp = monthly[monthly["year"].isin(analogs.index) & monthly["month"].isin(months)]
    if comp.empty:
        status_banner("No historical rows for the analog years / months.", "warning")
        return
    a_unit = m["unit"] if m["anomaly_kind"] == "diff" else "%"
    comp = comp.copy()
    a_col = "anomaly" if m["anomaly_kind"] == "diff" else "anomaly_pct"
    summary = comp.groupby("area")[a_col].agg(["mean", "std", "count"])
    kpi_row([anomaly_kpi(f"{area_label(a)} · analog composite", summary.loc[a, "mean"] if a in summary.index else None,
                         a_unit, summary.loc[a, "mean"] if a in summary.index else None,
                         anomaly_unit=f" {a_unit}", warm_is_positive=m["warm_is_positive"], fmt="{:+.1f}") for a in areas])
    comp["period"] = [f"{int(y)}-{MONTH_NAMES[int(mm) - 1]}" for y, mm in zip(comp["year"], comp["month"])]
    st.plotly_chart(make_period_bars(comp, a_unit, f"{metric} anomaly in analog years", "period", a_col,
                                     m["warm_is_positive"]), use_container_width=True)


def render_historical():
    st.markdown("#### HISTORICAL & ANALYSIS")
    st.caption("Volue actuals vs 30-year normal by country since 2013 · monthly and weekly · multi-year and "
               "multi-month selections · analogues from weather indexes.")
    tabs = st.tabs(["History by country", "Analogues (weather indexes)"])
    with tabs[0]:
        _render_history()
    with tabs[1]:
        _render_analogues()
