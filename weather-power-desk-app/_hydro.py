"""Section 3 — Hydro Monitoring.

Live version of P:/QFA/TonyWeather/Hydro_Report/quantify_*.py: for each of
the three hydro families (reservoir levels, snow & groundwater, hydro
balance) and each country it shows the climatology chart (grey history, blue
recent years, dashed norm, red current year) and the report's numbers —
anomaly in GWh and % of normal, percentile of the current week, and the
change since last week — plus the stats_*.txt text the report writes.
The maths is in _hydro_quantify.py; data comes from {SBX_SCHEMA}.hydro_daily.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from _config import HYDRO_AREA_CODES, HYDRO_COMPONENTS
from _data import load_hydro_available, load_hydro_series
from _charts import make_hydro_climatology_chart, make_hydro_anomaly_bars
from _hydro_quantify import (
    HYDRO_FAMILIES, HYDRO_DEFAULT_COUNTRIES, build_climatology, quantify_anomaly, split_recent_hist, stats_text,
)
from _ui import kpi_card, kpi_row, status_banner, ordinal


def _pct_class(q: float | None) -> str:
    if q is None or np.isnan(q):
        return "kpi-card-neutral"
    if q <= 15:
        return "kpi-card-critical"
    if q <= 30:
        return "kpi-card-warning"
    if q >= 85:
        return "kpi-card-cool"
    return "kpi-card-neutral"


def _country_kpis(country: str, q: dict, unit: str) -> list[str]:
    if not q:
        return [kpi_card(country, "N/A")]
    anom = q["anomaly"]
    up = anom >= 0
    cls = "kpi-card-cool" if up else "kpi-card-warm"
    d_cls = "kpi-delta-down" if up else "kpi-delta-up"     # more water = blue
    pct = q["anomaly_percent"]
    delta = f'<div class="kpi-delta {d_cls}">{"▲" if up else "▼"} {abs(anom):,.0f} GWh vs norm · {pct}% of normal</div>' \
        if pct is not None else ""
    wk = q.get("week_change_pct_points")
    wk_html = (f'<div class="kpi-rank">{wk:+d} pts vs last week</div>' if wk is not None else "")
    val = kpi_card(f"{country} · latest", f"{q['latest_value']:,.0f} GWh", cls, delta, wk_html)
    qv = q["anomaly_quantile"]
    pc = kpi_card(f"{country} · percentile", f"{ordinal(round(qv))}" if qv is not None else "N/A", _pct_class(qv),
                  f'<div class="kpi-delta kpi-delta-flat">this week vs {q["n_hist_years"]} yrs</div>')
    return [val, pc]


def _compute(country: str, family: str) -> tuple[dict | None, dict, str | None]:
    try:
        actual, normal = load_hydro_series(country, family)
    except Exception as e:
        return None, {}, str(e)
    if actual.empty:
        return None, {}, "no rows"
    try:
        clim = build_climatology(actual, normal if HYDRO_FAMILIES[family]["has_norm"] else None)
        q = quantify_anomaly(clim)
    except Exception as e:
        return None, {}, str(e)
    return clim, q, None


def _render_family(family: str, avail: pd.DataFrame):
    fam = HYDRO_FAMILIES[family]
    comp = HYDRO_COMPONENTS[family]
    st.caption(f"{fam['description']} Volue curve family `res <area> hydro {fam['code']} gwh cet h sa`"
               + (" with the Volue normal `... h n`." if fam["has_norm"] else "; norm = mean across completed years."))

    have = set(avail[(avail["component"] == comp) & (avail["data_type"] == "SA")]["area"]) if not avail.empty else set()
    countries_all = [c for c, code in HYDRO_AREA_CODES.items() if code in have] or list(HYDRO_AREA_CODES.keys())
    defaults = [c for c in HYDRO_DEFAULT_COUNTRIES[family] if c in countries_all] or countries_all[:6]
    missing = [c for c in HYDRO_DEFAULT_COUNTRIES[family] if c not in countries_all]
    if missing and not avail.empty:
        status_banner(f"No {family.lower()} rows for {', '.join(missing)} in hydro_daily — check the Volue area "
                      f"codes ({', '.join(HYDRO_AREA_CODES[c] for c in missing)}).", "warning")

    c1, c2 = st.columns([4, 1.4])
    with c1:
        countries = st.multiselect("Countries", countries_all, defaults, key=f"hy_c_{comp}", label_visibility="collapsed")
    with c2:
        n_recent = st.slider("Recent years highlighted", 1, 5, 3, key=f"hy_n_{comp}")
    if not countries:
        st.info("Select at least one country.")
        return

    results: dict[str, tuple[dict | None, dict, str | None]] = {}
    with st.spinner("Computing climatologies…"):
        for c in countries:
            results[c] = _compute(c, family)

    errs = {c: r[2] for c, r in results.items() if r[2]}
    for c, e in errs.items():
        st.caption(f"{c}: {e}")

    cards = []
    for c in countries:
        cards.extend(_country_kpis(c, results[c][1], fam["unit"]))
    kpi_row(cards, max_cols=6)

    rows = [{"country": c, "anomaly": results[c][1]["anomaly"]} for c in countries if results[c][1]]
    if rows:
        st.plotly_chart(make_hydro_anomaly_bars(rows, f"{family} — anomaly vs normal today (GWh)"),
                        use_container_width=True)

    st.divider()
    st.markdown("##### Climatology by country")
    cols = st.columns(2)
    i = 0
    for c in countries:
        clim, q, err = results[c]
        if clim is None:
            continue
        hist, recent = split_recent_hist(clim["years"], n_recent)
        with cols[i % 2]:
            st.plotly_chart(make_hydro_climatology_chart(clim, hist, recent, f"{c} — {family.lower()}", fam["unit"]),
                            use_container_width=True)
            if q:
                st.caption(f"current anomaly = {q['anomaly']:+,} GWh, at {q['anomaly_percent']}% of seasonal normal · "
                           f"{q['anomaly_quantile']}th percentile · last week {q['anomaly_w-1']:+,} GWh "
                           f"({q['anomaly_percent_w-1']}%) · variation {q['week_change_pct_points']:+d}% "
                           + ("· norm = mean of years" if clim["norm_source"] == "mean_of_years" else ""))
        i += 1

    txt = stats_text(fam["stats_title"], {c: results[c][1] for c in countries})
    with st.expander(f"Report text — stats_{fam['code']}.txt"):
        st.code(txt, language="text")
        st.download_button("Download", txt.encode(), f"stats_{fam['code']}_{dt.date.today():%Y%m%d}.txt", "text/plain",
                           key=f"hy_dl_{comp}")


def render_hydro():
    st.markdown("#### HYDRO MONITORING")
    st.caption("Reservoir levels · snow & groundwater · hydro balance — Volue hydro curves since 2013 vs normal, "
               "the Hydro Report quantify_* figures and statistics computed live.")
    try:
        avail = load_hydro_available()
    except Exception as e:
        st.error(f"Cannot read hydro_daily ({e}). Has power_desk_refresh.py run?")
        return
    if not avail.empty:
        last = avail["last_day"].max()
        st.caption(f"Latest hydro observation: {pd.Timestamp(last):%d %b %Y}")
    tabs = st.tabs(list(HYDRO_FAMILIES.keys()))
    for tab, family in zip(tabs, HYDRO_FAMILIES.keys()):
        with tab:
            _render_family(family, avail)
