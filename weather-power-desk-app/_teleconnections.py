"""Historical & Analysis → Teleconnections.

How the weather indices (all_daily_indices.csv → weather_indexes) relate to what the desk
trades: temperature, wind, solar and precipitation energy by country, month by month or
quarter by quarter, across the years.

  Setup        one period (a month or a calendar quarter), a year range, an index lead,
               a target (metric × sources × countries) and the indices to work with —
               one or several; several can be blended into a single index
  Indices      where the indices are, and the selected period's index by year
  Correlations indices × countries for the period, the index's seasonal profile, a scatter
  By phase     the target's anomaly when the index is high / low / neutral, or when
               several indices line up at once (joint regime)
  Regression   the target on several indices together, with its out-of-sample skill
  Analogues    the years whose index looked most like now, and what the target did then

Sources of the same metric (Meteologica reanalysis and Volue actuals for wind and solar)
sit side by side as adjacent columns of the same grid, never in separate stacked tables.
The statistics are in _tele_stats.py (no Streamlit); this module is layout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import _tele_stats as ts
from _config import (
    METRICS, DEFAULT_METRIC, MONTH_NAMES, DEFAULT_AREAS, ANALOG_N_YEARS, SBX_SCHEMA, HIST_DEFAULT_SOURCE,
    TELE_INDICES, TELE_DEFAULT_INDICES, TELE_AREAS, TELE_SOURCES, TELE_SOURCE_LABELS, TELE_SOURCE_SHORT,
    TELE_REANALYSIS_BASE_YEARS,
)
from _data import (
    load_index_monthly, load_index_recent, load_reanalysis_monthly, load_hist_monthly_all, hist_sources,
)
from _charts import (
    make_corr_heatmap, make_index_period_bars, make_index_recent_chart, make_tele_scatter, make_composite_bars,
    make_regression_fit, make_year_month_heatmap,
)
from _ui import kpi_card, kpi_row, status_banner

CORR_NOTE = ("* p < 0.05, ** p < 0.01 for each cell on its own — with this many cells about one in twenty shows a "
             "star by chance, so trust patterns (neighbouring countries, adjacent months, a physical reason) over a "
             "single starred cell. Few years behind a cell (hover) means a wide margin: Volue's history is 13 years.")


# ══════════════════════════════════════════════════════════════════════════════
# DATA ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def _available_sources(metric: str) -> list[str]:
    """The sources of `metric` that actually have rows right now, in TELE_SOURCES order."""
    avail = hist_sources()
    out = []
    for s in TELE_SOURCES[metric]:
        if s == "Reanalysis":
            if not load_reanalysis_monthly(metric, ("DE",)).empty:
                out.append(s)
        elif s in avail.get(metric, []) or (not avail and s == HIST_DEFAULT_SOURCE):
            out.append(s)
    return out


def _monthly_targets(metric: str, sources: list[str], areas: list[str]) -> pd.DataFrame:
    """Monthly (area, source, year, month, actual, normal, n_days) across the chosen sources."""
    parts = []
    for src in sources:
        if src == "Reanalysis":
            m = ts.reanalysis_monthly(load_reanalysis_monthly(metric, tuple(areas)), TELE_REANALYSIS_BASE_YEARS)
        else:
            known = hist_sources()
            m = load_hist_monthly_all(metric, tuple(areas), src if known else None)
            m = m[["area", "year", "month", "actual", "normal", "n_days"]].copy() if not m.empty else m
        if not m.empty:
            m = m.copy()
            m["source"] = src
            parts.append(m[["area", "source", "year", "month", "actual", "normal", "n_days"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _column_labels(tgt: pd.DataFrame, areas: list[str], sources: list[str]) -> dict[tuple[str, str], str]:
    """(area, source) → axis label, areas in the user's order and, within a country, the
    sources side by side. Single-source selections drop the source from the label."""
    have = set(zip(tgt["area"], tgt["source"]))
    multi = len({s for _, s in have}) > 1
    out = {}
    for a in areas:
        for s in sources:
            if (a, s) in have:
                out[(a, s)] = f"{a}<br>{TELE_SOURCE_SHORT.get(s, s)}" if multi else a
    return out


def _period_label(resolution: str, period: int) -> str:
    return MONTH_NAMES[period - 1] if resolution == "Monthly" else ts.QUARTER_LABELS[period - 1]


def _grid(long: pd.DataFrame, labels: dict, value: str) -> np.ndarray:
    """long (area, source, <value>) → row vector aligned to `labels`' order."""
    s = long.set_index(["area", "source"])[value]
    return np.array([s.get(k, np.nan) for k in labels])


# ══════════════════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render_teleconnections():
    raw = load_index_monthly()
    raw = raw[raw["index_name"].isin(TELE_INDICES)] if not raw.empty else raw
    if raw.empty:
        status_banner(f"The weather indices are not loaded yet. This tab reads {SBX_SCHEMA}.weather_indexes "
                      "(index_name, date, value) — run pipeline/indices_upload.py (python run_daily.py --only indices) "
                      "to load all_daily_indices.csv.", "warning")
        st.dataframe(pd.DataFrame({"index": list(TELE_INDICES), "group": [v[0] for v in TELE_INDICES.values()],
                                   "description": [v[1] for v in TELE_INDICES.values()]}),
                     use_container_width=True, hide_index=True)
        return
    series = ts.index_series(raw)
    names_avail = [n for n in TELE_INDICES if n in series]

    # ── setup ────────────────────────────────────────────────────────────────
    now = pd.Timestamp.today()
    y_lo, y_hi = max(1979, int(raw["year"].min())), int(raw["year"].max())
    s1, s2, s3, s4, s5 = st.columns([1.1, 1.5, 2.4, 1.3, 1.2])
    with s1:
        resolution = st.radio("Resolution", list(ts.RESOLUTIONS), key="tc_res", horizontal=True)
    with s2:
        periods = list(range(1, ts.n_periods(resolution) + 1))
        default_p = now.month if resolution == "Monthly" else (now.month - 1) // 3 + 1
        period = st.selectbox("Period", periods, index=periods.index(default_p),
                              format_func=lambda p: _period_label(resolution, p), key=f"tc_period_{resolution}")
    with s3:
        years = st.slider("Years", y_lo, y_hi, (y_lo, y_hi), key="tc_years")
    with s4:
        lead = st.slider("Index leads by (months)", 0, 6, 0, key="tc_lead",
                         help="0 = the index in the same period as the target. 2 = the index as it was two months "
                              "before the target period — what is usable as a forecast signal.")
    with s5:
        detrend = st.toggle("Remove linear trend", True, key="tc_detrend",
                            help="Takes the linear trend out of both the index and the target before every statistic. "
                                 "Temperature warms, Volue's wind and solar fleet grows, AMO/PDO/NINO drift: without "
                                 "this the correlations partly measure the trend.")

    t1, t2, t3 = st.columns([1.3, 2.4, 3])
    with t1:
        metric = st.selectbox("Metric", list(METRICS.keys()), index=list(METRICS).index(DEFAULT_METRIC), key="tc_metric")
    sources_av = _available_sources(metric)
    with t2:
        sources = st.multiselect("Sources", sources_av, sources_av, format_func=lambda s: TELE_SOURCE_LABELS.get(s, s),
                                 key=f"tc_src_{metric}")
    with t3:
        areas = st.multiselect("Countries", list(TELE_AREAS), DEFAULT_AREAS, format_func=lambda a: f"{a} — {TELE_AREAS[a]}",
                               key="tc_areas")

    i1, i2 = st.columns([4, 1.4])
    with i1:
        sel = st.multiselect("Indices (one or several — several can be blended)", names_avail,
                             [n for n in TELE_DEFAULT_INDICES if n in names_avail],
                             format_func=lambda n: f"{n} — {TELE_INDICES[n][1]}", key="tc_idx")
    with i2:
        thr = st.slider("High / low beyond ±σ", 0.25, 1.5, 0.5, 0.05, key="tc_thr",
                        help="An index is 'high' above +σ, 'low' below −σ, in z-scores of its own history for the period.")
    if not sel:
        st.info("Pick at least one index.")
        return
    if not sources:
        status_banner(f"No data source with rows for {metric} — see the Sources selector.", "warning")
        return
    if not areas:
        st.info("Pick at least one country.")
        return

    signs: dict[str, int] = {}
    if len(sel) >= 2:
        st.caption("Blend — the mean of the selected indices' z-scores, standardised again. Tick *invert* to subtract an "
                   "index instead of adding it (NAO − SCAND is a classic).")
        ncol = min(len(sel), 8)
        cols = st.columns(ncol)
        for i, n in enumerate(sel):
            signs[n] = -1 if cols[i % ncol].checkbox(f"invert {n}", key=f"tc_inv_{n}") else 1

    # ── index frames (z per calendar period) ────────────────────────────────
    frames = {n: ts.zscores(ts.at_resolution(series[n], resolution, lead)) for n in sel}
    frames_m = {n: ts.zscores(ts.at_resolution(series[n], "Monthly", lead)) for n in sel}
    blend_name = f"Blend: {ts.blend_label(sel, signs)}" if len(sel) >= 2 else None
    blend_f = ts.blend(frames, signs) if blend_name else None
    blend_m = ts.blend(frames_m, signs) if blend_name else None
    focus_opts = sel + ([blend_name] if blend_name else [])
    focus = st.selectbox("Focus — the index the profile, scatter, phase and analogue views use", focus_opts,
                         index=len(focus_opts) - 1 if blend_name else 0, key="tc_focus")
    focus_f = blend_f if focus == blend_name else frames[focus]
    focus_m = blend_m if focus == blend_name else frames_m[focus]
    all_frames = dict(frames)
    if blend_name:
        all_frames[blend_name] = blend_f
    if focus_f is None or focus_f.empty or focus_f["z"].notna().sum() < ts.MIN_YEARS:
        status_banner("Not enough overlapping years of the selected indices (they have to exist together) — "
                      "remove the index with the shortest history (SOI starts in 1991, the NINO regions in 1981).", "warning")

    # ── target ──────────────────────────────────────────────────────────────
    m = METRICS[metric]
    a_unit = m["unit"] if m["anomaly_kind"] == "diff" else "%"
    try:
        monthly = _monthly_targets(metric, sources, areas)
    except Exception as e:
        st.error(f"Failed to load {metric}: {e}")
        return
    if monthly.empty:
        status_banner(f"No {metric} rows for the chosen countries and sources.", "warning")
        return
    tgt = ts.targets_at_resolution(monthly, resolution, m["anomaly_kind"])
    labels = _column_labels(tgt, areas, sources)
    if not labels:
        status_banner("None of the chosen countries has data in the chosen sources.", "warning")
        return
    plabel = _period_label(resolution, period)
    ctx = (f"{plabel} · {years[0]}–{years[1]} · index leads {lead} mo" if lead else f"{plabel} · {years[0]}–{years[1]}")
    trend_note = "linear trend removed" if detrend else "trend kept"
    if "Reanalysis" in sources:
        st.caption(f"Reanalysis: Meteologica ERA5-driven generation of today's fleet (one vintage, constant capacity — "
                   f"pure weather), anomaly = % of its {TELE_REANALYSIS_BASE_YEARS[0]}–{TELE_REANALYSIS_BASE_YEARS[1]} "
                   f"mean for the month. Volue: actuals vs the Volue normal, and its fleet grows over time.")

    tabs = st.tabs(["Indices", "Correlations", "Anomalies by phase", "Regression", "Analogue years"])

    # ══ INDICES ═══════════════════════════════════════════════════════════════
    with tabs[0]:
        _render_indices(sel, raw, all_frames, focus, focus_f, focus_m, period, plabel, thr, years)

    # ══ CORRELATIONS ══════════════════════════════════════════════════════════
    with tabs[1]:
        _render_correlations(all_frames, focus, focus_f, tgt, labels, areas, sources, resolution, period, plabel,
                             detrend, years, ctx, trend_note, metric, a_unit)

    # ══ BY PHASE ══════════════════════════════════════════════════════════════
    with tabs[2]:
        _render_phase(sel, frames, focus, focus_f, tgt, labels, period, plabel, detrend, years, thr, a_unit, metric, ctx)

    # ══ REGRESSION ════════════════════════════════════════════════════════════
    with tabs[3]:
        _render_regression(sel, frames, tgt, labels, period, plabel, detrend, years, a_unit, metric, ctx)

    # ══ ANALOGUES ═════════════════════════════════════════════════════════════
    with tabs[4]:
        _render_analogues(focus, focus_f, focus_m, tgt, labels, period, plabel, detrend, years, thr, a_unit, metric, ctx)


# ──────────────────────────────────────────────────────────────────────────────

def _render_indices(sel, raw, all_frames, focus, focus_f, focus_m, period, plabel, thr, years):
    recent = load_index_recent(45)
    cards = []
    for n in sel:
        d = recent[recent["index_name"] == n].sort_values("date") if not recent.empty else recent
        if d.empty:
            cards.append(kpi_card(n, "n/a", "kpi-card-neutral"))
            continue
        last = d.iloc[-1]
        # month-to-date z against the same calendar month's complete history
        r = raw[raw["index_name"] == n]
        cur = r.sort_values(["year", "month"]).iloc[-1]
        hist = r[(r["month"] == cur["month"]) & (r["year"] < cur["year"]) & (r["year"] >= 1979)]["value"]
        z = (cur["value"] - hist.mean()) / hist.std() if len(hist) >= ts.MIN_YEARS and hist.std() > 0 else np.nan
        cls = "kpi-card-neutral" if not np.isfinite(z) or abs(z) < 1 else ("kpi-card-warm" if z > 0 else "kpi-card-cool")
        delta = (f'<div class="kpi-delta">{last["date"]:%d %b} · {MONTH_NAMES[int(cur["month"]) - 1]} so far '
                 f'{z:+.1f}σ</div>') if np.isfinite(z) else f'<div class="kpi-delta">{last["date"]:%d %b}</div>'
        cards.append(kpi_card(f"{n} · latest daily", f"{last['value']:+.2f}", cls, delta))
    kpi_row(cards)
    c1, c2 = st.columns([1.5, 1])
    with c1:
        z = focus_f[focus_f["period"] == period].set_index("year")["z"]
        z = z[(z.index >= years[0]) & (z.index <= years[1])]
        st.plotly_chart(make_index_period_bars(z, f"{focus} · {plabel}", thr), use_container_width=True)
    with c2:
        if recent.empty:
            st.info("No daily values in the last 45 days.")
        else:
            st.plotly_chart(make_index_recent_chart(recent, sel), use_container_width=True)
    hm = focus_m.rename(columns={"period": "month", "z": "anomaly"}).assign(area=focus)
    hm = hm[(hm["year"] >= years[0]) & (hm["year"] <= years[1])]
    st.plotly_chart(make_year_month_heatmap(hm, focus, "σ", "anomaly", title=f"{focus} — monthly z-score by year"),
                    use_container_width=True)
    with st.expander("Values — table (z-scores of the selected period, by year)"):
        tab = pd.concat([f[f["period"] == period].set_index("year")["z"].rename(n) for n, f in all_frames.items()],
                        axis=1).sort_index(ascending=False)
        tab = tab[(tab.index >= years[0]) & (tab.index <= years[1])]
        st.dataframe(tab.round(2), use_container_width=True)
        st.download_button("Download CSV", tab.to_csv().encode(), f"indices_{plabel}.csv", "text/csv", key="tc_dl_idx")


def _render_correlations(all_frames, focus, focus_f, tgt, labels, areas, sources, resolution, period, plabel,
                         detrend, years, ctx, trend_note, metric, a_unit):
    grid = ts.corr_grid(all_frames, tgt, period, detrend, years)
    rows = list(all_frames)
    r = np.vstack([_grid(grid[grid["index"] == n], labels, "r") for n in rows])
    p = np.vstack([_grid(grid[grid["index"] == n], labels, "p") for n in rows])
    n_ = np.vstack([_grid(grid[grid["index"] == n], labels, "n") for n in rows])
    st.plotly_chart(make_corr_heatmap(r, p, n_, list(labels.values()), rows,
                                      f"{metric} anomaly vs index · {ctx} · {trend_note}"), use_container_width=True)
    st.caption(CORR_NOTE)

    prof = ts.profile_grid(focus_f, tgt, resolution, detrend, years)
    keys = list(labels)
    plabels = [_period_label(resolution, q) if resolution == "Quarterly" else MONTH_NAMES[q - 1]
               for q in range(1, ts.n_periods(resolution) + 1)]
    ylab = [f"{a} · {TELE_SOURCE_SHORT.get(s, s)}" if len({s for _, s in keys}) > 1 else a for a, s in keys]
    rr = np.full((len(keys), len(plabels)), np.nan)
    pp, nn = rr.copy(), rr.copy()
    for (a, s), g in prof.groupby(["area", "source"]):
        if (a, s) in keys:
            i = keys.index((a, s))
            rr[i, g["period"].values - 1], pp[i, g["period"].values - 1], nn[i, g["period"].values - 1] = g["r"], g["p"], g["n"]
    st.plotly_chart(make_corr_heatmap(rr, pp, nn, plabels, ylab,
                                      f"{focus} — correlation with the {metric} anomaly through the year · "
                                      f"{years[0]}–{years[1]} · {trend_note}"), use_container_width=True)

    st.markdown("##### Scatter — one country, sources side by side")
    area = st.selectbox("Country", areas, format_func=lambda a: f"{a} — {TELE_AREAS[a]}", key="tc_sc_area")
    srcs = [s for s in sources if (area, s) in labels]
    if not srcs:
        st.info("That country has no data in the chosen sources.")
    else:
        cols = st.columns(len(srcs))
        x_all = focus_f[focus_f["period"] == period].set_index("year")["z"].sort_index()
        for col, s in zip(cols, srcs):
            y_all = tgt[(tgt["area"] == area) & (tgt["source"] == s) & (tgt["period"] == period)].set_index("year")["anomaly"].sort_index()
            df = pd.concat([x_all.rename("x"), y_all.rename("y")], axis=1, join="inner").dropna()
            df = df[(df.index >= years[0]) & (df.index <= years[1])]
            r_, p_, _ = ts.pearson(df["x"], df["y"], detrend)
            if detrend and len(df) >= 3:
                df = pd.concat([ts.detrended(df["x"]).rename("x"), ts.detrended(df["y"]).rename("y")], axis=1)
            df["year"] = df.index
            with col:
                st.plotly_chart(make_tele_scatter(df, f"{focus} (σ)", f"anomaly ({a_unit})",
                                                  f"{area} · {TELE_SOURCE_LABELS.get(s, s)}", r_, p_), use_container_width=True)
    with st.expander("Correlations — table"):
        flat = grid.assign(column=[labels.get((a, s), "") .replace("<br>", " ") for a, s in zip(grid["area"], grid["source"])])
        st.dataframe(flat.pivot_table(index="index", columns="column", values="r", sort=False).round(2), use_container_width=True)
        st.download_button("Download CSV", grid.to_csv(index=False).encode(), f"tele_corr_{metric}_{plabel}.csv", "text/csv",
                           key="tc_dl_corr")


def _group_years(groups: pd.Series, order: list[str]) -> str:
    return " · ".join(f"**{g}** ({int((groups == g).sum())}): " +
                      (", ".join(str(int(y)) for y in groups[groups == g].index) or "—") for g in order)


def _composite_block(tgt, labels, period, groups, order, detrend, years, a_unit, title, key):
    comp = ts.composite(tgt, period, groups, order, detrend, years)
    st.plotly_chart(make_composite_bars(comp, labels, order, a_unit, title), use_container_width=True)
    with st.expander("Composites — table (mean anomaly; countries as rows, sources and groups as columns)"):
        t = comp.pivot_table(index="area", columns=["source", "group"], values="mean", sort=False).round(2)
        st.dataframe(t, use_container_width=True)
        st.download_button("Download CSV", comp.to_csv(index=False).encode(), f"{key}.csv", "text/csv", key=f"{key}_dl")
    return comp


def _render_phase(sel, frames, focus, focus_f, tgt, labels, period, plabel, detrend, years, thr, a_unit, metric, ctx):
    mode = st.radio("Group the years by", ["Phase of the focus index", "Joint regime — several indices at once"],
                    horizontal=True, key="tc_phase_mode")
    if mode.startswith("Phase"):
        z = focus_f[focus_f["period"] == period].set_index("year")["z"].dropna()
        z = z[(z.index >= years[0]) & (z.index <= years[1])]
        groups = ts.classify(z, thr)
        order = ["Low", "Neutral", "High"]
        st.caption(_group_years(groups, order))
        _composite_block(tgt, labels, period, groups, order, detrend, years, a_unit,
                         f"{metric} anomaly when {focus} is low / neutral / high · {ctx}", f"tele_phase_{metric}_{plabel}")
        return
    st.caption(f"Pick a condition per index. A year is in the regime when every condition holds at once "
               f"(positive = at or above +{thr:g}σ, negative = at or below −{thr:g}σ).")
    cond: dict[str, str] = {}
    cols = st.columns(min(len(sel), 6))
    for i, n in enumerate(sel):
        default = 0 if i == 0 else (1 if i == 1 else 2)
        cond[n] = cols[i % len(cols)].selectbox(n, ["Positive", "Negative", "Ignore"], index=default, key=f"tc_cond_{n}")
    zs = {n: f[f["period"] == period].set_index("year")["z"] for n, f in frames.items()}
    zs = {n: z[(z.index >= years[0]) & (z.index <= years[1])] for n, z in zs.items()}
    groups = ts.regime_years(zs, cond, thr)
    if groups.empty:
        st.info("Set at least one index to Positive or Negative.")
        return
    n_reg = int((groups == "Regime").sum())
    st.caption(_group_years(groups, ["Regime", "Rest"]))
    if n_reg < 4:
        status_banner(f"Only {n_reg} year(s) meet all the conditions — a composite of that few years is an anecdote, "
                      "not a statistic. Loosen the threshold or drop a condition.", "warning")
    if n_reg == 0:
        return
    _composite_block(tgt, labels, period, groups, ["Regime", "Rest"], detrend, years, a_unit,
                     f"{metric} anomaly in the joint regime vs all other years · {ctx}", f"tele_regime_{metric}_{plabel}")


def _render_regression(sel, frames, tgt, labels, period, plabel, detrend, years, a_unit, metric, ctx):
    pred = st.multiselect("Predictors (2–4 work best — with ~45 years more starts to fit noise)", sel, sel[:3], max_selections=4,
                          key="tc_reg_pred")
    if not pred:
        st.info("Pick at least one predictor.")
        return
    X = pd.concat([frames[n][frames[n]["period"] == period].set_index("year")["z"].rename(n) for n in pred], axis=1, join="inner")
    rows, fits = [], {}
    for (a, s), lab in labels.items():
        y = tgt[(tgt["area"] == a) & (tgt["source"] == s) & (tgt["period"] == period)].set_index("year")["anomaly"].sort_index()
        o = ts.ols(X, y, detrend, years)
        if o is None:
            continue
        fits[(a, s)] = o
        row = {"country": a, "source": TELE_SOURCE_SHORT.get(s, s), "years": o["n"],
               "R² (in-sample)": o["r2"], "adj. R²": o["adj_r2"], "R² leave-one-year-out": o["loo_r2"]}
        for n in pred:
            row[f"β {n}"] = f"{o['beta'][n]:+.2f}" + ("*" if o["p"][n] < 0.05 else "")
        rows.append(row)
    if not rows:
        st.info("Too few overlapping years for this many predictors.")
        return
    tab = pd.DataFrame(rows)
    st.dataframe(tab.style.format({c: "{:.2f}" for c in tab.columns if c.startswith(("R²", "adj"))}),
                 use_container_width=True, hide_index=True)
    st.caption(f"{metric} anomaly ({a_unit}) regressed on the standardised indices, {ctx}. β is per standard deviation of "
               "the index (both sides standardised), * p < 0.05. **R² leave-one-year-out** is the out-of-sample skill: each "
               "year predicted by a model that never saw it. If it is near zero or negative the in-sample R² is "
               "overfitting, whatever its size.")
    pick = st.selectbox("Show the fit for", list(fits), format_func=lambda k: f"{k[0]} · {TELE_SOURCE_LABELS.get(k[1], k[1])}",
                        key="tc_reg_pick")
    o = fits[pick]
    st.plotly_chart(make_regression_fit(o["years"], o["observed"], o["fitted"],
                                        f"{pick[0]} {metric} anomaly — observed vs fitted from {', '.join(pred)} · {plabel}",
                                        a_unit), use_container_width=True)


def _render_analogues(focus, focus_f, focus_m, tgt, labels, period, plabel, detrend, years, thr, a_unit, metric, ctx):
    z = focus_f[focus_f["period"] == period].set_index("year")["z"].dropna()
    z = z[(z.index >= years[0]) & (z.index <= years[1])]
    if z.empty:
        st.info("No index values for that period and years.")
        return
    latest = focus_m.sort_values(["year", "period"]).iloc[-1]
    c1, c2 = st.columns(2)
    with c1:
        n_an = st.slider("Analogue years", 3, 10, ANALOG_N_YEARS, key="tc_an_n")
    with c2:
        target = st.slider(f"Assumed {focus} (σ)", -3.0, 3.0, float(np.clip(round(float(latest["z"]), 2), -3, 3)), 0.05, key="tc_an_target",
                           help=f"Defaults to the latest value: {latest['z']:+.2f}σ in {MONTH_NAMES[int(latest['period']) - 1]} {int(latest['year'])}.")
    this_year = pd.Timestamp.today().year
    an = ts.nearest_years(z, target, n_an, exclude={this_year})
    st.caption("Analogue years: " + ", ".join(f"{int(y)} ({v:+.2f}σ)" for y, v in an.items()))
    st.plotly_chart(make_index_period_bars(z, f"{focus} · {plabel}", thr, list(an.index)), use_container_width=True)
    groups = pd.Series(np.where(z.index.isin(an.index), "Analogues", "Rest"), index=z.index)
    _composite_block(tgt, labels, period, groups, ["Analogues", "Rest"], detrend, years, a_unit,
                     f"{metric} anomaly in the analogue years vs the rest · {ctx}", f"tele_analogues_{metric}_{plabel}")
