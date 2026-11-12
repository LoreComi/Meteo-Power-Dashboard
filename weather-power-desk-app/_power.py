"""Power Demand — Section 5.

Two views on one question, "what does temperature do to electricity load":

  Load vs temperature      the record. Every day of actual load since 2015 for one
                           country against that day's temperature; pick the years and
                           months, see the cloud per year (or per month) with its fitted
                           curve, the slope of each curve at fixed temperatures side by
                           side, and a year × month grid of the load change per °C.
  Temperature sensitivity  the forecast. The country's curve evaluated on the current
                           temperature forecast: how much load the coming week gains or
                           loses if it comes in one degree colder or warmer than forecast,
                           per country and per day, with the members' own range.

The curves are the Gas Demand section's hinge curves (heating branch, cooling branch,
thermal inertia, recency weights) refitted on load, plus a seasonal term — see
_power_demand_model.py. The sensitivity is read off the fitted curve, not off the
forecast's members; _config.py (POWER DEMAND) says why, and the member-implied figure
stays in the grid as a cross-check.

Data: {SBX_SCHEMA}.power_load_daily (history) and power_load_fcst (the latest
temperature / load forecast per model), both written by Power_dashboard/pipeline
(eq_load.py, `run_daily.py --only load`).
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import streamlit as st

import _power_demand_model as pdm
from _config import (
    AREAS, MONTH_NAMES, area_label, POWER_LOAD_TABLE, POWER_LOAD_FCST_TABLE, POWER_DEFAULT_AREA, POWER_DEFAULT_AREAS,
    POWER_FCST_MODELS, POWER_DEFAULT_MONTHS, POWER_RECENT_DAYS, POWER_SEED_DAYS, POWER_SHIFT_STEPS_C,
    POWER_SHIFT_DEFAULT_C, POWER_SLOPE_REF_TEMPS_C, POWER_FLAT_PCT_PER_C, POWER_MEMBER_MIN_R2, POWER_PARTIAL_DAYS,
)
from _charts import (
    make_load_scatter, make_load_slope_heatmap, make_sensitivity_bars, make_sensitivity_days, make_response_curve,
    power_group_colours,
)
from _data import load_power_areas, load_power_load, load_power_fcst, data_version, prefetch, VERSIONED_TTL
from _ui import info_tip, kpi_card, kpi_row, status_banner, view_switch

VIEWS = ["Load vs temperature", "Temperature sensitivity"]
WINDOWS = ["Rest of the week", "Next week", "Week + 2"]
DEFAULT_WINDOW = "Next week"
FCST_STALE_HOURS = 36


@st.cache_data(ttl=VERSIONED_TTL, max_entries=32, show_spinner=False)
def _curve(area: str, version: str) -> pdm.LoadCurve | None:
    """The country's curve from its whole history — refitted when the table is rewritten.
    None when the history is too short to hold a year out."""
    return pdm.fit_curve(load_power_load(area), area)


def _fmt(v, fmt="{:+,.0f}", na="n/a") -> str:
    return na if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def _segments(curve: pdm.LoadCurve) -> str:
    return " · ".join(f"{label} {slope:+,.0f}" for label, slope in curve.segments())


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 1 — LOAD VS TEMPERATURE
# ══════════════════════════════════════════════════════════════════════════════

def _history(area: str, curve: pdm.LoadCurve | None, effective: bool) -> pd.DataFrame:
    """The country's actual days with the scatter's x (day or effective temperature),
    year, month and daytype."""
    hist = load_power_load(area)
    if hist.empty:
        return hist
    temp = hist.set_index("day")["temp"]
    temp = temp.reindex(pd.date_range(temp.index.min(), temp.index.max(), freq="D")).interpolate(limit=3)
    if effective and curve is not None:
        temp = pdm.effective_temperature(temp, curve.half_life_days)
    d = hist.dropna(subset=["load"]).copy()
    d["x"] = temp.reindex(d["day"]).to_numpy()
    d = d.dropna(subset=["x"])
    d["year"], d["month"] = d["day"].dt.year, d["day"].dt.month
    d["daytype"] = pdm.assign_daytype(d["day"], area)
    return d


def _slope_at(curve: pdm.LoadCurve, t: float) -> float:
    """The curve's MW per °C at `t`, NaN outside the temperatures it was fitted on."""
    return float(curve.slope(np.array([float(t)]))[0]) if curve.t_min <= t <= curve.t_max else np.nan


def _render_explorer(areas: list[str]) -> None:
    c1, c2, c3 = st.columns([1.3, 2.6, 2.6])
    with c1:
        area = st.selectbox("Country", areas, index=areas.index(POWER_DEFAULT_AREA) if POWER_DEFAULT_AREA in areas else 0,
                            format_func=area_label, key="pw_ex_area")
    try:
        with st.spinner("Loading the load history…"):
            curve = _curve(area, data_version(POWER_LOAD_TABLE))
    except Exception as e:
        st.error(f"Cannot read the load history from {POWER_LOAD_TABLE}: {e}")
        return
    hl = curve.half_life_days if curve is not None else 0.0
    o1, o2, o3 = st.columns([1.4, 1.6, 3.0])
    with o1:
        working = st.checkbox("Working days only", value=True, key="pw_ex_working",
                              help="Leaves out Saturdays, Sundays and public holidays, whose lower load would "
                                   "otherwise sit as a second cloud under the working days.")
    with o2:
        by = st.radio("Colour by", ["Year", "Month"], horizontal=True, key="pw_ex_by")
    with o3:
        temp_opts = ["Day"] + (["Effective"] if hl > 0 else [])
        temp_kind = st.radio("Temperature", temp_opts, horizontal=True, key=f"pw_ex_temp_{len(temp_opts)}",
                             format_func=lambda k: "Of the day" if k == "Day" else f"Effective ({hl:g}-day half-life)",
                             help="Effective = the day and the days before it, weighted with the half-life this "
                                  "country's curve was fitted with: load answers a cold spell, not one cold day.")
    d = _history(area, curve, temp_kind == "Effective")
    if d.empty:
        status_banner(f"No load for {area_label(area)} in {POWER_LOAD_TABLE}.", "warning")
        return

    years_avail = sorted(d["year"].unique())
    # opens on the last calendar months with load (over a turn of the year: both years)
    latest = pd.period_range(end=d["day"].max().to_period("M"), periods=POWER_DEFAULT_MONTHS, freq="M")
    with c2:
        years = st.multiselect("Years", years_avail, [y for y in years_avail if y in set(latest.year)],
                               key=f"pw_ex_years_{area}")
    with c3:
        months = st.multiselect("Months", list(range(1, 13)), sorted(set(latest.month)),
                                format_func=lambda m: MONTH_NAMES[m - 1], key="pw_ex_months")
    if not (years and months):
        st.info("Select at least one year and one month.")
        return
    if (d["load_source"] == "synthetic").any():
        syn = d[d["load_source"] == "synthetic"]["day"]
        status_banner(f"{area_label(area)}: no published actual load from {syn.min():%b %Y} — those days are the "
                      "provider's reconstructed consumption.", "warning")

    pool = d[d["daytype"] == "weekday"] if working else d
    sel = pool[pool["year"].isin(years) & pool["month"].isin(months)]
    if sel.empty:
        status_banner("No days in that selection.", "warning")
        return
    x_title = "Temperature of the day (°C)" if temp_kind == "Day" else "Effective temperature (°C)"

    # the selection as one curve (the cards), then one per year or month (the chart, the table)
    whole = pdm.fit_subset(sel["day"], sel["load"], sel["x"], area)
    cards = [kpi_card("Days selected", f"{len(sel):,}", "kpi-card-neutral",
                      f'<div class="kpi-delta">{sel["load"].mean() / 1000:,.1f} GW mean load</div>',
                      f'<div class="kpi-rank">{sel["x"].min():.0f} to {sel["x"].max():.0f} °C</div>')]
    if whole is not None:
        for label, slope in whole.segments():
            cards.append(kpi_card(label[0].upper() + label[1:], f"{slope:+,.0f} MW/°C",
                                  "kpi-card-cool" if slope < 0 else "kpi-card-warm",
                                  f'<div class="kpi-delta">{slope / whole.mean_load * 100:+.2f} % of load per °C</div>',
                                  f'<div class="kpi-rank">{"more load when colder" if slope < 0 else "more load when warmer"}</div>'))
        cards.append(kpi_card("Fit of the selection", f"R² {whole.r2:.2f}", "kpi-card-neutral",
                              f'<div class="kpi-delta">± {whole.resid_std:,.0f} MW day to day</div>'))
    kpi_row(cards)

    by_month = by == "Month"
    key_col = "month" if by_month else "year"
    labels = {k: (MONTH_NAMES[k - 1] if by_month else str(k)) for k in sorted(sel[key_col].unique())}
    colours = power_group_colours(list(labels.values()), by_month=by_month)
    points = sel.assign(group=sel[key_col].map(labels))
    fits, curves = {}, {}
    for k, label in labels.items():
        g = sel[sel[key_col] == k]
        fit = pdm.fit_subset(g["day"], g["load"], g["x"], area)
        if fit is None:
            continue
        fits[label] = (fit, g)
        xs = np.linspace(fit.t_min, fit.t_max, 80)
        curves[label] = (xs, fit.curve(xs))
    recent = pool.sort_values("day").tail(POWER_RECENT_DAYS)       # the latest days, whatever the selection
    st.plotly_chart(make_load_scatter(points, curves, colours,
                                      f"{area_label(area)} — daily load against temperature"
                                      f"{' · working days' if working else ''}", x_title, recent=recent),
                    use_container_width=True)
    info_tip(f"The dark path is the last {len(recent)} {'working ' if working else ''}days with load, in order, ending "
             f"on the labelled day ({recent['day'].max():%a %d %b %Y}) — where load has just been against the "
             "selection behind it. "
             "Each dot is one day; each line is the curve fitted to its own year (or month): straight segments that "
             "may bend at a heating and at a cooling threshold, as the Gas Demand curves do. The lines are the plain "
             "relation in the dots, so across a whole year they also carry the season (dark, cold months have more "
             "load for reasons other than temperature). Pick single months to take the season out — or read the grid "
             "below, which fits each month on its own. The Temperature sensitivity view uses the country's full "
             "curve, which removes the season explicitly.")

    if fits:
        recs = []
        for label, (fit, g) in fits.items():
            rec = {by: label, "Days": len(g), "Mean temp (°C)": g["x"].mean(), "Mean load (GW)": g["load"].mean() / 1000}
            for t in POWER_SLOPE_REF_TEMPS_C:
                rec[f"at {t} °C"] = _slope_at(fit, t)
            rec["R²"] = fit.r2
            rec["Curve (MW per °C)"] = _segments(fit)
            recs.append(rec)
        table = pd.DataFrame(recs)
        ref_cols = [c for c in table.columns if c.startswith("at ") and table[c].notna().any()]
        table = table[[by, "Days", "Mean temp (°C)", "Mean load (GW)"] + ref_cols + ["R²", "Curve (MW per °C)"]]
        st.markdown(f"###### Load change per °C, by {by.lower()} — each curve's slope at the same temperatures (MW/°C)")
        st.dataframe(table.style.format({"Mean temp (°C)": "{:.1f}", "Mean load (GW)": "{:,.1f}", "R²": "{:.2f}",
                                         **{c: "{:+,.0f}" for c in ref_cols}}, na_rep="–"),
                     use_container_width=True, hide_index=True)

    grid = pd.DataFrame([(y, m, *pdm.linear_slope(g["x"], g["load"])[:2]) for (y, m), g in pool.groupby(["year", "month"])],
                        columns=["year", "month", "slope", "r2"])
    st.plotly_chart(make_load_slope_heatmap(grid.pivot(index="year", columns="month", values="slope"),
                                            grid.pivot(index="year", columns="month", values="r2"),
                                            f"{area_label(area)} — load change per °C within each month (MW/°C), every year"),
                    use_container_width=True)
    info_tip("A straight line through the days of one month of one year. Blue: load rises as it gets colder (heating). "
             "Red: load rises as it gets warmer (cooling). A cell without a number has a line that explains little of "
             "that month (R² under 0.15) — usually a month in the comfort range, where temperature does not move load.")

    st.download_button("Download the selected days (CSV)",
                       sel[["day", "load", "temp", "x", "daytype", "load_normal", "temp_normal", "load_source"]]
                       .rename(columns={"load": "load_mw", "temp": "temp_c", "x": "temp_plotted_c"}).to_csv(index=False).encode(),
                       f"power_load_{area}_{min(years)}_{max(years)}.csv", "text/csv", key="pw_ex_dl")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 2 — TEMPERATURE SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

def forecast_windows(days: list[pd.Timestamp], issue_day: pd.Timestamp) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """The windows the forecast can fill, name -> (first day, last day). The issue day
    itself is never in one: it is half over when the run arrives."""
    d1, last = issue_day + pd.Timedelta(days=1), max(days)
    monday = issue_day + pd.Timedelta(days=7 - issue_day.dayofweek)
    sunday = monday - pd.Timedelta(days=1)
    out = {}
    if sunday >= d1:                                   # nothing left of the week on a Sunday run
        out["Rest of the week"] = (d1, sunday)
    out["Next week"] = (monday, monday + pd.Timedelta(days=6))
    out["Week + 2"] = (monday + pd.Timedelta(days=7), monday + pd.Timedelta(days=13))
    return {k: (a, min(b, last)) for k, (a, b) in out.items() if a <= last}


def _week_no(day: pd.Timestamp) -> str:
    """ISO week of a window's first day, as the desk writes it."""
    return f"W{day.isocalendar().week}"


def country_sensitivity(curve: pdm.LoadCurve, hist: pd.DataFrame, fc: pd.DataFrame, delta: float) -> pd.DataFrame:
    """One country's forecast, day by day, with the load change for a sustained shift.

    The forecast temperature is put behind the last POWER_SEED_DAYS of actual
    temperature so the curve's effective temperature is not cold-started (the Gas
    Demand section's seeding). The shift is sustained — the days leading into a day
    are shifted with it — so the effective temperature moves by the full `delta` and
    the change is response(T ± delta) − response(T): the curve read at the forecast,
    not a fixed MW per °C, which is what makes a week sitting near a threshold
    asymmetric. The members' P10 / P90 are read the same way."""
    f = fc.sort_values("day").drop_duplicates("day").set_index("day")
    h = hist.set_index("day")
    seed = h["temp"].dropna().loc[:f.index.min() - pd.Timedelta(days=1)].tail(POWER_SEED_DAYS)
    path = pd.concat([seed, f["temp"]])
    path = path.reindex(pd.date_range(path.index.min(), path.index.max(), freq="D")).interpolate(limit_direction="both")
    teff = pdm.effective_temperature(path, curve.half_life_days).reindex(f.index)
    teff = teff.fillna(f["temp"])                      # no seed in the table: the day's own temperature
    base = curve.response(teff.to_numpy())

    def shifted(by) -> np.ndarray:
        return curve.response(teff.to_numpy() + np.asarray(by, dtype=float)) - base

    out = f[["temp", "temp_p10", "temp_p90", "load", "mem_slope", "mem_se", "mem_r2"]].copy()
    out["teff"] = teff
    out["slope"] = curve.slope(teff.to_numpy())
    out["d_cold"], out["d_warm"] = shifted(-delta), shifted(+delta)
    out["d_p10"] = shifted((f["temp_p10"] - f["temp"]).fillna(0.0))
    out["d_p90"] = shifted((f["temp_p90"] - f["temp"]).fillna(0.0))
    out["model_load"] = curve.predict(teff.to_numpy(), f.index)
    out["temp_normal"] = h["temp_normal"].reindex(f.index)
    out["load_normal"] = h["load_normal"].reindex(f.index)
    return out.reset_index()


def _member_implied(w: pd.DataFrame) -> tuple[float, int]:
    """The window's member-implied MW per °C: the days' slopes weighted by their
    precision, days whose members do not line up (low R²) left out. (value, days used)."""
    m = w[(w["mem_r2"] >= POWER_MEMBER_MIN_R2) & w["mem_slope"].notna() & (w["mem_se"] > 0)]
    if m.empty:
        return np.nan, 0
    return float(np.average(m["mem_slope"], weights=1.0 / m["mem_se"] ** 2)), int(len(m))


def summarise(area: str, days: pd.DataFrame, w0: pd.Timestamp, w1: pd.Timestamp) -> dict | None:
    """The window's figures for one country (means over its days, MW)."""
    w = days[(days["day"] >= w0) & (days["day"] <= w1)]
    if w.empty:
        return None
    load = w["load"].mean() if w["load"].notna().any() else w["model_load"].mean()
    slope = float(w["slope"].mean())
    pct = abs(slope) / load * 100 if load else np.nan
    implied, n_implied = _member_implied(w)
    gwh = 24.0 * len(w) / 1000.0
    return {"area": area, "n_days": int(len(w)), "temp": float(w["temp"].mean()),
            "temp_anom": float((w["temp"] - w["temp_normal"]).mean()), "load": float(load),
            "load_is_model": not w["load"].notna().any(), "slope": slope,
            "regime": "Flat" if pct < POWER_FLAT_PCT_PER_C else ("Heating" if slope < 0 else "Cooling"),
            "d_cold": float(w["d_cold"].mean()), "d_warm": float(w["d_warm"].mean()),
            "gwh_cold": float(w["d_cold"].mean() * gwh), "gwh_warm": float(w["d_warm"].mean() * gwh),
            "d_p10": float(w["d_p10"].mean()), "d_p90": float(w["d_p90"].mean()),
            "spread": float((w["temp_p90"] - w["temp_p10"]).mean()) if w["temp_p10"].notna().any() else np.nan,
            "implied": implied, "n_implied": n_implied}


def _load_cell(v: float, unit: str = "") -> str:
    """Signed load change coloured by its price read, as the gas deltas: more load red."""
    if v is None or np.isnan(v):
        return '<span class="mc-muted">n/a</span>'
    if round(v) == 0:
        return f'<span class="mc-zero">{v:+,.0f}{unit}</span>'
    return f'<span class="{"mc-neg" if v > 0 else "mc-pos"}">{v:+,.0f}{unit}</span>'


def _sensitivity_table(rows: list[dict], delta: float) -> str:
    """Countries down, the two shifts side by side, then what the members say."""
    head = ("<tr><th>Country</th><th>Regime</th><th>Temp fcst</th><th>vs normal</th><th>Load fcst</th>"
            f"<th>{delta:g} °C colder</th><th>{delta:g} °C warmer</th><th>% of load</th><th>GWh, window</th>"
            "<th>Members P10–P90</th><th>Load-forecast implied</th></tr>")
    body = ""
    for r in rows:
        pct = f'{r["d_cold"] / r["load"] * 100:+.1f} / {r["d_warm"] / r["load"] * 100:+.1f} %' if r["load"] else "n/a"
        spread = ('<span class="mc-muted">n/a</span>' if np.isnan(r["spread"]) else
                  f'{_load_cell(r["d_p10"])} / {_load_cell(r["d_p90"])} <span class="mc-muted">({r["spread"]:.1f} °C)</span>')
        implied = ('<span class="mc-muted">n/a</span>' if np.isnan(r["implied"]) else
                   f'{r["implied"]:+,.0f} <span class="mc-muted">/°C · {r["n_implied"]} d</span>')
        body += (f'<tr><td class="mc-region">{area_label(r["area"])}</td><td>{r["regime"]}</td>'
                 f'<td>{r["temp"]:.1f} °C</td><td>{_fmt(r["temp_anom"], "{:+.1f} °C")}</td>'
                 f'<td>{r["load"] / 1000:,.1f} GW{"*" if r["load_is_model"] else ""}</td>'
                 f'<td>{_load_cell(r["d_cold"], " MW")}</td><td>{_load_cell(r["d_warm"], " MW")}</td><td>{pct}</td>'
                 f'<td>{_load_cell(r["gwh_cold"])} / {_load_cell(r["gwh_warm"])}</td><td>{spread}</td><td>{implied}</td></tr>')
    if len(rows) > 1:
        tot = {k: float(np.sum([r[k] for r in rows])) for k in ("load", "d_cold", "d_warm", "gwh_cold", "gwh_warm")}
        body += (f'<tr><td class="mc-region"><b>TOTAL</b></td><td></td><td></td><td></td><td><b>{tot["load"] / 1000:,.1f} GW</b></td>'
                 f'<td><b>{_load_cell(tot["d_cold"], " MW")}</b></td><td><b>{_load_cell(tot["d_warm"], " MW")}</b></td>'
                 f'<td>{tot["d_cold"] / tot["load"] * 100:+.1f} / {tot["d_warm"] / tot["load"] * 100:+.1f} %</td>'
                 f'<td><b>{_load_cell(tot["gwh_cold"])} / {_load_cell(tot["gwh_warm"])}</b></td><td></td><td></td></tr>')
    return f'<div class="mc-block"><table class="mc-table">{head}{body}</table></div>'


def _render_sensitivity(areas: list[str]) -> None:
    try:
        fc_all = load_power_fcst()
    except Exception as e:
        st.error(f"Cannot read the forecast from {POWER_LOAD_FCST_TABLE}: {e}")
        return
    models = [m for m, tag in POWER_FCST_MODELS.items() if not fc_all.empty and (fc_all["model"] == tag).any()]
    if not models:
        status_banner(f"No forecast in {POWER_LOAD_FCST_TABLE} yet — it is written by Power_dashboard/pipeline "
                      "(run_daily.py --only load).", "warning")
        return

    c1, c2, c3, c4 = st.columns([1.3, 1.6, 1.2, 3.4])
    with c1:
        model = st.radio("Forecast", models, horizontal=True, key="pw_se_model")
    fc = fc_all[fc_all["model"] == POWER_FCST_MODELS[model]]
    issue = fc["reference_date"].max()
    fc = fc[fc["reference_date"] == issue]
    windows = forecast_windows(sorted(fc["day"].unique()), issue.normalize())
    with c2:
        names = [w for w in WINDOWS if w in windows]
        window = st.selectbox("Window", names, index=names.index(DEFAULT_WINDOW) if DEFAULT_WINDOW in names else 0,
                              key="pw_se_window",
                              format_func=lambda w: f"{w} · {_week_no(windows[w][0])} · "
                                                    f"{windows[w][0]:%d %b} – {windows[w][1]:%d %b}")
    with c3:
        delta = st.selectbox("Temperature shift", POWER_SHIFT_STEPS_C, index=POWER_SHIFT_STEPS_C.index(POWER_SHIFT_DEFAULT_C),
                             format_func=lambda v: f"± {v:g} °C", key="pw_se_delta")
    with c4:
        offered = [a for a in areas if (fc["area"] == a).any()]
        sel = st.multiselect("Countries", offered, [a for a in POWER_DEFAULT_AREAS if a in offered],
                             format_func=area_label, key="pw_se_areas")
    w0, w1 = windows[window]
    n_days = (w1 - w0).days + 1
    st.caption(f"**{model}** {issue:%H}z · {issue:%a %d %b} — {window.lower()}, {_week_no(w0)}: {w0:%a %d %b} to "
               f"{w1:%a %d %b}" + (f" ({n_days} of 7 days: the forecast ends there)" if window == "Week + 2" and n_days < 7 else ""))
    age_h = (pd.Timestamp.now(tz="UTC").tz_localize(None) - issue).total_seconds() / 3600
    if age_h > FCST_STALE_HOURS:
        status_banner(f"The latest {model} forecast in the table was issued {issue:%a %d %b %H}z — "
                      f"{age_h / 24:.0f} days ago. The feed (run_daily.py --only load) has not run since.", "warning")
    if not sel:
        status_banner("Pick at least one country.", "warning")
        return

    version = data_version(POWER_LOAD_TABLE)
    with st.spinner("Fitting the temperature-response curves…"):
        prefetch(*[partial(_curve, a, version) for a in sel])      # each loads its history, then fits
        per_area, rows, no_curve = {}, [], []
        for a in sel:
            try:
                curve = _curve(a, version)
            except Exception as e:
                st.error(f"Cannot read the load history of {area_label(a)}: {e}")
                return
            if curve is None:
                no_curve.append(a)
                continue
            days = country_sensitivity(curve, load_power_load(a), fc[fc["area"] == a], float(delta))
            row = summarise(a, days, w0, w1)
            if row is not None:
                per_area[a] = (curve, days)
                rows.append(row)
    if no_curve:
        status_banner("Not enough load history for a curve (a held-out year plus a year to fit on): "
                      + ", ".join(area_label(a) for a in no_curve) + ".", "warning")
    if not rows:
        status_banner("Nothing to show for that selection.", "warning")
        return

    tot = {k: float(np.sum([r[k] for r in rows])) for k in ("load", "d_cold", "d_warm", "gwh_cold", "gwh_warm")}
    top = max(rows, key=lambda r: abs(r["slope"]) / r["load"])
    scope = f"{len(rows)} countries" if len(rows) > 1 else area_label(rows[0]["area"])
    kpi_row([
        kpi_card(f"{delta:g} °C colder · {scope}", f"{tot['d_cold'] / 1000:+,.2f} GW", "kpi-card-cool",
                 f'<div class="kpi-delta">{tot["d_cold"] / tot["load"] * 100:+.1f} % of load</div>',
                 f'<div class="kpi-rank">{tot["gwh_cold"]:+,.0f} GWh over the window</div>'),
        kpi_card(f"{delta:g} °C warmer · {scope}", f"{tot['d_warm'] / 1000:+,.2f} GW", "kpi-card-warm",
                 f'<div class="kpi-delta">{tot["d_warm"] / tot["load"] * 100:+.1f} % of load</div>',
                 f'<div class="kpi-rank">{tot["gwh_warm"]:+,.0f} GWh over the window</div>'),
        kpi_card("Most temperature-driven", area_label(top["area"]), "kpi-card-neutral",
                 f'<div class="kpi-delta">{top["slope"]:+,.0f} MW per °C</div>',
                 f'<div class="kpi-rank">{abs(top["slope"]) / top["load"] * 100:.1f} % of its load · {top["regime"].lower()}</div>'),
    ])

    st.markdown(_sensitivity_table(rows, float(delta)), unsafe_allow_html=True)
    info_tip(f"Load change, as the mean over the window's days, if the whole window comes in {delta:g} °C colder or warmer "
             "than the forecast. It is read off each country's fitted curve at the forecast temperature, so the two "
             "directions differ where the week sits near a heating or cooling threshold. Regime: heating = load rises "
             "when colder, cooling = load rises when warmer, flat = the comfort range. GWh = the change summed over the "
             "window. Members P10–P90: the load change if the temperature lands on the members' cold (P10) / warm (P90) "
             "edge instead of their mean, with the width of that range in °C — the move the ensemble itself considers "
             "plausible. Load-forecast implied: the MW per °C read across the members of the provider's own load "
             "forecast (their load against their temperature, day by day, days where the members line up) — how far the "
             "published load forecast moves per degree, which is less than actual load has moved. * = no load forecast "
             "for this run, the curve's own level is shown.")

    labels = [area_label(r["area"]) for r in rows]
    st.plotly_chart(make_sensitivity_bars(labels, [r["d_cold"] for r in rows], [r["d_warm"] for r in rows], float(delta),
                                          f"Load change over {w0:%d %b} – {w1:%d %b} per country"), use_container_width=True)

    pick = st.selectbox("Country to chart", [r["area"] for r in rows], format_func=area_label, key="pw_se_chart_area")
    curve, days = per_area[pick]
    row = next(r for r in rows if r["area"] == pick)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(make_sensitivity_days(days, float(delta), w0, w1,
                                              f"{area_label(pick)} — {row['slope']:+,.0f} MW per °C in the window"),
                        use_container_width=True)
    with right:
        mid = w0 + (w1 - w0) / 2
        w = days[(days["day"] >= w0) & (days["day"] <= w1)]
        hist = load_power_load(pick).dropna(subset=["load", "temp"]).set_index("day")
        teff = pdm.effective_temperature(hist["temp"].asfreq("D").interpolate(limit=3), curve.half_life_days).reindex(hist.index)
        recent = pd.DataFrame({"x": teff, "load": hist["load"]}).dropna().tail(POWER_PARTIAL_DAYS)
        # each actual day moved to a working day at the window's time of year: curve + that day's residual
        recent["load"] = curve.curve(recent["x"].to_numpy(), mid) + (recent["load"] - curve.predict(recent["x"].to_numpy(), recent.index))
        xs = np.linspace(min(curve.t_min, w["teff"].min() - float(delta)), max(curve.t_max, w["teff"].max() + float(delta)), 160)
        st.plotly_chart(make_response_curve(xs, curve.curve(xs, mid), recent,
                                            pd.DataFrame({"x": w["teff"], "load": curve.curve(w["teff"].to_numpy(), mid), "day": w["day"]}),
                                            float(delta), f"{area_label(pick)} — where the window sits on the curve",
                                            "Effective temperature (°C)" if curve.half_life_days > 0 else "Temperature (°C)"),
                        use_container_width=True)

    with st.expander("Fitted curves — shape, thermal inertia and backtest"):
        recs = [{"Country": area_label(a), "Curve (MW per °C)": _segments(c),
                 "Thermal inertia": "day's own temperature" if c.half_life_days <= 0 else f"{c.half_life_days:g}-day half-life",
                 "R²": round(c.r2, 3), "Backtest MAE (MW)": round(c.backtest_mae),
                 "Naive MAE (MW)": round(c.backtest_mae_naive), "MAE % of load": round(c.backtest_mae / c.mean_load * 100, 1),
                 "Days": c.n_obs} for a, (c, _) in per_area.items()]
        st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)
        st.caption("Fitted on every day since 2015, recent years weighted most (1.5-year half-life), the level anchored "
                   "on the last 365 days. Day type and time of year are fitted alongside, so the slopes are the response "
                   "to temperature at a fixed calendar position. Backtest: the curve fitted without the last year, "
                   "scored on it; naive = day type plus a straight line in temperature.")

    out = pd.DataFrame(rows).drop(columns=["load_is_model"]).rename(columns={
        "temp": "temp_fcst_c", "temp_anom": "temp_vs_normal_c", "load": "load_fcst_mw", "slope": "mw_per_c",
        "d_cold": "mw_colder", "d_warm": "mw_warmer", "d_p10": "mw_members_p10", "d_p90": "mw_members_p90",
        "spread": "members_p10_p90_c", "implied": "load_forecast_implied_mw_per_c", "n_implied": "implied_days"})
    out.insert(0, "shift_c", float(delta))
    out.insert(0, "window_to", w1.date())
    out.insert(0, "window_from", w0.date())
    out.insert(0, "run", f"{model} {issue:%Y-%m-%d %H}z")
    st.download_button("Download the grid (CSV)", out.to_csv(index=False).encode(),
                       f"power_sensitivity_{POWER_FCST_MODELS[model]}_{issue:%Y%m%d_%H}z.csv", "text/csv", key="pw_se_dl")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION
# ══════════════════════════════════════════════════════════════════════════════

def render_power_demand():
    st.markdown("#### POWER DEMAND")
    try:
        avail = load_power_areas()
    except Exception as e:
        st.error(f"Cannot read from the sandbox table {POWER_LOAD_TABLE}: {e}")
        return
    if avail.empty:
        status_banner(f"No rows in {POWER_LOAD_TABLE} yet — it is written by Power_dashboard/pipeline "
                      "(run_daily.py --only load; the first run loads the history since 2015).", "warning")
        return
    areas = [a for a in AREAS if a in set(avail["area"])]
    last = avail["last_day"].max()
    if (pd.Timestamp.today().normalize() - last).days > 5:
        status_banner(f"The load history ends on {last:%a %d %b %Y} — the feed (run_daily.py --only load) has not "
                      "run since.", "warning")

    view = view_switch("View", VIEWS, "pw_view", {VIEWS[0]: ("pw_ex_",), VIEWS[1]: ("pw_se_",)})
    if view == VIEWS[0]:
        _render_explorer(areas)
    else:
        _render_sensitivity(areas)
