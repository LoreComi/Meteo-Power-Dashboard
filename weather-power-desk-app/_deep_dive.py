"""Morning Call → Country deep dive.

One country, the whole picture, read off the SAME reference run, compare columns and Δ
pairing as the Morning Report grid next to it:

  Forecast     every variable the report carries for the country — temperature, wind,
               solar, precipitation energy — as one grid with the variables as rows and
               the windows × runs as columns (the report's cells), then the daily
               series: the reference run against the run it is paired with, so what
               moved in the last 24 h is read day by day, with the normal and the other
               runs of the grid as thin lines.
  Hydro        the country's hydro state (reservoir levels, snow water, groundwater,
               hydro balance — the Hydro section's own figures) and the two-week
               outlook: production observed and Volue's forecast, precipitation energy
               observed and expected, each against its normal, with the change of the
               expected precipitation and production since the previous issue.
  Scenarios    the Forecast section's scenarios with its defaults (the EC-ENS members
               clustered on their weekly-mean Z500 field over the next two trading
               weeks, k = 2, named by the regime their centroid resembles), read for this country:
               temperature, wind and solar by scenario, side by side, on the land
               variables of the same member IDs.

Country → sources in _config.DEEP_DIVE_COUNTRIES. A source the country does not have
(hydro for the UK, temperature for SEE) is left out and said so. Precipitation-energy
members are not in the member tables (notebook cell 4 skips them), so the scenarios cannot
yet be read on precipitation — the hydro outlook's run-over-run change stands in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    DEEP_DIVE_COUNTRIES, DEEP_DIVE_DEFAULT, DEEP_DIVE_HYDRO_WINDOW_DAYS, HYDRO_OVERVIEW_LAYERS, HYDRO_PROD_AREAS,
    HYDRO_PROD_PRECIP_PATTERNS, HYDRO_PROD_TABLE, SCENARIO_DEFAULT_SOURCE, SCENARIO_K_DEFAULT,
)
from _data import load_hydro_prod
from _charts import make_country_run_chart, make_hydro_production_chart, make_country_scenario_panel
from _hydro import _overview_metrics, _layer_kpi, _flags, _chips
from _hydro_quantify import production_series, production_summary
from _forecast import default_scenarios, land_members, scenario_cards, scenario_ids, _side_by_side_html, _UNITS
from _morning import (
    RunCol, compute_grid, _series, _cell_html, _col_tip, _fmt_val, _signed, _short_title, MORNING_PRECIP_SUM_DAYS,
)
from _scenarios import apply_scenarios, scenario_daily_summary, scenario_weekly_values
from _ui import kpi_card, kpi_row, status_banner


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST — the grid's cells for one country, variables as rows
# ══════════════════════════════════════════════════════════════════════════════

def _country_blocks(country: str, cfg: dict, blocks: dict) -> dict:
    """The report's blocks cut down to this country's single row each; a block the
    country has no region for is dropped."""
    out = {}
    for name, block in blocks.items():
        region = cfg.get(block["family"])
        if region is None:
            continue
        label = cfg.get("rre_label", country) if block["family"] == "rre" else country
        out[name] = {**block, "rows": [(label, region)]}
    return out


def _variables_grid_html(cells: pd.DataFrame, cblocks: dict, cols: list[RunCol], win: dict,
                         view_key: str | None, delta_label: str) -> str:
    """One table: rows = variables, column groups = window → run (+ spread) — the Morning
    Report grid turned on its side for one country."""
    n = len(cols)
    show_spread = n >= 2
    per_win = n + (1 if show_spread else 0)
    windows = [("w1", win["t1"], win["w1"]), ("w2", win["t2"], win["w2"])]
    corner_sub = {"pair": f"Δ norm ▲▼ {delta_label} · abs", None: f"abs · {delta_label} · Δ norm", "d_norm": "Δ norm",
                  "d_run": delta_label, "d_ref": "Δ vs reference run", "value": "abs. value"}[view_key]
    colgroup = '<colgroup><col class="mcg-c0">' + "<col>" * (2 * per_win) + "</colgroup>"
    head1 = (f'<tr><th class="mcg-corner" rowspan="2"><div><span class="mcg-corner-name">Variable</span></div>'
             f'<div class="mcg-corner-sub">{corner_sub}</div></th>')
    for _, title, (s, e) in windows:
        head1 += (f'<th class="mcg-win" colspan="{per_win}">{title} '
                  f'<span class="mcg-range">{s:%a %d %b} – {e:%a %d %b}</span></th>')
    head1 += "</tr>"
    head2 = "<tr>"
    for _ in windows:
        for i, c in enumerate(cols):
            cls = "mcg-col" + (" mcg-first" if i == 0 else "") + (" mcg-ref" if c.is_ref else "")
            head2 += (f'<th class="{cls}" title="{_col_tip(c, delta_label)}">'
                      f'<span class="mcg-model">{c.model}</span><span class="mcg-date">{c.init:%a %d %b}</span></th>')
        if show_spread:
            head2 += '<th class="mcg-col mcg-spread">spread</th>'
    head2 += "</tr>"

    tint_ref = view_key not in ("pair", "d_norm", "d_run", "d_ref")
    body = ""
    for name, block in cblocks.items():
        is_sum = block["agg"] == "sum"
        region = block["rows"][0][0]
        short = name.split(" (")[0]
        sub = (f'<div class="mcg-corner-sub">{region} · sum of the first {MORNING_PRECIP_SUM_DAYS} forecast days</div>'
               if is_sum else "")
        body += f'<tr><td class="mc-region">{short} <span class="mcg-corner-unit">[{block["unit"]}]</span>{sub}</td>'
        for wkey, title, (s, e) in windows:
            blank = is_sum and wkey == "w2"
            span = None if is_sum else (e - s).days + 1
            sel = None if blank else cells[(cells["window"] == wkey) & (cells["block"] == name)].set_index("col")
            for i, c in enumerate(cols):
                cls = ("mcg-cell" + (" mcg-first" if i == 0 else "")
                       + (" mcg-refcell" if c.is_ref and tint_ref and not blank else ""))
                if blank or i not in sel.index:
                    body += f'<td class="{cls}"></td>'
                    continue
                body += _cell_html(sel.loc[i], c, region, title, block, span, view_key, cls)
            if show_spread:
                sp = np.nan if (blank or sel.empty) else float(sel["spread"].iloc[0])
                body += f'<td class="mcg-cell mcg-spread">{"" if blank else _fmt_val(sp, block["fmt"])}</td>'
        body += "</tr>"
    return f'<div class="mc-block"><table class="mc-table mcg-table">{colgroup}{head1}{head2}{body}</table></div>'


def _variable_cards(cells: pd.DataFrame, cblocks: dict, win: dict, delta_label: str) -> list[str]:
    """The reference run's first window per variable: value, Δ norm (coloured like the
    grid: red = warmer / less wind, solar, precipitation) and the change since the paired run."""
    cards = []
    for name, block in cblocks.items():
        is_sum = block["agg"] == "sum"
        r = cells[(cells["window"] == "w1") & (cells["block"] == name) & (cells["col"] == 0)]
        if r.empty or np.isnan(float(r["value"].iloc[0])):
            continue
        r = r.iloc[0]
        fmt, unit = block["fmt"], block["unit"]
        d_norm, d_run = float(r["d_norm"]), float(r["d_run"])
        cls, delta = "kpi-card-neutral", ""
        if not np.isnan(d_norm) and round(d_norm, 1) != 0:
            warm = (d_norm > 0) == block["warm_is_positive"]
            cls = "kpi-card-warm" if warm else "kpi-card-cool"
            delta = (f'<div class="kpi-delta {"kpi-delta-up" if warm else "kpi-delta-down"}">'
                     f'{"▲" if d_norm > 0 else "▼"} {fmt.format(abs(d_norm))} {unit} vs normal</div>')
        elif not np.isnan(d_norm):
            delta = '<div class="kpi-delta kpi-delta-flat">on the normal</div>'
        run = ("n/a" if np.isnan(d_run) else _signed(d_run, fmt))
        rank = f'<div class="kpi-rank">{delta_label}: {run} {unit}</div>'
        label = f"{name.split(' (')[0]} · " + (f"{MORNING_PRECIP_SUM_DAYS}-day sum" if is_sum else _short_title(win["t1"]))
        cards.append(kpi_card(label, f"{fmt.format(float(r['value']))} {unit}", cls, delta, rank))
    return cards


def _daily_panels(cblocks: dict, cols: list[RunCol]) -> list[dict]:
    ref = cols[0]
    panels = []
    for name, block in cblocks.items():
        fam, region, scale = block["family"], block["rows"][0][1], block["scale"]
        f, n = _series(ref.rows, fam, region)
        if f.empty:
            continue
        lo, hi = f.index.min(), f.index.max()          # the paired and compare runs on the reference's days only
        fp, _ = _series(ref.prev_rows, fam, region)
        fp = fp.loc[lo:hi]
        others = {}
        for c in cols[1:]:
            fo, _ = _series(c.rows, fam, region)
            if not fo.empty:
                others[f"{c.model} {c.init:%d %b}"] = fo.loc[lo:hi] * scale
        panels.append({"name": name.split(" (")[0], "unit": block["unit"], "ref": f * scale, "prev": fp * scale,
                       "normal": n * scale, "others": others})
    return panels


def _render_forecast(country: str, cfg: dict, cols: list[RunCol], win: dict, delta_label: str,
                     view_key: str | None, blocks: dict) -> None:
    ref = cols[0]
    cblocks = _country_blocks(country, cfg, blocks)
    if not cblocks:
        status_banner(f"The Morning Report has no curve for {country} in this source.", "warning")
        return
    missing = [n.split(" (")[0] for n, b in blocks.items() if n not in cblocks]
    st.markdown(f"##### Forecast · {ref.model} {ref.init:%a %d %b} — {delta_label} vs {ref.prev_label}"
                + ("" if ref.prev_exact else " (nearest earlier run)"))
    if missing:
        st.caption(f"The report carries no {', '.join(missing).lower()} for {country}.")
    cells = compute_grid(cols, win, cblocks)
    kpi_row(_variable_cards(cells, cblocks, win, delta_label), max_cols=4)
    st.markdown(_variables_grid_html(cells, cblocks, cols, win, view_key, delta_label), unsafe_allow_html=True)
    panels = _daily_panels(cblocks, cols)
    if panels:
        windows = [(_short_title(win["t1"]), *win["w1"]), (_short_title(win["t2"]), *win["w2"])]
        st.plotly_chart(make_country_run_chart(panels, windows, f"{ref.model} {ref.init:%d %b %H}z",
                                               None if ref.prev_init is None else f"{ref.prev_init:%d %b %H}z"),
                        use_container_width=True, config={"displayModeBar": False})
        st.caption("Solid = the reference run · dashed = the run its Δ is taken against (where the two part is the "
                   "change) · dotted = normal · thin = the other runs of the grid · shaded = the report's windows.")


# ══════════════════════════════════════════════════════════════════════════════
# HYDRO — state and two-week outlook
# ══════════════════════════════════════════════════════════════════════════════

def _issue_change(df: pd.DataFrame, area: str, variable: str, pattern: str, q: dict) -> dict | None:
    """The forward-window sum of the latest issue of a forecast against the issue 24 h
    earlier (or the latest earlier one), over the days both cover: dict(latest, prev,
    delta, prev_issue, n_days). None with fewer than two issues."""
    f = df[(df["area"] == area) & (df["variable"] == variable) & (df["data_type"] == "f") & (df["pattern"] == pattern)]
    f = f[f["init_time"].notna()]
    issues = sorted(f["init_time"].unique())
    if len(issues) < 2:
        return None
    latest = issues[-1]
    earlier = [i for i in issues if i <= latest - pd.Timedelta(hours=24)]
    prev = earlier[-1] if earlier else issues[-2]
    fwd = (q["last_obs"] + pd.Timedelta(days=1), q["fwd_to"])
    a = f[f["init_time"] == latest].set_index("day")["value"].astype(float).sort_index().loc[fwd[0]:fwd[1]]
    b = f[f["init_time"] == prev].set_index("day")["value"].astype(float).sort_index().loc[fwd[0]:fwd[1]]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return None
    return {"latest": float(a.sum()), "prev": float(b.loc[common].sum()),
            "delta": float(a.loc[common].sum() - b.loc[common].sum()), "prev_issue": pd.Timestamp(prev),
            "n_days": int(len(common))}


def _pct_card(label: str, value: float | None, normal: float | None, change: dict | None, what: str) -> str:
    """GWh over a window with % of normal (blue = above normal, red = short of it) and, when
    there is one, the change since the previous issue."""
    if value is None:
        return kpi_card(label, "n/a")
    pct = (value / normal * 100) if normal else None
    cls, delta = "kpi-card-neutral", ""
    if pct is not None:
        up = value >= normal
        cls = "kpi-card-cool" if up else ("kpi-card-critical" if pct < 70 else "kpi-card-warm")
        delta = (f'<div class="kpi-delta {"kpi-delta-down" if up else "kpi-delta-up"}">{"▲" if up else "▼"} '
                 f'{abs(value - normal):,.0f} GWh vs normal · {pct:.0f}% of normal</div>')
    rank = ""
    if change:
        rank = (f'<div class="kpi-rank">{change["delta"]:+,.0f} GWh vs the issue of {change["prev_issue"]:%a %d %b %H}z'
                f' ({what}, {change["n_days"]} common days)</div>')
    return kpi_card(label, f"{value:,.0f} GWh", cls, delta, rank)


def _render_hydro(country: str, cfg: dict) -> None:
    area_lvl, area_prod = cfg.get("hydro_level"), cfg.get("hydro_prod")
    st.markdown(f"##### Hydro · {country}")
    if not area_lvl and not area_prod:
        st.caption(f"No hydro series for {country} — the Hydro section covers the Alpine, Nordic, Iberian and "
                   "south-east European systems.")
        return
    # state — the Hydro overview's own figures for this area
    if area_lvl:
        try:
            metrics = _overview_metrics()
        except Exception as e:
            metrics = {}
            st.warning(f"Hydro state unavailable: {e}")
        layers = [l for l in HYDRO_OVERVIEW_LAYERS if metrics.get(l, {}).get(area_lvl)]
        if layers:
            kpi_row([_layer_kpi(l, metrics[l][area_lvl]) for l in layers], max_cols=4)
            flags = [(lvl, l, t) for l in layers for lvl, t in _flags(metrics[l][area_lvl], l)]
            if flags:
                _chips(flags)
        else:
            st.caption(f"No reservoir / snow / groundwater series for {country} yet.")
    # outlook — production and precipitation energy, last two weeks vs the next two
    if not area_prod:
        return
    W = DEEP_DIVE_HYDRO_WINDOW_DAYS
    try:
        df = load_hydro_prod((area_prod,))
    except Exception as e:
        st.warning(f"Cannot read {HYDRO_PROD_TABLE}: {e}")
        return
    if df.empty:
        st.caption(f"No production rows for {HYDRO_PROD_AREAS.get(area_prod, area_prod)} in {HYDRO_PROD_TABLE} yet "
                   "(pipeline/run_daily.py --only hydro-prod).")
        return
    s = production_series(df[df["area"] == area_prod], HYDRO_PROD_PRECIP_PATTERNS)
    q = production_summary(s, W, HYDRO_PROD_PRECIP_PATTERNS)
    if not q:
        st.caption(f"No observed production for {HYDRO_PROD_AREAS.get(area_prod, area_prod)} yet.")
        return
    ref_p = HYDRO_PROD_PRECIP_PATTERNS[0]
    d_rre = _issue_change(df, area_prod, "rre", ref_p, q)
    d_tot = _issue_change(df, area_prod, "tot", "volue", q)
    name = HYDRO_PROD_AREAS.get(area_prod, area_prod)
    st.caption(f"{name} · last observed day {q['last_obs']:%d %b} · the {W} days to it against the {W} after it"
               + (f" · Volue production forecast issued {q['fc_issued']:%d %b}" if q.get("fc_issued") is not None else ""))
    kpi_row([
        _pct_card(f"Production · last {W} d", q["obs_tot"], q["obs_tot_n"], None, ""),
        _pct_card(f"Production forecast · next {W} d", q["fwd_tot"], q["fwd_tot_n"], d_tot, "Volue forecast"),
        _pct_card(f"Precip. energy · last {W} d", q["obs_rre"], q["obs_rre_n"], None, ""),
        _pct_card(f"Precip. energy expected · next {W} d ({ref_p})", q["fwd_rre_ref"], q["fwd_rre_n"], d_rre, ref_p),
    ], max_cols=4)
    others = {p: v for p, v in (q.get("fwd_rre") or {}).items() if p != ref_p and v is not None}
    bal = q.get("fwd_balance")
    st.caption((f"Other ensembles, next {W} d: " + " · ".join(f"{p} {v:,.0f} GWh" for p, v in others.items()) + " · " if others else "")
               + (f"precipitation − production, next {W} d: {bal:+,.0f} GWh" if bal is not None else ""))
    st.plotly_chart(make_hydro_production_chart(s, name, W, HYDRO_PROD_PRECIP_PATTERNS, height=420),
                    use_container_width=True, key=f"dd_hydro_chart_{area_prod}")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIOS — the Forecast section's defaults, read for this country
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_members(pm: pd.DataFrame, areas: list[str], label: str, how: str) -> pd.DataFrame:
    """Member rows of several countries as one area `label`: temperature averaged, wind and
    solar summed, per member and day (a day counts only when every country has it)."""
    if pm.empty:
        return pm
    if len(areas) == 1:
        return pm.assign(area=label)
    keys = [c for c in ("provider", "model", "reference_date", "metric", "day", "lead_day", "member") if c in pm.columns]
    agg = {c: how for c in ("value", "normal", "anomaly") if c in pm.columns}
    g = pm.groupby(keys, dropna=False)
    out = g.agg(agg)
    n = g["value"].count()
    out = out[n >= len(areas)].reset_index()
    return out.assign(area=label)


def _render_scenarios(country: str, cfg: dict) -> None:
    st.markdown(f"##### Scenarios · {country}")
    try:
        res, info, weeks_sel, weeks_all = default_scenarios(SCENARIO_DEFAULT_SOURCE)
    except Exception as e:
        st.warning(f"Scenario inputs unavailable: {e}")
        return
    if info is None:
        st.caption("No Z500 member field in the sandbox yet — the Forecast → Scenarios tab is empty too "
                   "(pipeline run_scenarios.bat for EC-ENS, power_desk_refresh.py cell 4b for EC-AI).")
        return
    if not res:
        st.caption("The run does not cover enough days of any trading week to form scenarios yet.")
        return
    weeks = list(weeks_sel["week_start"])
    scen_ids = scenario_ids(res)
    areas = list(cfg.get("scenario", []))
    st.caption(f"{info['label']} · run {pd.Timestamp(info['init']):%a %d %b %Hz} · {res['n_members']} members · "
               f"clustered on the weekly-mean Z500 field over {', '.join(weeks_sel['week_label'])} · "
               f"k = {SCENARIO_K_DEFAULT} — the Forecast → Scenarios tab's defaults; change them there for another split. "
               f"Temperature, wind and solar read on the {info['land']['label']} of the same member IDs.")
    kpi_row(scenario_cards(res, info, weeks), max_cols=4)

    if not areas:
        st.caption(f"No member rows for {country}.")
        return
    tagged = {}
    for metric in ("Temperature", "Wind", "Solar"):
        pm = land_members(info, metric, tuple(areas))
        pm = _aggregate_members(pm, areas, country, "mean" if metric == "Temperature" else "sum")
        t = apply_scenarios(pm, res["assign"]) if not pm.empty else pd.DataFrame()
        if not t.empty:
            tagged[metric] = t
    if not tagged:
        st.caption(f"No member rows for {country} ({', '.join(areas)}) in the {info['land']['label']} yet.")
        return
    weekly = {m: scenario_weekly_values(t, weeks, "value") for m, t in tagged.items()}
    daily = {m: scenario_daily_summary(t, "value") for m, t in tagged.items()}
    if len(areas) > 1:
        st.caption(f"{country} = {', '.join(areas)} per member: temperature averaged, wind and solar summed.")
    st.markdown(_side_by_side_html(country, weekly, scen_ids, weeks_sel, res["sizes"], res["n_members"], res["names"]),
                unsafe_allow_html=True)
    st.plotly_chart(make_country_scenario_panel(daily, country, _UNITS, list(weeks_all["week_start"])),
                    use_container_width=True, key=f"dd_scen_{country}")
    st.caption("Precipitation energy by scenario is not available: the member tables carry temperature, "
               "wind and solar only. The hydro outlook above gives the "
               "ensemble-mean precipitation and its change since the previous issue instead.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def render_country_deep_dive(cols: list[RunCol], win: dict, delta_label: str, view_key: str | None,
                             blocks: dict, source: str) -> None:
    """Everything the desk has on one country, on the grid's reference run, compare
    columns and Δ pairing (chosen above the tabs)."""
    names = list(DEEP_DIVE_COUNTRIES)
    c1, c2 = st.columns([1.4, 4.6])
    with c1:
        country = st.selectbox("Country", names, index=names.index(DEEP_DIVE_DEFAULT), key="dd_country")
    with c2:
        st.caption(f"The reference run, compare columns and Δ pairing are the grid's (set above). Source: {source}. "
                   "Forecast — the report's variables for the country, then the daily series run against run; "
                   "hydro — state and two-week outlook; scenarios — the Forecast section's split, read here.")
    cfg = DEEP_DIVE_COUNTRIES[country]
    _render_forecast(country, cfg, cols, win, delta_label, view_key, blocks)
    st.divider()
    _render_hydro(country, cfg)
    st.divider()
    _render_scenarios(country, cfg)
