"""Morning Call → Country deep dive.

One country, the whole picture, read off the SAME reference run, compare columns and Δ
pairing as the Morning Report grid next to it. On top a header with the outlook in two or
three words ("Cold & still") for the report's first window, the next window's and what moved
since the paired run, and one signal chip per variable plus hydro and the leading scenario.
Then three section cards:

  Forecast     every variable the report carries for the country — temperature, wind,
               solar, precipitation energy — as stat tiles per window (the report's
               first week, then the second, each with its outlook in words), then one
               grid with the variables as rows and the windows × runs as columns.
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
from _charts import make_hydro_production_chart, make_country_scenario_panel
from _hydro import _overview_metrics, _layer_kpi, _flags, _chips
from _hydro_quantify import production_series, production_summary
from _forecast import default_scenarios, land_members, scenario_cards, scenario_ids, week_title, _side_by_side_html, _UNITS
from _morning import (
    RunCol, compute_grid, _cell_html, _col_tip, _fmt_val, _signed, _short_title, MORNING_PRECIP_SUM_DAYS,
)
from _scenarios import apply_scenarios, scenario_daily_summary, scenario_weekly_values, short_name
from _style import DEEP_DIVE_CSS, DD_WARM, DD_COOL
from _charts import scenario_color
from _ui import (kpi_card, kpi_row, status_banner, icon_box as _icon, section as _section,
                 sec_header as _sec_header, tile as _tile, tiles as _tiles, note as _note)


# ══════════════════════════════════════════════════════════════════════════════
# LOOK — icons, section cards, stat tiles, the outlook in words
# ══════════════════════════════════════════════════════════════════════════════

# words for the outlook: (above normal, below normal) and the move (up, down)
_OUT_WORDS = {"wnd": ("Windy", "Still"), "spv": ("Sunny", "Dull"), "rre": ("Wet", "Dry")}
_MOVE_WORDS = {"tt": ("warmer", "colder"), "wnd": ("windier", "less windy"), "spv": ("sunnier", "duller"),
               "rre": ("wetter", "drier")}
_FAMILY_ORDER = ["tt", "wnd", "spv", "rre"]
OUTLOOK_MIN_SCORE = 0.25        # |Δ norm| / heat_scale below this is "near normal" for the outlook
OUTLOOK_MOVE_MIN = 0.15         # |Δ run| / heat_scale below this is not a move worth naming


def _temp_words(month: int) -> tuple[str, str]:
    if month in (11, 12, 1, 2, 3):
        return "Mild", "Cold"
    if month in (6, 7, 8):
        return "Hot", "Cool"
    return "Warm", "Cool"


def _tone(d: float, block: dict) -> str:
    if np.isnan(d) or round(d, 1) == 0:
        return "flat"
    return "warm" if (d > 0) == block["warm_is_positive"] else "cool"


def _ref_cell(cells: pd.DataFrame, wkey: str, name: str) -> pd.Series | None:
    r = cells[(cells["window"] == wkey) & (cells["block"] == name) & (cells["col"] == 0)]
    return None if r.empty else r.iloc[0]


def _outlook(cells: pd.DataFrame, cblocks: dict, wkey: str, start: pd.Timestamp) -> tuple[str, str]:
    """Two or three words for one window of the reference run ("Cold & still"), and the
    tone of the strongest signal (warm / cool / flat)."""
    sig = []
    for name, block in cblocks.items():
        r = _ref_cell(cells, wkey, name)
        d = np.nan if r is None else float(r["d_norm"])
        if np.isnan(d):
            continue
        score = abs(d) / block["heat_scale"]
        if score < OUTLOOK_MIN_SCORE:
            continue
        fam = block["family"]
        up, down = _temp_words(start.month) if fam == "tt" else _OUT_WORDS[fam]
        sig.append((score, up if d > 0 else down, _tone(d, block), _FAMILY_ORDER.index(fam)))
    if not sig:
        return "Near normal", "flat"
    sig.sort(key=lambda t: -t[0])                      # the three strongest, read temperature first
    words = [w for _, w, _, _ in sorted(sig[:3], key=lambda t: t[3])]
    if len(words) == 1:
        text = ("Very " + words[0].lower()) if sig[0][0] >= 1 else words[0]
    else:
        text = ", ".join([words[0]] + [w.lower() for w in words[1:-1]]) + " & " + words[-1].lower()
    return text, sig[0][2]


def _moves(cells: pd.DataFrame, cblocks: dict) -> str:
    """What moved since the paired run, in words ("colder, windier"), first window."""
    mv = []
    for name, block in cblocks.items():
        r = _ref_cell(cells, "w1", name)
        d = np.nan if r is None else float(r["d_run"])
        if np.isnan(d) or abs(d) / block["heat_scale"] < OUTLOOK_MOVE_MIN:
            continue
        up, down = _MOVE_WORDS[block["family"]]
        mv.append((abs(d) / block["heat_scale"], up if d > 0 else down))
    return ", ".join(w for _, w in sorted(mv, key=lambda t: -t[0])[:3]) if mv else "little changed"


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


def _variable_tiles(cells: pd.DataFrame, cblocks: dict, win: dict, delta_label: str, wkey: str) -> list[str]:
    """The reference run's tiles for one window: per variable the value, Δ norm on a gauge
    (red = warmer / less wind, solar, precipitation; blue = colder / more) and the change since
    the paired run. Precipitation is a 15-day sum from the init day, so it sits in the first window only."""
    tiles = []
    for name, block in cblocks.items():
        is_sum = block["agg"] == "sum"
        if is_sum and wkey != "w1":
            continue
        r = _ref_cell(cells, wkey, name)
        if r is None or np.isnan(float(r["value"])):
            continue
        fmt, unit = block["fmt"], block["unit"]
        d_norm, d_run = float(r["d_norm"]), float(r["d_run"])
        tone = _tone(d_norm, block)
        pill = ("n/a vs normal" if np.isnan(d_norm) else
                "on the normal" if tone == "flat" else f"{_signed(d_norm, fmt)} {unit} vs normal")
        if np.isnan(d_run):
            foot, ftone = f"{delta_label} n/a", ""
        else:
            arrow = "▲" if round(d_run, 1) > 0 else ("▼" if round(d_run, 1) < 0 else "=")
            foot, ftone = f"{arrow} {_signed(d_run, fmt)} {delta_label}", ("" if arrow == "=" else _tone(d_run, block))
        s, e = win[wkey]
        span = (e - s).days + 1
        n_days = int(r["n_days"]) if not pd.isna(r["n_days"]) else span
        sub = (f"{MORNING_PRECIP_SUM_DAYS}-day sum from the init day" if is_sum else
               "weekly average" + (f" · {n_days} of {span} days covered" if 0 < n_days < span else ""))
        tiles.append(_tile(block["family"], name.split(" (")[0], sub, fmt.format(float(r["value"])), unit,
                           d_norm / block["heat_scale"] if not np.isnan(d_norm) else None, tone, pill, foot, ftone))
    return tiles


def _week_band(cells: pd.DataFrame, cblocks: dict, win: dict, wkey: str) -> str:
    """The heading of one window's tiles: its name and dates, and its outlook in words."""
    title = win["t1"] if wkey == "w1" else win["t2"]
    s, e = win[wkey]
    kind = win.get("kind1") if wkey == "w1" else None
    out, tone = _outlook(cells, cblocks, wkey, s)
    return (f'<div class="dd-wk"><div><span class="dd-wk-title">{_short_title(title)}</span>'
            f'<span class="dd-wk-dates">{s:%a %d %b} – {e:%a %d %b}'
            + (" · weekend" if kind == "weekend" else "") + "</span></div>"
            f'<span class="dd-wk-out dd-wk-{tone}">{out}</span></div>')


def _render_forecast(country: str, cblocks: dict, cells: pd.DataFrame, cols: list[RunCol], win: dict,
                     delta_label: str, view_key: str | None, blocks: dict) -> None:
    ref = cols[0]
    missing = [n.split(" (")[0].lower() for n in blocks if n not in cblocks]
    _sec_header("forecast", "Forecast",
                f"{ref.model} {ref.init:%a %d %b %H}z · {delta_label} vs {ref.prev_label}"
                + ("" if ref.prev_exact else " (nearest earlier run)")
                + (f" · no {', '.join(missing)} for {country} in the report" if missing else ""))
    for wkey in ("w1", "w2"):
        tiles = _variable_tiles(cells, cblocks, win, delta_label, wkey)
        if not tiles:
            continue
        st.markdown(_week_band(cells, cblocks, win, wkey), unsafe_allow_html=True)
        _tiles(tiles)
    st.markdown(_variables_grid_html(cells, cblocks, cols, win, view_key, delta_label), unsafe_allow_html=True)


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


def _pct_tile(icon: str, label: str, sub: str, value: float | None, normal: float | None,
              change: dict | None) -> str:
    """GWh over a window on a gauge of % of normal (blue = above normal, red = short of it)
    and, when there is one, the change since the previous issue."""
    if value is None:
        return _tile(icon, label, sub, "n/a", "", None, "flat", "no data")
    pct = (value / normal * 100) if normal else None
    tone, pill, gauge = "flat", "no normal", None
    if pct is not None:
        tone = "cool" if value >= normal else "warm"
        pill = f"{pct:.0f}% of normal"
        gauge = (pct - 100) / 50
    foot, ftone = "", ""
    if change:
        d = change["delta"]
        foot = f"{'▲' if d > 0 else ('▼' if d < 0 else '=')} {d:+,.0f} GWh vs {change['prev_issue']:%a %H}z issue"
        ftone = "cool" if d > 0 else ("warm" if d < 0 else "")
    return _tile(icon, label, sub, f"{value:,.0f}", "GWh", gauge, tone, pill, foot, ftone)


def _hydro_outlook(cfg: dict) -> dict:
    """The production / precipitation-energy summary of the country's hydro area, or the
    reason there is none: dict(area, name, q, error[, df, s])."""
    area = cfg.get("hydro_prod")
    out = {"area": area, "name": HYDRO_PROD_AREAS.get(area, area) if area else None, "q": None, "error": None}
    if not area:
        return out
    try:
        df = load_hydro_prod((area,))
    except Exception as e:
        out["error"] = f"Cannot read {HYDRO_PROD_TABLE}: {e}"
        return out
    if df.empty:
        out["error"] = (f"No production rows for {out['name']} in {HYDRO_PROD_TABLE} yet "
                        "(pipeline/run_daily.py --only hydro-prod).")
        return out
    s = production_series(df[df["area"] == area], HYDRO_PROD_PRECIP_PATTERNS)
    q = production_summary(s, DEEP_DIVE_HYDRO_WINDOW_DAYS, HYDRO_PROD_PRECIP_PATTERNS)
    if not q:
        out["error"] = f"No observed production for {out['name']} yet."
        return out
    out.update(df=df, s=s, q=q)
    return out


def _render_hydro(country: str, cfg: dict, hy: dict) -> None:
    area_lvl = cfg.get("hydro_level")
    W = DEEP_DIVE_HYDRO_WINDOW_DAYS
    q = hy.get("q")
    if q:
        sub = (f"{hy['name']} · last observed day {q['last_obs']:%d %b} · the {W} days to it against the {W} after it"
               + (f" · Volue forecast issued {q['fc_issued']:%d %b}" if q.get("fc_issued") is not None else ""))
    else:
        sub = "reservoirs, snow, groundwater and the two-week outlook"
    _sec_header("hydro", "Hydro", sub)
    if not area_lvl and not hy.get("area"):
        _note(f"No hydro series for {country}: the Hydro section covers the Alpine, Nordic, Iberian and "
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
            _note(f"No reservoir / snow / groundwater series for {country} yet.")
    # outlook — production and precipitation energy, last two weeks vs the next two
    if not hy.get("area"):
        return
    if hy.get("error"):
        _note(hy["error"])
        return
    df, area_prod = hy["df"], hy["area"]
    ref_p = HYDRO_PROD_PRECIP_PATTERNS[0]
    d_rre = _issue_change(df, area_prod, "rre", ref_p, q)
    d_tot = _issue_change(df, area_prod, "tot", "volue", q)
    _tiles([
        _pct_tile("prod", "Production", f"last {W} days · observed", q["obs_tot"], q["obs_tot_n"], None),
        _pct_tile("prod", "Production forecast", f"next {W} days · Volue", q["fwd_tot"], q["fwd_tot_n"], d_tot),
        _pct_tile("rre", "Precip. energy", f"last {W} days · observed", q["obs_rre"], q["obs_rre_n"], None),
        _pct_tile("rre", "Precip. energy expected", f"next {W} days · {ref_p}", q["fwd_rre_ref"], q["fwd_rre_n"], d_rre),
    ])
    others = {p: v for p, v in (q.get("fwd_rre") or {}).items() if p != ref_p and v is not None}
    bal = q.get("fwd_balance")
    parts = []
    if others:
        parts.append(f"Other ensembles, next {W} d: " + " · ".join(f"{p} {v:,.0f} GWh" for p, v in others.items()))
    if bal is not None:
        parts.append(f"precipitation − production, next {W} d: {bal:+,.0f} GWh")
    if parts:
        _note(" · ".join(parts))
    st.plotly_chart(make_hydro_production_chart(hy["s"], hy["name"], W, HYDRO_PROD_PRECIP_PATTERNS, height=420),
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


def _load_scenarios() -> tuple:
    """The Forecast tab's default scenarios: (results per week, info, weeks_sel, weeks_all, error)."""
    try:
        return (*default_scenarios(SCENARIO_DEFAULT_SOURCE), None)
    except Exception as e:
        return None, None, None, None, f"Scenario inputs unavailable: {e}"


def _render_scenarios(country: str, cfg: dict, scen: tuple) -> None:
    results, info, weeks_sel, weeks_all, err = scen
    if results and info is not None:
        sub = (f"{info['label']} · run {pd.Timestamp(info['init']):%a %d %b %Hz} · "
               f"{next(iter(results.values()))['n_members']} members · each trading week clustered on its own "
               f"weekly-mean Z500 field, k = {SCENARIO_K_DEFAULT}")
    else:
        sub = "the Forecast section's split, read for this country"
    _sec_header("scen", "Scenarios", sub)
    if err:
        st.warning(err)
        return
    if info is None:
        st.caption("No Z500 member field in the sandbox yet — the Forecast → Scenarios tab is empty too "
                   "(pipeline run_scenarios.bat for EC-ENS, power_desk_refresh.py cell 4b for EC-AI).")
        return
    if not results:
        st.caption("The run does not cover enough days of any trading week to form scenarios yet.")
        return
    areas = list(cfg.get("scenario", []))
    _note(f"The Forecast → Scenarios tab's defaults; change them there for another split. "
          f"Temperature, wind and solar read on the {info['land']['label']} of the same member IDs.")
    land = {}
    for metric in ("Temperature", "Wind", "Solar"):
        pm = land_members(info, metric, tuple(areas)) if areas else pd.DataFrame()
        pm = _aggregate_members(pm, areas, country, "mean" if metric == "Temperature" else "sum") if not pm.empty else pm
        if not pm.empty:
            land[metric] = pm
    if len(areas) > 1:
        st.caption(f"{country} = {', '.join(areas)} per member: temperature averaged, wind and solar summed.")
    for i, (_, wk) in enumerate(weeks_sel.iterrows()):
        ws = pd.Timestamp(wk["week_start"]).normalize()
        res = results.get(ws)
        if not res:
            continue
        st.markdown(f'<div class="dd-week">{week_title(wk)}</div>', unsafe_allow_html=True)
        kpi_row(scenario_cards(res, info, [ws]), max_cols=4)
        if not areas:
            st.caption(f"No member rows for {country}.")
            continue
        tagged = {m: apply_scenarios(t, res["assign"]) for m, t in land.items()}
        tagged = {m: t for m, t in tagged.items() if not t.empty}
        if not tagged:
            st.caption(f"No member rows for {country} ({', '.join(areas)}) in the {info['land']['label']} yet.")
            continue
        weekly = {m: scenario_weekly_values(t, [ws], "value") for m, t in tagged.items()}
        daily = {m: scenario_daily_summary(t[t["week_start"] == ws], "value") for m, t in tagged.items()}
        st.markdown(_side_by_side_html(country, weekly, scenario_ids(res), weeks_sel[weeks_sel["week_start"] == ws],
                                       res["sizes"], res["n_members"], res["names"]), unsafe_allow_html=True)
        st.plotly_chart(make_country_scenario_panel(daily, country, _UNITS, [ws]),
                        use_container_width=True, key=f"dd_scen_{country}_{wk['week_label']}")
    _note("Precipitation energy by scenario is not available: the member tables carry temperature, wind and "
          "solar only. The hydro outlook above gives the ensemble-mean precipitation and its change since the "
          "previous issue instead.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def _sig(color: str, label: str, value: str) -> str:
    return (f'<span class="dd-sig"><span class="dd-dot" style="background:{color};"></span>'
            f'{label} <b>{value}</b></span>')


def _hero_html(country: str, source: str, ref: RunCol, win: dict, delta_label: str, cells: pd.DataFrame,
               cblocks: dict, hy: dict, scen: tuple) -> str:
    out1, tone1 = _outlook(cells, cblocks, "w1", win["w1"][0])
    out2, _ = _outlook(cells, cblocks, "w2", win["w2"][0])
    moves = _moves(cells, cblocks) if ref.prev_init is not None else "no earlier run"
    sig = []
    for name, block in cblocks.items():
        r = _ref_cell(cells, "w1", name)
        d = np.nan if r is None else float(r["d_norm"])
        if np.isnan(d):
            continue
        t = _tone(d, block)
        color = DD_WARM if t == "warm" else (DD_COOL if t == "cool" else "#a9a89f")
        sig.append(_sig(color, name.split(" (")[0], f"{_signed(d, block['fmt'])} {block['unit']}"))
    q = hy.get("q")
    if q and q.get("fwd_rre_ref") is not None and q.get("fwd_rre_n"):
        pct = q["fwd_rre_ref"] / q["fwd_rre_n"] * 100
        sig.append(_sig(DD_COOL if pct >= 100 else DD_WARM, f"Hydro inflow, next {DEEP_DIVE_HYDRO_WINDOW_DAYS} d",
                        f"{pct:.0f}% of normal"))
    results, _, weeks_sel, _, _ = scen
    res = None
    if results and weeks_sel is not None and len(weeks_sel):
        res = results.get(pd.Timestamp(weeks_sel["week_start"].iloc[0]).normalize())
    if res:
        ids = [i for i in scenario_ids(res) if i != 0]
        if ids:
            s1 = max(ids, key=lambda i: res["sizes"][i])
            label = short_name(res["names"].get(s1)) or f"Scenario {s1}"
            sig.append(_sig(scenario_color(s1), f"Leading scenario · {weeks_sel.iloc[0]['week_label']}",
                            f"{label} · {res['sizes'][s1] / res['n_members']:.0%}"))
    return (f'<div class="dd-hero"><div class="dd-hero-top">'
            f'<div><div class="dd-eyebrow">Country deep dive · {source}</div>'
            f'<div class="dd-country">{country}</div>'
            f'<div class="dd-meta">{ref.model} {ref.init:%a %d %b %H}z · {delta_label} vs {ref.prev_label}</div></div>'
            f'<div class="dd-out-box"><div class="dd-out-label">Outlook · {_short_title(win["t1"])} · '
            f'{win["w1"][0]:%d %b} – {win["w1"][1]:%d %b}</div>'
            f'<div class="dd-out dd-tone-{tone1}">{out1}</div>'
            f'<div class="dd-out-next">{_short_title(win["t2"])}: <b>{out2}</b> · since last run: <b>{moves}</b></div>'
            f'</div></div>'
            + (f'<div class="dd-signals">{"".join(sig)}</div>' if sig else "") + "</div>")


def _pick_country() -> str:
    names = list(DEEP_DIVE_COUNTRIES)
    last = st.session_state.get("dd_country_last", DEEP_DIVE_DEFAULT)
    pills = getattr(st, "pills", None)
    if pills is not None:
        if "dd_country" not in st.session_state:
            st.session_state["dd_country"] = last
        picked = pills("Country", names, selection_mode="single", key="dd_country", label_visibility="collapsed")
    else:
        picked = st.selectbox("Country", names, index=names.index(last), key="dd_country")
    country = picked if picked in names else last      # a pill clicked off keeps the last country
    st.session_state["dd_country_last"] = country
    return country


def render_country_deep_dive(cols: list[RunCol], win: dict, delta_label: str, view_key: str | None,
                             blocks: dict, source: str) -> None:
    """Everything the desk has on one country, on the grid's reference run, compare
    columns and Δ pairing (chosen above the tabs)."""
    st.markdown(DEEP_DIVE_CSS, unsafe_allow_html=True)
    country = _pick_country()
    cfg = DEEP_DIVE_COUNTRIES[country]
    cblocks = _country_blocks(country, cfg, blocks)
    if not cblocks:
        status_banner(f"The Morning Report has no curve for {country} in this source.", "warning")
        return
    cells = compute_grid(cols, win, cblocks)
    hy = _hydro_outlook(cfg)
    scen = _load_scenarios()
    st.markdown(_hero_html(country, source, cols[0], win, delta_label, cells, cblocks, hy, scen),
                unsafe_allow_html=True)
    with _section("dd_sec_forecast"):
        _render_forecast(country, cblocks, cells, cols, win, delta_label, view_key, blocks)
    with _section("dd_sec_hydro"):
        _render_hydro(country, cfg, hy)
    with _section("dd_sec_scen"):
        _render_scenarios(country, cfg, scen)
