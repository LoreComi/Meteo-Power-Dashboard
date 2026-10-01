"""Section 3 — Hydro Monitoring.

Overview tab   What you see on entering: a Europe map of the hydro outlook per
               country. Each hydro area is painted by one layer — reservoir
               levels as % of normal by default; snow water equivalent, ground-
               water and hydro balance are the other layers — with its numbers
               written on the country and the full read-out on hover. River
               temperature stations (Energy Quantified and the Volue share) sit
               on the map as markers coloured by their anomaly vs normal. Under
               the map: the criticalities (series at or below the 15th / 30th
               percentile of their history, at or above the 85th, moving 5+ pts
               of normal in a week; rivers hot or warm) and a grid with every
               area × every layer side by side.
Deep dive      Click a country (or a river station) on the map — or pick it in
               the selector — and the section below the map shows that area
               alone: a KPI card per layer, its flags, the climatology chart of
               every layer, and each river station in the country with observed
               temperature, normal and the latest forecast issues.
Family tabs    Live version of P:/QFA/TonyWeather/Hydro_Report/quantify_*.py:
               for each Volue hydro family (reservoir levels, groundwater = the
               `sgw` snow + groundwater stock, hydro balance) and each country
               the climatology chart and the report's numbers plus the
               stats_*.txt text the report writes.
The maths is in _hydro_quantify.py; hydro data comes from {SBX_SCHEMA}.hydro_daily,
SWE from swe_daily (pipeline/), rivers from river_temp_eq (pipeline/) and
river_temp_volue (notebook cell 6b).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    HYDRO_AREA_CODES, HYDRO_COMPONENTS, HYDRO_MAP_REGIONS, HYDRO_MAP_LABEL_POS, HYDRO_OVERVIEW_LAYERS,
    HYDRO_OVERVIEW_DEFAULT_LAYER, HYDRO_COLOUR_MODES, HYDRO_PCT_OF_NORMAL_RANGE,
    HYDRO_PCTL_CRITICAL, HYDRO_PCTL_LOW, HYDRO_PCTL_HIGH, HYDRO_WEEK_MOVE_PTS,
    HYDRO_SWE_REGIONS, HYDRO_DEEP_DIVE_RIVER_MONTHS, HYDRO_STATION_MODES,
    RIVER_TEMP_WARM_ANOMALY_C, RIVER_TEMP_HOT_C, RIVER_TEMP_COLOUR_RANGE_C,
    RIVER_FLOW_CRITICAL_PCT, RIVER_FLOW_LOW_PCT, RIVER_FLOW_HIGH_PCT, RIVER_FLOW_COLOUR_RANGE_PCT, RIVER_STALE_DAYS,
)
from _data import (
    load_hydro_available, load_hydro_series, load_hydro_component, load_swe_country_daily, load_river_latest,
    load_river_series,
)
from _charts import (
    make_hydro_climatology_chart, make_hydro_anomaly_bars, make_hydro_europe_map, make_river_chart,
)
from _style import DIV_NEG, DIV_MID, DIV_POS
from _hydro_quantify import (
    HYDRO_FAMILIES, HYDRO_DEFAULT_COUNTRIES, build_climatology, quantify_anomaly, split_recent_hist, stats_text,
)
from _ui import kpi_card, kpi_row, status_banner, ordinal

NONE_LABEL = "— none —"


# ══════════════════════════════════════════════════════════════════════════════
# STATUS — one reading of the thresholds for cards, map labels, flags and grid
# ══════════════════════════════════════════════════════════════════════════════

def _pct_status(q: float | None) -> str | None:
    """'critical' | 'low' | 'high' | None from the percentile of this week vs history."""
    if q is None or (isinstance(q, float) and np.isnan(q)):
        return None
    if q <= HYDRO_PCTL_CRITICAL:
        return "critical"
    if q <= HYDRO_PCTL_LOW:
        return "low"
    if q >= HYDRO_PCTL_HIGH:
        return "high"
    return None


def _pct_class(q: float | None) -> str:
    return {"critical": "kpi-card-critical", "low": "kpi-card-warning", "high": "kpi-card-cool"}.get(
        _pct_status(q), "kpi-card-neutral")


def _level_text(q: dict, layer: str) -> str:
    """'92%' for a level series, '-1,215' (unit vs norm) for an anomaly series."""
    if HYDRO_OVERVIEW_LAYERS[layer]["level"] == "percent" and q.get("anomaly_percent") is not None:
        return f"{q['anomaly_percent']:.0f}%"
    return f"{q['anomaly']:+,.0f}"


def _week_move(q: dict, layer: str) -> tuple[float | None, str]:
    """(move over the last week, unit): pts of normal for a level series, the
    layer's unit for an anomaly series whose % of normal means nothing."""
    cfg = HYDRO_OVERVIEW_LAYERS[layer]
    if cfg["level"] == "percent":
        wk = q.get("week_change_pct_points")
        return (float(wk) if wk is not None else None), "pts"
    a, aw = q.get("anomaly"), q.get("anomaly_w-1")
    if a is None or aw is None:
        return None, cfg["unit"]
    return float(a - aw), cfg["unit"]


def _fmt_move(wk: float | None, unit: str, arrow: bool = False) -> str:
    if wk is None:
        return "—"
    head = ("▲ " if wk > 0 else "▼ " if wk < 0 else "■ ") if arrow else ""
    body = f"{wk:+.0f} pts" if unit == "pts" else f"{wk:+,.0f} {unit}"
    return head + (body.lstrip("+-") if arrow else body)


def _flags(q: dict, layer: str) -> list[tuple[str, str]]:
    """Criticality flags for one series' quantify dict: (level, text) with level in
    critical / warning / high. Percentile first, then — for level series — the
    week-on-week move in pts of normal."""
    out: list[tuple[str, str]] = []
    pq = q.get("anomaly_quantile")
    status = _pct_status(pq)
    if status == "critical":
        out.append(("critical", f"{ordinal(round(pq))} percentile of {q.get('n_hist_years', '?')} yrs — critically low"))
    elif status == "low":
        out.append(("warning", f"{ordinal(round(pq))} percentile of {q.get('n_hist_years', '?')} yrs — low"))
    elif status == "high":
        out.append(("high", f"{ordinal(round(pq))} percentile of {q.get('n_hist_years', '?')} yrs — very high"))
    wk, unit = _week_move(q, layer)
    if unit == "pts" and wk is not None and abs(wk) >= HYDRO_WEEK_MOVE_PTS:
        out.append(("warning" if wk < 0 else "high",
                    f"{wk:+.0f} pts of normal in a week — fast {'drawdown' if wk < 0 else 'refill'}"))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SERIES AND METRICS — every area × every layer, from each layer's source
# ══════════════════════════════════════════════════════════════════════════════

def _layer_series(layer: str) -> dict[str, tuple[pd.Series, pd.Series | None]]:
    """area code -> (actual, normal) daily series for one layer.

    Sources (HYDRO_OVERVIEW_LAYERS[layer]['source']):
      hydro_daily  a Volue component (actual SA, normal N where the family has one)
      swe_daily    the internal Exolabs SWE model's mean mm per Alpine country,
                   uploaded by pipeline/swe_upload.py; no normal → mean of years
    To add another (river levels, …): load its (area, day, actual[, normal])
    series here and register the layer — metrics, map, flags, grid and the deep
    dive pick it up unchanged. The loaders are cached, so this is cheap to call.
    """
    cfg = HYDRO_OVERVIEW_LAYERS[layer]
    src = cfg.get("source", "hydro_daily")
    series: dict[str, tuple[pd.Series, pd.Series | None]] = {}
    try:
        if src == "hydro_daily":
            family = cfg["family"]
            df = load_hydro_component(HYDRO_COMPONENTS[family])
            for area, g in (df.groupby("area") if not df.empty else []):
                actual = g[g["data_type"] == "SA"].set_index("day")["value"].sort_index()
                normal = g[g["data_type"] == "N"].set_index("day")["value"].sort_index()
                series[str(area)] = (actual, normal if (HYDRO_FAMILIES[family]["has_norm"] and not normal.empty) else None)
        elif src == "swe_daily":
            df = load_swe_country_daily()
            for region, g in (df.groupby("region") if not df.empty else []):
                area = HYDRO_SWE_REGIONS.get(str(region))
                if area:
                    series[area] = (g.set_index("day")["swe_mean_mm"].sort_index(), None)
    except Exception:
        return {}
    return series


@st.cache_data(ttl=3600, show_spinner=False)
def _overview_metrics() -> dict[str, dict[str, dict]]:
    """layer -> hydro area code -> quantify_anomaly dict, for every area with a
    series — the report maths per area, exactly as the family tabs do it."""
    out: dict[str, dict[str, dict]] = {}
    for layer in HYDRO_OVERVIEW_LAYERS:
        per_area: dict[str, dict] = {}
        for area, (actual, normal) in _layer_series(layer).items():
            if actual.dropna().empty:
                continue
            try:
                q = quantify_anomaly(build_climatology(actual, normal))
            except Exception:
                continue
            if q:
                per_area[area] = q
        out[layer] = per_area
    return out


# ══════════════════════════════════════════════════════════════════════════════
# RIVER TEMPERATURE STATIONS (EQ via pipeline/, Volue via notebook cell 6b)
# ══════════════════════════════════════════════════════════════════════════════

def _river_flags(value, anom, fc_max7, flow_pct=None) -> list[tuple[str, str]]:
    """Temperature flags first (hot / warm / forecast peak), then flow vs its normal."""
    out: list[tuple[str, str]] = []
    if value is not None and value >= RIVER_TEMP_HOT_C:
        out.append(("critical", f"{value:.1f} °C — at or above {RIVER_TEMP_HOT_C:.0f} °C"))
    elif anom is not None and anom >= RIVER_TEMP_WARM_ANOMALY_C:
        out.append(("warning", f"{anom:+.1f} °C vs normal — warm"))
    if fc_max7 is not None and fc_max7 >= RIVER_TEMP_HOT_C and not (value is not None and value >= RIVER_TEMP_HOT_C):
        out.append(("warning", f"forecast peaks at {fc_max7:.1f} °C within 7 days"))
    if flow_pct is not None:
        if flow_pct <= RIVER_FLOW_CRITICAL_PCT:
            out.append(("critical", f"flow {flow_pct:.0f}% of normal — very low water"))
        elif flow_pct <= RIVER_FLOW_LOW_PCT:
            out.append(("warning", f"flow {flow_pct:.0f}% of normal — low water"))
        elif flow_pct >= RIVER_FLOW_HIGH_PCT:
            out.append(("high", f"flow {flow_pct:.0f}% of normal — high water"))
    return out


def _river_rows(mode: str = "temperature") -> list[dict]:
    """One dict per station, every source: for the map (lat, lon, z, hover, code),
    the tables and the deep dive. `mode` picks the marker colour value: the
    temperature anomaly, or the flow as % of normal."""
    df = load_river_latest()
    if df is None or df.empty:
        return []
    rows = []
    for r in df.itertuples(index=False):
        def f(x):
            return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
        value, normal, anom = f(r.value), f(r.normal), f(r.anomaly)
        fc_max, fc_min, fc_norm = f(r.fc_max7), f(r.fc_min7), f(r.fc_normal7)
        flow, flow_norm, flow_pct = f(getattr(r, "flow", None)), f(getattr(r, "flow_normal", None)), f(getattr(r, "flow_pct", None))
        flow_day = pd.Timestamp(r.flow_day) if hasattr(r, "flow_day") and not pd.isna(r.flow_day) else None
        day = pd.Timestamp(r.day) if not pd.isna(r.day) else None
        # a series that stopped updating (EQ's Bugey flow ends in 2019) is shown with
        # its date but is neither flagged nor used to colour the marker
        today = pd.Timestamp.today().normalize()
        t_stale = day is not None and (today - day).days > RIVER_STALE_DAYS
        q_stale = flow_day is not None and (today - flow_day).days > RIVER_STALE_DAYS
        flags = _river_flags(None if t_stale else value, None if t_stale else anom, None if t_stale else fc_max,
                             None if q_stale else flow_pct)
        area = str(r.area).upper() if r.area is not None else ""
        lines = [f"<b>{r.station}</b> · river station ({r.source})"]
        if value is not None:
            lines.append(f"Temperature {value:.1f} °C" + (f" · {day:%d %b %Y}" if t_stale else (f" · {day:%d %b}" if day is not None else ""))
                         + (f" · normal {normal:.1f} °C · <b>{anom:+.1f} °C</b>" if anom is not None else " · no normal")
                         + (" · <i>stale</i>" if t_stale else ""))
        if fc_max is not None and not t_stale:
            issued = pd.Timestamp(r.fc_issued) if not pd.isna(r.fc_issued) else None
            lines.append(f"Next 7 days ({r.fc_tag}{f', issued {issued:%d %b %Hz}' if issued is not None else ''}): "
                         f"{fc_min:.1f}–{fc_max:.1f} °C" + (f" · normal {fc_norm:.1f} °C" if fc_norm is not None else ""))
        if flow is not None:
            lines.append(f"Flow {flow:,.0f} m³/s" + (f" · {flow_day:%d %b %Y}" if q_stale else (f" · {flow_day:%d %b}" if flow_day is not None else ""))
                         + (f" · normal {flow_norm:,.0f} · <b>{flow_pct:.0f}% of normal</b>" if flow_pct is not None else " · no normal")
                         + (" · <i>stale</i>" if q_stale else ""))
        for _l, t in flags:
            lines.append(f"⚑ {t}")
        lines.append("<i>click for the country's deep dive</i>")
        rows.append({"station_key": r.station_key, "name": r.station, "river": r.river, "site": r.site, "area": area,
                     "source": r.source, "lat": f(r.latitude), "lon": f(r.longitude), "day": day, "value": value,
                     "normal": normal, "anomaly": anom, "fc_tag": r.fc_tag, "fc_max7": fc_max, "fc_min7": fc_min,
                     "fc_normal7": fc_norm, "flow": flow, "flow_normal": flow_norm, "flow_pct": flow_pct,
                     "flow_day": flow_day, "stale": {"temperature": t_stale, "flow": q_stale},
                     "z": ((None if q_stale else flow_pct) if mode == "flow" else (None if t_stale else anom)),
                     "flags": flags, "hover": "<br>".join(lines), "code": f"station:{r.station_key}|{area}"})
    return rows


def _flow_class(pct: float | None) -> str:
    if pct is None:
        return ""
    if pct <= RIVER_FLOW_CRITICAL_PCT:
        return "hy-crit"
    if pct <= RIVER_FLOW_LOW_PCT:
        return "hy-low"
    if pct >= RIVER_FLOW_HIGH_PCT:
        return "hy-high"
    return ""


def _river_table_html(rows: list[dict]) -> str:
    head = ("<tr><th rowspan='2' style='text-align:left;vertical-align:bottom'>Station</th>"
            "<th rowspan='2' style='text-align:left;vertical-align:bottom'>Source</th>"
            "<th colspan='4' class='hy-layer'>Temperature °C</th><th colspan='2' class='hy-layer'>7-day forecast</th>"
            "<th colspan='3' class='hy-layer'>Flow m³/s</th><th rowspan='2' style='text-align:left;vertical-align:bottom'>Flags</th></tr>"
            "<tr><th class='hy-sub hy-first'>Latest</th><th class='hy-sub'>°C</th><th class='hy-sub'>Normal</th><th class='hy-sub'>Δ</th>"
            "<th class='hy-sub hy-first'>Range</th><th class='hy-sub'>vs normal</th>"
            "<th class='hy-sub hy-first'>Latest</th><th class='hy-sub'>Normal</th><th class='hy-sub'>% norm</th></tr>")
    body = []
    for r in sorted(rows, key=lambda x: -(x["value"] if x["value"] is not None else -99)):
        stale = r.get("stale", {})
        t_stale, q_stale = stale.get("temperature", False), stale.get("flow", False)
        acls = "" if t_stale else ("hy-crit" if (r["value"] is not None and r["value"] >= RIVER_TEMP_HOT_C) else
                                   ("hy-low" if (r["anomaly"] is not None and r["anomaly"] >= RIVER_TEMP_WARM_ANOMALY_C) else ""))
        fcls = "hy-low" if (r["fc_max7"] is not None and r["fc_max7"] >= RIVER_TEMP_HOT_C and not t_stale) else ""
        fc = (f"{r['fc_min7']:.1f}–{r['fc_max7']:.1f} <span class='hy-agg'>{r['fc_tag']}</span>"
              if r["fc_max7"] is not None else "—")
        fcd = f"{r['fc_max7'] - r['fc_normal7']:+.1f}" if (r["fc_max7"] is not None and r["fc_normal7"] is not None) else "—"
        qcls = "" if q_stale else _flow_class(r.get("flow_pct"))
        flags = " · ".join(t for _l, t in r["flags"]) or "—"
        day_txt = (f"{r['day']:%b %Y} <span class='hy-agg'>stale</span>" if t_stale else f"{r['day']:%d %b}") if r["day"] is not None else None
        cells = [
            f"<td class='mc-region'>{r['name']} <span class='hy-agg'>{r['area']}</span></td>",
            f"<td class='hy-agg' style='text-align:left'>{r['source']}</td>",
            f"<td class='hy-first'>{day_txt}</td>" if day_txt else "<td class='hy-na hy-first'>—</td>",
            f"<td class='{acls}'>{r['value']:.1f}</td>" if r["value"] is not None else "<td class='hy-na'>—</td>",
            f"<td>{r['normal']:.1f}</td>" if r["normal"] is not None else "<td class='hy-na'>—</td>",
            f"<td class='{acls}'>{r['anomaly']:+.1f}</td>" if r["anomaly"] is not None else "<td class='hy-na'>—</td>",
            f"<td class='hy-first {fcls}'>{fc}</td>",
            f"<td class='{fcls}'>{fcd}</td>",
            (f"<td class='hy-first {qcls}'>{r['flow']:,.0f}" + (f" <span class='hy-agg'>{r['flow_day']:%b %Y} stale</span>" if q_stale else "") + "</td>")
            if r.get("flow") is not None else "<td class='hy-na hy-first'>—</td>",
            f"<td>{r['flow_normal']:,.0f}</td>" if r.get("flow_normal") is not None else "<td class='hy-na'>—</td>",
            f"<td class='{qcls}'>{r['flow_pct']:.0f}%</td>" if r.get("flow_pct") is not None else "<td class='hy-na'>—</td>",
            f"<td class='hy-agg' style='text-align:left'>{flags}</td>",
        ]
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<div class='mc-block'><table class='mc-table hy-table'><thead>{head}</thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


# ══════════════════════════════════════════════════════════════════════════════
# MAP PAINTING
# ══════════════════════════════════════════════════════════════════════════════

def _colour_value(q: dict, colour_by: str) -> float | None:
    v = q.get(colour_by)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def _label(area: str, q: dict, layer: str) -> str:
    """The numbers written on the country: code, level vs normal, percentile, week move."""
    pq = q.get("anomaly_quantile")
    line2 = _level_text(q, layer) + (f" · P{pq:.0f}" if pq is not None else "")
    wk, unit = _week_move(q, layer)
    line3 = _fmt_move(wk, unit, arrow=True) if wk is not None else ""
    return "<br>".join(x for x in (f"<b>{area}</b>", line2, line3) if x)


def _hover_html(area: str, cfg: dict, layer: str, metrics: dict[str, dict[str, dict]]) -> str:
    """The full read-out for one area: every layer's numbers, the chosen layer in
    bold, then its flags."""
    name = cfg["name"] + (" — aggregate" if cfg.get("aggregate") else "")
    lines = [f"<b>{name}</b>"]
    for lyr in HYDRO_OVERVIEW_LAYERS:
        q = metrics.get(lyr, {}).get(area)
        if not q:
            continue
        unit = HYDRO_OVERVIEW_LAYERS[lyr]["unit"]
        as_of = q.get("as_of")
        as_of_s = f" · {pd.Timestamp(as_of):%d %b}" if as_of is not None else ""
        head = f"<b>{lyr}</b>" if lyr == layer else lyr
        lines.append(f"{head}: {q['latest_value']:,.0f} {unit}{as_of_s}")
        bits = [f"{q['anomaly']:+,.0f} {unit} vs norm"]
        if HYDRO_OVERVIEW_LAYERS[lyr]["level"] == "percent" and q.get("anomaly_percent") is not None:
            bits.append(f"{q['anomaly_percent']}% of normal")
        if q.get("anomaly_quantile") is not None:
            bits.append(f"{ordinal(round(q['anomaly_quantile']))} pct of {q['n_hist_years']} yrs")
        wk, wunit = _week_move(q, lyr)
        if wk is not None:
            bits.append(f"Δ week {_fmt_move(wk, wunit)}")
        lines.append("&nbsp;&nbsp;" + " · ".join(bits))
        for _lvl, text in _flags(q, lyr):
            lines.append(f"&nbsp;&nbsp;⚑ {text}")
    lines.append("<i>click for the deep dive</i>")
    return "<br>".join(lines)


def _paint(layer: str, colour_by: str, metrics: dict[str, dict[str, dict]]) -> list[dict]:
    """One row per ISO-3 country to draw. A country's own series wins; an
    aggregate (Nordics, SEE) paints the members that have none and is labelled
    once; what is left is drawn without a value. `code` is what a click returns."""
    lm = metrics.get(layer, {})
    rows: dict[str, dict] = {}
    own = [(a, c) for a, c in HYDRO_MAP_REGIONS.items() if not c.get("aggregate")]
    aggs = [(a, c) for a, c in HYDRO_MAP_REGIONS.items() if c.get("aggregate")]
    for area, cfg in own + aggs:
        q = lm.get(area)
        if not q:
            continue
        labelled = False
        for iso in cfg["iso3"]:
            if iso in rows:
                continue
            lat, lon = HYDRO_MAP_LABEL_POS.get(iso, (None, None))
            label = "" if (cfg.get("aggregate") and labelled) else _label(area, q, layer)
            labelled = True
            rows[iso] = {"iso3": iso, "area": area, "z": _colour_value(q, colour_by), "label": label,
                         "hover": _hover_html(area, cfg, layer, metrics), "lat": lat, "lon": lon,
                         "code": f"area:{area}"}
    for area, cfg in own + aggs:
        for iso in cfg["iso3"]:
            if iso in rows:
                continue
            lat, lon = HYDRO_MAP_LABEL_POS.get(iso, (None, None))
            rows[iso] = {"iso3": iso, "area": area, "z": None,
                         "label": "" if cfg.get("aggregate") else f"{area}<br>—",
                         "hover": f"<b>{cfg['name']}</b><br>no {layer.lower()} series<br><i>click for the deep dive</i>",
                         "lat": lat, "lon": lon, "code": f"area:{area}"}
    return list(rows.values())


def _grid_html(metrics: dict[str, dict[str, dict]]) -> str:
    """Areas as rows, layers as column groups — each layer's level, percentile and
    week move side by side, shaded by status (the number and the word carry it too)."""
    layers = list(HYDRO_OVERVIEW_LAYERS)
    known = list(HYDRO_MAP_REGIONS)
    extra = sorted({a for l in layers for a in metrics.get(l, {}) if a not in known})
    areas = [a for a in known + extra if any(a in metrics.get(l, {}) for l in layers)]

    def level_head(l: str) -> str:
        return "% norm" if HYDRO_OVERVIEW_LAYERS[l]["level"] == "percent" else f"{HYDRO_OVERVIEW_LAYERS[l]['unit']} vs norm"

    head = ("<tr><th rowspan='2' style='text-align:left;vertical-align:bottom'>Area</th>"
            + "".join(f"<th colspan='3' class='hy-layer'>{l}</th>" for l in layers) + "</tr>"
            + "<tr>" + "".join(f"<th class='hy-sub hy-first'>{level_head(l)}</th><th class='hy-sub'>Pctl</th>"
                               f"<th class='hy-sub'>Δ wk</th>" for l in layers) + "</tr>")
    body = []
    for a in areas:
        cfg = HYDRO_MAP_REGIONS.get(a, {"name": a})
        name = cfg["name"] + (" <span class='hy-agg'>aggregate</span>" if cfg.get("aggregate") else "")
        cells = [f"<td class='mc-region'>{name}</td>"]
        for l in layers:
            q = metrics.get(l, {}).get(a)
            if not q:
                cells += ["<td class='hy-na hy-first'>—</td>", "<td class='hy-na'>—</td>", "<td class='hy-na'>—</td>"]
                continue
            pq = q.get("anomaly_quantile")
            level = _level_text(q, l)
            wk, wunit = _week_move(q, l)
            pcls = {"critical": "hy-crit", "low": "hy-low", "high": "hy-high"}.get(_pct_status(pq), "")
            ptxt = f"{ordinal(round(pq))}" if pq is not None else "—"
            if pcls == "hy-crit":
                ptxt += " crit. low"
            elif pcls == "hy-low":
                ptxt += " low"
            elif pcls == "hy-high":
                ptxt += " high"
            wcls = "" if (wunit != "pts" or wk is None or abs(wk) < HYDRO_WEEK_MOVE_PTS) else ("hy-low" if wk < 0 else "hy-high")
            wtxt = _fmt_move(wk, wunit)
            cells += [f"<td class='hy-first'>{level}</td>", f"<td class='{pcls}'>{ptxt}</td>", f"<td class='{wcls}'>{wtxt}</td>"]
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<div class='mc-block'><table class='mc-table hy-table'><thead>{head}</thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def _overview_table(metrics: dict[str, dict[str, dict]]) -> pd.DataFrame:
    rows = []
    for layer, per_area in metrics.items():
        for area, q in per_area.items():
            wk, wunit = _week_move(q, layer)
            rows.append({"area": area, "name": HYDRO_MAP_REGIONS.get(area, {}).get("name", area), "layer": layer,
                         "as_of": pd.Timestamp(q["as_of"]).date() if q.get("as_of") is not None else None,
                         "latest_value": q.get("latest_value"), "norm_today": q.get("norm_today"),
                         "anomaly": q.get("anomaly"),
                         "anomaly_percent": q.get("anomaly_percent") if HYDRO_OVERVIEW_LAYERS[layer]["level"] == "percent" else None,
                         "anomaly_quantile": q.get("anomaly_quantile"),
                         "week_move": wk, "week_move_unit": wunit,
                         "n_hist_years": q.get("n_hist_years"),
                         "flags": "; ".join(t for _l, t in _flags(q, layer))})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# DEEP DIVE — one area, every variable
# ══════════════════════════════════════════════════════════════════════════════

def _selection_from_event(event) -> tuple[str | None, str | None]:
    """(area code, station key) from a plotly_chart selection — the second item of
    customdata is 'area:FR' for a country or 'station:<key>|FR' for a marker.
    Falls back to the choropleth location (ISO-3) when customdata is missing."""
    try:
        sel = event.selection if hasattr(event, "selection") else (event or {}).get("selection")
        pts = sel.points if hasattr(sel, "points") else (sel or {}).get("points", [])
    except Exception:
        pts = []
    iso_to_area = {iso: a for a, c in HYDRO_MAP_REGIONS.items() if not c.get("aggregate") for iso in c["iso3"]}
    for a, c in HYDRO_MAP_REGIONS.items():
        if c.get("aggregate"):
            for iso in c["iso3"]:
                iso_to_area.setdefault(iso, a)
    for p in pts or []:
        get = p.get if isinstance(p, dict) else (lambda k, d=None: getattr(p, k, d))
        cd = get("customdata")
        code = cd[1] if isinstance(cd, (list, tuple)) and len(cd) >= 2 else (cd if isinstance(cd, str) else None)
        if isinstance(code, str):
            if code.startswith("area:"):
                return code[5:], None
            if code.startswith("station:"):
                key, _, area = code[8:].partition("|")
                return (area or None), key
        loc = get("location")
        if isinstance(loc, str) and loc in iso_to_area:
            return iso_to_area[loc], None
    return None, None


def _close_deep_dive() -> None:
    st.session_state["hy_dd_area"] = None
    st.session_state["hy_dd_station"] = None


def _sync_select(name_to_area: dict[str, str]) -> None:
    v = st.session_state.get("hy_dd_select")
    st.session_state["hy_dd_area"] = name_to_area.get(v) if v and v != NONE_LABEL else None
    st.session_state["hy_dd_station"] = None


def _layer_kpi(layer: str, q: dict) -> str:
    cfg = HYDRO_OVERVIEW_LAYERS[layer]
    unit = cfg["unit"]
    pq = q.get("anomaly_quantile")
    cls = _pct_class(pq)
    level = _level_text(q, layer) + (" of normal" if cfg["level"] == "percent" else f" {unit} vs norm")
    delta = (f"<div class='kpi-delta kpi-delta-flat'>{level}"
             + (f" · {ordinal(round(pq))} pctl of {q.get('n_hist_years', '?')} yrs" if pq is not None else "") + "</div>")
    wk, wunit = _week_move(q, layer)
    rank = f"<div class='kpi-rank'>{_fmt_move(wk, wunit)} in a week</div>" if wk is not None else ""
    as_of = q.get("as_of")
    label = layer + (f" · {pd.Timestamp(as_of):%d %b}" if as_of is not None else "")
    return kpi_card(label, f"{q['latest_value']:,.0f} {unit}", cls, delta, rank)


def _chips(items: list[tuple[str, str, str]]) -> None:
    """items: (level, head, text)."""
    order = {"critical": 0, "warning": 1, "high": 2}
    items = sorted(items, key=lambda t: (order.get(t[0], 9), t[1]))
    html = "".join(f"<span class='hy-flag hy-flag-{lvl}'><b>{head}</b> {text}</span>" for lvl, head, text in items)
    st.markdown(f"<div class='hy-flags'>{html}</div>", unsafe_allow_html=True)


def _render_deep_dive(area: str, metrics: dict[str, dict[str, dict]], rivers: list[dict],
                      focus_station: str | None) -> None:
    cfg = HYDRO_MAP_REGIONS.get(area, {"name": area, "iso3": []})
    name = cfg["name"] + (" (aggregate)" if cfg.get("aggregate") else "")
    h1, h2 = st.columns([6, 1])
    with h1:
        st.markdown(f"##### Deep dive · {name}")
    with h2:
        st.button("Close", key="hy_dd_close", on_click=_close_deep_dive, use_container_width=True)

    layers_with = [l for l in HYDRO_OVERVIEW_LAYERS if metrics.get(l, {}).get(area)]
    stations = [r for r in rivers if r["area"] == area]
    if not layers_with and not stations:
        status_banner(f"No hydro series or river stations for {name} yet.", "warning")
        return

    if layers_with:
        kpi_row([_layer_kpi(l, metrics[l][area]) for l in layers_with], max_cols=4)
        flags = [(lvl, l, t) for l in layers_with for lvl, t in _flags(metrics[l][area], l)]
        flags += [(lvl, r["name"], t) for r in stations for lvl, t in r["flags"]]
        if flags:
            _chips(flags)
        else:
            st.caption("No criticalities flagged for this area.")
        cols = st.columns(2)
        i = 0
        for l in layers_with:
            series = _layer_series(l).get(area)
            if not series:
                continue
            actual, normal = series
            try:
                clim = build_climatology(actual, normal)
            except Exception:
                continue
            hist, recent = split_recent_hist(clim["years"], 3)
            with cols[i % 2]:
                st.plotly_chart(make_hydro_climatology_chart(clim, hist, recent, f"{cfg['name']} — {l.lower()}",
                                                             HYDRO_OVERVIEW_LAYERS[l]["unit"], height=360),
                                use_container_width=True, key=f"hy_dd_chart_{area}_{i}")
                if clim["norm_source"] == "mean_of_years":
                    st.caption("norm = mean of completed years (no provider normal for this series)")
            i += 1
    elif stations:
        flags = [(lvl, r["name"], t) for r in stations for lvl, t in r["flags"]]
        if flags:
            _chips(flags)

    if stations:
        st.markdown(f"**River stations** · {len(stations)} · temperature (observed, normal, the latest forecast "
                    f"issues; dotted line = {RIVER_TEMP_HOT_C:.0f} °C) and, where EQ has it, flow against its normal")
        st.markdown(_river_table_html(stations), unsafe_allow_html=True)
        order = sorted(stations, key=lambda r: (r["station_key"] != focus_station,
                                                -(r["value"] if r["value"] is not None else -99)))
        series_by_source: dict[str, pd.DataFrame] = {}
        with st.spinner("Loading river series…"):
            for src in sorted({r["source"] for r in order}):
                keys = tuple(sorted(r["station_key"] for r in order if r["source"] == src))
                series_by_source[src] = load_river_series(src, keys, HYDRO_DEEP_DIVE_RIVER_MONTHS)
        for r in order:
            df = series_by_source.get(r["source"], pd.DataFrame())
            if not df.empty:
                df = df[df["station_key"] == r["station_key"]]
            temp = df[df["variable"] == "temperature"] if not df.empty else df
            flow = df[df["variable"] == "flow"] if not df.empty else pd.DataFrame()
            # one row per station: temperature on the left, flow on the right when it exists
            c_t, c_f = st.columns(2)
            with c_t:
                st.plotly_chart(make_river_chart(temp, f"{r['name']} — temperature ({r['source']})", "°C", RIVER_TEMP_HOT_C),
                                use_container_width=True, key=f"hy_dd_riv_t_{r['source']}_{r['station_key']}")
            with c_f:
                if not flow.empty:
                    st.plotly_chart(make_river_chart(flow, f"{r['name']} — flow ({r['source']})", "m³/s", None, fmt=",.0f"),
                                    use_container_width=True, key=f"hy_dd_riv_q_{r['source']}_{r['station_key']}")
                else:
                    st.caption(f"{r['name']}: no flow series in {r['source']}.")


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW TAB
# ══════════════════════════════════════════════════════════════════════════════

def _render_overview(avail: pd.DataFrame):
    c1, c2, c3 = st.columns([2.6, 1.8, 1.2])
    with c1:
        layers = list(HYDRO_OVERVIEW_LAYERS)
        layer = st.radio("Map layer", layers, index=layers.index(HYDRO_OVERVIEW_DEFAULT_LAYER),
                         horizontal=True, key="hy_ov_layer")
    with c3:
        show_rivers = st.checkbox("River stations", value=True, key="hy_ov_rivers",
                                  help="Latest temperature vs normal and flow vs normal per station, 7-day forecast "
                                       "peak on hover. EQ stations from Power_dashboard/pipeline, Volue stations from "
                                       "the refresh notebook.")
        station_mode = st.radio("Colour stations by", list(HYDRO_STATION_MODES), horizontal=True,
                                format_func=HYDRO_STATION_MODES.get, key="hy_ov_station_mode",
                                label_visibility="collapsed", disabled=not show_rivers)
    with c2:
        # a deviation series has no meaningful % of normal to colour by
        modes = [m for m in HYDRO_COLOUR_MODES
                 if not (m == "anomaly_percent" and HYDRO_OVERVIEW_LAYERS[layer]["level"] != "percent")]
        default_mode = HYDRO_OVERVIEW_LAYERS[layer]["colour_by"]
        colour_by = st.radio("Colour by", modes, index=modes.index(default_mode), horizontal=True,
                             format_func=HYDRO_COLOUR_MODES.get, key=f"hy_ov_mode_{layer}")

    with st.spinner("Quantifying every hydro series…"):
        metrics = _overview_metrics()
        rivers = _river_rows(station_mode) if show_rivers else []
    lm = metrics.get(layer, {})
    if not lm:
        if HYDRO_OVERVIEW_LAYERS[layer].get("source") == "swe_daily":
            status_banner("No SWE rows in swe_daily yet — run Power_dashboard/pipeline/run_daily.bat "
                          "(it uploads the Exolabs model's CSVs).", "warning")
        else:
            status_banner(f"No {layer.lower()} series in hydro_daily — nothing to paint. "
                          "Has power_desk_refresh.py run, and do the Volue area codes match?", "warning")
        return
    if show_rivers and not rivers:
        st.caption("No river temperature stations yet — river_temp_eq is written by Power_dashboard/pipeline, "
                   "river_temp_volue by the refresh notebook.")

    painted = _paint(layer, colour_by, metrics)
    if colour_by == "anomaly_percent":
        zmin, zmax = HYDRO_PCT_OF_NORMAL_RANGE
        zmid, cb = 100.0, "% of normal"
    elif colour_by == "anomaly_quantile":
        zmin, zmax, zmid, cb = 0.0, 100.0, 50.0, "percentile of this week"
    else:
        m = max([abs(r["z"]) for r in painted if r.get("z") is not None] or [1.0])
        zmin, zmax, zmid, cb = -m, m, 0.0, f"{HYDRO_OVERVIEW_LAYERS[layer]['unit']} vs norm"

    as_of = [pd.Timestamp(q["as_of"]) for q in lm.values() if q.get("as_of") is not None]
    n_yrs = sorted({q.get("n_hist_years") for q in lm.values() if q.get("n_hist_years")})
    yrs = "–".join(str(n) for n in (n_yrs[0], n_yrs[-1])) if len(n_yrs) > 1 else (str(n_yrs[0]) if n_yrs else "?")
    aggs = [f"{a} paints {', '.join(c['iso3'])} where they have no series of their own"
            for a, c in HYDRO_MAP_REGIONS.items() if c.get("aggregate") and a in lm]
    legend = (layer + (f" as of {max(as_of):%d %b %Y}" if as_of else "")
              + " · red = below normal / low in the history, blue = above / high · on each country: level vs "
              f"normal · percentile of this week against the same week in {yrs} years · move in a week · hover for "
              "every layer and the flags · <b>click a country or a station for its deep dive</b>"
              + (" · " + "; ".join(aggs) if aggs else "")
              + ((f" · ● river stations coloured by the latest temperature vs normal, red = warm, ±{RIVER_TEMP_COLOUR_RANGE_C:.0f} °C"
                  if station_mode == "temperature" else
                  f" · ● river stations coloured by the latest flow as % of normal, red = low water, "
                  f"{RIVER_FLOW_COLOUR_RANGE_PCT[0]:.0f}–{RIVER_FLOW_COLOUR_RANGE_PCT[1]:.0f} %") if rivers else "")
              + ".")
    station_scale = None
    if station_mode == "flow":
        station_scale = dict(cmin=RIVER_FLOW_COLOUR_RANGE_PCT[0], cmax=RIVER_FLOW_COLOUR_RANGE_PCT[1], cmid=100.0,
                             colorscale=[[0.0, DIV_POS], [0.5, DIV_MID], [1.0, DIV_NEG]])

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        fig = make_hydro_europe_map(painted, zmin, zmax, zmid, cb, stations=rivers,
                                    station_range=RIVER_TEMP_COLOUR_RANGE_C, station_scale=station_scale)
        event = st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False},
                                key="hy_map", on_select="rerun", selection_mode="points")
        area_sel, station_sel = _selection_from_event(event)
        sig = f"{area_sel}|{station_sel}"
        if area_sel and sig != st.session_state.get("hy_map_last_sel"):
            # a new click: open that area (a repeated selection on rerun is ignored,
            # so the selector below can still change or close the deep dive)
            st.session_state["hy_map_last_sel"] = sig
            st.session_state["hy_dd_area"] = area_sel
            st.session_state["hy_dd_station"] = station_sel
        st.markdown(f"<div class='mc-sub'>{legend}</div>", unsafe_allow_html=True)

    # ── Criticalities across every layer, beside the map ──────────────────────
    with right:
        st.markdown("##### Criticalities")
        items = []
        for lyr in HYDRO_OVERVIEW_LAYERS:
            for area, q in metrics.get(lyr, {}).items():
                for level, text in _flags(q, lyr):
                    items.append((level, HYDRO_MAP_REGIONS.get(area, {}).get("name", area), f"{lyr.lower()}: {text}"))
        for r in rivers:
            for level, text in r["flags"]:
                items.append((level, r["name"], f"river temperature: {text}"))
        if items:
            _chips(items)
        else:
            status_banner(f"None flagged — every series sits between the {ordinal(HYDRO_PCTL_LOW)} and "
                          f"{ordinal(HYDRO_PCTL_HIGH)} percentile of its history and moved less than "
                          f"{HYDRO_WEEK_MOVE_PTS} pts of normal in the week.", "good")
        st.caption(f"Flags: at or below the {ordinal(HYDRO_PCTL_CRITICAL)} percentile = critically low, the "
                   f"{ordinal(HYDRO_PCTL_LOW)} = low, at or above the {ordinal(HYDRO_PCTL_HIGH)} = very high; "
                   f"|Δ| ≥ {HYDRO_WEEK_MOVE_PTS} pts of normal in a week = fast drawdown / refill (level series "
                   "only — the hydro balance and SWE are shown as anomalies, so their week move is in GWh / mm and "
                   f"not flagged). River temperature: ≥ {RIVER_TEMP_HOT_C:.0f} °C = hot, ≥ {RIVER_TEMP_WARM_ANOMALY_C:.0f} °C "
                   f"above normal = warm; river flow: ≤ {RIVER_FLOW_CRITICAL_PCT:.0f} % of normal = very low, "
                   f"≤ {RIVER_FLOW_LOW_PCT:.0f} % = low, ≥ {RIVER_FLOW_HIGH_PCT:.0f} % = high water (indicative — "
                   "discharge limits and minimum flows differ by plant and river). Thresholds in _config "
                   "(HYDRO_PCTL_*, HYDRO_WEEK_MOVE_PTS, RIVER_TEMP_*, RIVER_FLOW_*).")

    # ── Deep dive: the clicked area, or the one picked here ───────────────────
    river_areas = sorted({r["area"] for r in rivers if r["area"] and r["area"] not in HYDRO_MAP_REGIONS})
    area_to_name = {a: c["name"] + (" (aggregate)" if c.get("aggregate") else "") for a, c in HYDRO_MAP_REGIONS.items()}
    area_to_name.update({a: f"{a} (river stations only)" for a in river_areas})
    name_to_area = {v: k for k, v in area_to_name.items()}
    current = st.session_state.get("hy_dd_area")
    if current and current not in area_to_name:
        area_to_name[current] = current
        name_to_area[current] = current
    st.session_state["hy_dd_select"] = area_to_name.get(current, NONE_LABEL)
    d1, d2 = st.columns([1.6, 4])
    with d1:
        st.selectbox("Deep dive on", [NONE_LABEL] + list(area_to_name.values()), key="hy_dd_select",
                     on_change=_sync_select, args=(name_to_area,),
                     help="Or click a country / a river station on the map.")
    if current:
        _render_deep_dive(current, metrics, rivers, st.session_state.get("hy_dd_station"))

    # ── Every area × every layer, side by side ────────────────────────────────
    st.markdown("##### All areas · all layers")
    st.markdown(_grid_html(metrics), unsafe_allow_html=True)
    table = _overview_table(metrics)
    if not table.empty:
        st.download_button("Download overview CSV", table.to_csv(index=False).encode(),
                           f"hydro_overview_{dt.date.today():%Y%m%d}.csv", "text/csv", key="hy_ov_dl")

    if rivers:
        st.markdown("##### River stations · temperature and flow")
        st.markdown(_river_table_html(rivers), unsafe_allow_html=True)
        riv = pd.DataFrame([{k: v for k, v in r.items() if k not in ("hover", "flags", "z", "code")}
                            | {"flags": "; ".join(t for _l, t in r["flags"])} for r in rivers])
        st.download_button("Download river temperatures CSV", riv.to_csv(index=False).encode(),
                           f"river_temps_{dt.date.today():%Y%m%d}.csv", "text/csv", key="hy_ov_riv_dl")


# ══════════════════════════════════════════════════════════════════════════════
# FAMILY TABS — the report's figures per country
# ══════════════════════════════════════════════════════════════════════════════

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
    st.caption("The hydro outlook on a map of Europe — reservoirs, snow water equivalent, groundwater, hydro balance "
               "and river temperatures, criticalities flagged, a deep dive per country on click — then the Hydro "
               "Report quantify_* figures and statistics per family, computed live from the Volue hydro curves "
               "since 2013 vs normal.")
    try:
        avail = load_hydro_available()
    except Exception as e:
        st.error(f"Cannot read hydro_daily ({e}). Has power_desk_refresh.py run?")
        return
    if not avail.empty:
        last = avail["last_day"].max()
        st.caption(f"Latest hydro observation: {pd.Timestamp(last):%d %b %Y}")
    for k, v in (("hy_dd_area", None), ("hy_dd_station", None)):
        st.session_state.setdefault(k, v)
    tabs = st.tabs(["Overview"] + list(HYDRO_FAMILIES.keys()))
    with tabs[0]:
        _render_overview(avail)
    for tab, family in zip(tabs[1:], HYDRO_FAMILIES.keys()):
        with tab:
            _render_family(family, avail)
