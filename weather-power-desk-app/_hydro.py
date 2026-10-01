"""Section 3 — Hydro Monitoring.

Overview tab   What you see on entering: a Europe map of the hydro outlook per
               country. Each hydro area is painted by its reservoir level as %
               of normal (or by percentile, or by the anomaly in GWh) with its
               numbers written on the country, and the full read-out on hover —
               reservoirs, snow & groundwater, hydro balance, and the flags.
               Under the map: the criticalities (series at or below the 15th /
               30th percentile of their history, at or above the 85th, or moving
               5+ pts of normal in a week) and a grid with every area × every
               layer side by side. Layers are HYDRO_OVERVIEW_LAYERS in _config;
               SWE, river levels or temperatures slot in there once their series
               exist in the sandbox (see _overview_metrics).
Family tabs    Live version of P:/QFA/TonyWeather/Hydro_Report/quantify_*.py:
               for each of the three hydro families (reservoir levels, snow &
               groundwater, hydro balance) and each country the climatology
               chart (grey history, blue recent years, dashed norm, red current
               year) and the report's numbers — anomaly in GWh and % of normal,
               percentile of the current week, and the change since last week —
               plus the stats_*.txt text the report writes.
The maths is in _hydro_quantify.py; data comes from {SBX_SCHEMA}.hydro_daily.
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
    HYDRO_SWE_REGIONS, HYDRO_RIVER_FORECAST_TAG, RIVER_TEMP_WARM_ANOMALY_C, RIVER_TEMP_HOT_C,
    RIVER_TEMP_COLOUR_RANGE_C,
)
from _data import (
    load_hydro_available, load_hydro_series, load_hydro_component, load_swe_country_daily, load_river_latest,
)
from _charts import make_hydro_climatology_chart, make_hydro_anomaly_bars, make_hydro_europe_map
from _hydro_quantify import (
    HYDRO_FAMILIES, HYDRO_DEFAULT_COUNTRIES, build_climatology, quantify_anomaly, split_recent_hist, stats_text,
)
from _ui import kpi_card, kpi_row, status_banner, ordinal


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
    """'92%' for a level series, '-1,215' (GWh vs norm) for a deviation series."""
    if HYDRO_OVERVIEW_LAYERS[layer]["level"] == "percent" and q.get("anomaly_percent") is not None:
        return f"{q['anomaly_percent']:.0f}%"
    return f"{q['anomaly']:+,.0f}"


def _week_move(q: dict, layer: str) -> tuple[float | None, str]:
    """(move over the last week, unit): pts of normal for a level series, GWh of
    anomaly for a deviation series whose % of normal means nothing."""
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
# OVERVIEW — every area × every layer quantified once
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _overview_metrics() -> dict[str, dict[str, dict]]:
    """layer -> hydro area code -> quantify_anomaly dict, for every area with a
    series. One query per layer, then the report maths per area, exactly as the
    family tabs do it one country at a time.

    Sources (HYDRO_OVERVIEW_LAYERS[layer]['source']):
      hydro_daily  a Volue component (actual SA, normal N where the family has one)
      swe_daily    the internal Exolabs SWE model's mean mm per Alpine country,
                   uploaded by pipeline/swe_upload.py; no normal → mean of years
    To add another (river levels, …): load its (area, day, actual[, normal])
    series here, run it through build_climatology / quantify_anomaly the same
    way, and register the layer — nothing downstream changes.
    """
    def quantify(series: dict[str, tuple[pd.Series, pd.Series | None]]) -> dict[str, dict]:
        per_area: dict[str, dict] = {}
        for area, (actual, normal) in series.items():
            if actual.dropna().empty:
                continue
            try:
                q = quantify_anomaly(build_climatology(actual, normal))
            except Exception:
                continue
            if q:
                per_area[str(area)] = q
        return per_area

    out: dict[str, dict[str, dict]] = {}
    for layer, cfg in HYDRO_OVERVIEW_LAYERS.items():
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
            series = {}
        out[layer] = quantify(series)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# RIVER TEMPERATURE STATIONS (EQ, via pipeline/eq_river_temps.py)
# ══════════════════════════════════════════════════════════════════════════════

def _river_flags(value, anom, fc_max7) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if value is not None and value >= RIVER_TEMP_HOT_C:
        out.append(("critical", f"{value:.1f} °C — at or above {RIVER_TEMP_HOT_C:.0f} °C"))
    elif anom is not None and anom >= RIVER_TEMP_WARM_ANOMALY_C:
        out.append(("warning", f"{anom:+.1f} °C vs normal — warm"))
    if fc_max7 is not None and fc_max7 >= RIVER_TEMP_HOT_C and not (value is not None and value >= RIVER_TEMP_HOT_C):
        out.append(("warning", f"forecast peaks at {fc_max7:.1f} °C within 7 days"))
    return out


def _river_rows() -> list[dict]:
    """One dict per station for the map (lat, lon, z, hover) and the table."""
    df = load_river_latest(HYDRO_RIVER_FORECAST_TAG)
    if df is None or df.empty:
        return []
    rows = []
    for r in df.itertuples(index=False):
        def f(x):
            return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
        value, normal, anom = f(r.value), f(r.normal), f(r.anomaly)
        fc_max, fc_min, fc_norm = f(r.fc_max7), f(r.fc_min7), f(r.fc_normal7)
        flags = _river_flags(value, anom, fc_max)
        day = pd.Timestamp(r.day) if not pd.isna(r.day) else None
        lines = [f"<b>{r.station}</b> · river temperature (EQ)"]
        if value is not None:
            lines.append(f"Latest {value:.1f} °C" + (f" · {day:%d %b}" if day is not None else "")
                         + (f" · normal {normal:.1f} °C · <b>{anom:+.1f} °C</b>" if anom is not None else ""))
        if fc_max is not None:
            issued = pd.Timestamp(r.fc_issued) if not pd.isna(r.fc_issued) else None
            lines.append(f"Next 7 days ({r.fc_tag}{f', issued {issued:%d %b %Hz}' if issued is not None else ''}): "
                         f"{fc_min:.1f}–{fc_max:.1f} °C" + (f" · normal {fc_norm:.1f} °C" if fc_norm is not None else ""))
        for _l, t in flags:
            lines.append(f"⚑ {t}")
        rows.append({"station_key": r.station_key, "name": r.station, "river": r.river, "site": r.site, "area": r.area,
                     "lat": f(r.latitude), "lon": f(r.longitude), "day": day, "value": value, "normal": normal,
                     "anomaly": anom, "fc_tag": r.fc_tag, "fc_max7": fc_max, "fc_min7": fc_min, "fc_normal7": fc_norm,
                     "z": anom, "flags": flags, "hover": "<br>".join(lines)})
    return rows


def _river_table_html(rows: list[dict]) -> str:
    head = ("<tr><th style='text-align:left'>Station</th><th>Latest</th><th>°C</th><th>Normal</th><th>Δ</th>"
            f"<th class='hy-first'>7-day forecast ({HYDRO_RIVER_FORECAST_TAG})</th><th>vs normal</th><th style='text-align:left'>Flags</th></tr>")
    body = []
    for r in sorted(rows, key=lambda x: -(x["value"] if x["value"] is not None else -99)):
        acls = "hy-crit" if (r["value"] is not None and r["value"] >= RIVER_TEMP_HOT_C) else \
            ("hy-low" if (r["anomaly"] is not None and r["anomaly"] >= RIVER_TEMP_WARM_ANOMALY_C) else "")
        fcls = "hy-low" if (r["fc_max7"] is not None and r["fc_max7"] >= RIVER_TEMP_HOT_C) else ""
        fc = f"{r['fc_min7']:.1f}–{r['fc_max7']:.1f}" if r["fc_max7"] is not None else "—"
        fcd = f"{r['fc_max7'] - r['fc_normal7']:+.1f}" if (r["fc_max7"] is not None and r["fc_normal7"] is not None) else "—"
        flags = " · ".join(t for _l, t in r["flags"]) or "—"
        cells = [
            f"<td class='mc-region'>{r['name']}</td>",
            f"<td>{r['day']:%d %b}</td>" if r["day"] is not None else "<td class='hy-na'>—</td>",
            f"<td class='{acls}'>{r['value']:.1f}</td>" if r["value"] is not None else "<td class='hy-na'>—</td>",
            f"<td>{r['normal']:.1f}</td>" if r["normal"] is not None else "<td class='hy-na'>—</td>",
            f"<td class='{acls}'>{r['anomaly']:+.1f}</td>" if r["anomaly"] is not None else "<td class='hy-na'>—</td>",
            f"<td class='hy-first {fcls}'>{fc}</td>",
            f"<td class='{fcls}'>{fcd}</td>",
            f"<td class='hy-agg' style='text-align:left'>{flags}</td>",
        ]
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<div class='mc-block'><table class='mc-table hy-table'><thead>{head}</thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


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
    return "<br>".join(lines)


def _paint(layer: str, colour_by: str, metrics: dict[str, dict[str, dict]]) -> list[dict]:
    """One row per ISO-3 country to draw. A country's own series wins; an
    aggregate (Nordics, SEE) paints the members that have none; what is left is
    drawn without a value."""
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
            # an aggregate is labelled once, on its first painted member
            label = "" if (cfg.get("aggregate") and labelled) else _label(area, q, layer)
            labelled = True
            rows[iso] = {"iso3": iso, "area": area, "z": _colour_value(q, colour_by),
                         "label": label, "hover": _hover_html(area, cfg, layer, metrics),
                         "lat": lat, "lon": lon}
    for area, cfg in own + aggs:
        for iso in cfg["iso3"]:
            if iso in rows:
                continue
            lat, lon = HYDRO_MAP_LABEL_POS.get(iso, (None, None))
            rows[iso] = {"iso3": iso, "area": area, "z": None,
                         "label": "" if cfg.get("aggregate") else f"{area}<br>—",
                         "hover": f"<b>{cfg['name']}</b><br>no {layer.lower()} series in hydro_daily",
                         "lat": lat, "lon": lon}
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


def _render_overview(avail: pd.DataFrame):
    c1, c2, c3 = st.columns([2.2, 1.8, 1.2])
    with c1:
        layers = list(HYDRO_OVERVIEW_LAYERS)
        layer = st.radio("Map layer", layers, index=layers.index(HYDRO_OVERVIEW_DEFAULT_LAYER),
                         horizontal=True, key="hy_ov_layer")
    with c3:
        show_rivers = st.checkbox("River temperature stations", value=True, key="hy_ov_rivers",
                                  help="Latest EQ backcast vs normal per station (red = warm), 7-day forecast peak on hover. "
                                       "Needs river_temp_eq from Power_dashboard/pipeline.")
    with c2:
        # a deviation series (hydro balance) has no meaningful % of normal to colour by
        modes = [m for m in HYDRO_COLOUR_MODES
                 if not (m == "anomaly_percent" and HYDRO_OVERVIEW_LAYERS[layer]["level"] != "percent")]
        default_mode = HYDRO_OVERVIEW_LAYERS[layer]["colour_by"]
        colour_by = st.radio("Colour by", modes, index=modes.index(default_mode), horizontal=True,
                             format_func=HYDRO_COLOUR_MODES.get, key=f"hy_ov_mode_{layer}")

    with st.spinner("Quantifying every hydro series…"):
        metrics = _overview_metrics()
        rivers = _river_rows() if show_rivers else []
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
        st.caption("No river temperature stations yet — river_temp_eq is written by Power_dashboard/pipeline.")

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
              "every layer and the flags" + (" · " + "; ".join(aggs) if aggs else "")
              + (f" · ● river temperature stations: latest EQ backcast vs normal, red = warm, ±{RIVER_TEMP_COLOUR_RANGE_C:.0f} °C"
                 if rivers else "") + ".")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.plotly_chart(make_hydro_europe_map(painted, zmin, zmax, zmid, cb, stations=rivers,
                                              station_range=RIVER_TEMP_COLOUR_RANGE_C),
                        use_container_width=True, config={"scrollZoom": False})
        st.caption(legend)

    # ── Criticalities across every layer, beside the map ──────────────────────
    with right:
        st.markdown("##### Criticalities")
        order = {"critical": 0, "warning": 1, "high": 2}
        items = []
        for lyr in HYDRO_OVERVIEW_LAYERS:
            for area, q in metrics.get(lyr, {}).items():
                for level, text in _flags(q, lyr):
                    items.append((order[level], HYDRO_MAP_REGIONS.get(area, {}).get("name", area), lyr, level, text))
        for r in rivers:
            for level, text in r["flags"]:
                items.append((order[level], r["name"], "river temperature", level, text))
        items.sort(key=lambda t: (t[0], t[1], t[2]))
        if items:
            chips = "".join(f"<span class='hy-flag hy-flag-{lvl}'><b>{name}</b> {lyr.lower()}: {text}</span>"
                            for _o, name, lyr, lvl, text in items)
            st.markdown(f"<div class='hy-flags'>{chips}</div>", unsafe_allow_html=True)
        else:
            status_banner(f"None flagged — every series sits between the {ordinal(HYDRO_PCTL_LOW)} and "
                          f"{ordinal(HYDRO_PCTL_HIGH)} percentile of its history and moved less than "
                          f"{HYDRO_WEEK_MOVE_PTS} pts of normal in the week.", "good")
        st.caption(f"Flags: at or below the {ordinal(HYDRO_PCTL_CRITICAL)} percentile = critically low, the "
                   f"{ordinal(HYDRO_PCTL_LOW)} = low, at or above the {ordinal(HYDRO_PCTL_HIGH)} = very high; "
                   f"|Δ| ≥ {HYDRO_WEEK_MOVE_PTS} pts of normal in a week = fast drawdown / refill (level series "
                   "only — the hydro balance and SWE are shown as anomalies, so their week move is in GWh / mm and "
                   f"not flagged). River temperature: ≥ {RIVER_TEMP_HOT_C:.0f} °C = hot, ≥ {RIVER_TEMP_WARM_ANOMALY_C:.0f} °C "
                   "above normal = warm (indicative — discharge limits differ by plant). Thresholds in _config "
                   "(HYDRO_PCTL_*, HYDRO_WEEK_MOVE_PTS, RIVER_TEMP_*).")

    # ── Every area × every layer, side by side ────────────────────────────────
    st.markdown("##### All areas · all layers")
    st.markdown(_grid_html(metrics), unsafe_allow_html=True)
    table = _overview_table(metrics)
    if not table.empty:
        st.download_button("Download overview CSV", table.to_csv(index=False).encode(),
                           f"hydro_overview_{dt.date.today():%Y%m%d}.csv", "text/csv", key="hy_ov_dl")

    if rivers:
        st.markdown("##### River temperatures · stations")
        st.markdown(_river_table_html(rivers), unsafe_allow_html=True)
        riv = pd.DataFrame([{k: v for k, v in r.items() if k not in ("hover", "flags", "z")}
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
    st.caption("The hydro outlook on a map of Europe — reservoirs first, snow & groundwater and hydro balance "
               "alongside, criticalities flagged — then the Hydro Report quantify_* figures and statistics per "
               "family, computed live from the Volue hydro curves since 2013 vs normal.")
    try:
        avail = load_hydro_available()
    except Exception as e:
        st.error(f"Cannot read hydro_daily ({e}). Has power_desk_refresh.py run?")
        return
    if not avail.empty:
        last = avail["last_day"].max()
        st.caption(f"Latest hydro observation: {pd.Timestamp(last):%d %b %Y}")
    tabs = st.tabs(["Overview"] + list(HYDRO_FAMILIES.keys()))
    with tabs[0]:
        _render_overview(avail)
    for tab, family in zip(tabs[1:], HYDRO_FAMILIES.keys()):
        with tab:
            _render_family(family, avail)
