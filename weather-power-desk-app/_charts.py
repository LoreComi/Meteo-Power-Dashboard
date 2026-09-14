"""Chart builders — Power Desk Weather Dashboard.

All charts are Plotly figures in the light desk theme. Color follows the job:
  - provider / area identity  -> fixed categorical hue per entity
  - ensemble spread           -> one hue at stepped opacity (fan)
  - anomaly polarity          -> diverging blue (below) / red (above)
  - scenario identity         -> SCENARIO_COLORS by cluster rank
  - hydro climatology         -> grey history, blue ramp recent years, red current
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from _config import AREAS, MONTH_NAMES, SPREAD_RATIO_HIGH, SPREAD_RATIO_LOW
from _style import (
    PLOTLY_LAYOUT, INK_PRIMARY, INK_SECONDARY, INK_MUTED, BASELINE, GRIDLINE,
    CATEGORICAL, PROVIDER_COLORS, SCENARIO_COLORS, ENS_FAN_ALPHA,
    DIV_NEG, DIV_POS, DIV_MID, STATUS_WARNING, STATUS_GOOD,
    HYDRO_HIST_GREY, HYDRO_CURRENT_RED, hydro_recent_colours, hex_to_rgba,
    CAT_BLUE, CAT_ORANGE,
)

AREA_COLORS: dict[str, str] = {code: CATEGORICAL[i % len(CATEGORICAL)] for i, code in enumerate(AREAS)}
RUN_OPACITY = {1: 1.0, 2: 0.55, 3: 0.4, 4: 0.3, 5: 0.22, 6: 0.16}
DIVERGING_SCALE = [[0.0, DIV_NEG], [0.5, DIV_MID], [1.0, DIV_POS]]


def _base_fig(title: str = "", height: int = 400) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=14)), height=height)
    return fig


def _area_name(code: str) -> str:
    return AREAS.get(code, code)


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST — values
# ══════════════════════════════════════════════════════════════════════════════

def make_fan_chart(df: pd.DataFrame, area: str, unit: str, color: str = CAT_BLUE,
                   prev_runs: pd.DataFrame | None = None, other_means: pd.DataFrame | None = None,
                   title: str | None = None) -> go.Figure:
    """Ensemble fan for one area: P10-P90 and P25-P75 bands, median, mean, normal,
    earlier runs' means (faded), and other providers' means (their fixed hue)."""
    fig = _base_fig(title or f"{_area_name(area)} — ensemble forecast", height=380)
    d = df.sort_values("day")
    if d.empty:
        return fig

    has_members = d["n_members"].fillna(0).max() > 0
    if has_members:
        fig.add_trace(go.Scatter(x=d["day"], y=d["p90"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d["day"], y=d["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=hex_to_rgba(color, ENS_FAN_ALPHA["outer"]), name="P10–P90",
                                 hovertemplate="P10 %{y:.1f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=d["day"], y=d["p75"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d["day"], y=d["p25"], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=hex_to_rgba(color, ENS_FAN_ALPHA["inner"]), name="P25–P75",
                                 hovertemplate="P25 %{y:.1f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=d["day"], y=d["p50"], mode="lines", name="Median",
                                 line=dict(color=color, width=1.5, dash="dot"),
                                 hovertemplate="Median %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["day"], y=d["ens_mean"], mode="lines", name="Ensemble mean (Volue)",
                             line=dict(color=color, width=2.5),
                             hovertemplate="Mean %{y:.1f}<extra></extra>"))
    if d["normal"].notna().any():
        fig.add_trace(go.Scatter(x=d["day"], y=d["normal"], mode="lines", name="Normal",
                                 line=dict(color=INK_PRIMARY, width=1.8, dash="dash"),
                                 hovertemplate="Normal %{y:.1f}<extra></extra>"))

    if prev_runs is not None and not prev_runs.empty:
        for rank, rdf in prev_runs.sort_values("run_rank").groupby("run_rank"):
            rdf = rdf.sort_values("day")
            fig.add_trace(go.Scatter(x=rdf["day"], y=rdf["ens_mean"], mode="lines",
                                     name=str(rdf["run_label"].iloc[0]),
                                     line=dict(color=INK_MUTED, width=1.2),
                                     opacity=RUN_OPACITY.get(int(rank), 0.15),
                                     hovertemplate=f"{rdf['run_label'].iloc[0]} %{{y:.1f}}<extra></extra>"))

    if other_means is not None and not other_means.empty:
        for fam, odf in other_means.groupby("model_family"):
            odf = odf.sort_values("day")
            fig.add_trace(go.Scatter(x=odf["day"], y=odf["ens_mean"], mode="lines+markers", name=f"{fam} mean",
                                     line=dict(color=PROVIDER_COLORS.get(fam, INK_MUTED), width=1.6),
                                     marker=dict(size=5),
                                     hovertemplate=f"{fam} %{{y:.1f}}<extra></extra>"))

    fig.update_yaxes(title_text=unit)
    return fig


def make_anomaly_heatmap(df: pd.DataFrame, value_col: str = "anomaly", unit: str = "°C",
                         title: str = "Daily anomaly vs normal — all selected countries",
                         symmetric: bool = True) -> go.Figure:
    """Area × day heatmap of ensemble-mean anomaly. Diverging blue/red, zero at grey."""
    if df.empty:
        return _base_fig(title, height=300)
    piv = df.pivot_table(index="area", columns="day", values=value_col, aggfunc="mean")
    piv = piv.reindex([a for a in AREAS if a in piv.index])
    z = piv.values.astype(float)
    lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
    lim = max(lim, 0.1)
    fig = go.Figure(go.Heatmap(
        z=z, x=piv.columns, y=[_area_name(a) for a in piv.index],
        colorscale=DIVERGING_SCALE, zmid=0, zmin=-lim if symmetric else None, zmax=lim if symmetric else None,
        colorbar=dict(title=unit, thickness=12, len=0.9),
        hovertemplate="%{y} · %{x|%a %d %b}<br>%{z:+.1f} " + unit + "<extra></extra>",
        xgap=2, ygap=2,
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=14)),
                      height=max(260, 40 + 28 * len(piv.index)))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    return fig


def make_multi_area_lines(df: pd.DataFrame, value_col: str, unit: str, title: str,
                          areas: list[str]) -> go.Figure:
    """One line per country (fixed hue per country) — mean anomaly by day."""
    fig = _base_fig(title, height=360)
    for a in areas:
        adf = df[df["area"] == a].sort_values("day")
        if adf.empty:
            continue
        fig.add_trace(go.Scatter(x=adf["day"], y=adf[value_col], mode="lines", name=_area_name(a),
                                 line=dict(color=AREA_COLORS.get(a, INK_MUTED), width=2),
                                 hovertemplate=f"{_area_name(a)} %{{y:+.1f}}<extra></extra>"))
    fig.add_hline(y=0, line_color=BASELINE, line_width=1.2)
    fig.update_yaxes(title_text=unit)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST — uncertainty
# ══════════════════════════════════════════════════════════════════════════════

def make_spread_vs_normal_chart(cur: pd.DataFrame, clim: pd.DataFrame, area: str, unit: str) -> go.Figure:
    """Today's ensemble spread (std across members) per lead day against the
    climatological spread for that lead day (mean and P25-P75 band)."""
    fig = _base_fig(f"{_area_name(area)} — ensemble spread vs normal spread by lead day", height=340)
    c = clim.sort_values("lead_day")
    if not c.empty:
        fig.add_trace(go.Scatter(x=c["lead_day"], y=c["spread_std_p75"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=c["lead_day"], y=c["spread_std_p25"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=hex_to_rgba(INK_MUTED, 0.18),
                                 name="Normal spread P25–P75", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=c["lead_day"], y=c["spread_std_mean"], mode="lines", name="Normal spread (mean)",
                                 line=dict(color=INK_PRIMARY, width=1.8, dash="dash"),
                                 hovertemplate="normal σ %{y:.2f}<extra></extra>"))
    d = cur.sort_values("lead_day")
    if not d.empty:
        fig.add_trace(go.Scatter(x=d["lead_day"], y=d["spread_std"], mode="lines+markers", name="This run's spread",
                                 line=dict(color=CAT_ORANGE, width=2.5), marker=dict(size=7),
                                 hovertemplate="lead %{x} · σ %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Lead day", dtick=1)
    fig.update_yaxes(title_text=f"σ across members ({unit})", rangemode="tozero")
    return fig


def make_spread_ratio_heatmap(ratio: pd.DataFrame, title: str = "Spread ratio: this run / normal spread") -> go.Figure:
    """Area × lead-day heatmap of spread ratio; 1 = normal, >1.3 unusually uncertain."""
    if ratio.empty:
        return _base_fig(title, height=300)
    piv = ratio.pivot_table(index="area", columns="lead_day", values="ratio", aggfunc="mean")
    piv = piv.reindex([a for a in AREAS if a in piv.index])
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=[_area_name(a) for a in piv.index],
        colorscale=[[0.0, DIV_NEG], [0.5, DIV_MID], [1.0, DIV_POS]], zmid=1.0, zmin=0.4, zmax=1.6,
        colorbar=dict(title="ratio", thickness=12, len=0.9),
        hovertemplate="%{y} · lead %{x}<br>%{z:.2f}× normal<extra></extra>", xgap=2, ygap=2,
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=14)),
                      height=max(260, 40 + 28 * len(piv.index)))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="Lead day", dtick=1)
    fig.update_yaxes(autorange="reversed")
    return fig


def make_member_distribution(members: pd.DataFrame, area: str, unit: str, normal: float | None,
                             day_label: str) -> go.Figure:
    """Histogram of member values for one country/day, with the normal marked."""
    fig = _base_fig(f"{_area_name(area)} — member distribution, {day_label}", height=300)
    if members.empty:
        return fig
    fig.add_trace(go.Histogram(x=members["value"], nbinsx=15, marker_color=hex_to_rgba(CAT_BLUE, 0.6),
                               marker_line=dict(color="#ffffff", width=1), name="Members",
                               hovertemplate="%{x} · %{y} members<extra></extra>"))
    mean = float(members["value"].mean())
    fig.add_vline(x=mean, line_color=CAT_BLUE, line_width=2, annotation_text=f"mean {mean:.1f}",
                  annotation_position="top right")
    if normal is not None and not np.isnan(normal):
        fig.add_vline(x=normal, line_color=INK_PRIMARY, line_dash="dash", line_width=1.6,
                      annotation_text=f"normal {normal:.1f}", annotation_position="top left")
    fig.update_layout(hovermode="closest", bargap=0.04)
    fig.update_xaxes(title_text=unit)
    fig.update_yaxes(title_text="members")
    return fig


def make_member_strips(members: pd.DataFrame, area: str, unit: str) -> go.Figure:
    """Box per day of member values for one country — the day-by-day distribution."""
    fig = _base_fig(f"{_area_name(area)} — daily member distribution", height=360)
    if members.empty:
        return fig
    d = members.sort_values("day")
    fig.add_trace(go.Box(x=d["day"], y=d["value"], name="Members", marker_color=CAT_BLUE,
                         line=dict(width=1.4), fillcolor=hex_to_rgba(CAT_BLUE, 0.25), boxpoints="outliers",
                         marker=dict(size=4, opacity=0.6)))
    if "normal" in d.columns and d["normal"].notna().any():
        nrm = d.groupby("day")["normal"].mean().reset_index()
        fig.add_trace(go.Scatter(x=nrm["day"], y=nrm["normal"], mode="lines", name="Normal",
                                 line=dict(color=INK_PRIMARY, width=1.8, dash="dash")))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(title_text=unit)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST — scenarios
# ══════════════════════════════════════════════════════════════════════════════

def scenario_color(scenario: int) -> str:
    if int(scenario) == 0:
        return INK_MUTED
    return SCENARIO_COLORS[(int(scenario) - 1) % len(SCENARIO_COLORS)]


def make_scenario_lines(summary: pd.DataFrame, area: str, unit: str, value_label: str,
                        members_tagged: pd.DataFrame | None = None) -> go.Figure:
    """Per-scenario mean path for one country (with member spaghetti faded behind),
    the full-ensemble mean in ink and the normal dashed."""
    fig = _base_fig(f"{_area_name(area)} — {value_label} by scenario", height=380)
    s = summary[summary["area"] == area].sort_values("day")
    if s.empty:
        return fig
    if members_tagged is not None and not members_tagged.empty:
        m = members_tagged[members_tagged["area"] == area]
        for (scen, mem), g in m.groupby(["scenario", "member_id"]):
            g = g.sort_values("day")
            fig.add_trace(go.Scatter(x=g["day"], y=g["value"], mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color=scenario_color(scen), width=0.7), opacity=0.25))
    for scen, g in s.groupby("scenario"):
        g = g.sort_values("day")
        name = "Other" if scen == 0 else f"Scenario {scen}"
        fig.add_trace(go.Scatter(x=g["day"], y=g["mean"], mode="lines", name=f"{name} (n={int(g['n'].iloc[0])})",
                                 line=dict(color=scenario_color(scen), width=2.6),
                                 hovertemplate=f"{name} %{{y:.1f}}<extra></extra>"))
    ens = s.drop_duplicates("day").sort_values("day")
    fig.add_trace(go.Scatter(x=ens["day"], y=ens["ens_mean"], mode="lines", name="Ensemble mean",
                             line=dict(color=INK_PRIMARY, width=1.8)))
    if "normal" in ens.columns and ens["normal"].notna().any():
        fig.add_trace(go.Scatter(x=ens["day"], y=ens["normal"], mode="lines", name="Normal",
                                 line=dict(color=INK_PRIMARY, width=1.4, dash="dash")))
    fig.update_yaxes(title_text=unit)
    return fig


def make_scenario_table_heatmap(table: pd.DataFrame, unit: str, title: str) -> go.Figure:
    """Scenario × country heatmap of horizon-mean anomaly."""
    if table.empty:
        return _base_fig(title, height=240)
    z = table.values.astype(float)
    lim = max(float(np.nanmax(np.abs(z))), 0.1)
    fig = go.Figure(go.Heatmap(
        z=z, x=[_area_name(a) for a in table.columns], y=list(table.index),
        colorscale=DIVERGING_SCALE, zmid=0, zmin=-lim, zmax=lim,
        text=[[f"{v:+.1f}" for v in row] for row in z], texttemplate="%{text}",
        textfont=dict(size=11, color=INK_PRIMARY),
        colorbar=dict(title=unit, thickness=12), xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}<br>%{z:+.2f} " + unit + "<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=14)),
                      height=max(220, 80 + 40 * len(table.index)))
    fig.update_layout(hovermode="closest")
    return fig


def make_silhouette_chart(sil_by_k: dict[int, float], chosen_k: int) -> go.Figure:
    fig = _base_fig("Cluster count selection (silhouette)", height=220)
    if not sil_by_k:
        return fig
    ks = sorted(sil_by_k)
    fig.add_trace(go.Bar(x=[str(k) for k in ks], y=[sil_by_k[k] for k in ks],
                         marker_color=[CAT_BLUE if k == chosen_k else hex_to_rgba(INK_MUTED, 0.5) for k in ks],
                         hovertemplate="k=%{x} · silhouette %{y:.3f}<extra></extra>", showlegend=False))
    fig.update_layout(hovermode="closest", bargap=0.3)
    fig.update_xaxes(title_text="k")
    fig.update_yaxes(title_text="silhouette", rangemode="tozero")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL
# ══════════════════════════════════════════════════════════════════════════════

def make_year_month_heatmap(monthly: pd.DataFrame, area: str, unit: str, value_col: str = "anomaly") -> go.Figure:
    """Year × month anomaly heatmap for one country."""
    m = monthly[monthly["area"] == area]
    title = f"{_area_name(area)} — monthly anomaly vs normal"
    if m.empty:
        return _base_fig(title, height=300)
    piv = m.pivot_table(index="year", columns="month", values=value_col, aggfunc="mean").reindex(columns=range(1, 13))
    z = piv.values.astype(float)
    lim = max(float(np.nanmax(np.abs(z[np.isfinite(z)]))) if np.isfinite(z).any() else 1.0, 0.1)
    fig = go.Figure(go.Heatmap(
        z=z, x=MONTH_NAMES, y=[str(int(y)) for y in piv.index],
        colorscale=DIVERGING_SCALE, zmid=0, zmin=-lim, zmax=lim,
        text=[[("" if not np.isfinite(v) else f"{v:+.1f}") for v in row] for row in z], texttemplate="%{text}",
        textfont=dict(size=10, color=INK_PRIMARY),
        colorbar=dict(title=unit, thickness=12), xgap=2, ygap=2,
        hovertemplate="%{y} %{x}<br>%{z:+.2f} " + unit + "<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=14)),
                      height=max(280, 60 + 26 * len(piv.index)))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    return fig


def make_period_bars(agg: pd.DataFrame, unit: str, title: str, x_col: str, value_col: str = "anomaly",
                     warm_is_positive: bool = True) -> go.Figure:
    """Grouped bars: one group per period (month or week), one bar per country; color by country."""
    fig = _base_fig(title, height=380)
    if agg.empty:
        return fig
    for a, g in agg.groupby("area"):
        g = g.sort_values(x_col)
        fig.add_trace(go.Bar(x=g[x_col].astype(str), y=g[value_col], name=_area_name(a),
                             marker_color=AREA_COLORS.get(a, INK_MUTED), marker_line_width=0,
                             hovertemplate=f"{_area_name(a)} · %{{x}}<br>%{{y:+.2f}} {unit}<extra></extra>"))
    fig.add_hline(y=0, line_color=BASELINE, line_width=1.2)
    fig.update_layout(barmode="group", bargap=0.18, bargroupgap=0.06, hovermode="closest")
    fig.update_yaxes(title_text=f"anomaly ({unit})")
    return fig


def make_period_lines(agg: pd.DataFrame, unit: str, title: str, x_col: str, value_col: str) -> go.Figure:
    """Actual and normal per period for each country (solid actual, dashed normal, same hue)."""
    fig = _base_fig(title, height=380)
    if agg.empty:
        return fig
    for a, g in agg.groupby("area"):
        g = g.sort_values(x_col)
        col = AREA_COLORS.get(a, INK_MUTED)
        fig.add_trace(go.Scatter(x=g[x_col], y=g[value_col], mode="lines+markers", name=_area_name(a),
                                 line=dict(color=col, width=2), marker=dict(size=5)))
        if "normal" in g.columns:
            fig.add_trace(go.Scatter(x=g[x_col], y=g["normal"], mode="lines", name=f"{_area_name(a)} normal",
                                     line=dict(color=col, width=1.2, dash="dash"), showlegend=False, opacity=0.7))
    fig.update_yaxes(title_text=unit)
    return fig


def make_index_chart(idx: pd.DataFrame, index_name: str, highlight_years: list[int] | None = None) -> go.Figure:
    fig = _base_fig(f"{index_name} — history", height=300)
    d = idx.sort_values("date")
    if d.empty:
        return fig
    fig.add_trace(go.Scatter(x=d["date"], y=d["value"], mode="lines", name=index_name,
                             line=dict(color=CAT_BLUE, width=1.6)))
    fig.add_hline(y=0, line_color=BASELINE, line_width=1.2)
    for y in highlight_years or []:
        fig.add_vrect(x0=f"{y}-01-01", x1=f"{y}-12-31", fillcolor=hex_to_rgba(CAT_ORANGE, 0.15), line_width=0)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HYDRO — climatology chart (Plotly port of hydro_plot_style.plot_climatology)
# ══════════════════════════════════════════════════════════════════════════════

def make_hydro_climatology_chart(clim: dict, hist_years: list[int], recent_years: list[int],
                                 title: str, unit: str, height: int = 400,
                                 show_legend: bool = True) -> go.Figure:
    """Grey historical spread, blue ramp for recent years, dashed norm, red current year
    with the latest value direct-labelled — same picture as the Hydro Report figures."""
    df_years: pd.DataFrame = clim["climatology"]
    this_year: pd.Series = clim["current_year"]
    fig = _base_fig(title, height=height)
    x = df_years.index

    for i, y in enumerate(hist_years):
        if y not in df_years.columns:
            continue
        fig.add_trace(go.Scatter(x=x, y=df_years[y], mode="lines", line=dict(color=HYDRO_HIST_GREY, width=0.8),
                                 name=f"{min(hist_years)}–{max(hist_years)}", legendgroup="hist",
                                 showlegend=(i == 0 and show_legend), hovertemplate=f"{y} %{{y:,.0f}}<extra></extra>"))
    for col, y in zip(hydro_recent_colours(len(recent_years)), recent_years):
        if y not in df_years.columns:
            continue
        fig.add_trace(go.Scatter(x=x, y=df_years[y], mode="lines", line=dict(color=col, width=1.6), name=str(y),
                                 showlegend=show_legend, hovertemplate=f"{y} %{{y:,.0f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=df_years["norm"], mode="lines", line=dict(color=INK_PRIMARY, width=2, dash="dash"),
                             name="norm", showlegend=show_legend, hovertemplate="norm %{y:,.0f}<extra></extra>"))
    ty = this_year.dropna()
    if not ty.empty:
        cur_year = pd.Timestamp.today().year
        fig.add_trace(go.Scatter(x=ty.index, y=ty.values, mode="lines", line=dict(color=HYDRO_CURRENT_RED, width=2.8),
                                 name=str(cur_year), showlegend=show_legend,
                                 hovertemplate=f"{cur_year} %{{y:,.0f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[ty.index[-1]], y=[ty.iloc[-1]], mode="markers+text",
                                 marker=dict(size=9, color=HYDRO_CURRENT_RED, line=dict(color="#ffffff", width=1.5)),
                                 text=[f"{ty.iloc[-1]:,.0f}"], textposition="middle right" if ty.index[-1].dayofyear < 300 else "middle left",
                                 textfont=dict(color=HYDRO_CURRENT_RED, size=11, family="JetBrains Mono, monospace"),
                                 showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(tickformat="%b", dtick="M1", range=[x[0], x[-1]])
    fig.update_yaxes(title_text=unit, tickformat=",.0f")
    fig.update_layout(hovermode="x")
    return fig


def make_hydro_anomaly_bars(rows: list[dict], title: str = "Anomaly vs normal today (GWh)") -> go.Figure:
    """Diverging bars of today's anomaly per country (blue above, red below normal)."""
    fig = _base_fig(title, height=300)
    if not rows:
        return fig
    names = [r["country"] for r in rows]
    vals = [r["anomaly"] for r in rows]
    fig.add_trace(go.Bar(x=names, y=vals, marker_color=[DIV_NEG if v >= 0 else DIV_POS for v in vals],
                         marker_line_width=0, text=[f"{v:+,.0f}" for v in vals], textposition="outside",
                         textfont=dict(color=INK_PRIMARY, size=11), showlegend=False,
                         hovertemplate="%{x}<br>%{y:+,.0f} GWh<extra></extra>"))
    fig.add_hline(y=0, line_color=BASELINE, line_width=1.2)
    fig.update_layout(hovermode="closest", bargap=0.3)
    fig.update_yaxes(title_text="GWh")
    return fig
