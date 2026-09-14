"""Hydro monitoring maths — port of P:/QFA/TonyWeather/Hydro_Report/quantify_*.py.

The three report scripts (quantify_reservoir_levels.py, quantify_snow_groundwater.py,
quantify_hydro_balance.py) all do the same thing on a different Volue curve:

  1. pull daily actuals since 2013 (and the Volue normal, where one exists)
  2. drop 29 Feb, pivot to one column per calendar year on a dummy-year index
  3. take the norm from Volue ('... h n' curve) — or, for hydro balance where
     Volue has no normal, the mean across completed historical years
  4. compute today's anomaly vs norm (GWh and % of normal), the same a week
     ago, and the percentile of this week's mean against the same week in
     every historical year

This module reproduces those numbers from plain pandas Series so the source
(Volue delta-share table in Databricks here, wapi in the original) is
irrelevant. Charts live in _charts.py (make_hydro_climatology_chart), which
mirrors hydro_plot_style.plot_climatology in Plotly.

Curve families (Volue naming, area code inserted):
  reservoir   'res {area} hydro wtr gwh cet h sa'   normal: '... h n'
  snow+ground 'res {area} hydro sgw gwh cet h sa'   normal: '... h n'
  balance     'res {area} hydro bal gwh cet h sa'   normal: none (mean of years)
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy import stats

PLOT_YEAR = 2013            # dummy year every climatology is mapped onto
HIST_START_YEAR = 2013      # first year of the Volue history pulled by the report
RECENT_YEARS_N = 3          # last N completed years drawn in the blue ramp

HYDRO_FAMILIES: dict[str, dict] = {
    "Reservoir levels": {
        "code": "wtr", "has_norm": True,
        "unit": "Accumulated GWh", "stats_title": "RESERVOIR LEVELS",
        "description": "Stored water in hydro reservoirs, energy-equivalent.",
    },
    "Snow & groundwater": {
        "code": "sgw", "has_norm": True,
        "unit": "Accumulated GWh", "stats_title": "SNOW AND GROUND WATERS",
        "description": "Snowpack plus groundwater, energy-equivalent — the inflow still to come.",
    },
    "Hydro balance": {
        "code": "bal", "has_norm": False,
        "unit": "GWh", "stats_title": "HYDRO BALANCE",
        "description": "Reservoir + snow + groundwater deviation from normal. No Volue normal exists, "
                       "so the norm is the mean across completed years.",
    },
}

HYDRO_AREAS: dict[str, str] = {
    "France": "fr",
    "Switzerland": "ch",
    "Austria": "at",
    "Italy": "it-nord",
    "Nordics": "np",
    "Spain": "es",
    "SEE": "see",
}

# Default country sets per family — mirrors the __main__ blocks of the three scripts.
HYDRO_DEFAULT_COUNTRIES: dict[str, list[str]] = {
    "Reservoir levels": ["France", "Switzerland", "Austria", "Italy", "Spain", "SEE"],
    "Snow & groundwater": ["France", "Switzerland", "Italy", "Austria", "Spain", "SEE"],
    "Hydro balance": ["France", "Switzerland", "Austria", "Italy", "Nordics"],
}


def volue_hydro_curve(area_code: str, family_code: str, normal: bool = False) -> str:
    """Volue curve name for a hydro family, e.g. 'res fr hydro wtr gwh cet h sa'."""
    return f"res {area_code} hydro {family_code} gwh cet h {'n' if normal else 'sa'}"


def _strip_leap_day(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s = s[~((s.index.month == 2) & (s.index.day == 29))]
    return s.sort_index()


def build_climatology(actual: pd.Series, norm: pd.Series | None = None,
                      today: dt.date | None = None) -> dict:
    """Reproduce the report's df_years pivot + this_year series.

    Returns dict with:
      climatology  DataFrame indexed on the dummy PLOT_YEAR calendar (365 rows),
                   one float column per completed historical year plus 'norm'
      current_year Series of this year's values on the same dummy calendar
      years        list of completed historical years (ints)
      norm_source  'volue' or 'mean_of_years'
    Mirrors the originals exactly: leap day removed, partial years skipped for
    hydro balance, the Volue norm read off the most recent completed year.
    """
    today = today or dt.date.today()
    actual = _strip_leap_day(actual.dropna())
    if actual.empty:
        raise ValueError("no actual data")

    dates = pd.date_range(dt.date(PLOT_YEAR, 1, 1), dt.date(PLOT_YEAR, 12, 31))
    all_years = sorted(actual.index.year.unique())
    hist_years = [y for y in all_years if y < today.year]

    df_years = pd.DataFrame(index=dates, columns=hist_years, dtype=float)
    for y in hist_years:
        vals = actual[actual.index.year == y].values
        if len(vals) == 365:
            df_years[y] = vals
        elif len(vals) > 0:
            # A year that started tracking late: align by day-of-year instead of dropping
            yd = actual[actual.index.year == y]
            doy = yd.index.dayofyear - (yd.index.is_leap_year & (yd.index.month > 2)).astype(int)
            col = pd.Series(np.nan, index=dates)
            col.iloc[doy.values - 1] = yd.values
            df_years[y] = col.values
    # hydro balance drops incomplete years entirely; keep the same behaviour
    df_years = df_years.dropna(axis=1, how="all")
    hist_years = [int(c) for c in df_years.columns]

    if norm is not None and not norm.dropna().empty:
        norm = _strip_leap_day(norm.dropna())
        ny = [y for y in sorted(norm.index.year.unique()) if (norm.index.year == y).sum() == 365]
        ref_year = ny[-1] if ny else None
        if ref_year is not None:
            df_years["norm"] = norm[norm.index.year == ref_year].values
            norm_source = "volue"
        else:
            df_years["norm"] = df_years[hist_years].mean(axis=1)
            norm_source = "mean_of_years"
    else:
        df_years["norm"] = df_years[hist_years].mean(axis=1)
        norm_source = "mean_of_years"

    df_years["Month"] = df_years.index.month
    df_years["Day"] = df_years.index.day

    this_year = actual[actual.index.year == today.year]
    this_year = pd.Series(this_year.values, index=dates[: len(this_year)], name="current")

    return {"climatology": df_years, "current_year": this_year,
            "years": hist_years, "norm_source": norm_source}


def quantify_anomaly(clim: dict, today: dt.date | None = None) -> dict:
    """Today / last-week anomaly and the percentile of this week's mean.

    Exactly the arithmetic in the report scripts:
      anomaly           = last value - norm on that dummy-calendar day   (GWh)
      anomaly_percent   = last value / norm * 100
      anomaly_w-1       = same, 7 rows earlier
      anomaly_quantile  = percentileofscore(mean of the same 7 days across
                          historical years, mean of the last 8 days this year)
    """
    today = today or dt.date.today()
    df_years: pd.DataFrame = clim["climatology"]
    this_year: pd.Series = clim["current_year"].dropna()
    years = clim["years"]
    if this_year.empty:
        return {}

    last_idx = this_year.index[-1]
    norm_today = float(df_years.loc[last_idx, "norm"])
    anom = float(this_year.iloc[-1] - norm_today)
    anom_pct = float(this_year.iloc[-1] / norm_today * 100) if norm_today else np.nan

    if len(this_year) > 7:
        w_idx = this_year.index[-7]
        norm_w = float(df_years.loc[w_idx, "norm"])
        anom_w = float(this_year.iloc[-7] - norm_w)
        anom_w_pct = float(this_year.iloc[-7] / norm_w * 100) if norm_w else np.nan
    else:
        anom_w, anom_w_pct = np.nan, np.nan

    # percentile of this week's mean vs the same calendar week across history
    pos = df_years.index.get_loc(last_idx)
    lo = max(0, pos - 7)
    this_week_hist = df_years.iloc[lo:pos][years].astype(float).mean(axis=0).dropna()
    this_week = float(this_year.iloc[-8:].mean())
    quant = float(stats.percentileofscore(this_week_hist.values, this_week)) if len(this_week_hist) else np.nan

    return {
        "as_of": last_idx.replace(year=today.year) if last_idx.month <= today.month else last_idx,
        "latest_value": float(this_year.iloc[-1]),
        "norm_today": norm_today,
        "anomaly": round(anom),
        "anomaly_percent": round(anom_pct) if not np.isnan(anom_pct) else None,
        "anomaly_quantile": round(quant, 1) if not np.isnan(quant) else None,
        "anomaly_w-1": round(anom_w) if not np.isnan(anom_w) else None,
        "anomaly_percent_w-1": round(anom_w_pct) if not np.isnan(anom_w_pct) else None,
        "week_change_pct_points": (round(anom_pct - anom_w_pct)
                                   if not (np.isnan(anom_pct) or np.isnan(anom_w_pct)) else None),
        "n_hist_years": int(len(this_week_hist)),
    }


def split_recent_hist(years: list[int], n_recent: int = RECENT_YEARS_N) -> tuple[list[int], list[int]]:
    """Grey historical spread vs the last n completed years in the blue ramp."""
    years = sorted(int(y) for y in years)
    recent = years[-n_recent:] if n_recent else []
    hist = [y for y in years if y not in recent]
    return hist, recent


def stats_text(family_title: str, per_country: dict[str, dict]) -> str:
    """The stats_*.txt block the report writes, reproduced verbatim."""
    lines = [f"{family_title} ", ""]
    for country, q in per_country.items():
        lines.append(f"{country} ")
        if not q:
            lines += ["no data", "*** "]
            continue
        lines.append(f"current anomaly = {q['anomaly']}GWh, at {q['anomaly_percent']}% of seasonal normal")
        lines.append(f"standing at {q['anomaly_quantile']}th percentile of the distribution")
        lines.append(f"last week anomaly = {q['anomaly_w-1']}GWh, at {q['anomaly_percent_w-1']}% of seasonal normal")
        lines.append(f"variation of {q['week_change_pct_points']}% compared to previous week")
        lines.append("*** ")
    lines.append("")
    return "\n".join(lines)
