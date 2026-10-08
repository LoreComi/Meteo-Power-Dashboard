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
    "Groundwater": {
        "code": "sgw", "has_norm": True,
        "unit": "Accumulated GWh", "stats_title": "SNOW AND GROUND WATERS",
        "description": "Volue's snow + groundwater stock (`sgw`), energy-equivalent — the inflow still to come. "
                       "Shown as Groundwater here; the snowpack itself is the Snow water equivalent layer.",
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
    "Groundwater": ["France", "Switzerland", "Italy", "Austria", "Spain", "SEE"],
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
      norm_source  'volue' (a complete normal year), 'mean_of_years' (no provider
                   normal) or 'volue_partial' (a series younger than a year: the
                   provider normal by day of year, no percentile possible)
    Mirrors the originals exactly where the originals apply: leap day removed,
    partial years skipped for hydro balance, the Volue norm read off the most
    recent completed year. This year's values are placed by day of year, which
    is the report's positional mapping whenever the year is complete from 1 Jan.
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

    def _doy(idx: pd.DatetimeIndex) -> np.ndarray:
        """Day of year on the 365-day dummy calendar (29 Feb already removed)."""
        return (idx.dayofyear - (idx.is_leap_year & (idx.month > 2)).astype(int)).values

    if norm is not None and not norm.dropna().empty:
        norm = _strip_leap_day(norm.dropna())
        ny = [y for y in sorted(norm.index.year.unique()) if (norm.index.year == y).sum() == 365]
        ref_year = ny[-1] if ny else None
        if ref_year is not None:
            df_years["norm"] = norm[norm.index.year == ref_year].values
            norm_source = "volue"
        elif hist_years:
            df_years["norm"] = df_years[hist_years].mean(axis=1)
            norm_source = "mean_of_years"
        else:
            # A series that started this year (the Volue share's Eastern European
            # areas, from May 2026): no complete normal year yet, no history — take
            # the provider normal by day of year from what exists. Anomaly and the
            # week move work; the percentile needs completed years and stays empty.
            col = pd.Series(np.nan, index=dates)
            by_doy = norm.groupby(_doy(norm.index)).mean()
            col.iloc[by_doy.index.values - 1] = by_doy.values
            df_years["norm"] = col.values
            norm_source = "volue_partial"
    else:
        df_years["norm"] = df_years[hist_years].mean(axis=1) if hist_years else np.nan
        norm_source = "mean_of_years"

    df_years["Month"] = df_years.index.month
    df_years["Day"] = df_years.index.day

    # This year's values on the dummy calendar, placed by day of year — identical
    # to the report's positional mapping when the year is complete from 1 Jan,
    # and right for a series that starts (or has a gap) later in the year.
    ty = actual[actual.index.year == today.year]
    this_year = pd.Series(np.nan, index=dates, name="current")
    if not ty.empty:
        doy = _doy(ty.index)
        this_year.iloc[doy - 1] = ty.values
        this_year = this_year.iloc[: int(doy[-1])]
    else:
        this_year = this_year.iloc[:0]

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
    norm_today = float(df_years.loc[last_idx, "norm"])          # NaN when the normal does not cover the day
    anom = float(this_year.iloc[-1] - norm_today)
    anom_pct = float(this_year.iloc[-1] / norm_today * 100) if norm_today and not np.isnan(norm_today) else np.nan

    if len(this_year) > 7:
        w_idx = this_year.index[-7]
        norm_w = float(df_years.loc[w_idx, "norm"])
        anom_w = float(this_year.iloc[-7] - norm_w)
        anom_w_pct = float(this_year.iloc[-7] / norm_w * 100) if norm_w and not np.isnan(norm_w) else np.nan
    else:
        anom_w, anom_w_pct = np.nan, np.nan

    # percentile of this week's mean vs the same calendar week across history
    pos = df_years.index.get_loc(last_idx)
    lo = max(0, pos - 7)
    this_week_hist = df_years.iloc[lo:pos][years].astype(float).mean(axis=0).dropna() if years else pd.Series(dtype=float)
    this_week = float(this_year.iloc[-8:].mean())
    quant = float(stats.percentileofscore(this_week_hist.values, this_week)) if len(this_week_hist) else np.nan

    def _r(x, nd=None):
        """round(), or None when the value is undefined (a normal that does not cover the day)."""
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return round(x, nd) if nd is not None else round(x)

    return {
        "as_of": last_idx.replace(year=today.year) if last_idx.month <= today.month else last_idx,
        "latest_value": float(this_year.iloc[-1]),
        "norm_today": None if np.isnan(norm_today) else norm_today,
        "anomaly": _r(anom),
        "anomaly_percent": _r(anom_pct),
        "anomaly_quantile": _r(quant, 1),
        "anomaly_w-1": _r(anom_w),
        "anomaly_percent_w-1": _r(anom_w_pct),
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
        na = lambda v: "n/a" if v is None else v   # noqa: E731 — a series without a normal for the day
        lines.append(f"current anomaly = {na(q['anomaly'])}GWh, at {na(q['anomaly_percent'])}% of seasonal normal")
        lines.append(f"standing at {na(q['anomaly_quantile'])}th percentile of the distribution")
        lines.append(f"last week anomaly = {na(q['anomaly_w-1'])}GWh, at {na(q['anomaly_percent_w-1'])}% of seasonal normal")
        lines.append(f"variation of {na(q['week_change_pct_points'])}% compared to previous week")
        lines.append("*** ")
    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCTION — hydro_prod_daily (pipeline/volue_hydro_prod.py)
# ══════════════════════════════════════════════════════════════════════════════
# Per area and day, GWh/day: run-of-river ('ror') and total ('tot') production,
# actual / normal / Volue's forecast, and precipitation energy ('rre') actual /
# normal / ensemble means. Reservoir production = tot − ror, the split
# Hydro_Report/analysis_ror_reservoir_italy.py makes (it includes pumped-storage
# turbining; Volue's `pump` curve is the pumping load, not subtracted here).

def _latest_issue(g: pd.DataFrame) -> tuple[pd.Series, pd.Timestamp | None]:
    """The most recent issue of a forecast frame as a day-indexed series, with its init time."""
    if g.empty:
        return pd.Series(dtype=float), None
    ref = g["reference_date"].max()
    g = g[g["reference_date"] == ref]
    issued = g["init_time"].iloc[0] if "init_time" in g.columns and pd.notna(g["init_time"].iloc[0]) else ref
    return g.set_index("day")["value"].astype(float).sort_index(), issued


def production_series(df: pd.DataFrame, precip_patterns: list[str]) -> dict:
    """The daily GWh series of ONE area from the long hydro_prod_daily frame:
    ror / res / tot observed, `_n` their normals, `_f` the latest Volue production
    issue (fc_issued = its init time), rre / rre_n precipitation energy observed and
    normal, rre_f = {pattern: (series, issued)} for the latest issue of each
    ensemble, last_obs = the last observed production day."""
    def ts(variable: str, data_type: str) -> pd.Series:
        g = df[(df["variable"] == variable) & (df["data_type"] == data_type) & (df["pattern"] == "")]
        return g.set_index("day")["value"].astype(float).sort_index()

    out: dict = {}
    for suffix, data_type in (("", "sa"), ("_n", "n")):
        ror, tot = ts("ror", data_type), ts("tot", data_type)
        out["ror" + suffix], out["tot" + suffix], out["res" + suffix] = ror, tot, (tot - ror).dropna()
    fc = df[(df["data_type"] == "f") & (df["pattern"] == "volue")]
    ror_f, issued = _latest_issue(fc[fc["variable"] == "ror"])
    tot_f, _ = _latest_issue(fc[fc["variable"] == "tot"])
    out["ror_f"], out["tot_f"], out["res_f"], out["fc_issued"] = ror_f, tot_f, (tot_f - ror_f).dropna(), issued
    out["rre"], out["rre_n"] = ts("rre", "sa"), ts("rre", "n")
    out["rre_f"] = {}
    for p in precip_patterns:
        s, iss = _latest_issue(df[(df["variable"] == "rre") & (df["data_type"] == "f") & (df["pattern"] == p)])
        if len(s):
            out["rre_f"][p] = (s, iss)
    out["last_obs"] = out["tot"].index.max() if len(out["tot"]) else None
    return out


def _window_sum(series: pd.Series | None, w: tuple[pd.Timestamp, pd.Timestamp]) -> tuple[float | None, int]:
    if series is None or len(series) == 0:
        return None, 0
    x = series.loc[w[0]:w[1]].dropna()
    return (float(x.sum()) if len(x) else None), int(len(x))


def _pct(a: float | None, b: float | None) -> float | None:
    return (a / b * 100.0) if (a is not None and b not in (None, 0)) else None


def production_summary(s: dict, window: int, precip_patterns: list[str]) -> dict:
    """GWh sums over the observed window (the `window` days to the last observed
    production day) and the forward window (the `window` days after it): obs_* /
    fwd_* for ror, res, tot with their normals (`_n`) and % of normal, the
    precipitation energy observed / expected (per pattern, `fwd_rre_ref` = the
    first pattern), and the precipitation − production balance of each window.
    n_* are the days found in each window, so a short series is visible."""
    last = s.get("last_obs")
    if last is None:
        return {}
    obs = (last - pd.Timedelta(days=window - 1), last)
    fwd = (last + pd.Timedelta(days=1), last + pd.Timedelta(days=window))
    r: dict = {"last_obs": last, "obs_from": obs[0], "fwd_to": fwd[1], "fc_issued": s.get("fc_issued")}
    for k in ("ror", "res", "tot"):
        r[f"obs_{k}"], _ = _window_sum(s[k], obs)
        r[f"obs_{k}_n"], _ = _window_sum(s[k + "_n"], obs)
        r[f"fwd_{k}"], _ = _window_sum(s[k + "_f"], fwd)
        r[f"fwd_{k}_n"], _ = _window_sum(s[k + "_n"], fwd)
    _, r["n_obs"] = _window_sum(s["tot"], obs)
    _, r["n_fwd"] = _window_sum(s["tot_f"], fwd)
    r["obs_tot_pct"], r["fwd_tot_pct"] = _pct(r["obs_tot"], r["obs_tot_n"]), _pct(r["fwd_tot"], r["fwd_tot_n"])
    r["obs_rre"], r["n_obs_rre"] = _window_sum(s["rre"], obs)
    r["obs_rre_n"], _ = _window_sum(s["rre_n"], obs)
    r["obs_rre_pct"] = _pct(r["obs_rre"], r["obs_rre_n"])
    r["fwd_rre_n"], _ = _window_sum(s["rre_n"], fwd)
    r["fwd_rre"], r["n_fwd_rre"], r["rre_issued"] = {}, {}, {}
    for p, (ser, iss) in s["rre_f"].items():
        r["fwd_rre"][p], r["n_fwd_rre"][p] = _window_sum(ser, fwd)
        r["rre_issued"][p] = iss
    ref = precip_patterns[0] if precip_patterns else None
    r["fwd_rre_ref"] = r["fwd_rre"].get(ref)
    r["fwd_rre_ref_pct"] = _pct(r["fwd_rre_ref"], r["fwd_rre_n"])
    r["obs_balance"] = (r["obs_rre"] - r["obs_tot"]) if (r["obs_rre"] is not None and r["obs_tot"] is not None) else None
    r["fwd_balance"] = (r["fwd_rre_ref"] - r["fwd_tot"]) if (r["fwd_rre_ref"] is not None and r["fwd_tot"] is not None) else None
    return r
