"""The French reservoir outlook → {schema}.hydro_reservoir_outlook (+ _runs).

Runs the model fitted by fit_reservoir_model.py (weather-power-desk-app/reservoir_model_fr.json)
on today's inputs and writes one daily path per series for the next 365 days:

  normal        the Volue normal level ('res fr hydro wtr gwh cet h n')
  baseline      S0 + the normal's change — the seasonal path from the latest observed level
  weather       the weather term alone: k(h, season) × cumulative precipitation-energy forecast
                anomaly (ec00ens members for days 1–16, the ecmonthly 'Avg' for 17–46), ensemble mean
  state         the state term alone: mean reversion of the level anomaly (+ the SWE term Dec–Jun),
                ramped in between 14 and 56 days
  model         baseline + weather + state;  model_p10 / p25 / p75 / p90 combine the member spread of
                the weather term with the state regression's residual std (independent, in quadrature)
  volue         Volue's own reservoir forecast ('res fr hydro wtr gwh cet h f'), re-anchored on S0
  analogue_<y>  the k nearest years on (level anomaly, SWE anomaly) at this week — their actual path
                from the same calendar day, re-anchored on S0 (context, not a forecast)

hydro_reservoir_outlook (MERGE key: area, issue_date, series, day): area, issue_date, series, day, value, loaded_at.
hydro_reservoir_outlook_runs (key: area, issue_date): the inputs of the run — s0, s0_day, s_a0, swe_a0,
swe_day, season, n_members, precip_anom_16d, precip_anom_46d, volue_offset, analogue_years, model_version.
Issues older than RETENTION_DAYS are pruned. Credentials as volue_morning.py.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from volue_morning import _credentials, _retry

log = logging.getLogger("pipeline.reservoir")

HERE = Path(__file__).resolve().parent
MODEL_JSON = Path(os.environ.get("RESERVOIR_MODEL_JSON", str(HERE.parent / "weather-power-desk-app" / "reservoir_model_fr.json")))
TABLE = os.environ.get("RESERVOIR_TABLE", "hydro_reservoir_outlook")
RUNS_TABLE = os.environ.get("RESERVOIR_RUNS_TABLE", "hydro_reservoir_outlook_runs")
HORIZON_DAYS = int(os.environ.get("RESERVOIR_HORIZON_DAYS", "365"))
N_ANALOGUES = int(os.environ.get("RESERVOIR_ANALOGUES", "5"))
RETENTION_DAYS = int(os.environ.get("RESERVOIR_RETENTION_DAYS", "60"))
Z = {"p10": -1.2816, "p25": -0.6745, "p75": 0.6745, "p90": 1.2816}

COLUMNS = [("area", "STRING"), ("issue_date", "DATE"), ("series", "STRING"), ("day", "DATE"), ("value", "DOUBLE"), ("loaded_at", "TIMESTAMP")]
KEYS = ["area", "issue_date", "series", "day"]
RUN_COLUMNS = [("area", "STRING"), ("issue_date", "DATE"), ("s0", "DOUBLE"), ("s0_day", "DATE"), ("s_a0", "DOUBLE"), ("swe_a0", "DOUBLE"),
               ("swe_day", "DATE"), ("season", "STRING"), ("n_members", "INT"), ("precip_anom_16d", "DOUBLE"), ("precip_anom_46d", "DOUBLE"),
               ("volue_offset", "DOUBLE"), ("analogue_years", "STRING"), ("model_version", "STRING"), ("loaded_at", "TIMESTAMP")]
RUN_KEYS = ["area", "issue_date"]


def _daily(ts, scale: float = 1.0) -> pd.Series:
    x = ts.to_pandas()
    if getattr(x.index, "tz", None) is not None:
        x = x.tz_localize(None)
    x.index = pd.DatetimeIndex(x.index).normalize()
    return x.astype(float) * scale


def _interp(grid: list[int], values: list[float], h: np.ndarray) -> np.ndarray:
    """Linear interpolation on the fitted grid, held flat beyond its ends."""
    return np.interp(h, np.asarray(grid, float), np.asarray(values, float))


# ── the model, as pure functions of the inputs (unit-testable without Volue) ──

def weather_term(model: dict, season: str, members: list[pd.Series], ext: pd.Series | None, rre_n: pd.Series,
                 days: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Per member, k(h, season) × cumulative precipitation-energy forecast anomaly over the first
    ens_days (member) + ext_days (ecmonthly mean) days, held afterwards. Returns a frame
    (days × members) and the mean cumulative anomalies at 16 / 46 days."""
    wcfg = model["weather"]
    ens_days, ext_days = int(wcfg["ens_days"]), int(wcfg["ext_days"])
    h = np.arange(1, len(days) + 1)
    k_grid = [wcfg["k"][str(x)].get(season, wcfg["k"][str(x)]["all"]) for x in wcfg["h_days"]]
    k = _interp(wcfg["h_days"], k_grid, h)
    n = rre_n.reindex(days)
    ext_a = ((ext.reindex(days) - n).fillna(0.0) if ext is not None else pd.Series(0.0, index=days)).values
    cols, a16, a46 = {}, [], []
    for i, m in enumerate(members):
        ma = (m.reindex(days) - n).fillna(0.0).values
        cum = np.zeros(len(days))
        cum[:ens_days] = np.cumsum(ma[:ens_days])
        cum[ens_days:ext_days] = cum[ens_days - 1] + np.cumsum(ext_a[ens_days:ext_days])
        cum[ext_days:] = cum[ext_days - 1]
        cols[i] = k * cum
        a16.append(cum[ens_days - 1]); a46.append(cum[ext_days - 1])
    return pd.DataFrame(cols, index=days), {"precip_anom_16d": float(np.mean(a16)), "precip_anom_46d": float(np.mean(a46))}


def state_term(model: dict, season: str, month: int, s_a0: float, swe_a0: float | None, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """The mean-reversion (+ SWE) term and its residual std per day, ramped in over ramp_days."""
    scfg = model["state"]
    hw_grid = scfg["h_weeks"]
    use_swe = swe_a0 is not None and month in scfg["swe_months"]
    mean_g, sig_g = [], []
    for hw in hw_grid:
        b = scfg["beta"][str(hw)][season]
        if not b.get("use_level", True):              # no skill vs the normal path at this horizon / season
            mean_g.append(0.0); sig_g.append(b["sigma"]); continue
        if use_swe and b.get("use_swe") and b.get("b_swe") is not None:
            mean_g.append(b["b0_swe"] + b["b_level_swe"] * s_a0 + b["b_swe"] * swe_a0)
        else:
            mean_g.append(b["b0"] + b["b_level"] * s_a0)
        sig_g.append(b["sigma"])
    h = np.arange(1, len(days) + 1)
    r0, r1 = scfg["ramp_days"]
    ramp = np.clip((h - r0) / (r1 - r0), 0.0, 1.0)
    hw = h / 7.0
    return ramp * _interp(hw_grid, mean_g, hw), ramp * _interp(hw_grid, sig_g, hw)


def analogue_years(model: dict, t0: pd.Timestamp, s_a0: float, swe_a0: float | None, k: int) -> list[int]:
    lib = model["analogue_library"]
    L = pd.DataFrame({"S_a": lib["S_a"], "swe_a": lib["swe_a"]}, index=pd.to_datetime(lib["weeks"]))
    L["woy"] = L.index.isocalendar().week.astype(int).values
    woy0 = int(t0.isocalendar()[1])
    cand = L[(abs(L["woy"] - woy0) <= 1) & (L.index.year < t0.year)]
    dist = ((cand["S_a"] - s_a0) / lib["scale"]["S_a"]) ** 2
    if swe_a0 is not None:
        sw = ((cand["swe_a"].astype(float) - swe_a0) / lib["scale"]["swe_a"]) ** 2
        dist = dist + sw.fillna(sw.mean() if sw.notna().any() else 0.0)
    best = dist.groupby(cand.index.year).min().nsmallest(k)
    return [int(y) for y in best.index]


def analogue_path(wtr: pd.Series, year: int, t0: pd.Timestamp, s0: float, days: pd.DatetimeIndex) -> pd.Series:
    """That year's actual path from the same calendar day, re-anchored on S0."""
    try:
        ty = t0.replace(year=year)
    except ValueError:                       # 29 Feb
        ty = t0.replace(year=year, day=28)
    base = wtr.asof(ty)
    shifted = pd.Series([wtr.asof(ty + (d - t0)) for d in days], index=days)
    return s0 + (shifted - base)


def run(settings, writer, backfill: bool = False, today: dt.date | None = None) -> dict:
    import wapi
    today = today or dt.date.today()
    model = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    area = model["area"]
    cid, sec = _credentials()
    s = wapi.Session(client_id=cid, client_secret=sec, timeout=60000)
    t_issue = pd.Timestamp(today)
    days = pd.date_range(t_issue, periods=HORIZON_DAYS, freq="D")

    # observed level and normal
    wtr = _daily(_retry(lambda: s.get_curve(name=f"res {area} hydro wtr gwh cet h sa").get_data(
        data_from=pd.Timestamp("2013-01-01"), data_to=t_issue + pd.Timedelta(days=1), function="AVERAGE", frequency="D"), "wtr sa")).dropna()
    wtr = wtr[wtr.index < t_issue]                                    # today is partial
    wtr_n = _daily(_retry(lambda: s.get_curve(name=f"res {area} hydro wtr gwh cet h n").get_data(
        data_from=t_issue - pd.Timedelta(days=400), data_to=days[-1] + pd.Timedelta(days=1), function="AVERAGE", frequency="D"), "wtr n"))
    rre_n = _daily(_retry(lambda: s.get_curve(name=f"rre {area} gwh cet min15 n").get_data(
        data_from=t_issue, data_to=t_issue + pd.Timedelta(days=60), function="SUM", frequency="D"), "rre n"))
    t0 = wtr.index[-1]
    s0 = float(wtr.iloc[-1]); s_a0 = s0 - float(wtr_n.asof(t0))
    season = model["seasons"][str(t_issue.month)]

    # precipitation-energy ensemble (members) and the extended mean
    c_ens = s.get_curve(name=f"rre {area} ec00ens gwh cet min15 f")
    tags = [t for t in c_ens.get_tags() if t != "Avg"]
    found = _retry(lambda: c_ens.search_instances(issue_date_from=t_issue - pd.Timedelta(days=4), issue_date_to=t_issue + pd.Timedelta(days=1),
                                                  tags=["Avg"], with_data=False), "ec00ens issues")
    ens_issue = max(pd.Timestamp(i.issue_date).tz_localize(None) if pd.Timestamp(i.issue_date).tzinfo else pd.Timestamp(i.issue_date) for i in found)
    latest = _retry(lambda: c_ens.get_instance(issue_date=ens_issue, tag=tags, function="SUM", frequency="D"), "ec00ens members")
    members = [_daily(ts) for ts in (latest if isinstance(latest, list) else [latest])]
    members = [m for m in members if len(m.dropna())]
    ext = None
    try:
        ext = _daily(s.get_curve(name=f"rre {area} ecmonthly gwh cet min15 f").get_latest(tags=["Avg"], function="SUM", frequency="D"))
    except Exception as e:
        log.warning("no ecmonthly issue: %s", str(e)[:80])

    # SWE anomaly (Exolabs CSV), vs the fitted week-of-year climatology
    swe_a0, swe_day = None, None
    try:
        swe = pd.read_csv(settings.swe_csv_dir / "m-SWE_countries_timeseries.csv", header=[0, 1], index_col=0, parse_dates=True)
        sw = swe[(model["swe"]["country"], "total")].astype(float).dropna()
        swe_day = sw.index[-1]
        clim = model["swe"]["clim_by_week"].get(str(int(swe_day.isocalendar()[1])))
        if clim is not None and (t_issue - swe_day).days <= 21:
            swe_a0 = float(sw.iloc[-1]) - float(clim)
    except Exception as e:
        log.warning("SWE unavailable: %s", str(e)[:80])

    # the terms
    baseline = s0 + (wtr_n.reindex(days) - float(wtr_n.asof(t0)))
    W, meta = weather_term(model, season, members, ext, rre_n, days)
    w_mean = W.mean(axis=1).values
    st_mean, st_sig = state_term(model, season, t_issue.month, s_a0, swe_a0, days)
    mdl = baseline.values + w_mean + st_mean
    series = {"normal": wtr_n.reindex(days).values, "baseline": baseline.values, "weather": w_mean, "state": st_mean, "model": mdl}
    for q, z in Z.items():
        dw = np.nanpercentile(W.values, float(q[1:]), axis=1) - w_mean
        series[f"model_{q}"] = mdl + np.sign(z) * np.sqrt(dw ** 2 + (z * st_sig) ** 2)
    volue_offset = None
    try:
        vf = _daily(s.get_curve(name=f"res {area} hydro wtr gwh cet h f").get_latest(function="AVERAGE", frequency="D",
                                                                                      data_to=days[-1] + pd.Timedelta(days=1)))
        volue_offset = s0 - float(vf.iloc[0])
        series["volue"] = (vf.reindex(days) + volue_offset).values
    except Exception as e:
        log.warning("no Volue reservoir forecast: %s", str(e)[:80])
    years = analogue_years(model, t0, s_a0, swe_a0, N_ANALOGUES)
    for y in years:
        series[f"analogue_{y}"] = analogue_path(wtr, y, t0, s0, days).values

    frames = [pd.DataFrame({"area": area, "issue_date": today, "series": name, "day": [d.date() for d in days], "value": np.asarray(v, float)})
              for name, v in series.items()]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    df["loaded_at"] = pd.Timestamp.utcnow().tz_localize(None)
    runs = pd.DataFrame([{
        "area": area, "issue_date": today, "s0": s0, "s0_day": t0.date(), "s_a0": s_a0, "swe_a0": swe_a0,
        "swe_day": swe_day.date() if swe_day is not None else None, "season": season, "n_members": len(members),
        **meta, "volue_offset": volue_offset, "analogue_years": ",".join(str(y) for y in years), "model_version": model["version"],
        "loaded_at": pd.Timestamp.utcnow().tz_localize(None)}])
    table, rtable = f"{settings.schema}.{TABLE}", f"{settings.schema}.{RUNS_TABLE}"
    writer.ensure_table(table, COLUMNS, "French reservoir outlook — daily paths per series (normal, baseline, weather, state, model + quantiles, "
                                        "volue, analogue_<year>), GWh. Written by Power_dashboard/pipeline (volue_reservoir_outlook.py).")
    writer.ensure_table(rtable, RUN_COLUMNS, "Inputs of each reservoir outlook run.")
    n = writer.merge(table, df, KEYS)
    writer.merge(rtable, runs, RUN_KEYS)
    writer.prune(table, f"issue_date < current_date() - INTERVAL {RETENTION_DAYS} DAYS")
    writer.prune(rtable, f"issue_date < current_date() - INTERVAL {RETENTION_DAYS} DAYS")
    summary = {"rows": n, "s0": round(s0), "s0_day": str(t0.date()), "s_a0": round(s_a0), "swe_a0": None if swe_a0 is None else round(swe_a0, 1),
               "members": len(members), "ens_issue": str(ens_issue)[:10] if ens_issue is not None else None, "analogues": years,
               "model_1m": round(float(mdl[29])), "baseline_1m": round(float(baseline.iloc[29])), "model_3m": round(float(mdl[89]))}
    log.info("reservoir outlook done: %s", summary)
    return summary
