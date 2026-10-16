"""Fit the French reservoir outlook model from Volue history and write reservoir_model_fr.json.

Run once a quarter (or when the hydro season turns), from the snow_obs env:

    python fit_reservoir_model.py                 # fetch from Volue (≈ 6 min: ~200 weekly issues), write the JSON
    python fit_reservoir_model.py --cache DIR     # reuse / store the fetched history and issues as pickles

The model, in three terms added to the Volue normal path from the latest observed level
(S0), all in GWh (the study behind it: 2013-2026 history, 197 weekly issues 2023-2026,
leave-one-year-out — see Power_dashboard/README.md, Method notes → Reservoir outlook):

  weather   k(h, season) × Σ_{d≤min(h,46)} (precipitation-energy forecast − normal)
            the reduced form of rain → reservoir: a direct regression of the h-day
            change of the level anomaly on the cumulative precipitation-energy forecast
            anomaly of the issue (ec00ens 'Avg' for days 1–16, ecmonthly 'Avg' for 17–46),
            per start season. It beats the normal path by ~10 % to 4 weeks and absorbs the
            production response that makes an inflow − production balance unusable on
            forecasts (inflow and turbining co-move; Volue's own reservoir forecast is
            worse than the normal path at every horizon).
  state     β0 + β1 · S_a0 (+ β2 · SWE_a0 for Dec–Jun starts), per start season and
            horizon in weeks — the mean reversion of the level anomaly (−20…−35 % RMSE at
            13–26 weeks for winter/spring starts) and the snowpack's refill signal.
            Ramped in between 2 and 8 weeks so the short term is the weather term's, and
            entered at half weight (STATE_SHRINK): head-to-head on the 45 monthly issues the
            full term over-predicted the 2–3-month drawdown, the half term beats the normal
            path and Volue's forecast at every horizon.
  analogues the k nearest years on (S_a0, SWE_a0) at the same week — context only: as a
            point forecast they were worse than the normal path.

Inputs: Volue 'res fr hydro wtr gwh cet h sa|n', 'rre fr gwh cet min15 sa|n', the issues
of 'rre fr ec00ens gwh cet min15 f' and 'rre fr ecmonthly gwh cet min15 f' ('Avg'), and
the Exolabs SWE country CSV (France, total). Credentials as volue_morning.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline_config import load_settings, setup_logging   # noqa: E402
from volue_morning import _credentials                       # noqa: E402

log = logging.getLogger("pipeline.reservoir_fit")

AREA = "fr"
SWE_COUNTRY = "France"
HISTORY_START = pd.Timestamp("2013-01-01")
ISSUES_START = pd.Timestamp("2023-01-02")         # weekly Monday issues of the precipitation ensembles
WEATHER_H = [7, 14, 21, 28, 35, 42, 49, 56, 70]                # days — grid of the weather coefficient (no skill left at 91 d)
STATE_HW = [4, 6, 8, 10, 13, 17, 22, 26, 30, 35, 39, 44, 48, 52]  # weeks — grid of the state regression
ENS_DAYS, EXT_DAYS = 16, 46
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SWE_MONTHS = [12, 1, 2, 3, 4, 5, 6]               # starts for which the snowpack term applies
STATE_RAMP_DAYS = (14, 56)                         # the state term weighs 0 at 14 d, 1 from 56 d
STATE_SHRINK = 0.5                                 # the state term enters at half weight: on the 45 monthly issues 2023-26 the full
                                                   # regression over-predicted the drawdown at 2-3 months (RMSE 792/923 vs normal
                                                   # 707/877 GWh); at 0.5 it beats the normal path at every horizon (699/853/912)
MODEL_VERSION = "fr-2026.10"
OUT_JSON = HERE.parent / "weather-power-desk-app" / "reservoir_model_fr.json"


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def rmse(e) -> float:
    e = np.asarray(e, float)
    return float(np.sqrt(np.nanmean(e ** 2)))


def _daily(ts, scale: float = 1.0) -> pd.Series:
    x = ts.to_pandas()
    if getattr(x.index, "tz", None) is not None:
        x = x.tz_localize(None)
    x.index = pd.DatetimeIndex(x.index).normalize()
    return x.astype(float) * scale


# ── data ───────────────────────────────────────────────────────────────────────

def fetch_history(session, cache: Path | None) -> pd.DataFrame:
    f = cache / "history.pkl" if cache else None
    if f and f.exists():
        return pickle.load(open(f, "rb"))
    last = pd.Timestamp(dt.date.today() + dt.timedelta(days=60))
    cols = {}
    for col, name, fn in (("wtr", f"res {AREA} hydro wtr gwh cet h sa", "AVERAGE"), ("wtr_n", f"res {AREA} hydro wtr gwh cet h n", "AVERAGE"),
                          ("rre", f"rre {AREA} gwh cet min15 sa", "SUM"), ("rre_n", f"rre {AREA} gwh cet min15 n", "SUM")):
        c = session.get_curve(name=name)
        cols[col] = _daily(c.get_data(data_from=HISTORY_START, data_to=last, function=fn, frequency="D"))
        log.info("  %s: %d days", name, cols[col].notna().sum())
    d = pd.DataFrame(cols).sort_index()
    if f:
        pickle.dump(d, open(f, "wb"))
    return d


def fetch_issues(session, issues: pd.DatetimeIndex, cache: Path | None) -> dict:
    f = cache / "issues.pkl" if cache else None
    have = pickle.load(open(f, "rb")) if (f and f.exists()) else {}
    c_ens = session.get_curve(name=f"rre {AREA} ec00ens gwh cet min15 f")
    c_ext = session.get_curve(name=f"rre {AREA} ecmonthly gwh cet min15 f")
    n_new = 0
    for day in issues:
        if day in have:
            continue
        rec = {"ens": None, "ext": None}
        try:
            rec["ens"] = _daily(c_ens.get_instance(issue_date=day, tag="Avg", function="SUM", frequency="D"))
        except Exception as e:
            log.info("  no ec00ens issue %s: %s", day.date(), str(e)[:60])
        try:  # ecmonthly is issued Mondays and Thursdays — the latest at or before the Monday
            rec["ext"] = _daily(c_ext.get_latest(tags=["Avg"], issue_date_from=day - pd.Timedelta(days=4), issue_date_to=day + pd.Timedelta(hours=1),
                                                 function="SUM", frequency="D", data_to=day + pd.Timedelta(days=EXT_DAYS + 5)))
        except Exception as e:
            log.info("  no ecmonthly issue %s: %s", day.date(), str(e)[:60])
        have[day] = rec
        n_new += 1
        if n_new % 25 == 0:
            log.info("  %d issues fetched", n_new)
    if f:
        pickle.dump(have, open(f, "wb"))
    return have


def load_swe(csv_path: Path) -> pd.Series:
    """Exolabs France total SWE (mean mm), daily."""
    swe = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
    return swe[(SWE_COUNTRY, "total")].astype(float).sort_index()


def swe_anomaly(swe_daily: pd.Series, index: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    """Weekly SWE anomaly vs the week-of-year mean over the whole record, on `index` (weekly)."""
    wk = swe_daily.resample("W-SUN").last()
    woy = pd.Series(wk.index.isocalendar().week.astype(int).values, index=wk.index)
    clim = wk.groupby(woy.values).mean()
    anom = wk - woy.map(clim)
    return anom.reindex(index), {int(k): round(float(v), 2) for k, v in clim.items()}


# ── fit ────────────────────────────────────────────────────────────────────────

def fit_weather(d: pd.DataFrame, issues: dict) -> tuple[dict, pd.DataFrame]:
    """k(h, season): regression of the h-day change of the level anomaly on the cumulative
    precipitation-energy forecast anomaly (ens 1–16 d, ext 17–46 d). Also the LOYO scores."""
    rows = []
    for day, rec in sorted(issues.items()):
        if rec["ens"] is None:
            continue
        t0 = day - pd.Timedelta(days=1)
        if t0 not in d.index or pd.isna(d["wtr"].get(t0)):
            continue
        idx = pd.date_range(day, day + pd.Timedelta(days=max(WEATHER_H) - 1))
        y = (d["wtr"].reindex(idx) - d["wtr"].loc[t0]) - (d["wtr_n"].reindex(idx) - d["wtr_n"].loc[t0])
        ens_a = (rec["ens"].reindex(idx) - d["rre_n"].reindex(idx)).fillna(0.0)
        ext_a = (rec["ext"].reindex(idx) - d["rre_n"].reindex(idx)).fillna(0.0) if rec["ext"] is not None else pd.Series(0.0, index=idx)
        cum = pd.Series(0.0, index=idx)
        cum.iloc[:ENS_DAYS] = ens_a.iloc[:ENS_DAYS].cumsum()
        cum.iloc[ENS_DAYS:EXT_DAYS] = cum.iloc[ENS_DAYS - 1] + ext_a.iloc[ENS_DAYS:EXT_DAYS].cumsum()
        cum.iloc[EXT_DAYS:] = cum.iloc[EXT_DAYS - 1]
        r = {"issue": day, "year": day.year, "season": SEASONS[day.month]}
        for h in WEATHER_H:
            r[f"y_{h}"], r[f"x_{h}"] = y.iloc[h - 1], cum.iloc[h - 1]
        rows.append(r)
    B = pd.DataFrame(rows)
    k, scores = {}, {}
    for h in WEATHER_H:
        T = B.dropna(subset=[f"y_{h}"])
        k[h] = {"all": float(ols(T[[f"x_{h}"]].values, T[f"y_{h}"].values)[1])}
        for sea in ("DJF", "MAM", "JJA", "SON"):
            g = T[T["season"] == sea]
            k[h][sea] = float(ols(g[[f"x_{h}"]].values, g[f"y_{h}"].values)[1]) if len(g) > 12 else k[h]["all"]
        # leave-one-year-out score of the seasonal k, against the normal path
        errs = []
        for yr in sorted(T["year"].unique()):
            tr, te = T[T["year"] != yr], T[T["year"] == yr]
            if te.empty or len(tr) < 20:
                continue
            for sea in te["season"].unique():
                g = tr[tr["season"] == sea]
                kk = ols(g[[f"x_{h}"]].values, g[f"y_{h}"].values)[1] if len(g) > 12 else ols(tr[[f"x_{h}"]].values, tr[f"y_{h}"].values)[1]
                tt = te[te["season"] == sea]
                errs += list(tt[f"y_{h}"].values - kk * tt[f"x_{h}"].values)
        scores[h] = {"rmse_model": rmse(errs), "rmse_normal": rmse(T[f"y_{h}"]), "n": int(len(T))}
    log.info("weather term: %s", {h: round(v["all"], 3) for h, v in k.items()})
    return {"k": {str(h): v for h, v in k.items()}, "scores": {str(h): v for h, v in scores.items()}}, B


def fit_state(w: pd.DataFrame) -> dict:
    """β(season, hw): cumulative hw-week change of the level anomaly on S_a0 (all years) and on
    S_a0 + SWE_a0 (years with SWE), LOYO scores and residual std for the uncertainty band."""
    out = {}
    for hw in STATE_HW:
        fut = w["dS_a"].rolling(hw).sum().shift(-hw)
        T = pd.DataFrame({"y": fut, "S_a": w["S_a"], "swe_a": w["swe_a"], "season": w["season"], "yr": w.index.year}).dropna(subset=["y", "S_a"])
        T = T[T["yr"] >= 2014]
        out[str(hw)] = {}
        for sea in ("DJF", "MAM", "JJA", "SON"):
            g = T[T["season"] == sea]
            gs = g.dropna(subset=["swe_a"])
            b1 = ols(g[["S_a"]].values, g["y"].values)
            b2 = ols(gs[["S_a", "swe_a"]].values, gs["y"].values) if len(gs) > 40 else None

            def loyo(cols, gg):
                errs = []
                for yr in sorted(gg["yr"].unique()):
                    tr, te = gg[gg["yr"] != yr], gg[gg["yr"] == yr]
                    if len(tr) < 20:
                        continue
                    b = ols(tr[cols].values, tr["y"].values)
                    errs += list(te["y"].values - (b[0] + te[cols].values @ b[1:]))
                return errs
            e1 = loyo(["S_a"], g)
            e2 = loyo(["S_a", "swe_a"], gs) if b2 is not None else None
            resid = (g["y"].values - (b1[0] + g[["S_a"]].values @ b1[1:]))
            out[str(hw)][sea] = {
                "b0": float(b1[0]), "b_level": float(b1[1]),
                "b0_swe": float(b2[0]) if b2 is not None else None, "b_level_swe": float(b2[1]) if b2 is not None else None,
                "b_swe": float(b2[2]) if b2 is not None else None,
                "sigma": float(np.std(resid)), "rmse_model": rmse(e1), "rmse_model_swe": rmse(e2) if e2 is not None else None,
                "rmse_normal": rmse(g["y"]), "n": int(len(g)), "n_swe": int(len(gs)),
                "rmse_level_on_swe_sample": rmse(loyo(["S_a"], gs)) if b2 is not None else None,
            }
    # one decision per start season, so a path has no kink between horizons: the level term is
    # used when it beats the normal path on average over the grid, the SWE term when it beats
    # the level-only fit on the SWE years on average
    for sea in ("DJF", "MAM", "JJA", "SON"):
        rows = [out[str(hw)][sea] for hw in STATE_HW]
        lvl = np.mean([r["rmse_model"] / r["rmse_normal"] for r in rows]) < 1.0
        # the snowpack is a seasonal (2–6 month) signal: it is used over the longest contiguous run of
        # horizons where it beats the level-only fit on the SWE years; outside the run the mean term is
        # the level-only fit, and the interpolation between grid points joins the two smoothly
        helps = [bool(r["rmse_model_swe"] is not None and r["rmse_level_on_swe_sample"] and r["rmse_model_swe"] < r["rmse_level_on_swe_sample"]) for r in rows]
        best, cur = [], []
        for i, h in enumerate(helps + [False]):
            cur = cur + [i] if h else []
            if len(cur) > len(best):
                best = list(cur)
        for i, r in enumerate(rows):
            r["use_level"], r["use_swe"] = bool(lvl), bool(lvl and i in best)
            if not lvl:
                r["b0"], r["b_level"] = 0.0, 0.0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", help="directory for the fetched history / issues pickles (reused when present)")
    ap.add_argument("--out", default=str(OUT_JSON))
    a = ap.parse_args(argv)
    settings = load_settings()
    setup_logging(settings.log_dir, "reservoir_fit")
    cache = Path(a.cache) if a.cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    import wapi
    cid, sec = _credentials()
    session = wapi.Session(client_id=cid, client_secret=sec, timeout=60000)

    log.info("history …")
    d = fetch_history(session, cache)
    today = pd.Timestamp(dt.date.today())
    issues = pd.date_range(ISSUES_START, today - pd.Timedelta(days=8), freq="W-MON")
    log.info("issues %s -> %s (%d) …", issues[0].date(), issues[-1].date(), len(issues))
    inst = fetch_issues(session, issues, cache)

    swe_csv = settings.swe_csv_dir / "m-SWE_countries_timeseries.csv"
    swe_d = load_swe(swe_csv)
    w = pd.DataFrame({"S": d["wtr"].resample("W-SUN").last(), "S_n": d["wtr_n"].resample("W-SUN").last()})
    w["S_a"] = w["S"] - w["S_n"]
    w["dS_a"] = w["S_a"].diff()
    w["season"] = w.index.month.map(SEASONS)
    w["swe_a"], swe_clim = swe_anomaly(swe_d, w.index)

    weather, B = fit_weather(d, inst)
    state = fit_state(w)
    # analogue library: per year, the weekly level anomaly and SWE anomaly (for the nearest-year search at run time)
    lib = w[["S_a", "swe_a"]].dropna(subset=["S_a"]).copy()
    lib["woy"] = lib.index.isocalendar().week.astype(int).values
    library = {"weeks": [str(t.date()) for t in lib.index], "S_a": [round(float(v), 1) for v in lib["S_a"]],
               "swe_a": [None if pd.isna(v) else round(float(v), 2) for v in lib["swe_a"]],
               "scale": {"S_a": float(lib["S_a"].std()), "swe_a": float(lib["swe_a"].std())}}
    model = {
        "area": AREA, "version": MODEL_VERSION, "fitted_at": dt.date.today().isoformat(),
        "history": {"from": str(d.index.min().date()), "to": str(d["wtr"].last_valid_index().date())},
        "issues": {"from": str(min(inst).date()), "to": str(max(inst).date()), "n": int(len(B))},
        "weather": {"h_days": WEATHER_H, "ens_days": ENS_DAYS, "ext_days": EXT_DAYS, **weather},
        "state": {"h_weeks": STATE_HW, "swe_months": SWE_MONTHS, "ramp_days": list(STATE_RAMP_DAYS), "shrink": STATE_SHRINK, "beta": state},
        "swe": {"source": str(swe_csv), "country": SWE_COUNTRY, "clim_by_week": swe_clim},
        "analogue_library": library,
        "seasons": {str(k): v for k, v in SEASONS.items()},
    }
    Path(a.out).write_text(json.dumps(model, indent=1), encoding="utf-8")
    log.info("wrote %s (%d kB)", a.out, Path(a.out).stat().st_size // 1024)
    print("\nweather term — LOYO RMSE model / normal (GWh):")
    for h in WEATHER_H:
        s = weather["scores"][str(h)]
        print("  h=%3d d  %5.0f / %5.0f  (%+.0f %%)  k_all=%.3f  k=%s" % (
            h, s["rmse_model"], s["rmse_normal"], 100 * (s["rmse_model"] / s["rmse_normal"] - 1), weather["k"][str(h)]["all"],
            {k: round(v, 3) for k, v in weather["k"][str(h)].items() if k != "all"}))
    print("\nstate term — LOYO RMSE at 13 / 26 weeks by start season: normal | level | level+SWE (used?)")
    for sea in ("DJF", "MAM", "JJA", "SON"):
        print("  %s  " % sea + "   ".join("%5.0f | %5.0f%s | %s" % (
            state[str(hw)][sea]["rmse_normal"], state[str(hw)][sea]["rmse_model"], "" if state[str(hw)][sea]["use_level"] else " (off)",
            ("%5.0f %s" % (state[str(hw)][sea]["rmse_model_swe"], "yes" if state[str(hw)][sea]["use_swe"] else "no")) if state[str(hw)][sea]["rmse_model_swe"] else "   —") for hw in (13, 26)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
