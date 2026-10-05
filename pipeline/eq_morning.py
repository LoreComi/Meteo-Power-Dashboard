"""Morning Call input from Energy Quantified → {schema}.morning_daily_eq.

The Morning Call grid (weather-power-desk-app/_morning.py) reads, per run, the
daily values of four families on the Morning Report's regions, with the normal:

  tt   consumption temperature  °C      daily mean          rows: fr de uk it hu np ib  (+ be nl)
  wnd  wind power production    MWh/h   daily mean          rows: de uk fr it see np ib  (+ be nl)
  spv  solar PV production      MWh/h   daily mean          rows: de fr it see np ib  (+ be nl)
  rre  hydro precipitation energy GWh   daily sum           rows: cwe + it-nord (Alps), np, see, ib

The Gas Demand section reads the same table (its LDZ countries are DE UK FR BE NL
IT, its wind & solar regions those plus Iberia), which is why be and nl are
loaded for the three weather families although the Morning Call grid has no row
for them.

Volue delivered these as one 'Avg' curve per region and run pattern (notebook
cell 7, table morning_daily). This module delivers the same table shape from EQ
— every model EQ carries (ec-ens, ec, gfs-ens, gfs, aifs-ens, aifs, icon, ecsr;
ec-ext / gfs-ext from the Medium-term curves) and every cycle (00, 06, 12, 18) —
so the grid shows all the 00z runs in the morning and each later cycle as soon
as EQ publishes it. Run it often (run_morning.bat, e.g. every 2 hours): each run
loads the issues of the last MORNING_LOOKBACK_HOURS and MERGEs them, so a cycle
is picked up on the first run after it lands and a revised issue is updated.

Regions. EQ has NP (Nordic) as an area; the others are built from EQ's country
areas (MORNING_REGIONS): sums for production and precipitation energy, a
population-weighted mean for temperature (weights from pop_weights_0p5.csv —
the same weights the notebook uses for the Meteomatics means). An aggregate is
written only when every member has the value for that (model, issue, day), so a
model missing one member never biases the sum. The SEE and Alps (cwe) member
lists are configurable; Volue's own definitions of those aggregates are not
published, so they are our best reading of them.

Daily aggregation is EQ's (P1D, AVERAGE or SUM, CET days). threshold_pct=10
keeps the issue day when it misses at most 10 % of its points and drops any
shorter partial day, which is what the report's "drop the half day" does.
n_points is therefore a constant 96 (the app's partial-day filter keeps all).

Table morning_daily_eq (MERGE key: family, region, pattern, reference_date, day):
  provider 'EQ', family, region, pattern (EQ tag), curve_name, reference_date
  (issue time, UTC), day, value, n_points, normal, loaded_at
Values are in the units the app expects from morning_daily: °C, MWh/h, GWh.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("pipeline.morning")

MORNING_COLUMNS = [
    ("provider", "STRING"), ("family", "STRING"), ("region", "STRING"), ("pattern", "STRING"), ("curve_name", "STRING"),
    ("reference_date", "TIMESTAMP"), ("day", "DATE"), ("value", "DOUBLE"), ("n_points", "INT"), ("normal", "DOUBLE"),
    ("loaded_at", "TIMESTAMP"),
]
MORNING_KEYS = ["family", "region", "pattern", "reference_date", "day"]
N_POINTS = 96

# family -> EQ curve templates, EQ aggregation, scale to the app's unit, how members combine
MORNING_FAMILIES = {
    "tt":  {"curve": "{a} Consumption Temperature °C 15min Forecast",
            "medium": "{a} Consumption Temperature Medium-term °C 15min Forecast",
            "normal": "{a} Consumption Temperature °C 15min Normal", "agg": "AVERAGE", "scale": 1.0, "combine": "pop"},
    "wnd": {"curve": "{a} Wind Power Production MWh/h 15min Forecast",
            "medium": "{a} Wind Power Production Medium-term MWh/h 15min Forecast",
            "normal": "{a} Wind Power Production MWh/h 15min Normal", "agg": "AVERAGE", "scale": 1.0, "combine": "sum"},
    "spv": {"curve": "{a} Solar Photovoltaic Production MWh/h 15min Forecast",
            "medium": "{a} Solar Photovoltaic Production Medium-term MWh/h 15min Forecast",
            "normal": "{a} Solar Photovoltaic Production MWh/h 15min Normal", "agg": "AVERAGE", "scale": 1.0, "combine": "sum"},
    "rre": {"curve": "{a} Hydro Precipitation Energy MWh H Forecast",
            "medium": "{a} Hydro Precipitation Energy Medium-term MWh H Forecast",
            "normal": "{a} Hydro Precipitation Energy MWh H Normal", "agg": "SUM", "scale": 0.001, "combine": "sum"},  # MWh -> GWh
}
# region code (as the app's MORNING_BLOCKS rows) -> EQ areas
MORNING_REGIONS = {
    "fr": ["FR"], "de": ["DE"], "uk": ["GB"], "it": ["IT"], "hu": ["HU"], "np": ["NP"], "ib": ["ES", "PT"],
    "be": ["BE"], "nl": ["NL"],                           # Gas Demand countries without a Morning Call row
    "see": ["RO", "BG", "GR", "HR", "SI", "RS"],          # our reading of Volue's SEE aggregate
    "cwe": ["FR", "DE", "AT", "CH"],                      # Volue's CWE; + it-nord = the report's "Alps" precipitation
    "it-nord": ["IT-NORD"],
}
# which regions each family needs (the report's rows + the Gas Demand countries),
# so nothing superfluous is downloaded
FAMILY_REGIONS = {
    "tt": ["fr", "de", "uk", "it", "hu", "np", "ib", "be", "nl"],
    "wnd": ["de", "uk", "fr", "it", "see", "np", "ib", "be", "nl"],
    "spv": ["de", "fr", "it", "see", "np", "ib", "be", "nl"],
    "rre": ["cwe", "it-nord", "np", "see", "ib"],
}
# per-family member lists where EQ lacks a curve for one member: no Serbian solar
# curves, no Slovenian wind curves (checked 2026-10-02)
REGION_OVERRIDES = {"spv": {"see": ["RO", "BG", "GR", "HR", "SI"]},
                    "wnd": {"see": ["RO", "BG", "GR", "HR", "RS"]}}


def region_members(family: str, region: str) -> list[str]:
    return REGION_OVERRIDES.get(family, {}).get(region, MORNING_REGIONS[region])
MEDIUM_TAGS = ["ec-ext", "gfs-ext"]
MEDIUM_LOOKBACK_DAYS = 8                     # the extended runs come twice a week
# EQ caps `limit` at 24 instances per call. Tags are therefore requested in groups
# by cycle frequency, so that 24 covers every issue of the lookback window:
# the 12-hourly ECMWF pair (≤ 3 issues each in 30 h) and the 6-hourly models
# (≤ 5 issues each × 4 tags = 20); anything else (icon, ecsr) is a third group.
EQ_INSTANCE_LIMIT = 24
TAG_GROUPS = [["ec-ens", "ec"], ["gfs-ens", "gfs", "aifs-ens", "aifs"]]
NORMAL_AHEAD_DAYS = 50
RETRIES, BACKOFF_S = 3, 5
POP_WEIGHTS_CSV = Path(__file__).resolve().parents[1] / "weather-power-desk-app" / "pop_weights_0p5.csv"
POP_FALLBACK_M = {"ES": 47.4, "PT": 10.3, "FR": 66.9, "DE": 83.0, "AT": 9.0, "CH": 8.7, "RO": 19.0, "BG": 6.4,
                  "GR": 10.4, "HR": 3.9, "SI": 2.1, "RS": 6.6}


def _retry(fn, what: str):
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == RETRIES or "not found" in str(e).lower():     # a missing curve will not appear on retry
                raise
            log.warning("%s failed (%s) — retry %d/%d in %ds", what, str(e)[:120], attempt, RETRIES, BACKOFF_S * attempt)
            time.sleep(BACKOFF_S * attempt)


def eq_session(api_key: str, ssl_verify: bool):
    if not api_key:
        raise RuntimeError("No EQ_API_KEY — set it in pipeline/.env or keep eq_fundamentals.py reachable")
    from energyquantified import EnergyQuantified
    if not ssl_verify:
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    return EnergyQuantified(api_key=api_key, ssl_verify=ssl_verify)


def population_weights() -> dict[str, float]:
    """Persons per EQ area for the temperature aggregates — the notebook's weights file
    summed per country (GB = the UK row), falling back to rounded census figures."""
    try:
        w = pd.read_csv(POP_WEIGHTS_CSV).groupby("area")["population"].sum()
        out = {("GB" if a == "UK" else a): float(p) for a, p in w.items()}
        if out:
            return out
    except Exception as e:
        log.info("pop weights file not usable (%s) — using the fallback figures", str(e)[:80])
    return {k: v * 1e6 for k, v in POP_FALLBACK_M.items()}


# ─── EQ loading ────────────────────────────────────────────────────────────────

def _agg(name: str):
    from energyquantified.metadata import Aggregation
    return Aggregation.SUM if name == "SUM" else Aggregation.AVERAGE


def _ts_to_frame(ts, family: str, area: str, tag: str, issued) -> pd.DataFrame:
    df = ts.to_pandas_dataframe()
    if df.empty:
        return pd.DataFrame()
    s = df.iloc[:, 0]
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(s.values, index=idx.normalize()).dropna()
    if issued is not None:
        issued = pd.Timestamp(issued)
        issued = issued.tz_convert("UTC").tz_localize(None) if issued.tzinfo is not None else issued
    return pd.DataFrame({"family": family, "area": area, "pattern": tag, "issued": issued,
                         "day": [d.date() for d in s.index], "value": s.values.astype(float)})


def load_area_forecasts(eq, family: str, area: str, models: list[str], lookback_h: int, now: dt.datetime) -> pd.DataFrame:
    """Every issue of the last lookback hours for the models, plus the extended
    runs of the last MEDIUM_LOOKBACK_DAYS, as daily values."""
    from energyquantified.time import Frequency
    cfg = MORNING_FAMILIES[family]
    frames = []
    curve = cfg["curve"].format(a=area)
    short = [m for m in models if m not in MEDIUM_TAGS]
    groups = [[t for t in g if t in short] for g in TAG_GROUPS]
    groups.append([t for t in short if not any(t in g for g in TAG_GROUPS)])
    for tags in [g for g in groups if g]:
        try:
            lst = _retry(lambda: eq.instances.load(curve, tags=tags, issued_at_earliest=now - dt.timedelta(hours=lookback_h),
                                                   limit=EQ_INSTANCE_LIMIT, frequency=Frequency.P1D,
                                                   aggregation=_agg(cfg["agg"]), threshold_pct=10), f"instances {curve} {tags}")
            for ts in lst:
                frames.append(_ts_to_frame(ts, family, area, ts.instance.tag, ts.instance.issued))
        except Exception as e:
            log.warning("  %s %s: %s", curve, tags, str(e)[:160])
    medium = [m for m in models if m in MEDIUM_TAGS]
    if medium:
        mcurve = cfg["medium"].format(a=area)
        try:
            lst = _retry(lambda: eq.instances.load(mcurve, tags=medium, limit=6,
                                                   issued_at_earliest=now - dt.timedelta(days=MEDIUM_LOOKBACK_DAYS),
                                                   frequency=Frequency.P1D, aggregation=_agg(cfg["agg"]), threshold_pct=10),
                         f"instances {mcurve}")
            for ts in lst:
                frames.append(_ts_to_frame(ts, family, area, ts.instance.tag, ts.instance.issued))
        except Exception as e:
            log.info("  %s: %s", mcurve, str(e)[:120])
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["family", "area", "pattern", "issued", "day", "value"])


def load_area_normal(eq, family: str, area: str, today: dt.date) -> pd.DataFrame:
    from energyquantified.time import Frequency
    cfg = MORNING_FAMILIES[family]
    curve = cfg["normal"].format(a=area)
    try:
        ts = _retry(lambda: eq.timeseries.load(curve, begin=today - dt.timedelta(days=10),
                                               end=today + dt.timedelta(days=NORMAL_AHEAD_DAYS),
                                               frequency=Frequency.P1D, aggregation=_agg(cfg["agg"])), f"normal {curve}")
    except Exception as e:
        log.warning("  %s: %s", curve, str(e)[:160])
        return pd.DataFrame(columns=["family", "area", "day", "normal"])
    f = _ts_to_frame(ts, family, area, "", None)
    return f.rename(columns={"value": "normal"})[["family", "area", "day", "normal"]] if not f.empty else \
        pd.DataFrame(columns=["family", "area", "day", "normal"])


# ─── regions ───────────────────────────────────────────────────────────────────

def combine_regions(fc: pd.DataFrame, nm: pd.DataFrame, family: str, regions: list[str], pop: dict[str, float]) -> pd.DataFrame:
    """Member areas → the report's regions. Sums (or a population-weighted mean for
    temperature), written only where every member is present for the key."""
    cfg = MORNING_FAMILIES[family]
    out = []
    for region in regions:
        members = region_members(family, region)
        f = fc[fc["area"].isin(members)]
        n = nm[nm["area"].isin(members)]
        if f.empty:
            continue
        if cfg["combine"] == "pop" and len(members) > 1:
            w = {m: pop.get(m, np.nan) for m in members}
            if any(np.isnan(v) for v in w.values()):
                log.warning("  %s/%s: missing population weight for %s", family, region, w)
                continue
            f = f.assign(w=f["area"].map(w))
            n = n.assign(w=n["area"].map(w))
            g = f.groupby(["pattern", "issued", "day"], dropna=False)
            agg = g.apply(lambda x: pd.Series({"value": np.average(x["value"], weights=x["w"]), "k": len(x)}),
                          include_groups=False).reset_index()
            gn = n.groupby("day").apply(lambda x: pd.Series({"normal": np.average(x["normal"], weights=x["w"]), "k": len(x)}),
                                        include_groups=False).reset_index()
        else:
            agg = f.groupby(["pattern", "issued", "day"], dropna=False).agg(value=("value", "sum"), k=("value", "size")).reset_index()
            gn = n.groupby("day").agg(normal=("normal", "sum"), k=("normal", "size")).reset_index()
        agg = agg[agg["k"] == len(members)].drop(columns="k")
        gn = gn[gn["k"] == len(members)].drop(columns="k")
        agg = agg.merge(gn, on="day", how="left")
        agg["region"] = region
        out.append(agg)
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["family"] = family
    df["value"] = df["value"] * cfg["scale"]
    df["normal"] = df["normal"] * cfg["scale"]
    return df


# ─── orchestration ──────────────────────────────────────────────────────────────

def run(settings, writer, backfill: bool = False, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    now = dt.datetime.utcnow()
    lookback_h = settings.morning_lookback_hours * (6 if backfill else 1)
    eq = eq_session(settings.eq_api_key, settings.eq_ssl_verify)
    pop = population_weights()
    models = settings.morning_models
    writer.ensure_table(settings.morning_table_fq, MORNING_COLUMNS,
                        "Morning Call input from Energy Quantified: daily values per family / region / model / issue "
                        "with the EQ normal, every cycle. Written by Power_dashboard/pipeline (run_morning.bat).")
    frames = []
    for family, regions in FAMILY_REGIONS.items():
        areas = sorted({a for r in regions for a in region_members(family, r)})
        fc, nm = [], []
        for a in areas:
            fc.append(load_area_forecasts(eq, family, a, models, lookback_h, now))
            nm.append(load_area_normal(eq, family, a, today))
        fc = pd.concat([f for f in fc if not f.empty], ignore_index=True) if any(not f.empty for f in fc) else pd.DataFrame()
        nm = pd.concat([n for n in nm if not n.empty], ignore_index=True) if any(not n.empty for n in nm) else pd.DataFrame()
        if fc.empty:
            log.warning("%s: no forecasts returned", family)
            continue
        issues = fc.groupby("pattern")["issued"].agg(["nunique", "max"])
        log.info("%s: %d areas, issues per model: %s", family, len(areas),
                 ", ".join(f"{t} ×{int(r['nunique'])} (latest {pd.Timestamp(r['max']):%d %b %Hz})" for t, r in issues.iterrows()))
        frames.append(combine_regions(fc, nm, family, regions, pop))
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return {"rows": 0}
    df = pd.concat(frames, ignore_index=True)
    df["provider"] = "EQ"
    df["curve_name"] = "EQ " + df["family"] + " " + df["region"] + " " + df["pattern"]
    df["reference_date"] = pd.to_datetime(df["issued"])
    df["n_points"] = N_POINTS
    df["loaded_at"] = pd.Timestamp.utcnow().tz_localize(None)
    df = df[[c for c, _ in MORNING_COLUMNS]]
    n = writer.merge(settings.morning_table_fq, df, MORNING_KEYS)
    writer.prune(settings.morning_table_fq, f"reference_date < current_timestamp() - INTERVAL {settings.morning_retention_days} DAYS")
    latest = df.groupby("pattern")["reference_date"].max().sort_values(ascending=False)
    summary = {"rows": n, "families": sorted(df["family"].unique()),
               "latest_issue": {p: f"{t:%d %b %Hz}" for p, t in latest.items()}}
    log.info("morning done: %s", summary)
    return summary
