"""Configuration — Power Desk Weather Dashboard.

Sections: Forecast, Historical & Analysis, Hydro Monitoring, Gas Demand, Strategy.

Data sources
------------
Volue (delta share, silver layer)   {VOLUE_SCHEMA}.*          per-member ensembles via `tag`
Meteomatics (silver layer)          {METEOMATICS_SCHEMA}.*    gridded ensemble MEAN only
Meteomatics per-member table        METEOMATICS_MEMBER_TABLE  optional, enables ECAI/EC-ENS member clustering
Meteologica                         METEOLOGICA_TABLE         optional, not yet in Databricks
Weather indexes (NAO/AO/ENSO/...)   {SBX_SCHEMA}.weather_indexes  empty until loaded

Everything the app reads is pre-aggregated into {SBX_SCHEMA} by
power_desk_refresh.py (a Databricks notebook scheduled with Lakeflow Jobs).
The lookup tables below are mirrored in that notebook — keep both in sync.
"""
from __future__ import annotations

import os

from _style import CATEGORICAL

# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS / TABLES
# ══════════════════════════════════════════════════════════════════════════════
SBX_SCHEMA = os.environ.get("POWER_DESK_SCHEMA", "dna_snbx_weather.power_desk")

# The user-facing name for this source is "volue_deltashare". In the workspace
# the shared Volue tables are mounted under dna_prod_silver.volue; if the
# delta-share catalog is mounted elsewhere, set VOLUE_SCHEMA in app.yaml.
VOLUE_SCHEMA = os.environ.get("VOLUE_SCHEMA", "dna_prod_silver.volue_deltashare")
METEOMATICS_SCHEMA = os.environ.get("METEOMATICS_SCHEMA", "dna_prod_silver.meteomatics")

# Optional per-member Meteomatics table. Expected columns:
#   model STRING ('ecmwf-ens' | 'ecmwf-aifs-ens'), created_at TIMESTAMP,
#   delivery_start TIMESTAMP, member INT/STRING, latitude, longitude, value
# Leave empty if none exists — scenarios then cluster on Volue EC-ENS members.
METEOMATICS_MEMBER_TABLE = os.environ.get("METEOMATICS_MEMBER_TABLE", "")

# Optional Meteologica table (not in Databricks today). Expected columns:
#   reference_date TIMESTAMP, area STRING, metric STRING, delivery_date DATE,
#   member STRING, value DOUBLE
METEOLOGICA_TABLE = os.environ.get("METEOLOGICA_TABLE", "")

# Gold-layer Meteomatics climatology tables (ERA5 reanalysis, 0.5° grid,
# value/normal/anomaly per grid point per day). Used for the anomaly maps tab
# in the Historical section — read directly, not via the sandbox.
GOLD_SCHEMA = os.environ.get("GOLD_SCHEMA", "dna_prod_gold.weather")

MAP_METRICS: dict[str, dict] = {
    "Temperature": {
        "gold_table": "temperature_meteomatics_climatology",
        "curve_name": "t_mean_2m_24h_c_ecmwf_era5_p1d",
        "unit": "°C",
        "symmetric": True,
        "warm_is_positive": True,
    },
    "Wind speed (200 hPa)": {
        "gold_table": "wind_speed_meteomatics_climatology",
        "curve_name": "wind_speed_200hpa_ms_ecmwf_era5_p1d",
        "unit": "m/s",
        "symmetric": True,
        "warm_is_positive": False,
    },
    "Precipitation": {
        "gold_table": "precipitation_forecast_meteomatics_climatology",
        "curve_name": "precip_24h_mm_mix_p1d",
        "unit": "mm",
        "symmetric": True,
        "warm_is_positive": False,
    },
}

MAP_EUROPE_BBOX = {"lat_min": 35, "lat_max": 72, "lon_min": -12, "lon_max": 35}

# ══════════════════════════════════════════════════════════════════════════════
# SECTIONS (landing tiles + sidebar)
# ══════════════════════════════════════════════════════════════════════════════
SECTIONS: dict[str, dict] = {
    "Morning Call": {
        "num": "00",
        "desc": "The Morning Report, live from Volue or from the AI models on the Meteomatics method: "
                "what moved since the previous run and what is away from normal, up front — the biggest "
                "moves and anomalies as chips, an arrow chart per block, then the grid with the other "
                "models' runs as columns, each with its own change and its difference to the reference.",
        "color": CATEGORICAL[3], "locked": False, "wide": True,
    },
    "Forecast": {
        "num": "01",
        "desc": "Volue ensemble values by country, spread of the distribution vs its normal, "
                "and weather scenarios from member clustering.",
        "color": CATEGORICAL[0], "locked": False,
    },
    "Historical & Analysis": {
        "num": "02",
        "desc": "Monthly and weekly history by country — temperature from the Meteomatics / ERA5 "
                "climatology, population-weighted — multi-year / multi-month anomalies, ERA5 anomaly "
                "maps and analogues from weather indexes.",
        "color": CATEGORICAL[1], "locked": False,
    },
    "Hydro Monitoring": {
        "num": "03",
        "desc": "A Europe map of the hydro outlook — reservoirs, snow water equivalent, groundwater, hydro "
                "balance and river temperatures, criticalities flagged; click a country for its deep dive — "
                "then the Hydro Report quantify_* figures and stats per family, live.",
        "color": CATEGORICAL[4], "locked": False,
    },
    "Gas Demand": {
        "num": "04",
        "desc": "EU gas demand from the weather, on the Morning Call's Volue runs, or the AI models' temperature: "
                "LDZ heating demand from the fitted temperature-response curves, and wind + solar as "
                "gas-for-power displacement — run-over-run deltas and the trade signal per country.",
        "color": CATEGORICAL[5], "locked": False,
    },
    "Strategy": {
        "num": "05",
        "desc": "Positioning views built on the other three sections.",
        "color": CATEGORICAL[6], "locked": True,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# AREAS
# ══════════════════════════════════════════════════════════════════════════════
# Volue `area` codes with a TT/WND/SPV forecast. Names are what the UI shows.
AREAS: dict[str, str] = {
    "DE": "Germany", "FR": "France", "IT": "Italy", "ES": "Spain", "UK": "United Kingdom",
    "NL": "Netherlands", "BE": "Belgium", "PL": "Poland", "CZ": "Czech Republic",
    "HU": "Hungary", "NO": "Norway", "SE": "Sweden", "FI": "Finland", "DK": "Denmark",
    "PT": "Portugal", "SI": "Slovenia", "SK": "Slovakia", "HR": "Croatia",
    "EE": "Estonia", "LV": "Latvia", "LT": "Lithuania",
}
DEFAULT_AREAS = ["DE", "FR", "IT", "ES", "UK", "NL"]
AREA_NAME_TO_CODE = {v: k for k, v in AREAS.items()}

# Core countries used as the default clustering feature space
SCENARIO_DEFAULT_AREAS = ["DE", "FR", "IT", "ES", "UK", "NL", "BE", "PL"]

# Meteomatics country means are POPULATION-WEIGHTED in the refresh notebook:
# pop_weights_0p5.csv (built by build_pop_weights.py from GHS-POP 2025 and the
# country shapefile) gives, per country and 0.5° grid point, the persons of that
# country in that cell, and a country mean is sum(value × population) /
# sum(population). Upload the file next to wr_patterns.npz and set
# POP_WEIGHTS_PATH in the notebook. The app reads only the results.
POP_WEIGHTS_FILE = "pop_weights_0p5.csv"

# Approximate country boxes on the 0.5° Meteomatics grid — the notebook's
# FALLBACK for a country mean when the population weights are not loaded.
COUNTRY_BBOX: dict[str, dict] = {
    "DE": {"lat_min": 47.5, "lat_max": 55.0, "lon_min": 6.0, "lon_max": 15.0},
    "FR": {"lat_min": 42.5, "lat_max": 51.0, "lon_min": -4.5, "lon_max": 8.0},
    "IT": {"lat_min": 37.0, "lat_max": 47.0, "lon_min": 7.0, "lon_max": 18.5},
    "ES": {"lat_min": 36.0, "lat_max": 43.5, "lon_min": -9.5, "lon_max": 3.0},
    "UK": {"lat_min": 50.0, "lat_max": 58.5, "lon_min": -7.5, "lon_max": 1.5},
    "NL": {"lat_min": 50.5, "lat_max": 53.5, "lon_min": 3.5, "lon_max": 7.0},
    "BE": {"lat_min": 49.5, "lat_max": 51.5, "lon_min": 2.5, "lon_max": 6.5},
    "PL": {"lat_min": 49.0, "lat_max": 54.5, "lon_min": 14.0, "lon_max": 24.0},
    "CZ": {"lat_min": 48.5, "lat_max": 51.0, "lon_min": 12.0, "lon_max": 18.5},
    "HU": {"lat_min": 45.5, "lat_max": 48.5, "lon_min": 16.0, "lon_max": 22.5},
    "NO": {"lat_min": 58.0, "lat_max": 64.0, "lon_min": 5.0, "lon_max": 12.0},
    "SE": {"lat_min": 55.5, "lat_max": 64.0, "lon_min": 11.5, "lon_max": 19.0},
    "FI": {"lat_min": 60.0, "lat_max": 66.0, "lon_min": 21.0, "lon_max": 30.0},
    "DK": {"lat_min": 54.5, "lat_max": 57.5, "lon_min": 8.0, "lon_max": 12.5},
    "PT": {"lat_min": 37.0, "lat_max": 42.0, "lon_min": -9.5, "lon_max": -6.5},
    "SI": {"lat_min": 45.5, "lat_max": 46.5, "lon_min": 13.5, "lon_max": 16.5},
    "SK": {"lat_min": 47.5, "lat_max": 49.5, "lon_min": 17.0, "lon_max": 22.5},
    "HR": {"lat_min": 42.5, "lat_max": 46.5, "lon_min": 13.5, "lon_max": 19.5},
    "EE": {"lat_min": 57.5, "lat_max": 59.5, "lon_min": 22.0, "lon_max": 28.0},
    "LV": {"lat_min": 55.5, "lat_max": 58.0, "lon_min": 21.0, "lon_max": 28.0},
    "LT": {"lat_min": 54.0, "lat_max": 56.5, "lon_min": 21.0, "lon_max": 26.5},
}

# ══════════════════════════════════════════════════════════════════════════════
# METRICS (Volue categories)
# ══════════════════════════════════════════════════════════════════════════════
# scale: multiply raw Volue value by this before display (MWh/h -> GWh/h)
METRICS: dict[str, dict] = {
    "Temperature": {
        "cat": "TT", "unit": "°C", "scale": 1.0, "has_normal": True,
        "fcst_table": "temperature_consumption_forecast", "hist_table": "temperature_consumption",
        "anomaly_kind": "diff",      # anomaly shown as forecast - normal
        "warm_is_positive": True,    # red when above normal
    },
    "Wind": {
        "cat": "WND", "unit": "GWh/h", "scale": 0.001, "has_normal": True,
        "fcst_table": "production_forecast", "hist_table": "production",
        "anomaly_kind": "pct", "warm_is_positive": False,
    },
    "Solar": {
        "cat": "SPV", "unit": "GWh/h", "scale": 0.001, "has_normal": True,
        "fcst_table": "production_forecast", "hist_table": "production",
        "anomaly_kind": "pct", "warm_is_positive": False,
    },
    "Precipitation energy": {
        "cat": "RRE", "unit": "GWh", "scale": 1.0, "has_normal": True,
        "fcst_table": "precipitation_forecast", "hist_table": "precipitation",
        "anomaly_kind": "pct", "warm_is_positive": False,
    },
}
DEFAULT_METRIC = "Temperature"

# Volue model families: same NWP, different init cycles grouped together.
VOLUE_MODELS: dict[str, dict] = {
    "EC-ENS": {"patterns": ["ec00ens", "ec12ens"], "init_hours": [0, 12], "horizon_days": 15},
    "EC-Extended": {"patterns": ["ecmonthly"], "init_hours": [0], "horizon_days": 46},
    "GFS-ENS": {"patterns": ["gfs00ens"], "init_hours": [0], "horizon_days": 16},
}
DEFAULT_MODEL = "EC-ENS"
N_RUNS_KEPT = 6            # runs per model family materialised for run-to-run evolution

# Meteomatics models shown alongside Volue (ensemble mean only from silver)
METEOMATICS_MODELS: dict[str, dict] = {
    "Meteomatics EC-ENS": {"model": "ecmwf-ens", "curve": "t_mean_2m_24h_c_ecmwf_ens_p1d"},
    "Meteomatics AIFS-ENS": {"model": "ecmwf-aifs-ens", "curve": "t_mean_2m_24h_c_ecmwf_aifs_ens_p1d"},
}

# ══════════════════════════════════════════════════════════════════════════════
# UNCERTAINTY
# ══════════════════════════════════════════════════════════════════════════════
SPREAD_CLIM_LOOKBACK_DAYS = 365      # runs used to build the "normal spread" per lead day
SPREAD_RATIO_HIGH = 1.3              # spread / normal spread above this = unusually uncertain
SPREAD_RATIO_LOW = 0.7               # below this = unusually confident

# ══════════════════════════════════════════════════════════════════════════════
# MORNING CALL (port of Morning_Report/import_00z_add_solar_np_tot.py)
# ══════════════════════════════════════════════════════════════════════════════
# Blocks of the Excel table, in order. Each row: (display label, Volue area code
# used inside the curve name). Curve names are exactly the wapi ones, so the
# job matches on LOWER(curve_name) and never depends on the `area` column.
#
# heat_scale / warm_is_positive drive the grid's coloured views: a Δ of
# ±heat_scale saturates the shade, so the same colour means the same magnitude
# every morning. Polarity follows the app's diverging convention (_style.py):
# red = warmer, or less wind / solar / precipitation; blue = colder, or more.
MORNING_BLOCKS: dict[str, dict] = {
    "Temperatures": {
        "family": "tt", "unit": "°C", "scale": 1.0, "agg": "mean", "fmt": "{:.1f}",
        "heat_scale": 2.0, "warm_is_positive": True,
        "rows": [("FRA", "fr"), ("DE", "de"), ("UK", "uk"), ("ITA", "it"), ("HUN", "hu"),
                 ("Nordic", "np"), ("Iberia", "ib")],
    },
    "Wind": {
        "family": "wnd", "unit": "GW", "scale": 0.001, "agg": "mean", "fmt": "{:.1f}",
        "heat_scale": 2.0, "warm_is_positive": False,
        "rows": [("DE", "de"), ("UK", "uk"), ("FRA", "fr"), ("ITA", "it"), ("SEE", "see"),
                 ("Nordic", "np"), ("Iberia", "ib")],
    },
    "Solar PV": {
        "family": "spv", "unit": "GW", "scale": 0.001, "agg": "mean", "fmt": "{:.1f}",
        "heat_scale": 1.0, "warm_is_positive": False,
        "rows": [("DE", "de"), ("FRA", "fr"), ("ITA", "it"), ("SEE", "see"), ("Nordic", "np"), ("Iberia", "ib")],
    },
    "Precip (sum of coming 2 weeks)": {
        "family": "rre", "unit": "TWh", "scale": 0.001, "agg": "sum", "fmt": "{:.1f}",
        "heat_scale": 1.0, "warm_is_positive": False,
        # Alps = cwe + it-nord, exactly as the report does
        "rows": [("Alps", ["cwe", "it-nord"]), ("Nordic", "np"), ("SEE", "see"), ("Iberia", "ib")],
    },
}

# The report sums precipitation over EC-ENS's whole 15-day range. Longer runs
# (GFS 16 days, EC-Extended 46) are capped at the same length so the column
# is comparable across models.
MORNING_PRECIP_SUM_DAYS = 15

MORNING_CURVES: dict[str, dict] = {
    # family: forecast curve template, normal curve template  ({r} = region, {run} = run pattern)
    "tt":  {"fcst": "tt {r} con {run} °c cet min15 f",     "norm": "tt {r} con °c cet min15 n"},
    "wnd": {"fcst": "pro {r} wnd {run} mwh/h cet min15 f", "norm": "pro {r} wnd mwh/h cet min15 n"},
    "spv": {"fcst": "pro {r} spv {run} mwh/h cet min15 f", "norm": "pro {r} spv mwh/h cet min15 n"},
    "rre": {"fcst": "rre {r} {run} gwh cet min15 f",       "norm": "rre {r} gwh cet min15 n"},
}
MORNING_REGION_CODES = ["fr", "de", "uk", "it", "hu", "np", "ib", "see", "cwe", "it-nord"]

# Runs materialised for the Morning Call. EC 00z is the report's run; the rest
# power the "confront the models" view. Label -> Volue curve pattern.
MORNING_MODELS: dict[str, str] = {
    "EC-ENS 00z": "ec00ens",
    "EC-ENS 12z": "ec12ens",
    "GFS-ENS 00z": "gfs00ens",
    "EC-Extended": "ecmonthly",
}
MORNING_DEFAULT_MODEL = "EC-ENS 00z"

# The grid. Columns are runs — pattern + init time — in two groups (the two
# windows). Short model names per pattern for the column headers; the init
# hour is appended, so "ec00ens" and "ec12ens" both read "EC-ENS" + "00z"/"12z".
MORNING_MODEL_LABELS: dict[str, str] = {
    # Volue patterns (morning_daily, gas_demand_daily) and Meteomatics models
    "ec00ens": "EC-ENS", "ec12ens": "EC-ENS", "gfs00ens": "GFS-ENS", "gfs12ens": "GFS-ENS", "ecmonthly": "EC-Extended",
    "ec00": "EC Op", "gfs00": "GFS Op",
    "ecmwf-ens": "MM EC-ENS", "ecmwf-aifs-ens": "MM AIFS-ENS", "ncep-gfs-ens": "MM GFS-ENS",
    # Energy Quantified tags (morning_daily_eq) — the cycle hour is appended from the issue time
    "ec-ens": "EC-ENS", "ec": "EC Op", "gfs-ens": "GFS-ENS", "gfs": "GFS Op", "aifs-ens": "AIFS-ENS",
    "aifs": "AIFS Op", "icon": "ICON", "ecsr": "EC short-range", "ec-ext": "EC-Extended", "gfs-ext": "GFS-Extended",
}

# Where the Morning Call reads its runs from. EQ (default) is the local pipeline's
# morning_daily_eq — every model EQ has, every cycle (00/06/12/18), refreshed every
# couple of hours by pipeline/run_morning.bat, so the latest issue of the reference
# model is what the grid opens on and what the agent families brief about. Volue
# is the notebook's morning_daily (00z/12z 'Avg' curves, 6-hourly refresh).
MORNING_SOURCES: dict[str, dict] = {
    "Volue": {"table": "morning_daily_volue", "reference": "ec00ens", "enabled": True,
              "compare": ["gfs00ens", "ec12ens", "gfs12ens"], "lookback_days": 10,
              "desc": "Volue 'Avg' ensemble means (00z / 12z) with the Volue normal, the Morning Report's own curves, "
                      "loaded by Power_dashboard/pipeline (--only volue, run_morning.bat)"},
    "AI models": {"table": "morning_daily", "provider": "Meteomatics", "reference": "ecmwf-aifs-ens", "enabled": True,
                  "compare": ["ecmwf-ens", "ncep-gfs-ens"], "lookback_days": 10, "families": ["tt"],
                  "note": "Temperature only: Meteomatics carries no wind, solar or precipitation-energy production, "
                          "so the other blocks are not shown for this source.",
                  "desc": "AI weather models through the Meteomatics method: the ECMWF AIFS-ENS ensemble mean on the "
                          "Meteomatics grid, reduced to population-weighted means on the report's regions (notebook cell 7) "
                          "with the Volue normal; Meteomatics EC-ENS and GFS-ENS next to it, on the same method"},
    "EQ":    {"table": "morning_daily_eq", "reference": "ec-ens", "enabled": False,     # switched off: set True to bring it back
              "compare": ["gfs-ens", "aifs-ens", "ec", "gfs"], "lookback_days": 10,
              "desc": "Energy Quantified: every model and cycle, EQ normals; loaded by Power_dashboard/pipeline "
                      "(--only morning) every couple of hours"},
}


def enabled_sources(sources: dict) -> list[str]:
    """Source names shown in the switches (the ones not switched off)."""
    return [k for k, v in sources.items() if v.get("enabled", True)]


MORNING_DEFAULT_SOURCE = os.environ.get("MORNING_SOURCE", "Volue")
# Phone layout: one window at a time, the reference run plus at most this many compare columns
MORNING_PHONE_MAX_COMPARE = 1
# The reference run is the report's: its init day sets the windows and the Δ
# pairing, and it is what the agent families comment on.
MORNING_DEFAULT_REFERENCE_PATTERN = "ec00ens"
# The latest run of each of these is a compare column by default. EC-Extended
# is opt-in: a 46-day, coarser product is a different animal on a weekly window.
MORNING_DEFAULT_COMPARE_PATTERNS: list[str] = ["gfs00ens", "ec12ens", "ecmwf-ens", "ecmwf-aifs-ens"]
# Which earlier run each column's Δ run is taken against — applied to every
# column, so "did GFS move too?" is answered over the same interval. "report"
# is the Morning Report's pairing: the same model's run one day earlier, three
# days earlier on a Monday (the last report was Friday's).
MORNING_PREV_RULES: dict[str, str | int] = {
    "Report rule (Δ -24h, Δ -72h on Monday)": "report",
    "Previous run of the same model": "previous",
    "1 day earlier": 1, "2 days earlier": 2, "3 days earlier": 3, "7 days earlier": 7,
}

MORNING_RUN_HISTORY_DAYS = 8         # runs kept so Δ vs yesterday / Friday is always available
MORNING_MIN_DAY_COVERAGE = 0.9       # drop partial forecast days (report drops the half-day tail)

# Meteomatics country means mapped onto the report's regions (temperature only)
MORNING_METEOMATICS_REGIONS: dict[str, list[str]] = {
    "fr": ["FR"], "de": ["DE"], "uk": ["UK"], "it": ["IT"], "hu": ["HU"],
    "np": ["NO", "SE", "FI", "DK"], "ib": ["ES", "PT"], "see": ["SI", "HR", "SK", "HU"],
}

# ══════════════════════════════════════════════════════════════════════════════
# EUROPEAN WEATHER REGIMES (Forecast → Weather Regimes tab)
# ══════════════════════════════════════════════════════════════════════════════
# Port of Franziska_Intern/corso_model_wr/working_wr_tool (uber_main.py →
# run_WR_tool.py → max_proj_members.py), following Michel & Rivière (2011).
#
# Method, per member and per forecast day:
#   1. take the 500 hPa geopotential height ANOMALY on the fixed North-Atlantic /
#      European domain (lat 30–90 N, lon 80 W–40 E, 0.5° = 121 × 241 points),
#   2. divide it by that day-of-year's domain-average amplitude (`area_avg` in
#      the original LCD file) so winter and summer are on the same scale,
#   3. project it onto the 7 fixed regime patterns with cos(latitude) area
#      weighting — an inner product, which is why the refresh notebook can do
#      it as a Spark join rather than pulling the grid to the driver,
#   4. standardise each projection into an index (IWR) with that regime's own
#      long-term mean and spread,
#   5. assign the regime with the highest IWR, if it clears the threshold;
#      otherwise the day is "no regime".
#
# The 7 patterns and their normalisation constants are the fixed output of the
# original k-means study (1979–2019). They cannot be derived from Databricks,
# so they ship with the app in wr_patterns.npz — verified to reproduce the
# tool's own classified reanalysis exactly (see README).
WR_PATTERNS_FILE = "wr_patterns.npz"

WR_REGIMES: list[str] = ["ScTr", "GL", "EuBl", "AR", "AT", "ScBl", "ZO"]
WR_NO_REGIME = "no"
WR_ALL_LABELS = WR_REGIMES + [WR_NO_REGIME]
WR_REGIME_LONG: dict[str, str] = {
    "ScTr": "Scandinavian Trough",
    "GL":   "Greenland Blocking",
    "EuBl": "European Blocking",
    "AR":   "Atlantic Ridge",
    "AT":   "Atlantic Trough",
    "ScBl": "Scandinavian Blocking",
    "ZO":   "Zonal Regime",
    "no":   "No regime",
}

# Domain — fixed by the regime patterns; the forecast grid must match it exactly.
WR_LAT_MIN, WR_LAT_MAX = 30.0, 90.0
WR_LON_MIN, WR_LON_MAX = -80.0, 40.0
WR_GRID_STEP = 0.5

# Assignment threshold. The original uses a static 1.0 that relaxes with lead
# time, because a regime is harder to pin down further out: the fit of the
# climatological maximum IWR against lead day has slope WR_LEAD_FACTOR
# (reproduced from corso_model_wr/wr_lead/WR_15_day_running_mean_leadtime_*,
# a_opt = -0.018774217528098960). So the day-`lead` threshold is
# WR_THRESHOLD + lead * WR_LEAD_FACTOR, i.e. 1.00 at day 0 down to 0.74 at day 14.
WR_THRESHOLD = 1.0
WR_LEAD_FACTOR = -0.018774217528098960

# The threshold slope was fitted over leads 0–14, so 15 days is the honest
# horizon even though the Meteomatics table carries 17.
WR_DEFAULT_HORIZON_DAYS = 15
WR_MAX_HORIZON_DAYS = 17

# Forecast sources in dna_prod_silver.meteomatics.geopotential_height_forecast.
# The original tool runs on ECMWF-ENS (51 IFS members); the silver layer carries
# per-member geopotential only for AIFS-ENS and GFS-ENS, so AIFS — ECMWF's own
# ensemble, 50 members — is the default and GFS is offered alongside it.
WR_MODELS: dict[str, dict] = {
    "ECMWF AIFS-ENS": {"model": "ecmwf-aifs-ens",
                       "curve": "geopotential_height_500hpa_m_ecmwf_aifs_ens_p1d",
                       "n_members": 50},
    "NCEP GFS-ENS":   {"model": "ncep-gfs-ens",
                       "curve": "geopotential_height_500hpa_m_ncep_gfs_ens_p1d",
                       "n_members": 30},
}
WR_DEFAULT_MODEL = "ECMWF AIFS-ENS"
WR_RUN_HISTORY_DAYS = 8          # runs kept, so run-to-run evolution is available

# Reanalysis climatology. ERA5 actuals
# (dna_prod_silver.meteomatics.geopotential_height, model 'ecmwf-era5') are
# classified through the same projection, giving the climatological frequency
# of each regime, its persistence and the transition matrix.
WR_CLIM_START_YEAR = 1979        # backfill start; the notebook then appends daily
WR_CLIM_DOY_WINDOW = 7           # ± days pooled when computing a day-of-year frequency
WR_CLIM_MIN_YEARS = 20           # refuse to present a climatology thinner than this

# Regime colours — fixed per regime, matching the tool's plots so the two read
# the same. This is a categorical job: identity, not magnitude.
WR_COLORS: dict[str, str] = {
    "ScTr": "#ff4500",   # orangered
    "GL":   "#2a78d6",   # blue
    "EuBl": "#008b45",   # green4
    "AR":   "#ffd700",   # gold
    "AT":   "#551a8b",   # purple4
    "ScBl": "#006400",   # darkgreen
    "ZO":   "#e34948",   # red
    "no":   "#7f7f7f",   # gray50
}

# ══════════════════════════════════════════════════════════════════════════════
# GAS DEMAND (port of EU-gas-demand/ldz_forecast.py + rdl_forecast.py)
# ══════════════════════════════════════════════════════════════════════════════
# Two legs, both answering "how did this run move gas demand versus the run we
# compared against yesterday", which is what the desk trades off:
#
#   LDZ   temperature → local-distribution-zone (heating) gas demand, through
#         the fitted hinge curves in curve_models.json (gas_demand_model.py).
#         Warmer run = less heating = bearish; the delta sign is the signal.
#   RDL   wind + solar → the gas-fired generation they displace. Converted at
#         GAS_EFFICIENCY / GAS_LOWER_LOAD, so more renewables = less gas burn
#         = bearish. Sign is therefore the opposite of LDZ.
#
# Curves are fitted offline by EU-gas-demand/fit_demand_curves.py and shipped
# with the app as curve_models.json — refresh that file when the fit is redone.
GAS_CURVE_MODELS_FILE = "curve_models.json"

# LDZ: countries with a fitted curve. ldz_forecast.py runs de/uk/fr/be/nl
# (Italy is commented out there); Italy has a curve, so it is offered as an
# opt-in rather than dropped.
GAS_LDZ_AREAS: dict[str, str] = {"DE": "de", "UK": "uk", "FR": "fr", "BE": "be", "NL": "nl", "IT": "it"}
GAS_LDZ_DEFAULT_AREAS = ["DE", "UK", "FR", "BE", "NL"]

# RDL: label -> Volue area codes summed to form it. rdl_forecast.py uses
# Volue's 'ib' Iberia aggregate; the sandbox table carries countries, so
# Iberia is ES + PT — the same two grids that aggregate covers.
GAS_RDL_REGIONS: dict[str, list[str]] = {
    "DE": ["DE"], "UK": ["UK"], "FR": ["FR"], "BE": ["BE"], "NL": ["NL"],
    "IT": ["IT"], "Iberia": ["ES", "PT"],
}
GAS_RDL_DEFAULT_REGIONS = ["DE", "UK", "FR", "BE", "NL", "Iberia", "IT"]

# Gas-for-power conversion (rdl_forecast.py): GW of wind+solar -> GWh/day of
# gas not burned = GW * 24 / efficiency * lower_load.
GAS_EFFICIENCY = 0.5        # CCGT thermal efficiency
GAS_LOWER_LOAD = 0.8        # share of the renewable swing that actually displaces gas

# Runs compared. Each is scored against the run the script pairs it with:
# a 00z run vs the same pattern's previous 00z (Friday's on a Monday);
# a 12z run vs the same day's 00z (or the 00z two days earlier on a Monday).
GAS_RUNS: dict[str, str] = {
    "EC-ENS 00z": "ec00ens", "EC-ENS 12z": "ec12ens", "GFS-ENS 00z": "gfs00ens",
    "EC Op 00z": "ec00", "GFS Op 00z": "gfs00",
}
GAS_DEFAULT_RUNS = ["EC-ENS 00z", "GFS-ENS 00z"]

# Family grouping for gas demand (same concept as MORNING_FAMILIES)
GAS_FAMILIES: dict[str, list[str]] = {
    "EC-ENS": ["ec00ens", "ec12ens"],
    "GFS-ENS": ["gfs00ens"],
    "EC-Op": ["ec00"],
    "GFS-Op": ["gfs00"],
}
GAS_DEFAULT_FAMILY = "EC-ENS"

# Flat list of every gas-demand pattern — used by the UI which always shows
# EC and GFS side by side without a family selector.
GAS_ALL_PATTERNS: list[str] = ["ec00ens", "ec12ens", "gfs00ens", "ec00", "gfs00"]

# Where the Gas Demand section reads its runs from — the same two tables as the
# Morning Call (MORNING_SOURCES), so the GWh and the grid are built on the same
# numbers. EQ (default): morning_daily_eq, every model and cycle; the pipeline
# loads BE and NL for tt / wnd / spv on top of the report's regions so every LDZ
# country is there, and Iberia is EQ's ES + PT sum (region 'ib'). Wind and solar
# are MWh/h in that table and GW in gas_demand_daily, hence `prod_scale`.
# Volue: the notebook's gas_demand_daily (00z / 12z ensemble means per country).
GAS_SOURCES: dict[str, dict] = {
    "Volue": {
        "table": "morning_daily_volue", "lookback_days": 10, "enabled": True,
        "time_mode": "volue",                                    # reference_date = issue day (CET midnight); cycle from the pattern
        "patterns": ["ec00ens", "ec12ens", "gfs00ens", "gfs12ens"],
        "default_patterns": ["ec00ens", "gfs00ens"],
        "region_to_area": {"de": "DE", "uk": "UK", "fr": "FR", "be": "BE", "nl": "NL", "it": "IT", "ib": "IB"},
        "prod_scale": 0.001,                                     # MWh/h -> GW
        "rdl_regions": {"DE": ["DE"], "UK": ["UK"], "FR": ["FR"], "BE": ["BE"], "NL": ["NL"],
                        "IT": ["IT"], "Iberia": ["IB"]},         # Volue's own 'ib' aggregate, as rdl_forecast.py
        "legs": ["ldz", "rdl"],
        "desc": "Volue ensemble means (00z / 12z) per country, Volue normals, the scripts' own curves, "
                "loaded by Power_dashboard/pipeline (--only volue)",
    },
    "AI models": {
        "table": "morning_daily", "provider": "Meteomatics", "lookback_days": 10, "enabled": True,
        "time_mode": "volue",                                    # Meteomatics created_at snaps to the 00z / 12z cycle
        "patterns": ["ecmwf-aifs-ens", "ecmwf-ens", "ncep-gfs-ens"],
        "default_patterns": ["ecmwf-aifs-ens", "ecmwf-ens"],
        "region_to_area": {"de": "DE", "uk": "UK", "fr": "FR", "be": "BE", "nl": "NL", "it": "IT", "ib": "IB"},
        "prod_scale": 1.0, "rdl_regions": {}, "legs": ["ldz"],  # temperature only -> the LDZ leg only
        "note": "Temperature only: Meteomatics carries no wind or solar production, so this source has the LDZ leg "
                "alone. BE and NL appear once the notebook has run with cell 7's be / nl regions.",
        "desc": "AI weather models through the Meteomatics method: AIFS-ENS population-weighted temperature per country "
                "(notebook cell 7), Meteomatics EC-ENS / GFS-ENS for comparison; LDZ leg only",
    },
    "EQ": {
        "table": "morning_daily_eq", "lookback_days": 10, "enabled": False,                 # switched off
        "time_mode": "eq",
        "patterns": ["ec-ens", "gfs-ens", "ec", "gfs", "aifs-ens", "aifs"],
        "default_patterns": ["ec-ens", "gfs-ens"],              # the latest run of each is selected
        "region_to_area": {"de": "DE", "uk": "UK", "fr": "FR", "be": "BE", "nl": "NL", "it": "IT", "ib": "IB"},
        "prod_scale": 0.001,                                     # MWh/h -> GW
        "rdl_regions": {"DE": ["DE"], "UK": ["UK"], "FR": ["FR"], "BE": ["BE"], "NL": ["NL"],
                        "IT": ["IT"], "Iberia": ["IB"]},
        "legs": ["ldz", "rdl"],
        "desc": "Energy Quantified: the Morning Call's runs (every model and cycle), EQ normals",
    },
}
GAS_DEFAULT_SOURCE = os.environ.get("GAS_SOURCE", MORNING_DEFAULT_SOURCE)

GAS_FORECAST_DAYS = 14               # horizon pulled per run, as in dwld_fct
GAS_HIST_LOOKBACK_DAYS = 10          # trailing actual days: MAX_LAG_DAYS (6) + 4
GAS_TOTAL_SIGNAL_GWH = 1000          # |cumulative delta| above this = a trade signal

# ══════════════════════════════════════════════════════════════════════════════
# AI MORNING BRIEF (two agent families)
# ══════════════════════════════════════════════════════════════════════════════
# Replaces the Morning Report's free-text Pattern / Comment boxes. Each family
# reads the Morning Call numbers that are already on screen — nothing else —
# and writes a few sentences on what they mean for its own market.
#
#   Power family   temperature -> load · wind & solar -> residual load and the
#                  merit order · precipitation -> hydro. Then a synthesis.
#   Gas family     temperature -> LDZ heating demand · wind & solar -> gas-for-
#                  power displacement. Then a synthesis. When the Gas Demand
#                  section has been run, its LDZ / RDL deltas are handed to
#                  this family as quantified evidence.
AI_BRIEF_MODEL = os.environ.get("AI_BRIEF_MODEL", "gpt-4o")
AI_BRIEF_MAX_TOKENS = 320            # a few sentences, not a report
AI_BRIEF_SYNTHESIS_MAX_TOKENS = 420

AI_POWER_AGENTS: list[tuple[str, str, str]] = [
    ("pw_temp",       "🌡 Temperature",   "Load"),
    ("pw_wind_solar", "🌬 Wind & Solar",  "Residual load"),
    ("pw_precip",     "💧 Precipitation", "Hydro"),
]
AI_GAS_AGENTS: list[tuple[str, str, str]] = [
    ("gas_temp",       "🌡 Temperature",  "LDZ heating"),
    ("gas_wind_solar", "🌬 Wind & Solar", "Gas-for-power"),
]

# ══════════════════════════════════════════════════════════════════════════════
# SCENARIOS (member clustering)
# ══════════════════════════════════════════════════════════════════════════════
# Method: k-means on WEEKLY-MEAN temperature anomaly per country, weeks being
# real ISO weeks (Mon-Sun) because the desk trades weekly products. Two
# scenarios by default (the Morning Report's "alternative scenario"). No
# silhouette-based k selection. The proper method is clustering on 500 hPa
# geopotential members; that field is not in Databricks yet — when it lands,
# feed its member matrix into cluster_members() unchanged.
SCENARIO_K_DEFAULT = 2
SCENARIO_K_MAX = 4
SCENARIO_MIN_MEMBERS = 3             # clusters smaller than this are folded into "Other"
SCENARIO_MIN_WEEK_DAYS = 4           # a forecast week needs >= this many days to be a feature
# Regions shown side by side per scenario (Volue area codes present in fcst_members)
SCENARIO_OUTPUT_AREAS = ["DE", "FR", "UK", "IT", "ES", "NL", "BE", "PL"]

# ---------------------------------------------------------------------------
# Spatial clustering (gridded member maps)
# ---------------------------------------------------------------------------
# TODO(lorenzo): Switch to Z500 geopotential when ecmwf-ens gets Z500 members.
# Currently only ecmwf-aifs-ens has geopotential_height_500hpa members; the
# standard ecmwf-ens (whose 50 perturbations match Volue member IDs) does not.
# Using temperature spatial anomaly as a proxy until geopotential is available.
SCENARIO_USE_GEOPOTENTIAL = False   # flip to True + update SPATIAL_* when Z500 lands
SPATIAL_N_PCA = 10                  # PCA components for dimensionality reduction before k-means
SPATIAL_GRID_RES = 1.0              # degrees (refresh notebook coarsens to this from native 0.5°)

# Which member sources the scenario tab can cluster on. `table` is the sandbox
# table power_desk_refresh.py writes; `available` is resolved at runtime.
SCENARIO_SOURCES: dict[str, dict] = {
    "Meteomatics EC-ENS members": {"provider": "Meteomatics", "model": "ecmwf-ens"},
    "Meteomatics AIFS-ENS members": {"provider": "Meteomatics", "model": "ecmwf-aifs-ens"},
    "Volue EC-ENS members": {"provider": "Volue", "model": "EC-ENS"},
}

# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL
# ══════════════════════════════════════════════════════════════════════════════
HIST_START_YEAR = 2013               # Volue history
HIST_MM_START_YEAR = 1979            # Meteomatics / ERA5 temperature history (gold climatology)
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Which `source` of hist_daily the Historical section reads per metric. Temperature
# is the Meteomatics / ERA5 gold climatology — the same value / normal / anomaly
# rows the Anomaly Maps are drawn from — reduced to population-weighted country
# means by the refresh notebook, exactly like the Meteomatics forecast means.
# Wind, solar and precipitation energy stay Volue actuals vs the Volue normal.
# The Volue temperature is still in the table (source 'Volue'): the Gas Demand
# section seeds its LDZ curves with it and must keep doing so (the curves were
# fitted on Volue temperatures, and the forecast leg is Volue).
HIST_SOURCES: dict[str, str] = {"Temperature": "Meteomatics"}
HIST_DEFAULT_SOURCE = "Volue"
HIST_SOURCE_NOTES: dict[str, str] = {
    "Meteomatics": "Meteomatics / ERA5 climatology (gold layer) · population-weighted country means · the same "
                   "value, normal and anomaly fields as the Anomaly Maps tab",
    "Volue": "Volue actuals vs the Volue normal",
}


def hist_source(metric: str) -> str:
    return HIST_SOURCES.get(metric, HIST_DEFAULT_SOURCE)

# Weather indexes the analogue tab expects in {SBX_SCHEMA}.weather_indexes
# (index_name, date, value). Empty table until loaded.
WEATHER_INDEXES: dict[str, str] = {
    "NAO": "North Atlantic Oscillation",
    "AO": "Arctic Oscillation",
    "EA": "East Atlantic pattern",
    "SCAND": "Scandinavian pattern",
    "PNA": "Pacific/North American pattern",
    "ONI": "Oceanic Niño Index (ENSO)",
    "MJO_AMP": "MJO amplitude (RMM)",
    "MJO_PHASE": "MJO phase (RMM)",
    "QBO": "Quasi-Biennial Oscillation (30 hPa)",
    "SSW": "Stratospheric polar vortex (10 hPa 60N zonal wind)",
}
ANALOG_N_YEARS = 5

# ══════════════════════════════════════════════════════════════════════════════
# HYDRO (see _hydro_quantify.py for the maths)
# ══════════════════════════════════════════════════════════════════════════════
# Country -> Volue hydro `area` value in hydro_daily. Italy is stored as IT in
# the silver hydro table (the wapi report used 'it-nord'). The Western / Nordic
# areas come from the silver table (history since 2011–2013); SEE and the
# Eastern European countries exist only in the Volue share (since May 2026), so
# they carry an anomaly vs the provider normal but no percentile yet.
HYDRO_AREA_CODES: dict[str, str] = {
    "France": "FR", "Switzerland": "CH", "Austria": "AT", "Italy": "IT",
    "Nordics": "NP", "Spain": "ES", "SEE": "SEE", "Germany": "DE",
    "Norway": "NO", "Sweden": "SE", "Finland": "FI",
    "Slovenia": "SI", "Croatia": "HR", "Bosnia and Herzegovina": "BA", "Serbia": "RS",
    "North Macedonia": "MK", "Bulgaria": "BG", "Romania": "RO", "Greece": "GR",
}
HYDRO_COMPONENTS = {"Reservoir levels": "WTR", "Groundwater": "SGW", "Hydro balance": "BAL"}
HYDRO_START_YEAR = 2013

# ── Overview map (first tab of the section) ──────────────────────────────────
# Each hydro `area` is painted on a Europe map with its numbers. Values: the
# countries drawn (ISO-3, the codes Plotly's built-in outlines use) and the
# label. NP is the Nordic aggregate: it paints only the members that have no
# series of their own. SEE is Volue's South-East Europe aggregate — the list is
# the usual SEE hydro scope; adjust it if the share's definition differs. An
# aggregate is labelled once, on the first of its painted members, so order the
# list with the country that should carry the label first.
HYDRO_MAP_REGIONS: dict[str, dict] = {
    "FR":  {"name": "France",            "iso3": ["FRA"]},
    "CH":  {"name": "Switzerland",       "iso3": ["CHE"]},
    "AT":  {"name": "Austria",           "iso3": ["AUT"]},
    "IT":  {"name": "Italy",             "iso3": ["ITA"]},
    "ES":  {"name": "Spain",             "iso3": ["ESP"]},
    "DE":  {"name": "Germany",           "iso3": ["DEU"]},
    "NO":  {"name": "Norway",            "iso3": ["NOR"]},
    "SE":  {"name": "Sweden",            "iso3": ["SWE"]},
    "FI":  {"name": "Finland",           "iso3": ["FIN"]},
    # Eastern Europe — the Volue share's own country series (since May 2026)
    "SI":  {"name": "Slovenia",          "iso3": ["SVN"]},
    "HR":  {"name": "Croatia",           "iso3": ["HRV"]},
    "BA":  {"name": "Bosnia and Herzegovina", "iso3": ["BIH"]},
    "RS":  {"name": "Serbia",            "iso3": ["SRB"]},
    "MK":  {"name": "North Macedonia",   "iso3": ["MKD"]},
    "BG":  {"name": "Bulgaria",          "iso3": ["BGR"]},
    "RO":  {"name": "Romania",           "iso3": ["ROU"]},
    "GR":  {"name": "Greece",            "iso3": ["GRC"]},
    "NP":  {"name": "Nordics",           "iso3": ["SWE", "NOR", "FIN"], "aggregate": True},
    "SEE": {"name": "South-East Europe", "iso3": ["SRB", "HRV", "BIH", "SVN", "MNE", "MKD", "BGR", "ROU", "GRC"],
            "aggregate": True},
}
# Where each painted country's label sits (lat, lon) — nudged so that the Alpine
# trio and the Nordics do not overprint each other.
HYDRO_MAP_LABEL_POS: dict[str, tuple[float, float]] = {
    "FRA": (46.0, 0.6), "CHE": (47.3, 8.6), "AUT": (48.3, 15.2), "ITA": (42.6, 12.6), "ESP": (40.0, -3.7),
    "DEU": (51.6, 10.2), "NOR": (61.0, 7.8), "SWE": (63.2, 15.6), "FIN": (64.6, 27.2),
    "SVN": (46.3, 14.6), "HRV": (45.6, 16.0), "BIH": (44.0, 17.6), "SRB": (44.2, 20.8),
    "MNE": (42.8, 19.3), "MKD": (41.5, 21.7), "BGR": (42.8, 25.3), "ROU": (46.0, 25.0), "GRC": (39.6, 22.0),
}
HYDRO_MAP_EXTENT = {"lat_min": 35.5, "lat_max": 71.0, "lon_min": -11.0, "lon_max": 32.0}

# Layers the overview can be coloured by. Each needs, per area, the
# quantify_anomaly dict (latest_value, anomaly, anomaly_percent,
# anomaly_quantile, week_change_pct_points, n_hist_years, as_of). `source` says
# where the daily series come from: 'hydro_daily' (a Volue component, `family`)
# or 'swe_daily' (the internal Exolabs SWE model, uploaded by pipeline/). To add
# a layer: give it a source in _hydro._overview_metrics and register it here —
# map, criticality flags and the side-by-side grid pick it up unchanged.
#
# `level` says how "vs normal" is expressed: 'percent' (% of normal, and the week
# move in pts of normal) for a stock such as reservoir content; 'anomaly' (the
# anomaly in `unit`, and the week move in `unit`) for a series whose norm sits
# near zero part of the year — the hydro balance (a deviation) and SWE (no snow
# in summer) — where any percentage turns into noise. The fast-drawdown / refill
# flag applies to 'percent' layers only.
HYDRO_OVERVIEW_LAYERS: dict[str, dict] = {
    "Reservoir levels":      {"source": "hydro_daily", "family": "Reservoir levels", "colour_by": "anomaly_percent",
                              "level": "percent", "unit": "GWh"},
    "Snow water equivalent": {"source": "swe_daily", "colour_by": "anomaly_quantile", "level": "anomaly", "unit": "mm"},
    "Groundwater":           {"source": "hydro_daily", "family": "Groundwater", "colour_by": "anomaly_percent",
                              "level": "percent", "unit": "GWh"},
    "Hydro balance":         {"source": "hydro_daily", "family": "Hydro balance", "colour_by": "anomaly_quantile",
                              "level": "anomaly", "unit": "GWh"},
}
# swe_daily (pipeline/swe_upload.py): level = 'country', band = 'total', the model's
# mean SWE in mm per Alpine country → the hydro area it paints.
HYDRO_SWE_REGIONS: dict[str, str] = {"Austria": "AT", "Italy": "IT", "France": "FR", "Switzerland": "CH"}

# River temperature stations, drawn as markers on the overview map (coloured by
# the latest observation's anomaly vs its normal, red = warm) and charted in the
# country deep dive. Two sources share one table layout (curve_name, station_key,
# station, area, data_type, day, value, issued, tag) and one stations table
# (station_key, station, area, river, site, latitude, longitude):
#   EQ     river_temp_eq / river_stations_eq — Energy Quantified backcast, normal
#          and forecasts (ec-ens 15 d, ec-ext 45 d), since 2015; written by
#          Power_dashboard/pipeline (daily, local)
#   Volue  river_temp_volue / river_stations_volue — the deltashare's `riv` curves
#          (synthetic actual + ec00 / ec12 deterministic runs, no normal, since
#          May 2026); written by the refresh notebook (cell 6b)
# `lead_tags` are the forecast tags whose latest issue gives the 7-day peak on
# the map; `forecast_tags` are all the tags drawn in the deep dive.
# `flow_table` (EQ only): river flow in m³/s, same layout — hourly actual as a daily
# mean plus the daily normal; no backcast or forecast exists.
HYDRO_RIVER_SOURCES: dict[str, dict] = {
    "EQ":    {"table": "river_temp_eq", "flow_table": "river_flow_eq", "stations": "river_stations_eq",
              "lead_tags": ["ec-ens"], "forecast_tags": ["ec-ens", "ec-ext"]},
    "Volue": {"table": "river_temp_volue", "flow_table": None, "stations": "river_stations_volue",
              "lead_tags": ["ec00", "ec12"], "forecast_tags": ["ec00", "ec12"]},
}
HYDRO_DEEP_DIVE_RIVER_MONTHS = 14       # observed river temperature / flow drawn in the deep dive
# How the station markers are coloured: by the temperature anomaly (red = warm)
# or by the flow as % of its normal (red = low water).
HYDRO_STATION_MODES: dict[str, str] = {"temperature": "Temperature anomaly", "flow": "Flow % of normal"}
# Thresholds are indicative: discharge limits at the French plants bite from
# roughly 25–28 °C depending on the site; low-flow restrictions depend on the
# river's regulatory minimum flow.
RIVER_TEMP_WARM_ANOMALY_C = 2.0         # latest value this far above normal → warm (warning)
RIVER_TEMP_HOT_C = 25.0                 # absolute level → hot (critical); a forecast peak above it → warning
RIVER_TEMP_COLOUR_RANGE_C = 4.0         # ± anomaly that saturates the marker colour
RIVER_FLOW_CRITICAL_PCT = 40.0          # flow at or below this % of normal → very low (critical)
RIVER_FLOW_LOW_PCT = 60.0               # at or below → low (warning)
RIVER_FLOW_HIGH_PCT = 160.0             # at or above → high water (notice)
RIVER_FLOW_COLOUR_RANGE_PCT = (40.0, 160.0)   # % of normal saturating the marker colour
RIVER_STALE_DAYS = 30                   # an observation older than this is shown with its date but neither coloured nor flagged
HYDRO_OVERVIEW_DEFAULT_LAYER = "Reservoir levels"
HYDRO_COLOUR_MODES: dict[str, str] = {
    "anomaly_percent": "% of normal", "anomaly_quantile": "Percentile", "anomaly": "Anomaly (GWh)",
}
HYDRO_PCT_OF_NORMAL_RANGE = (60.0, 140.0)     # % of normal that saturates the map colour

# Criticality flags — the percentile of this week against the same week in
# every historical year, and the week-on-week move in % of normal. The same
# thresholds colour the percentile KPI cards in the family tabs.
HYDRO_PCTL_CRITICAL = 15      # at or below: critically low
HYDRO_PCTL_LOW = 30           # at or below: low
HYDRO_PCTL_HIGH = 85          # at or above: very high
HYDRO_WEEK_MOVE_PTS = 5       # |Δ % of normal| in a week at or above: fast drawdown / refill

# ── Production tab — hydro_prod_daily (pipeline/volue_hydro_prod.py) ─────────
# Per Volue production area and day, GWh/day: run-of-river and total production
# (reservoir = total − run-of-river), the Volue normal, Volue's own production
# forecast (pattern 'volue', one issue a day) and precipitation energy — actual,
# normal and the 'Avg' ensemble means of the patterns below. The areas are
# Volue's production areas, which differ from HYDRO_AREA_CODES: `it-nord` and
# `cwe` exist, the Eastern European countries only through `see`.
HYDRO_PROD_TABLE = "hydro_prod_daily"
HYDRO_PROD_AREAS: dict[str, str] = {
    "fr": "France", "ch": "Switzerland", "at": "Austria", "it-nord": "Italy North", "it": "Italy",
    "cwe": "CWE", "de": "Germany", "es": "Spain", "pt": "Portugal", "ib": "Iberia",
    "np": "Nordics", "no": "Norway", "se": "Sweden", "fi": "Finland", "see": "SEE",
}
HYDRO_PROD_DEFAULT_AREAS = ["fr", "ch", "at", "it-nord", "np", "ib", "see"]
HYDRO_PROD_WINDOW_DAYS = 14                                  # observed window, and the forward window it is compared with
HYDRO_PROD_PRECIP_PATTERNS = ["ec00ens", "ec12ens", "gfs00ens"]   # first = the reference the grid sums
HYDRO_PROD_LOOKBACK_DAYS = 10                                # forecast issues read from the table
HYDRO_PROD_HISTORY_DAYS = 120                                # actual / normal days read from the table


# ══════════════════════════════════════════════════════════════════════════════
# LIVE VOLUE ACCESS (bypass sandbox, query the delta-share tables directly)
# ══════════════════════════════════════════════════════════════════════════════
# Volue forecast / normal table pairs per Morning Call family, plus aggregation.
MORNING_VOLUE_TABLES: dict[str, tuple[str, str, str]] = {
    # family: (forecast_table, normal_table, daily_aggregation)
    # In volue_deltashare forecasts and normals live in the same view,
    # distinguished by data_type_name = 'Forecast' / 'Normal'.
    "tt":  ("temperature_consumption", "temperature_consumption", "AVG"),
    "wnd": ("production_wind",         "production_wind",         "AVG"),
    "spv": ("production_solar",        "production_solar",        "AVG"),
    "rre": ("precipitation_energy",    "precipitation_energy",    "SUM"),
}

# Gas Demand source tables (category, forecast_table, normal_table, scale)
GAS_VOLUE_FAMILIES: dict[str, tuple[str, str, str, float]] = {
    "tt":  ("TT",  "temperature_consumption", "temperature_consumption", 1.0),
    "wnd": ("WND", "production_wind",         "production_wind",         0.001),
    "spv": ("SPV", "production_solar",        "production_solar",        0.001),
}
GAS_VOLUE_PATTERNS: list[str] = ["ec00ens", "ec12ens", "gfs00ens", "ec00", "gfs00"]
GAS_VOLUE_AREAS: list[str] = ["DE", "UK", "FR", "BE", "NL", "IT", "ES", "PT"]

# Deltashare curve names use shortened pattern names (ec00 not ec00ens) and
# carry deterministic forecasts, not ensemble means (no tag='Avg').
# This map translates the app's internal pattern names to curve-name patterns.
DELTASHARE_PATTERN_MAP: dict[str, str] = {
    "ec00ens": "ec00", "ec12ens": "ec12",
    "gfs00ens": "gfs00", "ecmonthly": "ecmonthly",
    "ec00": "ec00", "gfs00": "gfs00",
}

# Wider lookback for manual run selection (sandbox only kept 8 days)
LIVE_HISTORY_DAYS = 14

# Expected forecast-day count per pattern (for completeness banner)
EXPECTED_HORIZON: dict[str, int] = {
    "ec00ens": 15, "ec12ens": 15, "gfs00ens": 16, "ecmonthly": 46,
    "ec00": 10, "gfs00": 16,
    # EQ tags (daily values after the partial last day is dropped)
    "ec-ens": 15, "ec": 15, "gfs-ens": 15, "gfs": 16, "aifs-ens": 15, "aifs": 15, "icon": 4, "ecsr": 5,
    # Meteomatics models (morning_daily, provider 'Meteomatics'): daily means over the ensemble range
    "ecmwf-ens": 15, "ecmwf-aifs-ens": 15, "ncep-gfs-ens": 16,
    "ec-ext": 46, "gfs-ext": 35,
}


def area_label(code: str) -> str:
    return AREAS.get(code, code)
