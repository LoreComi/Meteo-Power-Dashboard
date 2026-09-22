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
        "desc": "The Morning Report table, live: weekly means per region for temperature, wind, solar "
                "and 2-week precipitation — absolute value, change vs the previous run, deviation from "
                "normal — with the other models' runs as columns of the same grid, each with its own "
                "run-over-run change and its difference to the reference.",
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
        "desc": "Monthly and weekly history by country, multi-year / multi-month anomalies, "
                "and analogues from weather indexes.",
        "color": CATEGORICAL[1], "locked": False,
    },
    "Hydro Monitoring": {
        "num": "03",
        "desc": "Reservoir levels, snow & groundwater and hydro balance vs normal — the "
                "Hydro Report quantify_* figures and stats, live.",
        "color": CATEGORICAL[4], "locked": False,
    },
    "Gas Demand": {
        "num": "04",
        "desc": "EU gas demand from the weather: LDZ heating demand from the fitted "
                "temperature-response curves, and wind + solar as gas-for-power "
                "displacement — run-over-run deltas and the trade signal per country.",
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
    "ec00ens": "EC-ENS", "ec12ens": "EC-ENS", "gfs00ens": "GFS-ENS", "ecmonthly": "EC-Extended",
    "ecmwf-ens": "MM EC-ENS", "ecmwf-aifs-ens": "MM AIFS-ENS",
}
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
GAS_RUNS: dict[str, str] = {"EC-ENS 00z": "ec00ens", "EC-ENS 12z": "ec12ens", "GFS-ENS 00z": "gfs00ens"}
GAS_DEFAULT_RUNS = ["EC-ENS 00z", "GFS-ENS 00z"]

# Family grouping for gas demand (same concept as MORNING_FAMILIES)
GAS_FAMILIES: dict[str, list[str]] = {
    "EC-ENS": ["ec00ens", "ec12ens"],
    "GFS-ENS": ["gfs00ens"],
}
GAS_DEFAULT_FAMILY = "EC-ENS"

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
HIST_START_YEAR = 2013
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

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
# Country -> Volue hydro `area` value in the shared table. Italy is stored as
# IT in the silver hydro table (the wapi report used 'it-nord'); SEE as SEE.
HYDRO_AREA_CODES: dict[str, str] = {
    "France": "FR", "Switzerland": "CH", "Austria": "AT", "Italy": "IT",
    "Nordics": "NP", "Spain": "ES", "SEE": "SEE", "Germany": "DE",
    "Norway": "NO", "Sweden": "SE", "Finland": "FI",
}
HYDRO_COMPONENTS = {"Reservoir levels": "WTR", "Snow & groundwater": "SGW", "Hydro balance": "BAL"}
HYDRO_START_YEAR = 2013


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
GAS_VOLUE_PATTERNS: list[str] = ["ec00ens", "ec12ens", "gfs00ens"]
GAS_VOLUE_AREAS: list[str] = ["DE", "UK", "FR", "BE", "NL", "IT", "ES", "PT"]

# Deltashare curve names use shortened pattern names (ec00 not ec00ens) and
# carry deterministic forecasts, not ensemble means (no tag='Avg').
# This map translates the app's internal pattern names to curve-name patterns.
DELTASHARE_PATTERN_MAP: dict[str, str] = {
    "ec00ens": "ec00", "ec12ens": "ec12",
    "gfs00ens": "gfs00", "ecmonthly": "ecmonthly",
}

# Wider lookback for manual run selection (sandbox only kept 8 days)
LIVE_HISTORY_DAYS = 14

# Expected forecast-day count per pattern (for completeness banner)
EXPECTED_HORIZON: dict[str, int] = {
    "ec00ens": 15, "ec12ens": 15, "gfs00ens": 16, "ecmonthly": 46,
}


def area_label(code: str) -> str:
    return AREAS.get(code, code)
