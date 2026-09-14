"""Configuration — Power Desk Weather Dashboard.

Four sections: Forecast, Historical & Analysis, Hydro Monitoring, Strategy.

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
VOLUE_SCHEMA = os.environ.get("VOLUE_SCHEMA", "dna_prod_silver.volue")
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

# ══════════════════════════════════════════════════════════════════════════════
# SECTIONS (landing tiles + sidebar)
# ══════════════════════════════════════════════════════════════════════════════
SECTIONS: dict[str, dict] = {
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
    "Strategy": {
        "num": "04",
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

# Approximate country boxes on the 0.5° Meteomatics grid — used only to build a
# country-mean from gridded Meteomatics fields (member table / model means).
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
# SCENARIOS (member clustering)
# ══════════════════════════════════════════════════════════════════════════════
SCENARIO_K_RANGE = (2, 5)            # k chosen by silhouette inside this range unless fixed
SCENARIO_DEFAULT_HORIZON = (1, 10)   # lead days clustered on
SCENARIO_MIN_MEMBERS = 3             # clusters smaller than this are folded into "Other"

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


def area_label(code: str) -> str:
    return AREAS.get(code, code)
