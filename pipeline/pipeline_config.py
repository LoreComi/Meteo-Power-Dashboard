"""Settings for the local hydro pipeline (river temperatures from Energy Quantified,
SWE from the Exolabs model) that uploads into the Power Desk sandbox schema.

Everything is read from the environment, with `pipeline/.env` (KEY=VALUE lines,
never committed) loaded first. Copy `env_example.txt` to `.env` and fill in the
Databricks credentials; the Energy Quantified key is the one the EQ project
already uses and is read from eq_fundamentals.py when EQ_API_KEY is not set.
PIPELINE_ENV_FILE overrides the .env location.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("pipeline.config")

HERE = Path(__file__).resolve().parent
# Settings files, read in this order (a key set by an earlier one wins; both are
# git-ignored): pipeline/.env, or pipeline/local_settings.txt for tools that will
# not create dot-files. PIPELINE_ENV_FILE adds another path in front.
ENV_FILES = [Path(p) for p in ([os.environ["PIPELINE_ENV_FILE"]] if os.environ.get("PIPELINE_ENV_FILE") else [])] \
    + [HERE / ".env", HERE / "local_settings.txt"]
ENV_FILE = ENV_FILES[0]

# Defaults that mirror the rest of the repo
DEFAULT_SCHEMA = "dna_snbx_weather.power_desk"                      # power_desk_refresh.py / app.yaml
DEFAULT_HTTP_PATH = "/sql/1.0/warehouses/2f4ff6b7c65abb1a"          # the desk apps' warehouse (app.yaml)
DEFAULT_EQ_FUNDAMENTALS = r"P:\QFA\TonyWeather\Lorenzo_Trainee\EQ_project\eq_fundamentals.py"
DEFAULT_SWE_ROOT = Path(r"P:\QFA\TonyWeather\Hydro_Report\SWE_Exolabs")
DEFAULT_INDICES_CSV = HERE.parent / "all_daily_indices.csv"
# Meteomatics is reached through the desk's DSL gateway (CLIMATOLOGIES/update_climp.py,
# corso_model_wr/download_forecast_dsl.py): an OAuth client-credential token whose
# client secret sits in the MDS_SECRET environment variable of the account running the job.
DEFAULT_DSL_URL = "https://fs-gateway-prod.trd.axusr.net/dsl/v3/meteo"
DEFAULT_DSL_URL_V1 = "https://fs-gateway-prod.trd.axusr.net/dsl/v1/meteo"
DEFAULT_DSL_APP_ID = "5a838d41-4163-42ef-8479-48fe7db6d547"
# The app's country codes (weather-power-desk-app/_config.AREAS) — the EQ member feed
# loads these; EQ names the UK 'GB'.
DEFAULT_EQ_MEMBER_AREAS = ["DE", "FR", "IT", "ES", "UK", "NL", "BE", "PL", "CZ", "HU", "NO", "SE", "FI", "DK",
                           "PT", "SI", "SK", "HR", "EE", "LV", "LT"]


def load_env_file(path: Path = ENV_FILE) -> int:
    """Load KEY=VALUE lines into os.environ without overriding what is already set."""
    if not path.exists():
        return 0
    n = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _list(name: str, default: list[str]) -> list[str]:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return list(default)
    return [x.strip() for x in v.split(",") if x.strip()]


def _date(name: str, default: str) -> dt.date:
    return dt.date.fromisoformat(os.environ.get(name) or default)


def _eq_key_from_project(path: str) -> str | None:
    """The EQ project keeps its key as a module constant; reuse it rather than copy it."""
    try:
        src = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'EQ_API_KEY\s*=\s*["\']([^"\']+)["\']', src)
    return m.group(1) if m else None


@dataclass
class Settings:
    # ── Energy Quantified ───────────────────────────────────────────────────
    eq_api_key: str
    eq_ssl_verify: bool                  # the EQ project runs with ssl_verify=False behind the proxy
    river_areas: list[str]               # EQ area tags whose "River Temperature" curves are loaded; empty = all
    river_stations: list[str]            # place keys / name fragments to keep; empty = every station of the areas
    river_backcast_days: int             # incremental window for backcast / actual / normal
    river_history_start: dt.date         # first day of a backfill (first run, --backfill, or a new curve)
    river_normal_ahead_days: int         # how far into the future the normal curve is loaded
    river_forecast_runs: int             # instances kept per forecast tag per run (1 = latest only)
    river_forecast_tags: list[str]       # forecast tags to load; empty = every tag the curve has
    # ── Databricks ──────────────────────────────────────────────────────────
    dbx_host: str
    dbx_http_path: str
    dbx_token: str
    dbx_auth_type: str                   # '' = PAT in DATABRICKS_TOKEN; else passed to databricks.sql.connect(auth_type=...)
    dbx_tls_no_verify: bool
    schema: str
    river_table: str                     # temperatures
    river_flow_table: str                # flows (m³/s)
    river_station_table: str
    swe_table: str
    merge_chunk_rows: int
    # ── Morning Call (EQ) ───────────────────────────────────────────────────
    morning_table: str
    morning_models: list[str]            # EQ tags loaded; 'ec-ext' / 'gfs-ext' come from the Medium-term curves
    morning_lookback_hours: int          # issues loaded per run (×6 on --backfill)
    morning_retention_days: int          # older issues are pruned from the table
    # ── SWE model ───────────────────────────────────────────────────────────
    swe_csv_dir: Path
    swe_main_py: Path
    swe_refresh_days: int                # days re-uploaded every run (the model itself re-reads the last 14)
    swe_include_heightbands: bool
    swe_model_timeout_min: int
    # ── Local ───────────────────────────────────────────────────────────────
    log_dir: Path
    out_dir: Path
    # ── Teleconnection indices (all_daily_indices.csv → weather_indexes) ────
    indices_csv: Path = DEFAULT_INDICES_CSV
    indices_table: str = "weather_indexes"
    indices_start_year: int = 1979       # the targets (ERA5 / reanalysis) start in 1979; 1950 loads the whole file
    indices_refresh_days: int = 90       # days re-sent every run — providers revise recent daily values
    extra: dict = field(default_factory=dict)

    # ── Scenario inputs (mm_z500_members.py, eq_members.py) ──
    dsl_url: str = DEFAULT_DSL_URL       # the DSL gateway (Meteomatics behind it); the v1 base for get_time_range
    dsl_url_v1: str = DEFAULT_DSL_URL_V1
    dsl_app_id: str = DEFAULT_DSL_APP_ID
    dsl_secret_env: str = "MDS_SECRET"   # environment variable holding the client secret
    dsl_ssl_verify: bool = False         # the gateway is called with verify=False in the desk's scripts (proxy)
    z500_table: str = "fcst_member_spatial_ecens"
    z500_model: str = "ecmwf-ens"        # Meteomatics model name of the ensemble whose Z500 members are fetched
    z500_members: int = 50               # perturbed members m1..mN (member:1-N; m0 = control, not loaded)
    z500_members_per_call: int = 10      # members per gateway call (ensembleMemberSelection 'member:a-b')
    z500_days: int = 15                  # forecast days after the init day (lead 0..15)
    z500_time_step: str = "PT6H"         # sampling within a day; the values of a day are averaged (P1D = 12 UTC only)
    z500_grid_res: float = 2.0           # degrees — the app's SPATIAL_GRID_RES
    z500_lat: tuple = (30.0, 74.0)       # the app's SPATIAL_BBOX
    z500_lon: tuple = (-30.0, 40.0)
    z500_min_age_hours: int = 7          # a cycle is complete in Meteomatics about this long after its init
    z500_min_members: int = 40           # fewer members fetched → the run is not written
    z500_retention_runs: int = 2         # runs kept in the table (the app reads the latest)
    eq_members_table: str = "fcst_members_eq"
    eq_members_models: list[str] = field(default_factory=lambda: ["aifs-ens"])    # EQ ensemble tags with members
    eq_members_areas: list[str] = field(default_factory=lambda: list(DEFAULT_EQ_MEMBER_AREAS))
    eq_members_retention_days: int = 7
    insert_chunk_rows: int = 6000        # rows per INSERT statement (whole-run snapshot tables)
    wr_ecens_table: str = "wr_forecast_members_ecens"   # wr_ecens.py: EC-ENS members on the 7 weather regimes
    wr_model: str = "ecmwf-ens"
    wr_days: int = 15                    # lead days after the init (the gateway serves exactly 15)
    wr_include_control: bool = True      # member 0 too → 51 members, as the tool
    wr_members_per_call: int = 5         # 29 161 points × 16 days per member ≈ 23 MB of JSON each
    wr_retention_days: int = 8           # runs kept (WR_RUN_HISTORY_DAYS in the notebook)
    wr_cache_dir: Path = HERE / "cache"  # the ERA5 day-of-year climatology pulled once from wr_gph_clim

    @property
    def z500_table_fq(self) -> str:
        return f"{self.schema}.{self.z500_table}"

    @property
    def eq_members_table_fq(self) -> str:
        return f"{self.schema}.{self.eq_members_table}"

    @property
    def wr_ecens_table_fq(self) -> str:
        return f"{self.schema}.{self.wr_ecens_table}"

    def dsl_ready(self) -> bool:
        return bool(os.environ.get(self.dsl_secret_env))

    @property
    def indices_table_fq(self) -> str:
        return f"{self.schema}.{self.indices_table}"

    @property
    def river_table_fq(self) -> str:
        return f"{self.schema}.{self.river_table}"

    @property
    def river_flow_table_fq(self) -> str:
        return f"{self.schema}.{self.river_flow_table}"

    @property
    def river_station_table_fq(self) -> str:
        return f"{self.schema}.{self.river_station_table}"

    @property
    def swe_table_fq(self) -> str:
        return f"{self.schema}.{self.swe_table}"

    @property
    def morning_table_fq(self) -> str:
        return f"{self.schema}.{self.morning_table}"

    def databricks_ready(self) -> bool:
        return bool(self.dbx_host and self.dbx_http_path and (self.dbx_token or self.dbx_auth_type))


def load_settings() -> Settings:
    for p in ENV_FILES:
        n = load_env_file(p)
        if n:
            log.info("settings: %d keys from %s", n, p)
    key = os.environ.get("EQ_API_KEY", "")
    if not key:
        key = _eq_key_from_project(os.environ.get("EQ_FUNDAMENTALS_PY", DEFAULT_EQ_FUNDAMENTALS)) or ""
        if key:
            log.info("EQ_API_KEY not set — using the key from the EQ project's eq_fundamentals.py")
    swe_root = Path(os.environ.get("SWE_ROOT", str(DEFAULT_SWE_ROOT)))
    host = os.environ.get("DATABRICKS_HOST", "").strip()
    if host.startswith("https://"):
        host = host[len("https://"):]
    host = host.rstrip("/")
    return Settings(
        eq_api_key=key,
        eq_ssl_verify=_bool("EQ_SSL_VERIFY", False),
        river_areas=_list("RIVER_AREAS", []),          # empty = every area EQ has river curves for (FR, DE, HU today)
        river_stations=_list("RIVER_STATIONS", []),
        river_backcast_days=_int("RIVER_BACKCAST_DAYS", 45),
        river_history_start=_date("RIVER_HISTORY_START", "2014-01-01"),   # EQ's river backcasts start in 2015
        river_normal_ahead_days=_int("RIVER_NORMAL_AHEAD_DAYS", 400),
        river_forecast_runs=_int("RIVER_FORECAST_RUNS", 1),
        river_forecast_tags=_list("RIVER_FORECAST_TAGS", []),
        dbx_host=host,
        dbx_http_path=os.environ.get("DATABRICKS_HTTP_PATH", DEFAULT_HTTP_PATH),
        dbx_token=os.environ.get("DATABRICKS_TOKEN", ""),
        dbx_auth_type=os.environ.get("DATABRICKS_AUTH_TYPE", ""),
        dbx_tls_no_verify=_bool("DATABRICKS_TLS_NO_VERIFY", False),
        schema=os.environ.get("POWER_DESK_SCHEMA", DEFAULT_SCHEMA),
        river_table=os.environ.get("RIVER_TABLE", "river_temp_eq"),
        river_flow_table=os.environ.get("RIVER_FLOW_TABLE", "river_flow_eq"),
        river_station_table=os.environ.get("RIVER_STATION_TABLE", "river_stations_eq"),
        swe_table=os.environ.get("SWE_TABLE", "swe_daily"),
        merge_chunk_rows=_int("MERGE_CHUNK_ROWS", 2000),
        morning_table=os.environ.get("MORNING_TABLE", "morning_daily_eq"),
        # icon (4 days) and ecsr (5 days) exist on EQ but cannot fill a weekly window — add them here if wanted
        morning_models=_list("MORNING_MODELS", ["ec-ens", "ec", "gfs-ens", "gfs", "aifs-ens", "aifs", "ec-ext", "gfs-ext"]),
        morning_lookback_hours=_int("MORNING_LOOKBACK_HOURS", 30),
        morning_retention_days=_int("MORNING_RETENTION_DAYS", 30),
        swe_csv_dir=Path(os.environ.get("SWE_CSV_DIR", str(swe_root / "Output_files" / "CSVs"))),
        swe_main_py=Path(os.environ.get("SWE_MAIN_PY", str(swe_root / "Scripts" / "SWE_main.py"))),
        swe_refresh_days=_int("SWE_REFRESH_DAYS", 30),
        swe_include_heightbands=_bool("SWE_INCLUDE_HEIGHTBANDS", True),
        swe_model_timeout_min=_int("SWE_MODEL_TIMEOUT_MIN", 120),
        log_dir=Path(os.environ.get("PIPELINE_LOG_DIR", str(HERE / "logs"))),
        out_dir=Path(os.environ.get("PIPELINE_OUT_DIR", str(HERE / "out"))),
        indices_csv=Path(os.environ.get("INDICES_CSV", str(DEFAULT_INDICES_CSV))),
        indices_table=os.environ.get("INDICES_TABLE", "weather_indexes"),
        indices_start_year=_int("INDICES_START_YEAR", 1979),
        indices_refresh_days=_int("INDICES_REFRESH_DAYS", 90),
        dsl_url=os.environ.get("DSL_URL", DEFAULT_DSL_URL),
        dsl_url_v1=os.environ.get("DSL_URL_V1", DEFAULT_DSL_URL_V1),
        dsl_app_id=os.environ.get("DSL_APP_ID", DEFAULT_DSL_APP_ID),
        dsl_secret_env=os.environ.get("DSL_SECRET_ENV", "MDS_SECRET"),
        dsl_ssl_verify=_bool("DSL_SSL_VERIFY", False),
        z500_table=os.environ.get("Z500_TABLE", "fcst_member_spatial_ecens"),
        z500_model=os.environ.get("Z500_MODEL", "ecmwf-ens"),
        z500_members=_int("Z500_MEMBERS", 50),
        z500_members_per_call=_int("Z500_MEMBERS_PER_CALL", 10),
        z500_days=_int("Z500_DAYS", 15),
        z500_time_step=os.environ.get("Z500_TIME_STEP", "PT6H"),
        z500_grid_res=float(os.environ.get("Z500_GRID_RES", "2.0")),
        z500_lat=tuple(float(x) for x in _list("Z500_LAT", ["30", "74"])),
        z500_lon=tuple(float(x) for x in _list("Z500_LON", ["-30", "40"])),
        z500_min_age_hours=_int("Z500_MIN_AGE_HOURS", 7),
        z500_min_members=_int("Z500_MIN_MEMBERS", 40),
        z500_retention_runs=_int("Z500_RETENTION_RUNS", 2),
        eq_members_table=os.environ.get("EQ_MEMBERS_TABLE", "fcst_members_eq"),
        eq_members_models=_list("EQ_MEMBERS_MODELS", ["aifs-ens"]),
        eq_members_areas=_list("EQ_MEMBERS_AREAS", list(DEFAULT_EQ_MEMBER_AREAS)),
        eq_members_retention_days=_int("EQ_MEMBERS_RETENTION_DAYS", 7),
        insert_chunk_rows=_int("INSERT_CHUNK_ROWS", 6000),
        wr_ecens_table=os.environ.get("WR_ECENS_TABLE", "wr_forecast_members_ecens"),
        wr_model=os.environ.get("WR_MODEL", "ecmwf-ens"),
        wr_days=_int("WR_DAYS", 15),
        wr_include_control=_bool("WR_INCLUDE_CONTROL", True),
        wr_members_per_call=_int("WR_MEMBERS_PER_CALL", 5),
        wr_retention_days=_int("WR_RETENTION_DAYS", 8),
        wr_cache_dir=Path(os.environ.get("WR_CACHE_DIR", str(HERE / "cache"))),
    )


def setup_logging(log_dir: Path, name: str = "pipeline") -> Path:
    """Console + one file per day, UTF-8 (curve names carry °C and umlauts)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{name}_{dt.date.today():%Y%m%d}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    try:  # Windows consoles default to cp1252; keep the ° and ü readable
        sh.stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    root.addHandler(sh)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("databricks").setLevel(logging.WARNING)
    return path
