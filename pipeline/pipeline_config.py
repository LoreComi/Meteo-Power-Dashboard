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
    river_table: str
    river_station_table: str
    swe_table: str
    merge_chunk_rows: int
    # ── SWE model ───────────────────────────────────────────────────────────
    swe_csv_dir: Path
    swe_main_py: Path
    swe_refresh_days: int                # days re-uploaded every run (the model itself re-reads the last 14)
    swe_include_heightbands: bool
    swe_model_timeout_min: int
    # ── Local ───────────────────────────────────────────────────────────────
    log_dir: Path
    out_dir: Path
    extra: dict = field(default_factory=dict)

    @property
    def river_table_fq(self) -> str:
        return f"{self.schema}.{self.river_table}"

    @property
    def river_station_table_fq(self) -> str:
        return f"{self.schema}.{self.river_station_table}"

    @property
    def swe_table_fq(self) -> str:
        return f"{self.schema}.{self.swe_table}"

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
        river_station_table=os.environ.get("RIVER_STATION_TABLE", "river_stations_eq"),
        swe_table=os.environ.get("SWE_TABLE", "swe_daily"),
        merge_chunk_rows=_int("MERGE_CHUNK_ROWS", 2000),
        swe_csv_dir=Path(os.environ.get("SWE_CSV_DIR", str(swe_root / "Output_files" / "CSVs"))),
        swe_main_py=Path(os.environ.get("SWE_MAIN_PY", str(swe_root / "Scripts" / "SWE_main.py"))),
        swe_refresh_days=_int("SWE_REFRESH_DAYS", 30),
        swe_include_heightbands=_bool("SWE_INCLUDE_HEIGHTBANDS", True),
        swe_model_timeout_min=_int("SWE_MODEL_TIMEOUT_MIN", 120),
        log_dir=Path(os.environ.get("PIPELINE_LOG_DIR", str(HERE / "logs"))),
        out_dir=Path(os.environ.get("PIPELINE_OUT_DIR", str(HERE / "out"))),
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
