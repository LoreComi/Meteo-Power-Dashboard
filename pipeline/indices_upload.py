"""Daily teleconnection indices (all_daily_indices.csv) → {schema}.weather_indexes.

The CSV is one row per day and one column per index (1950 → today, 'M' = missing):
the NOAA CPC / BoM style set — SOI, the NINO regions, MODOKI, IOD, AMO, PDO, NPM, AO,
AAO, the stratospheric winds (U1060N, U10060N), NAO_EOF / NAO_STNPRS, PNA, EPO, WPO,
ABNA, SCAND, EAWR, EA, SGDIPOLE, GLAAM, the MJO (RMM1, RMM2, phase, amplitude) and the
GWO (GWO1, GWO2, phase, amplitude). Nothing is recomputed here: the table is the file,
reshaped to the long layout the notebook already creates (cell 8) and the Historical →
Teleconnections tab reads.

Table weather_indexes (MERGE key: index_name, date):
  index_name STRING   the CSV header, upper-cased with spaces as underscores ('MJO Phase' → MJO_PHASE)
  date DATE, value DOUBLE (missing values are not stored), source STRING, loaded_at TIMESTAMP

The tab analyses against reanalysis / ERA5 targets that start in 1979, so by default only
1979 → today is loaded (INDICES_START_YEAR=1950 for the rest; AO, the stratospheric winds,
NAO, PNA, EPO, WPO, ABNA, SCAND, EAWR and SGDIPOLE reach back to 1950). The first load is
~520 k rows (about 20 min in 2 000-row MERGEs); later runs re-send the last INDICES_REFRESH_DAYS
days, because the providers revise recent daily values.

    python indices_upload.py                  incremental (full load when the table is empty)
    python indices_upload.py --backfill       whole file again
    python indices_upload.py --dry-run        no Databricks: the long frame goes to pipeline/out/
    python indices_upload.py --csv path.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("pipeline.indices")

INDICES_COLUMNS = [
    ("index_name", "STRING"), ("date", "DATE"), ("value", "DOUBLE"), ("source", "STRING"), ("loaded_at", "TIMESTAMP"),
]
INDICES_KEYS = ["index_name", "date"]
SOURCE_LABEL = "all_daily_indices.csv"


def index_name(column: str) -> str:
    return "_".join(str(column).strip().upper().split())


def read_indices_csv(path: Path, start_year: int = 1979) -> pd.DataFrame:
    """The CSV as a long frame (index_name, date, value, source), no missing values."""
    wide = pd.read_csv(path, na_values=["M"])
    wide["Date"] = pd.to_datetime(wide["Date"], format="%m/%d/%Y")
    wide = wide[wide["Date"].dt.year >= start_year].set_index("Date")
    wide.columns = [index_name(c) for c in wide.columns]
    wide = wide.apply(pd.to_numeric, errors="coerce")
    wide = wide.replace([np.inf, -np.inf], np.nan)
    long = wide.rename_axis("date").reset_index().melt(id_vars="date", var_name="index_name", value_name="value")
    long = long.dropna(subset=["value"])
    long["date"] = long["date"].dt.date
    long["source"] = SOURCE_LABEL
    return long[["index_name", "date", "value", "source"]].sort_values(["index_name", "date"]).reset_index(drop=True)


def run(settings, writer, backfill: bool = False, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    path = Path(settings.indices_csv)
    if not path.exists():
        raise FileNotFoundError(f"indices CSV not found: {path} (set INDICES_CSV)")
    df = read_indices_csv(path, settings.indices_start_year)
    if df.empty:
        log.warning("no index rows read from %s", path)
        return {"rows": 0}
    last = max(df["date"])
    if (today - last).days > 5:
        log.warning("the indices CSV ends on %s — %d days old, was it refreshed?", last, (today - last).days)
    table = settings.indices_table_fq
    writer.ensure_table(table, INDICES_COLUMNS,
                        "Daily teleconnection indices (NAO, AO, PNA, SCAND, ENSO regions, MJO, ...) from "
                        "all_daily_indices.csv for the Historical -> Teleconnections tab. "
                        "Written by Power_dashboard/pipeline.")
    if not backfill:
        have = writer.table_max(table, "date")
        loaded = have is not None and not have.empty and pd.notna(have["mx"].iloc[0])
        if loaded:
            df = df[df["date"] >= today - dt.timedelta(days=settings.indices_refresh_days)]
        else:
            log.info("%s is empty or missing — loading the whole file", table)
    df["loaded_at"] = pd.Timestamp.utcnow().tz_localize(None)
    n = writer.merge(table, df, INDICES_KEYS)
    summary = {"rows": n, "indices": int(df["index_name"].nunique()), "last_date": str(last)}
    log.info("indices done: %s", summary)
    return summary


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline_config import load_settings, setup_logging
    from dbx_upload import DatabricksWriter, DryRunWriter

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--csv", type=Path, help="override INDICES_CSV")
    a = ap.parse_args(argv)
    settings = load_settings()
    if a.csv:
        settings.indices_csv = a.csv
    setup_logging(settings.log_dir)
    if a.dry_run or not settings.databricks_ready():
        writer = DryRunWriter(settings.out_dir, settings.merge_chunk_rows)
    else:
        writer = DatabricksWriter.from_settings(settings)
    try:
        run(settings, writer, backfill=a.backfill)
    finally:
        writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
