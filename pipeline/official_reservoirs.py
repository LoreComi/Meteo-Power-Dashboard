"""Official reservoir content per country, weekly → {schema}.hydro_official_weekly.

The Hydro section's Reservoir levels are shown on the official scale where an
official series exists (the app splices Volue's daily level onto it — see
_hydro_quantify.splice_official). Sources, all public:

  CH  SFOE / BFE energy dashboard, total Swiss storage lakes (OGD CSV, weekly
      since 2000, stamped Monday). ENTSO-E's CH figure covers only ~74 % of it.
  NO  NVE magasinstatistikk, national (weekly since 1995, stamped Sunday).
  FR AT IT ES SE FI BG GR HR RO RS MK SI
      ENTSO-E Transparency 16.1.D, "aggregated filling rate of water reservoirs
      and hydro storage plants", stored energy per ISO week (from 2014/15), read
      from dna_prod_silver.entsoe.hydro_reservoir_level and stamped on the
      week's Sunday (the end-of-week value: it matches NVE's Sunday reading).
  NP  NO (NVE) + SE + FI (ENTSO-E) on the Sundays all three report.

Volue vs these (2017–2026): CH, NO, SE, FI, IT and the Eastern countries agree
within a few %; FR is 3.3× ENTSO-E's scope, AT 1.4×, ES 1.2×, MK 0.3×.

Table hydro_official_weekly (MERGE key: area, day), the whole history every run (~13 k rows):
  area STRING (the dashboard's hydro area code), day DATE, value DOUBLE (GWh),
  capacity DOUBLE (GWh where the source gives it), source STRING, loaded_at TIMESTAMP

    python official_reservoirs.py              fetch and upload
    python official_reservoirs.py --dry-run    no Databricks writes: the frame goes to pipeline/out/
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger("pipeline.official_reservoirs")

TABLE_NAME = "hydro_official_weekly"
COLUMNS = [("area", "STRING"), ("day", "DATE"), ("value", "DOUBLE"), ("capacity", "DOUBLE"),
           ("source", "STRING"), ("loaded_at", "TIMESTAMP")]
KEYS = ["area", "day"]

SFOE_URL = "https://www.uvek-gis.admin.ch/BFE/ogd/17/ogd17_fuellungsgrad_speicherseen.csv"
NVE_URL = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"
ENTSOE_TABLE = "dna_prod_silver.entsoe.hydro_reservoir_level"
# ENTSO-E map code -> dashboard hydro area (CH and NO come from SFOE / NVE instead)
ENTSOE_AREAS = {a: a for a in ("FR", "AT", "IT", "ES", "SE", "FI", "BG", "GR", "HR", "RO", "RS", "MK", "SI")}


def _get(url: str, timeout: int = 90) -> bytes:
    """GET with TLS verification; behind the corporate proxy that can fail on the
    proxy's certificate, so retry once unverified (public, read-only data)."""
    import requests
    try:
        r = requests.get(url, timeout=timeout)
    except requests.exceptions.SSLError as e:
        log.warning("TLS verification failed for %s (%s) — retrying unverified", url.split("/")[2], str(e)[:120])
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        r = requests.get(url, timeout=timeout, verify=False)
    r.raise_for_status()
    return r.content


def fetch_sfoe() -> pd.DataFrame:
    d = pd.read_csv(io.BytesIO(_get(SFOE_URL)), parse_dates=["Datum"])
    out = pd.DataFrame({"area": "CH", "day": d["Datum"].dt.date,
                        "value": d["TotalCH_speicherinhalt_gwh"].astype(float),
                        "capacity": d["TotalCH_max_speicherinhalt_gwh"].astype(float), "source": "SFOE"})
    return out.dropna(subset=["value"])


def fetch_nve() -> pd.DataFrame:
    d = pd.read_json(io.BytesIO(_get(NVE_URL)))
    d = d[d["omrType"] == "NO"]
    out = pd.DataFrame({"area": "NO", "day": pd.to_datetime(d["dato_Id"]).dt.date,
                        "value": d["fylling_TWh"].astype(float) * 1000.0,
                        "capacity": d["kapasitet_TWh"].astype(float) * 1000.0, "source": "NVE"})
    return out.dropna(subset=["value"])


def fetch_entsoe(writer) -> pd.DataFrame:
    """The latest revision of every (area, ISO week), stamped on the week's Sunday."""
    codes = ",".join(f"'{c}'" for c in ENTSOE_AREAS)
    df = writer.query_df(f"""
        SELECT area_map_code AS area, CAST(year AS INT) AS year, CAST(week AS INT) AS week,
               stored_energy / 1000.0 AS value
        FROM {ENTSOE_TABLE}
        WHERE area_map_code IN ({codes}) AND unit = 'MWh' AND stored_energy IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY area_map_code, year, week
                                   ORDER BY updated_at_source DESC, updated_at DESC) = 1
    """)
    if df.empty:
        return pd.DataFrame(columns=[c for c, _ in COLUMNS if c != "loaded_at"])
    days = [dt.date.fromisocalendar(int(y), int(w), 7) for y, w in zip(df["year"], df["week"])]
    return pd.DataFrame({"area": df["area"].map(ENTSOE_AREAS), "day": days, "value": df["value"].astype(float),
                         "capacity": float("nan"), "source": "ENTSO-E"})


def nordic_sum(parts: pd.DataFrame) -> pd.DataFrame:
    """NP = NO + SE + FI on the days all three report."""
    p = parts[parts["area"].isin(["NO", "SE", "FI"])].pivot_table(index="day", columns="area", values="value")
    if not {"NO", "SE", "FI"} <= set(p.columns):
        return pd.DataFrame(columns=parts.columns)
    s = p[["NO", "SE", "FI"]].dropna().sum(axis=1)
    return pd.DataFrame({"area": "NP", "day": s.index, "value": s.values, "capacity": float("nan"),
                         "source": "NVE + ENTSO-E"})


def run(settings, writer, backfill: bool = False, today: dt.date | None = None) -> dict:
    parts, counts = [], {}
    for name, fn in (("SFOE", fetch_sfoe), ("NVE", fetch_nve), ("ENTSO-E", lambda: fetch_entsoe(writer))):
        try:
            p = fn()
            parts.append(p)
            counts[name] = len(p)
            if len(p):
                log.info("%s: %d weeks, %s → %s", name, len(p), min(p["day"]), max(p["day"]))
        except Exception as e:   # one source down must not drop the others
            log.warning("%s failed: %s", name, str(e)[:200])
            counts[name] = f"FAILED {str(e)[:80]}"
    if not parts:
        raise RuntimeError("no official source could be read")
    df = pd.concat(parts, ignore_index=True)
    df = pd.concat([df, nordic_sum(df)], ignore_index=True)
    df = df.drop_duplicates(KEYS, keep="first").sort_values(KEYS).reset_index(drop=True)
    df["loaded_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)

    table = f"{settings.schema}.{TABLE_NAME}"
    writer.ensure_table(table, COLUMNS, "Official reservoir content per country, weekly (SFOE, NVE, ENTSO-E 16.1.D)")
    n = writer.merge(table, df, KEYS)
    counts["rows"] = n
    counts["areas"] = int(df["area"].nunique())
    return counts


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline_config import load_settings, setup_logging
    from dbx_upload import DatabricksWriter, DryRunWriter
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    settings = load_settings()
    setup_logging(settings.log_dir)
    if a.dry_run:
        # ENTSO-E is read from Databricks even in a dry run (read-only); only the write is skipped
        reader = DatabricksWriter.from_settings(settings)
        writer = DryRunWriter(settings.out_dir)
        writer.query_df = reader.query_df
    else:
        writer = DatabricksWriter.from_settings(settings)
    try:
        log.info("result: %s", run(settings, writer))
    finally:
        if hasattr(writer, "close"):
            writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
