"""Daily hydro pipeline — run from run_daily.bat (Task Scheduler) in the snow_obs env.

Steps, each isolated so one failure does not stop the others:
  1. swe-model   run Hydro_Report/SWE_Exolabs/Scripts/SWE_main.py (what 102_SWE_lorenzo.bat does)
  2. swe         upload the model's CSVs to {schema}.swe_daily
  3. rivers      Energy Quantified river temperatures and flows to {schema}.river_temp_eq / river_flow_eq
  4. morning     Energy Quantified Morning Call input (every model, every cycle) to {schema}.morning_daily_eq

    python run_daily.py                      everything, incremental
    python run_daily.py --skip-swe-model     upload only (the model already ran today)
    python run_daily.py --backfill           full history for both tables (first run)
    python run_daily.py --dry-run            no Databricks: frames to pipeline/out/, SQL logged
    python run_daily.py --only rivers
    python run_daily.py --only morning       what run_morning.bat runs every couple of hours

Exit code 0 when every requested step succeeded, 1 otherwise. Logs in pipeline/logs/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_config import load_settings, setup_logging  # noqa: E402

log = logging.getLogger("pipeline")
STEPS = ["swe-model", "swe", "rivers", "morning"]


def run_swe_model(settings) -> None:
    """SWE_main.py as a subprocess of this interpreter (snow_obs), from its own
    folder so `import download_SWE` resolves, with a non-interactive matplotlib
    backend because Task Scheduler has no desktop."""
    script = settings.swe_main_py
    if not script.exists():
        raise FileNotFoundError(f"SWE_main.py not found at {script}")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    log.info("running %s (timeout %d min)", script, settings.swe_model_timeout_min)
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(script)], cwd=str(script.parent), env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=settings.swe_model_timeout_min * 60)
    tail = "\n".join((proc.stdout or "").splitlines()[-25:])
    log.info("SWE_main.py finished in %.0f s, rc=%d\n--- last lines ---\n%s", time.time() - t0, proc.returncode, tail)
    if proc.returncode != 0:
        err = "\n".join((proc.stderr or "").splitlines()[-25:])
        raise RuntimeError(f"SWE_main.py exited with {proc.returncode}\n{err}")


def make_writer(settings, dry_run: bool):
    from dbx_upload import DatabricksWriter, DryRunWriter
    if dry_run or not settings.databricks_ready():
        if not dry_run:
            log.warning("no Databricks credentials in pipeline/.env — falling back to a dry run (frames to %s)",
                        settings.out_dir)
        return DryRunWriter(settings.out_dir, settings.merge_chunk_rows)
    return DatabricksWriter.from_settings(settings)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=STEPS, help="run a single step")
    ap.add_argument("--skip-swe-model", action="store_true")
    ap.add_argument("--skip-swe", action="store_true")
    ap.add_argument("--skip-rivers", action="store_true")
    ap.add_argument("--skip-morning", action="store_true")
    ap.add_argument("--backfill", action="store_true", help="full history instead of the incremental window")
    ap.add_argument("--dry-run", action="store_true", help="do not touch Databricks; write frames to pipeline/out/")
    ap.add_argument("--no-heightbands", action="store_true", help="skip the per-catchment height-band rows")
    a = ap.parse_args(argv)

    settings = load_settings()
    if a.no_heightbands:
        settings.swe_include_heightbands = False
    log_path = setup_logging(settings.log_dir)
    log.info("hydro pipeline start — schema %s, log %s", settings.schema, log_path)

    steps = [a.only] if a.only else [s for s, skip in zip(STEPS, [a.skip_swe_model, a.skip_swe, a.skip_rivers, a.skip_morning])
                                    if not skip]
    results: dict[str, str] = {}
    writer = None
    try:
        if "swe-model" in steps:
            try:
                run_swe_model(settings)
                results["swe-model"] = "ok"
            except Exception as e:
                log.exception("swe-model failed")
                results["swe-model"] = f"FAILED: {str(e)[:200]}"
        if "swe" in steps or "rivers" in steps or "morning" in steps:
            writer = make_writer(settings, a.dry_run)
        if "swe" in steps:
            try:
                import swe_upload
                results["swe"] = f"ok {swe_upload.run(settings, writer, backfill=a.backfill)}"
            except Exception as e:
                log.exception("swe upload failed")
                results["swe"] = f"FAILED: {str(e)[:200]}"
        if "rivers" in steps:
            try:
                import eq_river_temps
                results["rivers"] = f"ok {eq_river_temps.run(settings, writer, backfill=a.backfill)}"
            except Exception as e:
                log.exception("rivers failed")
                results["rivers"] = f"FAILED: {str(e)[:200]}"
        if "morning" in steps:
            try:
                import eq_morning
                results["morning"] = f"ok {eq_morning.run(settings, writer, backfill=a.backfill)}"
            except Exception as e:
                log.exception("morning failed")
                results["morning"] = f"FAILED: {str(e)[:200]}"
    finally:
        if writer is not None:
            writer.close()

    log.info("summary %s:\n  %s", dt.date.today(), "\n  ".join(f"{k:10s} {v}" for k, v in results.items()))
    return 0 if results and all(not v.startswith("FAILED") for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
