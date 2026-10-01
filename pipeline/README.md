# Hydro pipeline — SWE model and river temperatures → Databricks

A local, daily job that feeds two inputs the Hydro overview can show but Databricks does not have:

| Feed | Source | Sandbox table | Rows / day |
|------|--------|---------------|------------|
| Snow water equivalent | the internal Exolabs model — `Hydro_Report/SWE_Exolabs/Scripts/SWE_main.py`, run by `102_SWE_lorenzo.bat` — read from its CSVs in `Output_files/CSVs` | `dna_snbx_weather.power_desk.swe_daily` | Alps + 4 countries × 3 bands + 34 catchments + 34 × 10 height bands ≈ 390 (× 30 days re-uploaded) |
| River temperatures | Energy Quantified, the client and key of `Lorenzo_Trainee/EQ_project/eq_fundamentals.py`; curves `FR @<River>-<Site> River Temperature °C H Backcast / Normal / Forecast` (Golfech, Belleville, Chooz, Cattenom, Fessenheim, Bugey, Saint-Alban, Tricastin — discovered from EQ's metadata, so a new station appears by itself; `RIVER_AREAS=FR,DE,HU` adds the German and Hungarian stations) | `…power_desk.river_temp_eq`, `…power_desk.river_stations_eq` | 8 stations × (45 backcast days + 45 + 400 normal days + 2 forecast issues) ≈ 4 000 |

Nothing is computed here: the SWE numbers are the model's, the river values are EQ's daily means. Uploads are `MERGE`s keyed on the natural key, so re-running a day updates instead of duplicating (the SWE model re-reads its last 14 days and EQ revises backcasts).

```
pipeline/
├── run_daily.bat        what Task Scheduler runs: activates snow_obs, calls run_daily.py
├── run_daily.py         orchestrator — steps swe-model · swe · rivers, each isolated; exit 1 if any failed
├── pipeline_config.py   settings from pipeline/.env (copy env_example.txt) and the environment
├── dbx_upload.py        MERGE-into-Delta writer over the Databricks SQL connector (+ a dry-run writer)
├── swe_upload.py        SWE CSVs → swe_daily
├── eq_river_temps.py    EQ curves → river_temp_eq / river_stations_eq
├── env_example.txt      template for .env / local_settings.txt (the real one is git-ignored)
└── logs/ out/           one log per day · dry-run frames and example SQL (both git-ignored)
```

## Setup (once)

1. Environment: `snow_obs` (miniforge) already has everything — `energyquantified`, `databricks-sql-connector`, pandas — because it is the SWE model's environment. Nothing to install.
2. Credentials: copy `env_example.txt` to `pipeline/.env` (or `pipeline/local_settings.txt`, same format, for tools that cannot create dot-files), set `DATABRICKS_HOST` (workspace host without `https://`) and `DATABRICKS_TOKEN` (a personal access token; the warehouse HTTP path defaults to the desk apps' warehouse). `snow_obs` carries `databricks-sql-connector` 2.0.2, which only knows token auth — `DATABRICKS_AUTH_TYPE` (OAuth / Azure CLI) needs connector ≥ 3 (`pip install -U databricks-sql-connector` in the env). The EQ key is read from `eq_fundamentals.py` unless `EQ_API_KEY` is set. `.env` is git-ignored. Tables created by the pipeline sit in the same schema the notebook grants to the app's service principal, so the app reads them without further grants.
3. First load, from a `cmd` window:
   ```
   P:\QFA\TonyWeather\Power_dashboard\pipeline\run_daily.bat --skip-swe-model --backfill
   ```
   `--backfill` uploads the whole SWE history (2017 →: Alps + countries + catchments ≈ 157 k rows, ~6 min in 2 000-row MERGEs; with height bands another ~1.1 M rows, ~45 min — hence `--no-heightbands` for the first load; the daily run then fills the height bands' last 30 days, or run `--only swe --backfill` later for their full history) and every river curve from `RIVER_HISTORY_START` (2014; EQ's river backcasts begin in 2015; ≈ 70 k rows, ~5 min). Later runs are incremental and take a couple of minutes. A curve that is new to the table is backfilled automatically. First load done on 2026-10-01 from this repo.
4. Schedule: Task Scheduler → Create Task → *Run whether user is logged on or not* → Action `cmd.exe` with arguments `/c "P:\QFA\TonyWeather\Power_dashboard\pipeline\run_daily.bat"` → daily trigger at **11:30** (Exolabs publishes the day's raster around 10:45; before that `SWE_main.py` processes yesterday). The P: drive must be mapped for the task's account, or use the UNC path `\\vfbdn111.prod.axponet.ch\Projekte$\QFA\TonyWeather\...` like the model's own `.bat` does.

Check `pipeline/logs/pipeline_YYYYMMDD.log`; the last block is a per-step summary.

## What each step does

**swe-model** runs `SWE_main.py` as a subprocess of the same interpreter from its own folder (so `import download_SWE` resolves), with `MPLBACKEND=Agg` because a scheduled task has no desktop. That script downloads the missing Exolabs rasters from S3, cuts them to countries / catchments / height bands and appends to the CSVs — unchanged. Skip it with `--skip-swe-model` when the model already ran that day.

**swe** reads the CSVs into one long table, `swe_daily` (`day, level, region, band, swe_total, swe_mean_mm, swe_total_roll, swe_mean_roll_mm, pixel_size_m, winter_season, day_of_winter`). `level` is `alps` (region `Alps`), `country` (Austria, Italy, France, Switzerland; band `total`, `below 1800`, `above 1800`), `catchment` (34 Axpo catchments) or `heightband` (band `1`…`10` per catchment). `t-*.csv` is the pixel **sum** of SWE (kg/m² per pixel, summed — a volume proxy), `m-*.csv` the pixel **mean** in mm; `_roll` is the model's 7-day centred mean. Pixel size is 300 m for the Alps and countries (the model resamples the 20 m product by `xdim/20`) and 20 m for catchments and height bands, so volume in m³ ≈ `swe_total × pixel_size_m² / 1000`. `winter_season` / `day_of_winter` follow `SWE_main.py` (season starts 1 Oct, ends July). Each run re-uploads the last `SWE_REFRESH_DAYS` (30, more than the model's 14-day refresh so the rolling means settle).

**rivers** discovers every `River Temperature` curve of `RIVER_AREAS` in EQ's metadata and loads, per curve type, at daily resolution (`P1D`, `AVERAGE`, the curve's CET days):

| EQ type | Loaded as | Window |
|---------|-----------|--------|
| Backcast, Actual (TIMESERIES) | `data_type = 'backcast' / 'actual'` | last `RIVER_BACKCAST_DAYS` (45); from `RIVER_HISTORY_START` on `--backfill` or for a curve not yet in the table |
| Normal (TIMESERIES) | `'normal'` | same start, to today + `RIVER_NORMAL_AHEAD_DAYS` (EQ publishes the normal to the end of the current year) |
| Forecast (INSTANCE) | `'forecast'`, with `issued` (UTC) and `tag` | the latest issue per tag — EQ has `ec-ens` (15 days, 00z and 12z) and `ec-ext` (45 days); `RIVER_FORECAST_RUNS` > 1 keeps more issues per run |

MERGE key: `(curve_name, data_type, day, issued, tag)` — `issued` is NULL and `tag` empty for the timeseries types, and the ON clause is null-safe. `river_stations_eq` holds the stations' coordinates (EQ places) for the map.

## Reading the tables

Latest observed river temperature against its normal, per station:

```sql
WITH b AS (SELECT station_key, MAX(day) AS day FROM dna_snbx_weather.power_desk.river_temp_eq WHERE data_type = 'backcast' GROUP BY station_key)
SELECT r.station, r.day, r.value AS backcast_c, n.value AS normal_c, r.value - n.value AS anomaly_c
FROM dna_snbx_weather.power_desk.river_temp_eq r
JOIN b ON b.station_key = r.station_key AND b.day = r.day AND r.data_type = 'backcast'
LEFT JOIN dna_snbx_weather.power_desk.river_temp_eq n ON n.curve_name = REPLACE(r.curve_name, 'Backcast', 'Normal') AND n.day = r.day
```

Latest forecast issue per station and tag: `WHERE data_type = 'forecast' QUALIFY issued = MAX(issued) OVER (PARTITION BY curve_name, tag)`.

Country SWE for the overview map: `SELECT day, region, swe_mean_mm FROM swe_daily WHERE level = 'country' AND band = 'total'`.

## Verified

- `dbx_upload`: the generated `CREATE TABLE` / `MERGE` run on DuckDB (same statements, dialect switch only for `<=>` → `IS NOT DISTINCT FROM`, the `D` double suffix and the VALUES parentheses): first MERGE inserts, second updates a revised backcast and adds a new forecast issue without touching the old one, duplicate source keys collapse to the last, NULLs are typed (`CAST(NULL AS TIMESTAMP)`) so a batch without forecasts still writes.
- `eq_river_temps`: discovery and loads run against EQ from this machine (24 FR curves, 8 stations with coordinates; daily backcast, normal to 31 Dec, `ec-ens` / `ec-ext` issues) in `--dry-run`.
- `swe_upload`: the real CSVs parse into the long table in `--dry-run` (see `out/`).
- Not verified here: the Databricks connection itself — this machine has no token. The first real run is the test; a failing MERGE prints the statement's first lines in the log.
