# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Power Desk Sandbox Refresh
"""
Power Desk Sandbox Refresh
==========================
Populates dna_snbx_weather.power_desk.* from the Volue delta-share tables and
the Meteomatics silver layer. Schedule: every 6 hours via Lakeflow Jobs
(after the 00z / 12z EC-ENS runs land, ~05:00 and ~17:00 UTC, plus two
in-between refreshes for the hydro / historical legs).

Tables written (all read by weather-power-desk-app/_data.py):
  pop_weights           persons per country per 0.5° grid point (GHS-POP 2025),
                        loaded once from pop_weights_0p5.csv — every Meteomatics
                        country mean below is population-weighted with it
  fcst_runs             one row per (model_family, pattern, reference_date) kept
  fcst_daily            daily ensemble stats per provider/model/run/metric/area:
                        ens_mean, p10..p90, spread_std, min, max, n_members,
                        normal, anomaly, lead_day
  fcst_spread_clim      "normal" ensemble spread per metric/area/lead_day
                        (and per run month) from the last 365 days of EC-ENS runs
  fcst_members          latest EC-ENS run, per member daily values (Volue) and
                        Meteomatics EC-ENS / AIFS-ENS members as population-
                        weighted country means
  hist_daily            Volue actuals + normal per metric/area/day since 2013
  hydro_daily           Volue hydro reservoir components (WTR/SGW/BAL),
                        actual (SA) and normal (N), per area/day since 2013
  gas_demand_daily      daily ens-mean ('Avg') temperature / wind / solar with
                        the normal, per pattern and run, last 8 days of runs —
                        feeds the Gas Demand section (LDZ + wind/solar RDL)
  wr_patterns           the 7 fixed weather-regime patterns on the 121 x 241
  wr_norm / wr_amplitude  domain, their normalisation constants and the
                        day-of-year amplitude — built once from wr_patterns.npz
  wr_gph_clim           ERA5 day-of-year mean Z500 per grid point (once)
  wr_forecast_members   per run / member / lead day: 7 IWR values + regime
  wr_reanalysis_daily   ERA5 classified into regimes, for the climatology
  weather_indexes       CREATE IF NOT EXISTS — loaded separately
  meteologica_members   CREATE IF NOT EXISTS — populated only if METEOLOGICA_TABLE set

Lookup tables below mirror weather-power-desk-app/_config.py — update both.
"""
import os
from datetime import datetime

VOLUE = "dna_prod_silver.volue"               # set to the delta-share catalog if mounted elsewhere
MM = "dna_prod_silver.meteomatics"
SBX = "dna_snbx_weather.power_desk"
APP_SP = "82058742-a661-4413-9d6c-9fffe0f6e45d"  # weather-power-desk-app service principal

METEOMATICS_MEMBER_TABLE = ""   # e.g. "dna_snbx_weather.meteomatics_members.temperature_forecast_members"
METEOLOGICA_TABLE = ""          # e.g. "dna_snbx_weather.meteologica.ensext_daily"

print(f"Refresh started: {datetime.now()}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SBX}")
try:
    spark.sql(f"GRANT USE CATALOG ON CATALOG dna_snbx_weather TO `{APP_SP}`")
    spark.sql(f"GRANT USE SCHEMA, SELECT ON SCHEMA {SBX} TO `{APP_SP}`")
    print("Grants re-applied for the app SP")
except Exception as e:  # governance jobs sometimes own these grants
    print(f"Grant skipped: {e}")

# COMMAND ----------

# DBTITLE 1,Lookup tables (mirror _config.py)
AREAS = ["DE", "FR", "IT", "ES", "UK", "NL", "BE", "PL", "CZ", "HU", "NO", "SE", "FI", "DK",
         "PT", "SI", "SK", "HR", "EE", "LV", "LT"]
AREA_SQL = ",".join(f"'{a}'" for a in AREAS)

METRICS = {
    # name: (category, forecast table, history table, scale to display unit)
    "Temperature":          ("TT",  "temperature_consumption_forecast", "temperature_consumption", 1.0),
    "Wind":                 ("WND", "production_forecast",              "production",              0.001),
    "Solar":                ("SPV", "production_forecast",              "production",              0.001),
    "Precipitation energy": ("RRE", "precipitation_forecast",           "precipitation",           1.0),
}

VOLUE_MODELS = {
    "EC-ENS":      ["ec00ens", "ec12ens"],
    "EC-Extended": ["ecmonthly"],
    "GFS-ENS":     ["gfs00ens"],
}
N_RUNS_KEPT = 6

METEOMATICS_MODELS = {
    "Meteomatics EC-ENS":   ("ecmwf-ens",      "t_mean_2m_24h_c_ecmwf_ens_p1d"),
    "Meteomatics AIFS-ENS": ("ecmwf-aifs-ens", "t_mean_2m_24h_c_ecmwf_aifs_ens_p1d"),
}

COUNTRY_BBOX = {
    "DE": (47.5, 55.0, 6.0, 15.0),   "FR": (42.5, 51.0, -4.5, 8.0),   "IT": (37.0, 47.0, 7.0, 18.5),
    "ES": (36.0, 43.5, -9.5, 3.0),   "UK": (50.0, 58.5, -7.5, 1.5),   "NL": (50.5, 53.5, 3.5, 7.0),
    "BE": (49.5, 51.5, 2.5, 6.5),    "PL": (49.0, 54.5, 14.0, 24.0),  "CZ": (48.5, 51.0, 12.0, 18.5),
    "HU": (45.5, 48.5, 16.0, 22.5),  "NO": (58.0, 64.0, 5.0, 12.0),   "SE": (55.5, 64.0, 11.5, 19.0),
    "FI": (60.0, 66.0, 21.0, 30.0),  "DK": (54.5, 57.5, 8.0, 12.5),   "PT": (37.0, 42.0, -9.5, -6.5),
    "SI": (45.5, 46.5, 13.5, 16.5),  "SK": (47.5, 49.5, 17.0, 22.5),  "HR": (42.5, 46.5, 13.5, 19.5),
    "EE": (57.5, 59.5, 22.0, 28.0),  "LV": (55.5, 58.0, 21.0, 28.0),  "LT": (54.0, 56.5, 21.0, 26.5),
}

HYDRO_AREAS = ["FR", "CH", "AT", "IT", "NP", "ES", "SEE", "DE", "NO", "SE", "FI"]
HYDRO_COMPONENTS = ["WTR", "SGW", "BAL"]


def country_case_sql(lat="latitude", lon="longitude") -> str:
    lines = ["CASE"]
    for code, (la0, la1, lo0, lo1) in COUNTRY_BBOX.items():
        lines.append(f"  WHEN {lat} >= {la0} AND {lat} <= {la1} AND {lon} >= {lo0} AND {lon} <= {lo1} THEN '{code}'")
    lines.append("END")
    return "\n".join(lines)


COUNTRY_CASE = country_case_sql()
CET_DAY = "DATE(from_utc_timestamp(delivery_start, 'CET'))"

# Meteomatics country means are POPULATION-WEIGHTED (cell 1a, build_pop_weights.py):
# {SBX}.pop_weights holds, per country and 0.5° grid point, the persons of that
# country in that cell, and a country mean is SUM(value * population) /
# SUM(population) over the cells that join. COUNTRY_BBOX / COUNTRY_CASE above are
# only the fallback for when the weights have not been loaded yet.
POP_WEIGHTS_PATH = os.environ.get(
    "POP_WEIGHTS_PATH",
    "/Workspace/Users/lorenzo.comi@axpo.com/Meteo-Power-Dashboard/weather-power-desk-app/pop_weights_0p5.csv")
POP_GRID_RES = 0.5                      # the weights' grid step — the Meteomatics silver grid
# Pruning box for every gridded Meteomatics read (alias `t`): the weights' extent.
MM_BBOX = "t.latitude BETWEEN 34 AND 72 AND t.longitude BETWEEN -12 AND 36"


def count(table: str) -> None:
    n = spark.sql(f"SELECT COUNT(*) FROM {SBX}.{table}").first()[0]
    print(f"{table}: {n:,} rows")


# COMMAND ----------

# DBTITLE 1,1a. Population weights for Meteomatics country means (loaded once)
# Built offline by build_pop_weights.py from GHS-POP 2025 (JRC GHSL, ~1 km) and
# Europe_merged.shp: per country and 0.5° grid point, the persons of that country
# living in that cell. Population is assigned to countries at ~1 km and only then
# summed into the 0.5° cells, so a border cell carries both countries' shares and
# coastal population is not lost with a cell whose centre happens to be at sea.
#
# A country mean of a gridded Meteomatics field is then the population-weighted
# mean over the cells that join — the consumption-weighted temperature the desk
# reads (Morning_Report/meteomatics_morning_report.py), not the arithmetic mean
# over a bounding box. Persons rather than fractions are stored so that a group
# of countries (Nordics, Iberia, SEE) is weighted right by summing over the group.
if spark.catalog.tableExists(f"{SBX}.pop_weights"):
    print("pop_weights already loaded — skipping (drop the table to reload a new file)")
elif os.path.exists(POP_WEIGHTS_PATH):
    import pandas as _pd
    _w = _pd.read_csv(POP_WEIGHTS_PATH)
    assert {"area", "latitude", "longitude", "population"} <= set(_w.columns), list(_w.columns)
    spark.createDataFrame(_w).write.mode("overwrite").saveAsTable(f"{SBX}.pop_weights")
    print(f"pop_weights: {len(_w):,} cells, {_w['area'].nunique()} countries, "
          f"{_w['population'].sum() / 1e6:,.1f} M persons")
else:
    print(f"!! {POP_WEIGHTS_PATH} not found — Meteomatics country means fall back to bounding-box "
          "averages until the file is uploaded and this cell re-run")
POP_WEIGHTED = spark.catalog.tableExists(f"{SBX}.pop_weights")


def snap(col: str) -> str:
    """Snap a silver-table coordinate onto the weights' grid, so the join does not
    depend on the two tables' floats agreeing to the last bit."""
    k = round(1 / POP_GRID_RES)
    return f"ROUND(CAST({col} AS DOUBLE) * {k}) / {k}"


# The fragments every Meteomatics country mean is written with. The gridded source
# is aliased `t`; `area` and `mean` go into the SELECT, `join` right after the FROM.
if POP_WEIGHTED:
    CM = {"join": f"JOIN {SBX}.pop_weights w ON w.latitude = {snap('t.latitude')} "
                  f"AND w.longitude = {snap('t.longitude')}",
          "area": "w.area", "weight": "w.population", "method": "population-weighted"}
else:
    CM = {"join": "", "area": country_case_sql("t.latitude", "t.longitude"), "weight": "1.0",
          "method": "bounding-box average"}
CM["mean"] = f"SUM(t.value * {CM['weight']}) / SUM({CM['weight']})"
print(f"Meteomatics country means: {CM['method']}")

# Do the weights meet the silver grid? Every populated cell should find a forecast
# point; a misaligned grid would silently drop people, so say so loudly.
if POP_WEIGHTED:
    _model = next(iter(METEOMATICS_MODELS.values()))[0]
    _cov = spark.sql(f"""
        WITH g AS (
          SELECT DISTINCT {snap('t.latitude')} AS latitude, {snap('t.longitude')} AS longitude
          FROM {MM}.temperature_forecast t
          WHERE t.model = '{_model}' AND {MM_BBOX}
            AND t.created_at = (SELECT MAX(created_at) FROM {MM}.temperature_forecast WHERE model = '{_model}')
        )
        SELECT SUM(w.population) AS pop,
               SUM(CASE WHEN g.latitude IS NOT NULL THEN w.population ELSE 0 END) AS pop_matched
        FROM {SBX}.pop_weights w
        LEFT JOIN g ON g.latitude = w.latitude AND g.longitude = w.longitude
    """).first()
    _share = (_cov["pop_matched"] or 0) / _cov["pop"] if _cov["pop"] else 0.0
    print(f"pop_weights coverage by the {_model} grid: {_share:.1%} of the population sits on a forecast point")
    if _share < 0.95:
        print("!! the silver grid does not line up with the 0.5° weights — check POP_GRID_RES and the "
              "latitude/longitude values of temperature_forecast before trusting the country means")

# COMMAND ----------

# DBTITLE 1,1. Forecast runs kept per model family
# Volue's reference_date is the run's ISSUE DAY at midnight CET, stored in UTC —
# 22:00 (summer) or 23:00 (winter) the evening before — and it is the SAME for
# the 00z and the 12z run of that day. So a run is (CET day of reference_date,
# init_hour), init_hour coming from the pattern name (ec00ens → 0, ec12ens → 12),
# and runs are ranked on exactly that: ordering on reference_date alone put a
# 12z run ahead of the next morning's 00z and made "Latest -1 / -2 / -3" jump
# between dates. The app derives the same init time (_data.volue_init_time).
INIT_DAY = "DATE(from_utc_timestamp(reference_date, 'CET'))"
run_unions = []
for family, patterns in VOLUE_MODELS.items():
    for pat in patterns:
        init_hour = 12 if "12" in pat else 0
        run_unions.append(f"""
        SELECT '{family}' AS model_family, '{pat}' AS pattern,
               {init_hour} AS init_hour, reference_date
        FROM {VOLUE}.temperature_consumption_forecast
        WHERE curve_name LIKE '%{pat}%' AND data_type = 'F' AND tag = 'Avg'
          AND array_contains(categories, 'TT')
          AND reference_date >= current_timestamp() - INTERVAL 14 DAYS
        GROUP BY reference_date
        """)

spark.sql(f"""
CREATE OR REPLACE TABLE {SBX}.fcst_runs AS
WITH all_runs AS ({" UNION ALL ".join(run_unions)}),
ranked AS (
  SELECT model_family, pattern, init_hour, reference_date, {INIT_DAY} AS init_day,
         DENSE_RANK() OVER (PARTITION BY model_family
                            ORDER BY {INIT_DAY} DESC, init_hour DESC) AS run_rank
  FROM all_runs
)
SELECT model_family, pattern, init_hour, reference_date, init_day, run_rank,
       CASE WHEN run_rank = 1 THEN 'Latest' ELSE CONCAT('Latest -', run_rank - 1) END AS run_label,
       current_timestamp() AS snapshot_ts
FROM ranked WHERE run_rank <= {N_RUNS_KEPT}
""")
count("fcst_runs")

# COMMAND ----------

# DBTITLE 1,2. Daily ensemble statistics per run / metric / area (Volue)
metric_blocks = []
for name, (cat, fcst_tbl, hist_tbl, scale) in METRICS.items():
    metric_blocks.append(f"""
    SELECT 'Volue' AS provider, r.model_family, r.pattern, r.init_hour, r.reference_date, r.run_rank, r.run_label,
           '{name}' AS metric, f.area, {CET_DAY} AS day, f.tag, AVG(f.value) * {scale} AS value
    FROM {VOLUE}.{fcst_tbl} f
    JOIN {SBX}.fcst_runs r
      ON f.curve_name LIKE CONCAT('%', r.pattern, '%')
     AND f.reference_date BETWEEN r.reference_date - INTERVAL 3 HOURS AND r.reference_date + INTERVAL 3 HOURS
    WHERE array_contains(f.categories, '{cat}') AND f.data_type = 'F'
      AND f.area IN ({AREA_SQL})
      AND f.delivery_start >= current_date() - INTERVAL 2 DAYS
    GROUP BY r.model_family, r.pattern, r.init_hour, r.reference_date, r.run_rank, r.run_label, f.area, {CET_DAY}, f.tag
    """)

normal_blocks = []
for name, (cat, fcst_tbl, hist_tbl, scale) in METRICS.items():
    normal_blocks.append(f"""
    SELECT '{name}' AS metric, area, {CET_DAY} AS day, AVG(value) * {scale} AS normal
    FROM {VOLUE}.{hist_tbl}
    WHERE array_contains(categories, '{cat}') AND data_type = 'N'
      AND area IN ({AREA_SQL})
      AND delivery_start BETWEEN current_date() - INTERVAL 2 DAYS AND current_date() + INTERVAL 50 DAYS
    GROUP BY area, {CET_DAY}
    """)

spark.sql(f"""
CREATE OR REPLACE TABLE {SBX}.fcst_daily AS
WITH member_daily AS ({" UNION ALL ".join(metric_blocks)}),
agg AS (
  SELECT provider, model_family, pattern, init_hour, reference_date, run_rank, run_label, metric, area, day,
         MAX(CASE WHEN tag = 'Avg' THEN value END)                          AS ens_mean,
         percentile_approx(CASE WHEN tag != 'Avg' THEN value END, 0.10)     AS p10,
         percentile_approx(CASE WHEN tag != 'Avg' THEN value END, 0.25)     AS p25,
         percentile_approx(CASE WHEN tag != 'Avg' THEN value END, 0.50)     AS p50,
         percentile_approx(CASE WHEN tag != 'Avg' THEN value END, 0.75)     AS p75,
         percentile_approx(CASE WHEN tag != 'Avg' THEN value END, 0.90)     AS p90,
         STDDEV(CASE WHEN tag != 'Avg' THEN value END)                      AS spread_std,
         MIN(CASE WHEN tag != 'Avg' THEN value END)                         AS ens_min,
         MAX(CASE WHEN tag != 'Avg' THEN value END)                         AS ens_max,
         COUNT(DISTINCT CASE WHEN tag != 'Avg' THEN tag END)                AS n_members
  FROM member_daily
  GROUP BY provider, model_family, pattern, init_hour, reference_date, run_rank, run_label, metric, area, day
),
normals AS ({" UNION ALL ".join(normal_blocks)})
SELECT a.provider, a.model_family, a.pattern, a.init_hour, a.reference_date, a.run_rank, a.run_label,
       a.metric, a.area, a.day,
       DATEDIFF(a.day, DATE(a.reference_date)) AS lead_day,
       COALESCE(a.ens_mean, a.p50) AS ens_mean,
       a.p10, a.p25, a.p50, a.p75, a.p90, a.spread_std, a.ens_min, a.ens_max, a.n_members,
       n.normal,
       COALESCE(a.ens_mean, a.p50) - n.normal AS anomaly,
       CASE WHEN n.normal IS NOT NULL AND n.normal != 0
            THEN (COALESCE(a.ens_mean, a.p50) / n.normal - 1) * 100 END AS anomaly_pct,
       current_timestamp() AS snapshot_ts
FROM agg a
LEFT JOIN normals n ON n.metric = a.metric AND n.area = a.area AND n.day = a.day
""")
count("fcst_daily")

# COMMAND ----------

# DBTITLE 1,2b. Meteomatics EC-ENS / AIFS-ENS ensemble means as population-weighted country means
# The silver Meteomatics tables' ensemble mean, reduced to one value per country
# with the population weights of cell 1a (CM fragments), so they contribute a
# second and third "mean" line to the forecast values view.
mm_blocks = []
for label, (model, curve) in METEOMATICS_MODELS.items():
    mm_blocks.append(f"""
    SELECT 'Meteomatics' AS provider, '{label}' AS model_family, '{model}' AS pattern,
           t.created_at AS reference_date, 1 AS run_rank, 'Latest' AS run_label,
           'Temperature' AS metric, {CM['area']} AS area, CAST(t.delivery_start AS DATE) AS day,
           {CM['mean']} AS ens_mean
    FROM {MM}.temperature_forecast t {CM['join']}
    WHERE t.model = '{model}' AND t.curve_name = '{curve}'
      AND t.created_at = (SELECT MAX(created_at) FROM {MM}.temperature_forecast WHERE model = '{model}')
      AND {MM_BBOX}
    GROUP BY t.created_at, {CM['area']}, CAST(t.delivery_start AS DATE)
    """)

spark.sql(f"""
INSERT INTO {SBX}.fcst_daily
WITH mm AS ({" UNION ALL ".join(mm_blocks)}),
normals AS (
  SELECT area, {CET_DAY} AS day, AVG(value) AS normal
  FROM {VOLUE}.temperature_consumption
  WHERE array_contains(categories, 'TT') AND data_type = 'N' AND area IN ({AREA_SQL})
    AND delivery_start BETWEEN current_date() - INTERVAL 2 DAYS AND current_date() + INTERVAL 50 DAYS
  GROUP BY area, {CET_DAY}
)
SELECT m.provider, m.model_family, m.pattern, 0 AS init_hour, m.reference_date, m.run_rank, m.run_label,
       m.metric, m.area, m.day, DATEDIFF(m.day, DATE(m.reference_date)) AS lead_day,
       m.ens_mean, NULL AS p10, NULL AS p25, NULL AS p50, NULL AS p75, NULL AS p90,
       NULL AS spread_std, NULL AS ens_min, NULL AS ens_max, 0 AS n_members,
       n.normal, m.ens_mean - n.normal AS anomaly,
       CASE WHEN n.normal IS NOT NULL AND n.normal != 0 THEN (m.ens_mean / n.normal - 1) * 100 END AS anomaly_pct,
       current_timestamp() AS snapshot_ts
FROM mm m LEFT JOIN normals n ON n.area = m.area AND n.day = m.day
WHERE m.area IS NOT NULL
""")
count("fcst_daily")

# COMMAND ----------

# DBTITLE 1,3. Normal ensemble spread per lead day (uncertainty baseline)
# For every EC-ENS 00z run in the last year: per member daily mean, then the
# across-member std per (metric, area, lead_day). Aggregated two ways — all
# runs, and runs grouped by calendar month — so the app can compare today's
# spread with "a normal spread for this lead day at this time of year".
spread_blocks = []
for name, (cat, fcst_tbl, hist_tbl, scale) in METRICS.items():
    if name == "Precipitation energy":
        continue
    spread_blocks.append(f"""
    SELECT '{name}' AS metric, area, reference_date, tag,
           {CET_DAY} AS day, AVG(value) * {scale} AS value
    FROM {VOLUE}.{fcst_tbl}
    WHERE array_contains(categories, '{cat}') AND data_type = 'F' AND tag != 'Avg'
      AND curve_name LIKE '%ec00ens%'
      AND area IN ({AREA_SQL})
      AND reference_date >= current_timestamp() - INTERVAL 365 DAYS
      AND delivery_start >= reference_date
    GROUP BY area, reference_date, tag, {CET_DAY}
    """)

spark.sql(f"""
CREATE OR REPLACE TABLE {SBX}.fcst_spread_clim AS
WITH member_daily AS ({" UNION ALL ".join(spread_blocks)}),
per_run AS (
  SELECT metric, area, reference_date, DATEDIFF(day, DATE(reference_date)) AS lead_day,
         MONTH(reference_date) AS run_month,
         STDDEV(value) AS spread_std, percentile_approx(value, 0.9) - percentile_approx(value, 0.1) AS spread_p90_p10,
         COUNT(DISTINCT tag) AS n_members
  FROM member_daily
  GROUP BY metric, area, reference_date, DATEDIFF(day, DATE(reference_date)), MONTH(reference_date)
  HAVING COUNT(DISTINCT tag) >= 5
)
SELECT metric, area, lead_day, 0 AS run_month,
       AVG(spread_std) AS spread_std_mean, percentile_approx(spread_std, 0.25) AS spread_std_p25,
       percentile_approx(spread_std, 0.75) AS spread_std_p75,
       AVG(spread_p90_p10) AS spread_p90_p10_mean, COUNT(*) AS n_runs
FROM per_run GROUP BY metric, area, lead_day
UNION ALL
SELECT metric, area, lead_day, run_month,
       AVG(spread_std), percentile_approx(spread_std, 0.25), percentile_approx(spread_std, 0.75),
       AVG(spread_p90_p10), COUNT(*)
FROM per_run GROUP BY metric, area, lead_day, run_month
""")
count("fcst_spread_clim")

# COMMAND ----------

# DBTITLE 1,4. Per-member daily values of the latest EC-ENS run (scenario inputs)
member_blocks = []
for name, (cat, fcst_tbl, hist_tbl, scale) in METRICS.items():
    if name == "Precipitation energy":
        continue
    member_blocks.append(f"""
    SELECT 'Volue' AS provider, 'EC-ENS' AS model, r.reference_date, '{name}' AS metric, f.area,
           {CET_DAY} AS day, f.tag AS member, AVG(f.value) * {scale} AS value
    FROM {VOLUE}.{fcst_tbl} f
    JOIN (SELECT pattern, reference_date FROM {SBX}.fcst_runs WHERE model_family = 'EC-ENS' AND run_rank = 1) r
      ON f.curve_name LIKE CONCAT('%', r.pattern, '%')
     AND f.reference_date BETWEEN r.reference_date - INTERVAL 3 HOURS AND r.reference_date + INTERVAL 3 HOURS
    WHERE array_contains(f.categories, '{cat}') AND f.data_type = 'F' AND f.tag != 'Avg'
      AND f.area IN ({AREA_SQL}) AND f.delivery_start >= current_date()
    GROUP BY r.reference_date, f.area, {CET_DAY}, f.tag
    """)

spark.sql(f"""
CREATE OR REPLACE TABLE {SBX}.fcst_members AS
WITH members AS ({" UNION ALL ".join(member_blocks)}),
normals AS (
  SELECT 'Temperature' AS metric, area, {CET_DAY} AS day, AVG(value) AS normal
  FROM {VOLUE}.temperature_consumption
  WHERE array_contains(categories, 'TT') AND data_type = 'N' AND area IN ({AREA_SQL})
    AND delivery_start BETWEEN current_date() AND current_date() + INTERVAL 20 DAYS
  GROUP BY area, {CET_DAY}
  UNION ALL
  SELECT 'Wind', area, {CET_DAY}, AVG(value) * 0.001
  FROM {VOLUE}.production
  WHERE array_contains(categories, 'WND') AND data_type = 'N' AND area IN ({AREA_SQL})
    AND delivery_start BETWEEN current_date() AND current_date() + INTERVAL 20 DAYS
  GROUP BY area, {CET_DAY}
  UNION ALL
  SELECT 'Solar', area, {CET_DAY}, AVG(value) * 0.001
  FROM {VOLUE}.production
  WHERE array_contains(categories, 'SPV') AND data_type = 'N' AND area IN ({AREA_SQL})
    AND delivery_start BETWEEN current_date() AND current_date() + INTERVAL 20 DAYS
  GROUP BY area, {CET_DAY}
)
SELECT m.provider, m.model, m.reference_date, m.metric, m.area, m.day,
       DATEDIFF(m.day, DATE(m.reference_date)) AS lead_day,
       m.member, m.value, n.normal, m.value - n.normal AS anomaly,
       current_timestamp() AS snapshot_ts
FROM members m LEFT JOIN normals n ON n.metric = m.metric AND n.area = m.area AND n.day = m.day
""")
count("fcst_members")

# COMMAND ----------

# DBTITLE 1,4b. Gridded member spatial anomaly (for spatial k-means clustering)
# ---------------------------------------------------------------------------
# Gridded per-member temperature anomaly for spatial k-means clustering.
#
# TODO(lorenzo): Switch to geopotential_height_forecast (Z500) when ecmwf-ens
# gets geopotential members. Currently only ecmwf-aifs-ens has Z500; the
# standard ecmwf-ens (whose 50 perturbations match Volue member IDs) does not.
# Using temperature spatial anomaly as a proxy until then.
# ---------------------------------------------------------------------------
SPATIAL_MODEL = "ecmwf-ens"
SPATIAL_CURVE = "t_mean_2m_24h_c_ecmwf_ens_p1d"
BB_LAT = (35, 72)
BB_LON = (-12, 35)
GRID_RES = 1.0  # coarsen from native 0.5° to 1° for manageable feature space

spark.sql(f"""
CREATE OR REPLACE TABLE {SBX}.fcst_member_spatial AS
WITH latest AS (
  SELECT MAX(created_at) AS max_ca
  FROM {MM}.temperature_forecast
  WHERE model = '{SPATIAL_MODEL}' AND curve_name = '{SPATIAL_CURVE}'
),
raw AS (
  SELECT curve_member AS member,
         DATE(delivery_start) AS day,
         ROUND(CAST(latitude AS DOUBLE) / {GRID_RES}) * {GRID_RES} AS latitude,
         ROUND(CAST(longitude AS DOUBLE) / {GRID_RES}) * {GRID_RES} AS longitude,
         AVG(value) AS value
  FROM {MM}.temperature_forecast, latest
  WHERE model = '{SPATIAL_MODEL}' AND curve_name = '{SPATIAL_CURVE}'
    AND created_at = latest.max_ca
    AND latitude BETWEEN {BB_LAT[0]} AND {BB_LAT[1]}
    AND longitude BETWEEN {BB_LON[0]} AND {BB_LON[1]}
    AND curve_member IS NOT NULL AND curve_member != ''
  GROUP BY curve_member, DATE(delivery_start),
           ROUND(CAST(latitude AS DOUBLE) / {GRID_RES}) * {GRID_RES},
           ROUND(CAST(longitude AS DOUBLE) / {GRID_RES}) * {GRID_RES}
),
ens_mean AS (
  SELECT day, latitude, longitude, AVG(value) AS ens_mean
  FROM raw GROUP BY day, latitude, longitude
)
SELECT 'Meteomatics' AS provider, '{SPATIAL_MODEL}' AS model,
       (SELECT max_ca FROM latest) AS reference_date,
       'Temperature' AS metric,
       r.member, r.day,
       DATEDIFF(r.day, DATE((SELECT max_ca FROM latest))) AS lead_day,
       r.latitude, r.longitude, r.value,
       e.ens_mean, r.value - e.ens_mean AS anomaly,
       current_timestamp() AS snapshot_ts
FROM raw r
JOIN ens_mean e ON e.day = r.day AND e.latitude = r.latitude AND e.longitude = r.longitude
""")
count("fcst_member_spatial")

# COMMAND ----------

# DBTITLE 1,4c. Meteomatics per-member country means (curve_member in the silver table)
# The silver temperature_forecast table carries individual members in
# `curve_member` (the spatial cell above relies on it). Population-weighted
# country means of those members (CM fragments, cell 1a) feed the "Meteomatics
# EC-ENS / AIFS-ENS members" scenario sources. METEOMATICS_MEMBER_TABLE can still
# point at another member table, as long as it has latitude / longitude.
MEMBER_SRC = METEOMATICS_MEMBER_TABLE or f"{MM}.temperature_forecast"
cols = {f.name.lower() for f in spark.table(MEMBER_SRC).schema.fields}
member_col = next((c for c in ("curve_member", "member", "ens_member", "perturbation", "number") if c in cols), None)
run_col = "created_at" if "created_at" in cols else "reference_date"
if member_col is None:
    print(f"{MEMBER_SRC} has no member column — Meteomatics members not available; "
          "scenarios will cluster Volue EC-ENS members")
else:
    for label, (model, curve) in METEOMATICS_MODELS.items():
        curve_filter = f"AND t.curve_name = '{curve}'" if "curve_name" in cols else ""
        spark.sql(f"""
            INSERT INTO {SBX}.fcst_members
            WITH latest AS (SELECT MAX(t.{run_col}) AS rd FROM {MEMBER_SRC} t
                            WHERE t.model = '{model}' {curve_filter}
                              AND t.{member_col} IS NOT NULL AND CAST(t.{member_col} AS STRING) != ''),
            country AS (
              SELECT t.{run_col} AS reference_date, {CM['area']} AS area, CAST(t.delivery_start AS DATE) AS day,
                     CAST(t.{member_col} AS STRING) AS member, {CM['mean']} AS value
              FROM {MEMBER_SRC} t CROSS JOIN latest {CM['join']}
              WHERE t.model = '{model}' {curve_filter} AND t.{run_col} = latest.rd
                AND t.{member_col} IS NOT NULL AND CAST(t.{member_col} AS STRING) != ''
                AND {MM_BBOX}
              GROUP BY t.{run_col}, {CM['area']}, CAST(t.delivery_start AS DATE), t.{member_col}
            ),
            normals AS (
              SELECT area, {CET_DAY} AS day, AVG(value) AS normal
              FROM {VOLUE}.temperature_consumption
              WHERE array_contains(categories, 'TT') AND data_type = 'N' AND area IN ({AREA_SQL})
                AND delivery_start BETWEEN current_date() AND current_date() + INTERVAL 20 DAYS
              GROUP BY area, {CET_DAY}
            )
            SELECT 'Meteomatics' AS provider, '{model}' AS model, c.reference_date, 'Temperature' AS metric,
                   c.area, c.day, DATEDIFF(c.day, DATE(c.reference_date)) AS lead_day,
                   c.member, c.value, n.normal, c.value - n.normal AS anomaly, current_timestamp() AS snapshot_ts
            FROM country c LEFT JOIN normals n ON n.area = c.area AND n.day = c.day
            WHERE c.area IS NOT NULL
            """)
    count("fcst_members")

# COMMAND ----------

# DBTITLE 1,5. Historical actuals + normals per metric/area/day (since 2013)
hist_blocks = []
for name, (cat, fcst_tbl, hist_tbl, scale) in METRICS.items():
    hist_blocks.append(f"""
    SELECT '{name}' AS metric, area, {CET_DAY} AS day, data_type, AVG(value) * {scale} AS value
    FROM {VOLUE}.{hist_tbl}
    WHERE array_contains(categories, '{cat}') AND data_type IN ('AF', 'N')
      AND area IN ({AREA_SQL})
      AND delivery_start >= '2013-01-01' AND delivery_start < current_date()
    GROUP BY area, {CET_DAY}, data_type
    """)

spark.sql(f"""
CREATE OR REPLACE TABLE {SBX}.hist_daily AS
WITH raw AS ({" UNION ALL ".join(hist_blocks)}),
pivoted AS (
  SELECT metric, area, day,
         MAX(CASE WHEN data_type = 'AF' THEN value END) AS actual,
         MAX(CASE WHEN data_type = 'N'  THEN value END) AS normal_dated
  FROM raw GROUP BY metric, area, day
),
-- day-of-year normal from whatever years the N curve covers, as a fallback
-- for older dates where the dated normal is absent
doy_normal AS (
  SELECT metric, area, DAYOFYEAR(day) AS doy, AVG(normal_dated) AS normal_doy
  FROM pivoted WHERE normal_dated IS NOT NULL GROUP BY metric, area, DAYOFYEAR(day)
)
SELECT p.metric, p.area, p.day, YEAR(p.day) AS year, MONTH(p.day) AS month,
       WEEKOFYEAR(p.day) AS iso_week, DATE_TRUNC('week', p.day) AS week_start,
       p.actual, COALESCE(p.normal_dated, d.normal_doy) AS normal,
       p.actual - COALESCE(p.normal_dated, d.normal_doy) AS anomaly,
       CASE WHEN COALESCE(p.normal_dated, d.normal_doy) != 0
            THEN (p.actual / COALESCE(p.normal_dated, d.normal_doy) - 1) * 100 END AS anomaly_pct
FROM pivoted p LEFT JOIN doy_normal d ON d.metric = p.metric AND d.area = p.area AND d.doy = DAYOFYEAR(p.day)
WHERE p.actual IS NOT NULL
""")
count("hist_daily")

# COMMAND ----------

# DBTITLE 1,6. Hydro components — actual (SA) and normal (N) since 2013
hydro_area_sql = ",".join(f"'{a}'" for a in HYDRO_AREAS)
comp_blocks = []
for comp in HYDRO_COMPONENTS:
    comp_blocks.append(f"""
    SELECT area, '{comp}' AS component, {CET_DAY} AS day, data_type, AVG(value) AS value
    FROM {VOLUE}.hydro_reservoir
    WHERE UPPER(area) IN ({hydro_area_sql}) AND data_type IN ('SA', 'N')
      AND array_contains(categories, 'RES') AND array_contains(categories, 'HYDRO')
      AND array_contains(categories, '{comp}')
      AND delivery_start >= '2013-01-01'
    GROUP BY area, {CET_DAY}, data_type
    """)

spark.sql(f"""
CREATE OR REPLACE TABLE {SBX}.hydro_daily AS
SELECT UPPER(area) AS area, component, day, data_type, value, current_timestamp() AS snapshot_ts
FROM ({" UNION ALL ".join(comp_blocks)})
""")
count("hydro_daily")

# COMMAND ----------

# DBTITLE 1,7. Morning Call — daily values per run for the report's exact Volue curves
# Port of Morning_Report/import_00z_add_solar_np_tot.py. The wapi script asks
# for `tt fr con ec00ens °c cet min15 f` etc.; here the same curve names are
# matched on LOWER(curve_name) so the `area` column convention is irrelevant.
# Daily CET mean (tt/wnd/spv) or daily sum (rre) of the 'Avg' tag, for every
# run of the last MORNING_RUN_HISTORY_DAYS days of each model pattern, plus the
# normal curve on the same days. The app picks today's 00z and the previous
# 00z (Friday's on a Monday) and forms the weekly windows.
MORNING_REGIONS = ["fr", "de", "uk", "it", "hu", "np", "ib", "see", "cwe", "it-nord"]
MORNING_PATTERNS = ["ec00ens", "ec12ens", "gfs00ens", "ecmonthly"]
MORNING_FAMILIES = {
    # family: (table, forecast curve template, normal table, normal curve template, daily agg)
    "tt":  ("temperature_consumption_forecast", "tt {r} con {run} °c cet min15 f",
            "temperature_consumption",          "tt {r} con °c cet min15 n",           "AVG"),
    "wnd": ("production_forecast",              "pro {r} wnd {run} mwh/h cet min15 f",
            "production",                       "pro {r} wnd mwh/h cet min15 n",       "AVG"),
    "spv": ("production_forecast",              "pro {r} spv {run} mwh/h cet min15 f",
            "production",                       "pro {r} spv mwh/h cet min15 n",       "AVG"),
    "rre": ("precipitation_forecast",           "rre {r} {run} gwh cet min15 f",
            "precipitation",                    "rre {r} gwh cet min15 n",             "SUM"),
}
MORNING_RUN_HISTORY_DAYS = 8

fcst_blocks, norm_blocks = [], []
for fam, (ftbl, ftpl, ntbl, ntpl, agg) in MORNING_FAMILIES.items():
    fcst_names = ",".join(f"'{ftpl.format(r=r, run=p).lower()}'" for r in MORNING_REGIONS for p in MORNING_PATTERNS)
    norm_names = ",".join(f"'{ntpl.format(r=r).lower()}'" for r in MORNING_REGIONS)
    fcst_blocks.append(f"""
    SELECT '{fam}' AS family, LOWER(curve_name) AS curve_name, reference_date,
           {CET_DAY} AS day, {agg}(value) AS value, COUNT(*) AS n_points
    FROM {VOLUE}.{ftbl}
    WHERE LOWER(curve_name) IN ({fcst_names}) AND data_type = 'F' AND tag = 'Avg'
      AND reference_date >= current_timestamp() - INTERVAL {MORNING_RUN_HISTORY_DAYS} DAYS
      AND delivery_start >= current_date() - INTERVAL {MORNING_RUN_HISTORY_DAYS + 1} DAYS
    GROUP BY LOWER(curve_name), reference_date, {CET_DAY}
    """)
    norm_blocks.append(f"""
    SELECT '{fam}' AS family, LOWER(curve_name) AS curve_name, {CET_DAY} AS day, {agg}(value) AS normal
    FROM {VOLUE}.{ntbl}
    WHERE LOWER(curve_name) IN ({norm_names}) AND data_type = 'N'
      AND delivery_start BETWEEN current_date() - INTERVAL {MORNING_RUN_HISTORY_DAYS + 1} DAYS
                             AND current_date() + INTERVAL 50 DAYS
    GROUP BY LOWER(curve_name), {CET_DAY}
    """)

# region / pattern are parsed back out of the curve name so the app can pivot on them
region_case = "CASE " + " ".join(
    f"WHEN f.curve_name LIKE '% {r} %' OR f.curve_name LIKE 'rre {r} %' THEN '{r}'"
    for r in sorted(MORNING_REGIONS, key=len, reverse=True)) + " END"
pattern_case = "CASE " + " ".join(f"WHEN f.curve_name LIKE '%{p}%' THEN '{p}'" for p in MORNING_PATTERNS) + " END"
norm_region_case = "CASE " + " ".join(
    f"WHEN n.curve_name LIKE '% {r} %' OR n.curve_name LIKE 'rre {r} %' THEN '{r}'"
    for r in sorted(MORNING_REGIONS, key=len, reverse=True)) + " END"

spark.sql(f"""
CREATE OR REPLACE TABLE {SBX}.morning_daily AS
WITH f AS ({" UNION ALL ".join(fcst_blocks)}),
n AS ({" UNION ALL ".join(norm_blocks)}),
f2 AS (
  SELECT f.family, f.curve_name, {region_case} AS region, {pattern_case} AS pattern,
         f.reference_date, f.day, f.value, f.n_points
  FROM f
),
n2 AS (SELECT n.family, {norm_region_case} AS region, n.day, n.normal FROM n)
SELECT 'Volue' AS provider, f2.family, f2.region, f2.pattern, f2.curve_name, f2.reference_date,
       f2.day, f2.value, f2.n_points, n2.normal, current_timestamp() AS snapshot_ts
FROM f2 LEFT JOIN n2 ON n2.family = f2.family AND n2.region = f2.region AND n2.day = f2.day
WHERE f2.region IS NOT NULL AND f2.pattern IS NOT NULL
""")
count("morning_daily")

# Meteomatics EC-ENS / AIFS-ENS temperature on the report's regions, for the model
# comparison. A region is a group of countries and its value is the population-
# weighted mean over ALL the group's cells (CM fragments, cell 1a) — the Nordics
# are NO+SE+FI+DK weighted by where people live, not an average of four country
# means. Hungary belongs to both 'hu' and 'see', which is why the mapping is a
# join and not a CASE (a CASE stops at the first match and left SEE without HU).
MM_REGION_GROUPS = {"fr": ["FR"], "de": ["DE"], "uk": ["UK"], "it": ["IT"], "hu": ["HU"],
                    "np": ["NO", "SE", "FI", "DK"], "ib": ["ES", "PT"], "see": ["SI", "HR", "SK", "HU"]}
mm_region_values = ", ".join(f"('{r}', '{c}')" for r, cs in MM_REGION_GROUPS.items() for c in cs)
mm_blocks = []
for label, (model, curve) in METEOMATICS_MODELS.items():
    mm_blocks.append(f"""
    SELECT 'tt' AS family, r.region, '{model}' AS pattern, c.reference_date, c.day,
           SUM(c.value * c.weight) / SUM(c.weight) AS value
    FROM (SELECT t.created_at AS reference_date, CAST(t.delivery_start AS DATE) AS day, t.value,
                 {CM['area']} AS area, {CM['weight']} AS weight
          FROM {MM}.temperature_forecast t {CM['join']}
          WHERE t.model = '{model}' AND t.curve_name = '{curve}'
            AND t.created_at >= current_timestamp() - INTERVAL {MORNING_RUN_HISTORY_DAYS} DAYS
            AND {MM_BBOX}) c
    JOIN regions r ON r.area = c.area
    GROUP BY r.region, c.reference_date, c.day
    """)
spark.sql(f"""
INSERT INTO {SBX}.morning_daily
WITH regions AS (SELECT * FROM VALUES {mm_region_values} AS r(region, area)),
mm AS ({" UNION ALL ".join(mm_blocks)}),
n AS (
  SELECT {CET_DAY} AS day, LOWER(curve_name) AS curve_name, AVG(value) AS normal
  FROM {VOLUE}.temperature_consumption
  WHERE LOWER(curve_name) IN ({",".join(f"'tt {r} con °c cet min15 n'" for r in MM_REGION_GROUPS)}) AND data_type = 'N'
    AND delivery_start BETWEEN current_date() - INTERVAL 2 DAYS AND current_date() + INTERVAL 20 DAYS
  GROUP BY {CET_DAY}, LOWER(curve_name)
),
n2 AS (SELECT day, normal, {"CASE " + " ".join(f"WHEN curve_name = 'tt {r} con °c cet min15 n' THEN '{r}'" for r in MM_REGION_GROUPS) + " END"} AS region FROM n)
SELECT 'Meteomatics', mm.family, mm.region, mm.pattern, mm.pattern AS curve_name, mm.reference_date, mm.day, mm.value,
       96 AS n_points, n2.normal, current_timestamp()
FROM mm LEFT JOIN n2 ON n2.region = mm.region AND n2.day = mm.day
WHERE mm.region IS NOT NULL
""")
count("morning_daily")

# COMMAND ----------

# DBTITLE 1,7b. Gas Demand — daily ens-mean temperature / wind / solar per run
# Feeds the Gas Demand section, which ports EU-gas-demand/ldz_forecast.py and
# rdl_forecast.py. Both scripts pull, through wapi, the ensemble-MEAN daily
# value of one curve per country for TODAY's run and for the run they compare
# against (yesterday's, or Friday's on a Monday) — so what the app needs is
# the same daily 'Avg' series, per run, for several days back.
#
# fcst_daily cannot serve this: it keeps only N_RUNS_KEPT=6 runs *per model
# family*, and the EC-ENS family interleaves 00z and 12z, so a Monday's
# "ec00ens vs the ec00ens of 3 days ago" comparison falls outside the window.
# This table is therefore keyed by pattern and keeps GAS_RUN_HISTORY_DAYS of
# runs, the same way morning_daily does.
#
# Units match the scripts' arithmetic after their own conversions:
#   tt  → °C          (scale 1.0)
#   wnd → GW          (MWh/h * 0.001;  rdl_forecast.py's `fcst_mean/1000`)
#   spv → GW          (same)
# The normal ('N') curve is joined on the same day, in the same units, which is
# rdl_forecast.py's `norm` line.
GAS_AREAS = ["DE", "UK", "FR", "BE", "NL", "IT", "ES", "PT"]
GAS_AREA_SQL = ",".join(f"'{a}'" for a in GAS_AREAS)
GAS_PATTERNS = ["ec00ens", "ec12ens", "gfs00ens"]
GAS_FAMILIES = {
    # family: (category, forecast table, history table, scale)
    "tt":  ("TT",  "temperature_consumption_forecast", "temperature_consumption", 1.0),
    "wnd": ("WND", "production_forecast",              "production",              0.001),
    "spv": ("SPV", "production_forecast",              "production",              0.001),
}
GAS_RUN_HISTORY_DAYS = 8

gas_fcst_blocks, gas_norm_blocks = [], []
for fam, (cat, ftbl, ntbl, scale) in GAS_FAMILIES.items():
    for pat in GAS_PATTERNS:
        gas_fcst_blocks.append(f"""
        SELECT '{fam}' AS family, '{pat}' AS pattern, area, reference_date,
               {CET_DAY} AS day, AVG(value) * {scale} AS value, COUNT(*) AS n_points
        FROM {VOLUE}.{ftbl}
        WHERE curve_name LIKE '%{pat}%' AND data_type = 'F' AND tag = 'Avg'
          AND array_contains(categories, '{cat}') AND area IN ({GAS_AREA_SQL})
          AND reference_date >= current_timestamp() - INTERVAL {GAS_RUN_HISTORY_DAYS} DAYS
          AND delivery_start >= current_date() - INTERVAL {GAS_RUN_HISTORY_DAYS + 1} DAYS
        GROUP BY area, reference_date, {CET_DAY}
        """)
    gas_norm_blocks.append(f"""
    SELECT '{fam}' AS family, area, {CET_DAY} AS day, AVG(value) * {scale} AS normal
    FROM {VOLUE}.{ntbl}
    WHERE data_type = 'N' AND array_contains(categories, '{cat}') AND area IN ({GAS_AREA_SQL})
      AND delivery_start BETWEEN current_date() - INTERVAL {GAS_RUN_HISTORY_DAYS + 1} DAYS
                             AND current_date() + INTERVAL 30 DAYS
    GROUP BY area, {CET_DAY}
    """)

spark.sql(f"""
CREATE OR REPLACE TABLE {SBX}.gas_demand_daily AS
WITH f AS ({" UNION ALL ".join(gas_fcst_blocks)}),
n AS ({" UNION ALL ".join(gas_norm_blocks)})
SELECT 'Volue' AS provider, f.family, f.pattern, f.area, f.reference_date, f.day,
       f.value, f.n_points, n.normal, current_timestamp() AS snapshot_ts
FROM f LEFT JOIN n ON n.family = f.family AND n.area = f.area AND n.day = f.day
""")
count("gas_demand_daily")

# COMMAND ----------

# DBTITLE 1,7c. European Weather Regimes — project Z500 members onto the 7 regimes
# Port of Franziska_Intern/corso_model_wr/working_wr_tool/max_proj_members.py
# (Michel & Rivière 2011), reading Databricks instead of the DSL NetCDF drops.
#
# The projection is an area-weighted inner product between the standardised
# geopotential anomaly and each fixed regime pattern, so it is a JOIN on
# (latitude, longitude) plus a GROUP BY (member, day) — the 22 M grid rows of a
# run never leave Spark, and only ~850 rows per run are written.
#
# Tables written:
#   wr_patterns          the 7 fixed patterns on the 121 x 241 domain, one row
#                        per grid point, pre-multiplied by cos(latitude).
#                        Built ONCE from wr_patterns.npz (shipped with the app);
#                        set WR_PATTERNS_PATH to wherever you put that file.
#   wr_gph_clim          ERA5 day-of-year mean Z500 per grid point — turns the
#                        absolute heights the silver table stores into the
#                        anomalies the projection needs.
#   wr_reanalysis_daily  ERA5 classified through the same projection: the
#                        climatological frequency, persistence and transitions.
#                        Backfilled once, then appended daily.
#   wr_forecast_members  per run / member / lead day: the 7 IWR values and the
#                        assigned regime. This is what the app reads.
#
# The patterns are the fixed output of the original 1979-2019 k-means study and
# cannot be derived from Databricks; everything else comes from the silver layer.
GPH_FCST = f"{MM}.geopotential_height_forecast"
GPH_ACT = f"{MM}.geopotential_height"
# Where wr_patterns.npz (shipped in weather-power-desk-app/) was uploaded.
# Read once, on the first run only; override with the WR_PATTERNS_PATH env var
# or just edit this line. A Workspace path or a UC volume path both work.
WR_PATTERNS_PATH = os.environ.get("WR_PATTERNS_PATH", "/Workspace/Users/lorenzo.comi@axpo.com/Meteo-Power-Dashboard/weather-power-desk-app/wr_patterns.npz")

WR_REGIMES = ["ScTr", "GL", "EuBl", "AR", "AT", "ScBl", "ZO"]
WR_LAT_MIN, WR_LAT_MAX, WR_LON_MIN, WR_LON_MAX = 30.0, 90.0, -80.0, 40.0
WR_THRESHOLD, WR_LEAD_FACTOR = 1.0, -0.018774217528098960
WR_MODELS = {
    "ecmwf-aifs-ens": "geopotential_height_500hpa_m_ecmwf_aifs_ens_p1d",
    "ncep-gfs-ens": "geopotential_height_500hpa_m_ncep_gfs_ens_p1d",
}
WR_ERA5_CURVE = "geopotential_height_500hpa_m_ecmwf_era5_p1d"
WR_RUN_HISTORY_DAYS = 8
WR_CLIM_START_YEAR = 1979
WR_MAX_HORIZON_DAYS = 17

def wr_domain(alias: str = "") -> str:
    """The domain predicate, qualified by table alias. It MUST be qualified in the
    projection queries: the pattern and climatology tables also carry
    latitude/longitude, so an unqualified reference is ambiguous."""
    p = f"{alias}." if alias else ""
    return (f"{p}latitude BETWEEN {WR_LAT_MIN} AND {WR_LAT_MAX} "
            f"AND {p}longitude BETWEEN {WR_LON_MIN} AND {WR_LON_MAX}")


def wr_tables_exist(*names) -> bool:
    return all(spark.catalog.tableExists(f"{SBX}.{n}") for n in names)


# --- 7c.1 patterns (once) -----------------------------------------------------
if spark.catalog.tableExists(f"{SBX}.wr_patterns"):
    print("wr_patterns already built — skipping")
else:
    import numpy as _np
    import pandas as _pd

    _z = _np.load(WR_PATTERNS_PATH)
    _pat, _lats, _lons = _z["patterns"], _z["lats"], _z["lons"]   # (7, lat, lon)
    assert _pat.shape == (7, len(_lats), len(_lons)), f"unexpected pattern shape {_pat.shape}"
    _coslat = _np.cos(_np.deg2rad(_lats))
    _rows = {
        "latitude": _np.repeat(_lats, len(_lons)),
        "longitude": _np.tile(_lons, len(_lats)),
    }
    # store the pattern already multiplied by cos(lat): the SQL then only sums
    _for_k = _pat * _coslat[None, :, None]
    for _k, _name in enumerate(WR_REGIMES):
        _rows[f"w_{_name}"] = _for_k[_k].ravel()
    spark.createDataFrame(_pd.DataFrame(_rows)).write.mode("overwrite").saveAsTable(f"{SBX}.wr_patterns")

    # normalisation constants + the day-of-year amplitude, as a second small table
    _norm = _pd.DataFrame({"regime": WR_REGIMES,
                           "proj_avg": _z["proj_avg"], "denom": _z["denom"]})
    spark.createDataFrame(_norm).write.mode("overwrite").saveAsTable(f"{SBX}.wr_norm")
    _amp = _pd.DataFrame({"month": _z["amp_month"].astype(int), "day_of_month": _z["amp_day"].astype(int),
                          "amplitude": _z["amp_value"]})
    spark.createDataFrame(_amp).write.mode("overwrite").saveAsTable(f"{SBX}.wr_amplitude")
    # the projection divides by sum(cos(lat)) * n_lon — a pure domain constant
    print(f"wr_patterns: {len(_lats)} x {len(_lons)} grid, "
          f"normalizer = {float(_coslat.sum()) * len(_lons):.6f}")
count("wr_patterns")

WR_NORMALIZER = spark.sql(f"""
    SELECT SUM(c) * (SELECT COUNT(DISTINCT longitude) FROM {SBX}.wr_patterns)
    FROM (SELECT DISTINCT latitude, COS(RADIANS(latitude)) AS c FROM {SBX}.wr_patterns)
""").first()[0]
print(f"normalizer = {WR_NORMALIZER}")

# --- 7c.2 ERA5 day-of-year climatology (once) ---------------------------------
# The silver tables store absolute heights; the projection needs anomalies.
# A day-of-year mean per grid point over the ERA5 record is what the original
# MATLAB climatology provided — verified to give identical regime assignments.
if spark.catalog.tableExists(f"{SBX}.wr_gph_clim"):
    print("wr_gph_clim already built — skipping")
else:
    spark.sql(f"""
    CREATE OR REPLACE TABLE {SBX}.wr_gph_clim AS
    SELECT latitude, longitude,
           DAYOFYEAR(delivery_start)
             - CASE WHEN MONTH(delivery_start) > 2
                     AND (YEAR(delivery_start) % 4 = 0
                          AND (YEAR(delivery_start) % 100 != 0 OR YEAR(delivery_start) % 400 = 0))
                    THEN 1 ELSE 0 END          AS doy,
           AVG(value)   AS gph_normal,
           COUNT(*)     AS n_years
    FROM {GPH_ACT}
    WHERE curve_name = '{WR_ERA5_CURVE}' AND {wr_domain()}
      AND YEAR(delivery_start) >= {WR_CLIM_START_YEAR}
    GROUP BY latitude, longitude, 3
    """)
    _cov = spark.sql(f"SELECT MIN(n_years), MAX(n_years), COUNT(DISTINCT doy) FROM {SBX}.wr_gph_clim").first()
    print(f"wr_gph_clim: {_cov[2]} days of year, {_cov[0]}-{_cov[1]} years per grid point")
count("wr_gph_clim")

# --- 7c.3 the projection, as a reusable SQL block -----------------------------
# anomaly = value - day-of-year normal; standardised by the day's domain
# amplitude; projected onto each cos-lat-weighted pattern; then normalised.
_proj_cols = ",\n           ".join(
    f"SUM((f.value - c.gph_normal) / a.amplitude * p.w_{r}) / {WR_NORMALIZER} AS raw_{r}"
    for r in WR_REGIMES)
_iwr_cols = ",\n       ".join(
    f"(raw_{r} - (SELECT proj_avg FROM {SBX}.wr_norm WHERE regime = '{r}')) "
    f"/ (SELECT denom FROM {SBX}.wr_norm WHERE regime = '{r}') AS iwr_{r}"
    for r in WR_REGIMES)
_greatest = "GREATEST(" + ", ".join(f"iwr_{r}" for r in WR_REGIMES) + ")"
_which = "CASE " + " ".join(
    f"WHEN max_iwr = iwr_{r} THEN '{r}'" for r in WR_REGIMES) + " END"

# --- 7c.4 forecast members ----------------------------------------------------
_fc_models = ",".join(f"'{m}'" for m in WR_MODELS)
spark.sql(f"""
CREATE OR REPLACE TABLE {SBX}.wr_forecast_members
TBLPROPERTIES('delta.feature.allowColumnDefaults' = 'supported') AS
WITH raw AS (
  SELECT f.model, f.created_at AS reference_date,
         DATE(f.delivery_start) AS day,
         CAST(f.curve_member AS STRING) AS member,
         {_proj_cols}
  FROM {GPH_FCST} f
  JOIN {SBX}.wr_patterns p ON p.latitude = f.latitude AND p.longitude = f.longitude
  JOIN {SBX}.wr_gph_clim c ON c.latitude = f.latitude AND c.longitude = f.longitude
       AND c.doy = DAYOFYEAR(f.delivery_start)
         - CASE WHEN MONTH(f.delivery_start) > 2
                 AND (YEAR(f.delivery_start) % 4 = 0
                      AND (YEAR(f.delivery_start) % 100 != 0 OR YEAR(f.delivery_start) % 400 = 0))
                THEN 1 ELSE 0 END
  JOIN {SBX}.wr_amplitude a ON a.month = MONTH(f.delivery_start)
       AND a.day_of_month = DAY(f.delivery_start)
  WHERE f.model IN ({_fc_models}) AND {wr_domain('f')}
    AND f.curve_member IS NOT NULL AND CAST(f.curve_member AS STRING) != ''
    AND f.created_at >= current_timestamp() - INTERVAL {WR_RUN_HISTORY_DAYS} DAYS
    AND f.delivery_start >= current_date() - INTERVAL {WR_RUN_HISTORY_DAYS + 1} DAYS
  GROUP BY f.model, f.created_at, DATE(f.delivery_start), CAST(f.curve_member AS STRING)
),
iwr AS (
  SELECT model, reference_date, day, member,
       {_iwr_cols}
  FROM raw
),
scored AS (
  SELECT *, DATEDIFF(day, DATE(reference_date)) AS lead_day, {_greatest} AS max_iwr FROM iwr
)
SELECT model, reference_date, day, lead_day, member,
       {", ".join(f"iwr_{r}" for r in WR_REGIMES)},
       max_iwr,
       {WR_THRESHOLD} + lead_day * {WR_LEAD_FACTOR} AS threshold,
       CASE WHEN max_iwr >= {WR_THRESHOLD} + lead_day * {WR_LEAD_FACTOR}
            THEN {_which} ELSE 'no' END AS regime,
       current_timestamp() AS snapshot_ts
FROM scored
WHERE lead_day BETWEEN 0 AND {WR_MAX_HORIZON_DAYS}
""")
count("wr_forecast_members")

# --- 7c.5 classified ERA5 reanalysis (backfill once, then append) -------------
# Same projection over the actuals table. The first run is the expensive one
# (the whole ERA5 record); afterwards only the days not yet classified are
# added, which is one or two per refresh.
_era5_exists = spark.catalog.tableExists(f"{SBX}.wr_reanalysis_daily")
_since = f"YEAR(delivery_start) >= {WR_CLIM_START_YEAR}"
if _era5_exists:
    _last = spark.sql(f"SELECT MAX(day) FROM {SBX}.wr_reanalysis_daily").first()[0]
    if _last is not None:
        _since = f"delivery_start > DATE '{_last}'"
    print(f"wr_reanalysis_daily exists — appending days after {_last}")
else:
    print(f"wr_reanalysis_daily: first build, classifying ERA5 from {WR_CLIM_START_YEAR} "
          "(this run is slow; later runs only append new days)")

_era5_proj = ",\n           ".join(
    f"SUM((f.value - c.gph_normal) / a.amplitude * p.w_{r}) / {WR_NORMALIZER} AS raw_{r}"
    for r in WR_REGIMES)
_era5_sql = f"""
WITH raw AS (
  SELECT DATE(f.delivery_start) AS day, {_era5_proj}
  FROM {GPH_ACT} f
  JOIN {SBX}.wr_patterns p ON p.latitude = f.latitude AND p.longitude = f.longitude
  JOIN {SBX}.wr_gph_clim c ON c.latitude = f.latitude AND c.longitude = f.longitude
       AND c.doy = DAYOFYEAR(f.delivery_start)
         - CASE WHEN MONTH(f.delivery_start) > 2
                 AND (YEAR(f.delivery_start) % 4 = 0
                      AND (YEAR(f.delivery_start) % 100 != 0 OR YEAR(f.delivery_start) % 400 = 0))
                THEN 1 ELSE 0 END
  JOIN {SBX}.wr_amplitude a ON a.month = MONTH(f.delivery_start)
       AND a.day_of_month = DAY(f.delivery_start)
  WHERE f.curve_name = '{WR_ERA5_CURVE}' AND {wr_domain('f')} AND {_since}
  GROUP BY DATE(f.delivery_start)
),
iwr AS (SELECT day, {_iwr_cols} FROM raw),
scored AS (SELECT *, {_greatest} AS max_iwr FROM iwr)
SELECT day, {", ".join(f"iwr_{r}" for r in WR_REGIMES)}, max_iwr,
       CASE WHEN max_iwr >= {WR_THRESHOLD} THEN {_which} ELSE 'no' END AS regime,
       current_timestamp() AS snapshot_ts
FROM scored
"""
if _era5_exists:
    spark.sql(f"INSERT INTO {SBX}.wr_reanalysis_daily {_era5_sql}")
else:
    spark.sql(f"CREATE OR REPLACE TABLE {SBX}.wr_reanalysis_daily AS {_era5_sql}")
count("wr_reanalysis_daily")

_span = spark.sql(f"SELECT MIN(day), MAX(day), COUNT(*) FROM {SBX}.wr_reanalysis_daily").first()
print(f"wr_reanalysis_daily spans {_span[0]} -> {_span[1]} ({_span[2]:,} days)")

# COMMAND ----------

# DBTITLE 1,8. Placeholders — weather indexes and Meteologica
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {SBX}.weather_indexes (
  index_name STRING COMMENT 'NAO, AO, EA, SCAND, PNA, ONI, MJO_AMP, MJO_PHASE, QBO, SSW',
  date       DATE,
  value      DOUBLE,
  source     STRING COMMENT 'e.g. NOAA CPC, BoM, ERA5-derived',
  loaded_at  TIMESTAMP
) COMMENT 'Daily/monthly teleconnection indexes for the analogue tab. Loaded by a separate job.'
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {SBX}.meteologica_members (
  provider       STRING,
  model          STRING,
  reference_date TIMESTAMP,
  metric         STRING COMMENT 'Temperature | Wind | Solar | Hydro',
  area           STRING,
  day            DATE,
  lead_day       INT,
  member         STRING,
  value          DOUBLE,
  snapshot_ts    TIMESTAMP
) COMMENT 'Meteologica ECMWF ENS / ENSEXT members, same layout as fcst_members. Empty until ingested.'
""")

if METEOLOGICA_TABLE:
    spark.sql(f"""
    INSERT OVERWRITE {SBX}.meteologica_members
    WITH latest AS (SELECT MAX(reference_date) AS rd FROM {METEOLOGICA_TABLE})
    SELECT 'Meteologica', 'ECMWF ENS', reference_date, metric, UPPER(area), delivery_date,
           DATEDIFF(delivery_date, DATE(reference_date)), CAST(member AS STRING), value, current_timestamp()
    FROM {METEOLOGICA_TABLE}, latest WHERE reference_date = latest.rd
    """)
    count("meteologica_members")
else:
    print("METEOLOGICA_TABLE not set — meteologica_members left as-is")

print(f"Refresh finished: {datetime.now()}")

# COMMAND ----------

# DBTITLE 1,8. Anomaly maps — monthly grid-point averages from gold layer (ERA5)
# Pre-aggregate the ERA5 climatology from the gold layer into monthly
# grid-point means. The app reads this sandbox table instead of querying
# the gold catalog directly (the app SP has no USE CATALOG on dna_prod_gold).
#
# Same approach as the gas desk anomaly_map table: notebook runs as the user
# (who does have access), sandbox table inherits the SP grants.

GOLD = "dna_prod_gold.weather"
BBOX = "CAST(latitude AS DOUBLE) BETWEEN 35 AND 72 AND CAST(longitude AS DOUBLE) BETWEEN -12 AND 35"

MAP_SOURCES = [
    ("Temperature",          "temperature_meteomatics_climatology",              "t_mean_2m_24h_c_ecmwf_era5_p1d",      "°C"),
    ("Wind speed (200 hPa)", "wind_speed_meteomatics_climatology",               "wind_speed_200hpa_ms_ecmwf_era5_p1d",  "m/s"),
    ("Precipitation",        "precipitation_forecast_meteomatics_climatology",   "precip_24h_mm_mix_p1d",                "mm"),
]

blocks = []
for metric_name, table, curve, unit in MAP_SOURCES:
    blocks.append(f"""
    SELECT '{metric_name}' AS metric,
           CAST(latitude AS DOUBLE) AS latitude,
           CAST(longitude AS DOUBLE) AS longitude,
           FIRST(city) AS city,
           YEAR(delivery_start) AS year,
           MONTH(delivery_start) AS month,
           AVG(value) AS value,
           AVG(normal) AS normal,
           AVG(anomaly) AS anomaly
    FROM {GOLD}.{table}
    WHERE curve_name = '{curve}' AND {BBOX}
    GROUP BY latitude, longitude, YEAR(delivery_start), MONTH(delivery_start)
    """)

full_sql = " UNION ALL ".join(blocks)
spark.sql(f"""
    CREATE OR REPLACE TABLE {SBX}.anomaly_map AS
    SELECT *, current_timestamp() AS snapshot_ts
    FROM ({full_sql})
""")
count("anomaly_map")
