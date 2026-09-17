# Power Desk — Weather Dashboard (Databricks App)

Built on the structure of `LPG-desk-dashboard`: a Streamlit Databricks App that reads
pre-aggregated sandbox tables, plus a Databricks notebook that refreshes those tables
on a Lakeflow schedule. Nothing here runs locally — deploy both pieces to the workspace.

```
Power_dashboard/
├── power_desk_refresh.py          Databricks notebook → dna_snbx_weather.power_desk.*  (schedule: every 6 h)
└── weather-power-desk-app/        Databricks App (Streamlit)
    ├── app.py                     landing page (4 tiles) + sidebar routing
    ├── app.yaml                   warehouse + env vars
    ├── requirements.txt
    ├── .streamlit/config.toml
    ├── _config.py                 schemas, areas, metrics, models, section definitions
    ├── _data.py                   Statement-API queries against the sandbox tables
    ├── _charts.py                 Plotly builders (fan, heatmaps, scenarios, hydro climatology)
    ├── _style.py                  design tokens + CSS (same palette as the LPG app)
    ├── _ui.py                     KPI cards / banners
    ├── _morning.py                Morning Call — the Morning Report table, live + model comparison
    ├── _forecast.py               Section 1 — Values · Uncertainty · Scenarios
    ├── _scenarios.py              member clustering engine (weekly means, fixed k)
    ├── _historical.py             Section 2 — history by country · index analogues
    ├── _hydro.py                  Section 3 — Hydro Report quantify_* live
    ├── _hydro_quantify.py         port of Hydro_Report/quantify_{reservoir,snow,balance}.py
    └── _strategy.py               Section 4 — locked / work in progress
```

## Sections

| # | Section | Content |
|---|---------|---------|
| 0 | **Morning Call** (first page, full-width tile) | The Morning Report table (`Morning_Report/import_00z_add_solar_np_tot.py` → `table_00z_add_solar_np_tot.xlsx`) computed live from the Volue tables instead of wapi + Excel. Same weekday logic (Mon/Tue: this week + next; Wed/Thu: weekend + next week; Fri–Sun: next week + week after), same rows (Temperatures FRA DE UK ITA HUN Nordic Iberia · Wind DE UK FRA ITA SEE Nordic Iberia · Solar DE FRA ITA SEE Nordic Iberia · Precip Alps(cwe+it-nord) Nordic SEE Iberia as a 2-week sum), same three columns (abs. value, Δ vs previous 00z — Δ -72h on Mondays —, Δ norm) with green/red signed deltas, free-text Pattern / Comment boxes and the Week 3 commentary block. Plus "confront the models": the same windows for EC 12z, GFS 00z, EC-Extended and the Meteomatics EC-ENS / AIFS-ENS country means, with the difference to EC 00z. |
| 1 | **Forecast** | *Values*: every Volue ensemble value by country (mean, median, P10/P25/P75/P90, min/max, normal, anomaly), fan chart per country, earlier runs faded, Meteomatics EC-ENS / AIFS-ENS country means on top. *Uncertainty*: across-member spread per lead day vs the normal spread for that lead day (last 365 days of 00z runs), ratio heatmap and verdicts, daily member box plots and a per-day histogram. *Scenarios*: k-means on members' **weekly-mean** temperature anomaly per country over real Mon–Sun trading weeks, **two scenarios fixed** (no silhouette selection), then **per country the two scenarios side by side** for temperature, wind and solar (weekly value, Δ vs ensemble mean, Δ vs normal) with a three-panel chart. Geopotential-based regime clustering is the intended method once that field is in Databricks; the engine takes any member × feature matrix. |
| 2 | **Historical & Analysis** | Monthly or weekly Volue actuals vs 30-yr normal by country; multi-select years and months; anomaly bars, actual-vs-normal lines, weekly anomaly heatmap, year × month heatmap per country; CSV export. Analogues tab reads `weather_indexes` and composites country anomalies over the closest analog years. |
| 3 | **Hydro Monitoring** | Reservoir levels (`wtr`), snow & groundwater (`sgw`), hydro balance (`bal`): climatology chart (grey history, blue recent years, dashed norm, red current year with direct label), anomaly GWh and % of normal, percentile of this week vs history, change vs last week, and the `stats_*.txt` text — the exact numbers `quantify_*.py` prints. |
| 4 | **Strategy** | Locked, work-in-progress sign. |

## Data sources and what was verified

| Source | Where | Notes |
|--------|-------|-------|
| Volue ("volue_deltashare") | `dna_prod_silver.volue.*` | Per-member ensembles via `tag` (`'Avg'` = mean, other tags = members). Columns: `curve_name, reference_date, delivery_start, value, area, data_type, tag, categories(array)`. No `volue_deltashare` catalog is referenced anywhere in the TonyWeather codebase; if the share is mounted under another name, change `VOLUE` in the notebook and `VOLUE_SCHEMA` in `app.yaml`. |
| Meteomatics | `dna_prod_silver.meteomatics.temperature_forecast` | Ensemble **mean only** — no member column exists in the silver layer (confirmed in `US-gas-dashboard/gas_desk_sandbox_refresh.py`). Used for the EC-ENS / AIFS-ENS mean lines. For member clustering on Meteomatics, point `METEOMATICS_MEMBER_TABLE` at a per-member table (`model, created_at, delivery_start, member, latitude, longitude, value`). Until then the scenario tab clusters Volue EC-ENS members and says so. |
| Meteologica | not in Databricks | Consumed only via REST in other projects. `meteologica_members` is created empty with the `fcst_members` layout; set `METEOLOGICA_TABLE` when an ingestion exists. |
| Weather indexes | `dna_snbx_weather.power_desk.weather_indexes` | Created empty (`index_name, date, value, source, loaded_at`). Load NAO/AO/EA/SCAND/PNA/ONI/MJO/QBO/SSW here and the analogue tab activates. |

## Deploy

1. Import `power_desk_refresh.py` as a notebook, run it once, then schedule it (Lakeflow Jobs, every 6 h; one run should sit at ~05:30 UTC so the Morning Call has today's 00z). It creates the schema, applies grants to the app service principal, and writes: `fcst_runs, fcst_daily, fcst_spread_clim, fcst_members, hist_daily, hydro_daily, morning_daily, weather_indexes, meteologica_members`.
2. Create a Databricks App from `weather-power-desk-app/` (same warehouse `2f4ff6b7c65abb1a` as the other desk apps). The app authenticates as its service principal and falls back to the forwarded user token.
3. First checks in the workspace:
   - `SELECT tag, COUNT(*) FROM dna_prod_silver.volue.temperature_consumption_forecast WHERE data_type='F' AND reference_date >= current_date()-1 GROUP BY tag` — should show ~51 member tags. If it shows P10…P90 style tags, the Volue share carries percentiles, not members, and the Uncertainty/Scenario tabs need a different source.
   - `SELECT DISTINCT area FROM dna_prod_silver.volue.hydro_reservoir` — confirm `IT`, `SEE`, `NP`, `CH`, `AT` codes match `HYDRO_AREA_CODES` in `_config.py` (the wapi report used `it-nord`).

## Method notes

- **Spread vs normal spread**: for each EC-ENS 00z run in the last 365 days, the std across members of the daily country mean, per lead day; the app compares today's std with the mean (and P25–P75) for the same lead day, preferring the same calendar month when ≥10 runs exist. Ratio ≥1.3 flags high uncertainty, ≤0.7 unusual confidence.
- **Scenarios**: features = weekly-mean temperature anomaly per (country × ISO week), weeks needing ≥4 forecast days; standardised; k-means (n_init=20) with k fixed at 2 (user can raise to 4); clusters relabelled by size (Scenario 1 = consensus); clusters <3 members folded into "Other". Member IDs are normalised (`M07`, `07`, `ens07` → 7) so a cluster found on one provider is read on another — all providers are the same 50 ECMWF perturbations. Silhouette is shown only as a separability diagnostic.
- **Morning Call**: `morning_daily` holds, per run of the last 8 days and per exact wapi curve name, the daily CET 'Avg' mean (tt/wnd/spv) or daily sum (rre) plus the normal. The app drops partial days (<90 % of the 15-min points, the report's `[:-1]`), picks today's 00z and the previous 00z (Friday's on Monday) and reproduces the weekly-window means and the 2-week precipitation sums exactly as the script does.
- **Hydro**: `_hydro_quantify.build_climatology` / `quantify_anomaly` replicate the report scripts line by line (leap day dropped, dummy-year pivot, Volue normal from the last full year, percentile of this week's mean against the same week in history; hydro balance uses mean-of-years as norm).
