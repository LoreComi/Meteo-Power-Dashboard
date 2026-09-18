# Power Desk — Weather Dashboard (Databricks App)

Built on the structure of `LPG-desk-dashboard`: a Streamlit Databricks App that reads
pre-aggregated sandbox tables, plus a Databricks notebook that refreshes those tables
on a Lakeflow schedule. Nothing here runs locally — deploy both pieces to the workspace.

```
Power_dashboard/
├── power_desk_refresh.py          Databricks notebook → dna_snbx_weather.power_desk.*  (schedule: every 6 h)
└── weather-power-desk-app/        Databricks App (Streamlit)
    ├── app.py                     landing page (tiles) + sidebar routing
    ├── app.yaml                   warehouse + env vars + Azure OpenAI secrets
    ├── requirements.txt
    ├── .streamlit/config.toml
    ├── _config.py                 schemas, areas, metrics, models, section definitions
    ├── _data.py                   Statement-API queries against the sandbox tables
    ├── _charts.py                 Plotly builders (fan, heatmaps, scenarios, hydro, gas demand)
    ├── _style.py                  design tokens + CSS (same palette as the LPG app)
    ├── _ui.py                     KPI cards / banners
    ├── _morning.py                Morning Call — the Morning Report table, live + model comparison
    ├── _ai_brief.py               Morning Call commentary — power and gas agent families
    ├── _forecast.py               Section 1 — Values · Uncertainty · Scenarios
    ├── _scenarios.py              member clustering engine (weekly means, fixed k)
    ├── _historical.py             Section 2 — history by country · index analogues
    ├── _hydro.py                  Section 3 — Hydro Report quantify_* live
    ├── _hydro_quantify.py         port of Hydro_Report/quantify_{reservoir,snow,balance}.py
    ├── _gas.py                    Section 4 — port of EU-gas-demand/{ldz,rdl}_forecast.py
    ├── _gas_demand_model.py       port of EU-gas-demand/gas_demand_model.py (prediction only)
    ├── curve_models.json          fitted LDZ/IND/GTP curves — copied from EU-gas-demand/outputs/
    └── _strategy.py               Section 5 — locked / work in progress
```

## Sections

| # | Section | Content |
|---|---------|---------|
| 0 | **Morning Call** (first page, full-width tile) | The Morning Report table (`Morning_Report/import_00z_add_solar_np_tot.py` → `table_00z_add_solar_np_tot.xlsx`) computed live from the Volue tables instead of wapi + Excel. Same weekday logic (Mon/Tue: this week + next; Wed/Thu: weekend + next week; Fri–Sun: next week + week after), same rows (Temperatures FRA DE UK ITA HUN Nordic Iberia · Wind DE UK FRA ITA SEE Nordic Iberia · Solar DE FRA ITA SEE Nordic Iberia · Precip Alps(cwe+it-nord) Nordic SEE Iberia as a 2-week sum), same three columns (abs. value, Δ vs previous 00z — Δ -72h on Mondays —, Δ norm) with green/red signed deltas. The Excel's free-text Pattern / Comment boxes and the Week 3 commentary block are **replaced by two agent families** that write that commentary from the same numbers (see below). Plus "confront the models": the same windows for EC 12z, GFS 00z, EC-Extended and the Meteomatics EC-ENS / AIFS-ENS country means, with the difference to EC 00z. |
| 1 | **Forecast** | *Values*: every Volue ensemble value by country (mean, median, P10/P25/P75/P90, min/max, normal, anomaly), fan chart per country, earlier runs faded, Meteomatics EC-ENS / AIFS-ENS country means on top. *Uncertainty*: across-member spread per lead day vs the normal spread for that lead day (last 365 days of 00z runs), ratio heatmap and verdicts, daily member box plots and a per-day histogram. *Scenarios*: k-means on members' **weekly-mean** temperature anomaly per country over real Mon–Sun trading weeks, **two scenarios fixed** (no silhouette selection), then **per country the two scenarios side by side** for temperature, wind and solar (weekly value, Δ vs ensemble mean, Δ vs normal) with a three-panel chart. A *Spatial grid clustering* switch clusters on the weekly-mean anomaly field (Meteomatics ecmwf-ens members, 1° grid, PCA) instead of country averages; it uses temperature as a proxy until Z500 geopotential members exist for ecmwf-ens. |
| 2 | **Historical & Analysis** | Monthly or weekly Volue actuals vs 30-yr normal by country; multi-select years and months; anomaly bars, actual-vs-normal lines, weekly anomaly heatmap, year × month heatmap per country; CSV export. *Anomaly Maps*: ERA5 gridded anomaly over Europe (temperature, 200 hPa wind, precipitation) for any set of years and months. Analogues tab reads `weather_indexes` and composites country anomalies over the closest analog years. |
| 3 | **Hydro Monitoring** | Reservoir levels (`wtr`), snow & groundwater (`sgw`), hydro balance (`bal`): climatology chart (grey history, blue recent years, dashed norm, red current year with direct label), anomaly GWh and % of normal, percentile of this week vs history, change vs last week, and the `stats_*.txt` text — the exact numbers `quantify_*.py` prints. |
| 4 | **Gas Demand** | Port of `EU-gas-demand/ldz_forecast.py` and `rdl_forecast.py`, live off the sandbox tables instead of wapi. *LDZ*: each run's ensemble-mean temperature through the fitted hinge curves (`curve_models.json`) → heating gas demand per country, seeded with trailing Volue actuals so the multi-day effective temperature is not cold-started; run-over-run cumulative delta, LONG/SHORT/NO TRADE per country and cumulatively; the script's two-panel chart (effective temperature and LDZ, both runs, delta window marked) and a fitted-curve table with shape, thermal inertia and backtest MAE vs the naive model. *Wind & Solar*: wind + solar converted to the gas-fired generation they displace (GW × 24 ÷ 50 % efficiency × 80 % displacement share) per region, against the normal, same delta and signal with the opposite sign convention. *Both legs*: the two side by side with the net. Everything downloadable as CSV. |
| 5 | **Strategy** | Locked, work-in-progress sign. |

### Morning Call — the two agent families

The free-text boxes are gone; `_ai_brief.py` runs two families over the numbers already on screen (and nothing else — no web, no news):

| Family | Specialists | Synthesis |
|---|---|---|
| **Power market** | temperature → load (France's electric heating, summer cooling and thermal derates, Nordic snowmelt) · wind & solar → residual load and the merit order (spreads, the midday block, negative-price setups) · precipitation → hydro by catchment (Alps, Nordic, SEE, Iberia) and over what horizon it bites | nets the three, names where they reinforce or offset, the part of the curve it applies to, and what would flip it |
| **Gas market** | temperature → LDZ heating demand (the non-linear heating range, thermal inertia) · wind & solar → gas-for-power displacement | nets the two demand legs, quoting the net in GWh |

Each specialist is shown only its own rows, plus the run pairing, the date, the season and — when the run covers a window only partly — the day count, so a four-day mean is never read as a full week. Each ends on a `SIGNAL: BULLISH|BEARISH|NEUTRAL` line, stripped from the prose and rendered as a card; the synthesis card is the family's net read. Two expanders show the specialist reads and the exact document each agent was given.

The gas family is additionally handed the Gas Demand section's LDZ and displaced-gas figures for the same runs (`_gas.gas_demand_snapshot`, computed on demand whether or not the section has been opened), with their sign conventions spelled out, so the commentary and the GWh cannot drift apart. Without Azure OpenAI credentials the table renders in full and only the commentary is unavailable.

## Data sources and what was verified

| Source | Where | Notes |
|--------|-------|-------|
| Volue ("volue_deltashare") | `dna_prod_silver.volue.*` | Per-member ensembles via `tag` (`'Avg'` = mean, other tags = members). Columns: `curve_name, reference_date, delivery_start, value, area, data_type, tag, categories(array)`. No `volue_deltashare` catalog is referenced anywhere in the TonyWeather codebase; if the share is mounted under another name, change `VOLUE` in the notebook and `VOLUE_SCHEMA` in `app.yaml`. |
| Meteomatics | `dna_prod_silver.meteomatics.temperature_forecast` | Individual members are in the `curve_member` column (rows with an empty `curve_member` are the ensemble mean). The refresh job builds country-mean members for `ecmwf-ens` and `ecmwf-aifs-ens` into `fcst_members` and a 1° gridded member anomaly (`fcst_member_spatial`, ecmwf-ens) for spatial clustering. Only `ecmwf-aifs-ens` has Z500 geopotential members today; `SCENARIO_USE_GEOPOTENTIAL` switches the spatial field once ecmwf-ens gets them. |
| Gold-layer climatology | `dna_prod_gold.weather.*_meteomatics_climatology` | ERA5 gridded value / normal / anomaly per day. The job pre-aggregates monthly grid-point means into `anomaly_map` (the app SP has no gold access) for the Historical → Anomaly Maps tab (matplotlib + cartopy). |
| Meteologica | not in Databricks | Consumed only via REST in other projects. `meteologica_members` is created empty with the `fcst_members` layout; set `METEOLOGICA_TABLE` when an ingestion exists. |
| Weather indexes | `dna_snbx_weather.power_desk.weather_indexes` | Created empty (`index_name, date, value, source, loaded_at`). Load NAO/AO/EA/SCAND/PNA/ONI/MJO/QBO/SSW here and the analogue tab activates. |
| Fitted gas demand curves | `weather-power-desk-app/curve_models.json` | Copied from `EU-gas-demand/outputs/curve_models.json`, fitted offline by `fit_demand_curves.py` (Bloomberg demand history + Volue temperature absolutes). Not read from Databricks — re-run that script and copy the file to refresh. `_gas_demand_model.py` is the prediction half of `gas_demand_model.py`; verified to reproduce it exactly on all 18 curves. |
| Azure OpenAI | not in Databricks | Consumed via REST for the Morning Call agent families. Credentials resolve `st.secrets` → Databricks secrets (scope `axpo`) → `AZURE_*` env vars; `app.yaml` wires the same `commodity-news-secrets` scope the coal dashboard uses. Deployment name from `AI_BRIEF_MODEL` (default `gpt-4o`). |

## Deploy

1. Import `power_desk_refresh.py` as a notebook, run it once, then schedule it (Lakeflow Jobs, every 6 h; one run should sit at ~05:30 UTC so the Morning Call has today's 00z). It creates the schema, applies grants to the app service principal, and writes: `fcst_runs, fcst_daily, fcst_spread_clim, fcst_members, fcst_member_spatial, hist_daily, hydro_daily, morning_daily, gas_demand_daily, anomaly_map, weather_indexes, meteologica_members`. The notebook must run as a user with `dna_prod_gold` access (the anomaly-map cell reads the gold layer). **`gas_demand_daily` is new** — an existing deployment must re-run the notebook before the Gas Demand section has anything to read.
2. Create a Databricks App from `weather-power-desk-app/` (same warehouse `2f4ff6b7c65abb1a` as the other desk apps). The app authenticates as its service principal and falls back to the forwarded user token. Grant it READ on the three `commodity-news-secrets` keys in `app.yaml` if the Morning Call commentary is wanted; without them every section still works.
3. First checks in the workspace:
   - `SELECT tag, COUNT(*) FROM dna_prod_silver.volue.temperature_consumption_forecast WHERE data_type='F' AND reference_date >= current_date()-1 GROUP BY tag` — should show ~51 member tags. If it shows P10…P90 style tags, the Volue share carries percentiles, not members, and the Uncertainty/Scenario tabs need a different source.
   - `SELECT DISTINCT area FROM dna_prod_silver.volue.hydro_reservoir` — confirm `IT`, `SEE`, `NP`, `CH`, `AT` codes match `HYDRO_AREA_CODES` in `_config.py` (the wapi report used `it-nord`).

## Method notes

- **Spread vs normal spread**: for each EC-ENS 00z run in the last 365 days, the std across members of the daily country mean, per lead day; the app compares today's std with the mean (and P25–P75) for the same lead day, preferring the same calendar month when ≥10 runs exist. Ratio ≥1.3 flags high uncertainty, ≤0.7 unusual confidence.
- **Scenarios**: features = weekly-mean temperature anomaly per (country × ISO week), weeks needing ≥4 forecast days; standardised; k-means (n_init=20) with k fixed at 2 (user can raise to 4); clusters relabelled by size (Scenario 1 = consensus); clusters <3 members folded into "Other". Member IDs are normalised (`M07`, `07`, `ens07` → 7) so a cluster found on one provider is read on another — all providers are the same 50 ECMWF perturbations. Silhouette is shown only as a separability diagnostic.
- **Morning Call**: `morning_daily` holds, per run of the last 8 days and per exact wapi curve name, the daily CET 'Avg' mean (tt/wnd/spv) or daily sum (rre) plus the normal. The app drops partial days (<90 % of the 15-min points, the report's `[:-1]`), picks today's 00z and the previous 00z (Friday's on Monday) and reproduces the weekly-window means and the 2-week precipitation sums exactly as the script does.
- **Hydro**: `_hydro_quantify.build_climatology` / `quantify_anomaly` replicate the report scripts line by line (leap day dropped, dummy-year pivot, Volue normal from the last full year, percentile of this week's mean against the same week in history; hydro balance uses mean-of-years as norm).
- **Gas Demand**: `gas_demand_daily` holds, per pattern and per run of the last 8 days, the daily CET mean of the `'Avg'` tag for temperature (°C), wind and solar (GW, i.e. MWh/h × 0.001) plus the normal, for DE UK FR BE NL IT ES PT. `fcst_daily` cannot serve this — it keeps only six runs *per model family* and the EC-ENS family interleaves 00z and 12z, so a Monday's "ec00ens vs the ec00ens of three days ago" falls outside its window. The run pairing and the delta window are the scripts' exactly: a 00z run against the same pattern's previous 00z (three days back on a Monday), a 12z run against the same `date_min`'s 00z (two days back on a Monday); the window opens the day after `date_min` and closes 13 days after it, or on `date_min` + 11 days on a Monday — which is in both cases the last day of the older run's 15-day horizon, hence the shorter Monday window. Deltas are summed inclusively over that window. LDZ is bullish-positive, displaced gas bearish-positive; ±1 000 GWh is the cumulative trade threshold. Trailing actual temperature comes from `hist_daily` (`'AF'`), which is the quantity `ldz_forecast.py` was approximating with a 1-day-ahead deterministic forecast; the combined actual-plus-forecast path is forced gap-free before smoothing, because `smooth_temperature` shifts by position and a missing day would silently misalign the weights. Iberia is ES + PT (the scripts use Volue's `ib` aggregate, which covers the same two grids); a region missing a grid is summed from what is present and flagged, since that shifts the level but not the run-over-run delta.
- **Morning Call commentary**: seven LLM calls at most (power 3 + 1, gas 2 + 1), temperature 0.3, one independent call per specialist so no specialist can anchor another, and a synthesis that sees only the specialists' prose plus — for gas — the desk's own LDZ and displaced-gas figures. Week 3 reaches the agents only when the run covers at least `SCENARIO_MIN_WEEK_DAYS` of it, the same minimum the scenario engine applies to a forecast week.
