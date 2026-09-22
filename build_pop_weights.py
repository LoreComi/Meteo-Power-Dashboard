"""Build the population weights the refresh notebook uses for Meteomatics country means.

Run once, offline, wherever rasterio + geopandas exist (the Morning_Report environment):

    python build_pop_weights.py
    python build_pop_weights.py --tif <GHS-POP tif> --shp <Europe_merged.shp> --out <csv>

Input   GHS-POP E2025 (JRC GHSL, 30 arcsec, EPSG:4326) — persons per ~1 km cell — and
        Europe_merged.shp with a COUNTRY column; both as used by
        Lorenzo_Trainee/Morning_Report/meteomatics_morning_report.py.
Output  weather-power-desk-app/pop_weights_0p5.csv with columns
        area, latitude, longitude, population — one row per 0.5° grid point per country:
        the persons of that country living in the 0.5° cell centred on the point.
        Upload it next to wr_patterns.npz and point POP_WEIGHTS_PATH at it; the notebook
        loads it once into {SBX}.pop_weights and every Meteomatics country mean becomes
        SUM(value * population) / SUM(population) over the cells that joined.

Method  Population is assigned to countries at the raster's own ~1 km resolution — the
        country polygon is rasterised onto the 30" grid (cell centres inside the polygon)
        — and only then summed into the 0.5° cells. The GHS grid is exactly 1/120°, so a
        0.5° cell is a 60 × 60 block of fine cells with edges on the .25 / .75 lines,
        and the aggregation is a plain block sum, exact. Two consequences that the usual
        "mask at the coarse grid" approach gets wrong:
          - a border cell carries population for both countries, each its own share;
          - coastal population is kept even when the 0.5° cell centre falls at sea
            (Lisbon, the Dutch coast, Marseille …), instead of being dropped with the cell.
        Storing persons rather than normalised fractions means any group of countries
        (Nordics, Iberia, SEE) is weighted correctly by simply summing over the group.

Grid    Points at multiples of 0.5° (…, 48.0, 48.5, …), the Meteomatics silver grid.
        Extent 34–71.5 N, −12–36 E: mainland Europe plus the near islands. The Canaries,
        Madeira, the Azores and Svalbard fall outside and are deliberately not counted.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
MR = Path(r"P:\QFA\TonyWeather\Lorenzo_Trainee\Morning_Report")
DEFAULT_TIF = MR / "GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif"
DEFAULT_SHP = MR / "Europe_merged.shp"
DEFAULT_OUT = HERE / "weather-power-desk-app" / "pop_weights_0p5.csv"

RES = 0.5                                   # target grid step (degrees)
FINE_PER_COARSE = 60                        # 0.5° / (1/120°)
# Coarse grid POINTS — the silver table's latitude / longitude values.
LAT_POINTS = np.round(np.arange(71.5, 34.0 - 1e-9, -RES), 2)     # north → south, like the raster
LON_POINTS = np.round(np.arange(-12.0, 36.0 + 1e-9, RES), 2)
# Their cell EDGES: a point at 48.5 stands for the cell 48.25–48.75.
LAT_EDGE_N, LAT_EDGE_S = LAT_POINTS[0] + RES / 2, LAT_POINTS[-1] - RES / 2      # 71.75, 33.75
LON_EDGE_W, LON_EDGE_E = LON_POINTS[0] - RES / 2, LON_POINTS[-1] + RES / 2      # -12.25, 36.25

# Dashboard area code → COUNTRY values in Europe_merged.shp. Mirrors AREAS in
# weather-power-desk-app/_config.py; extend both together.
COUNTRIES: dict[str, list[str]] = {
    "DE": ["Germany"], "FR": ["France"], "IT": ["Italy"], "ES": ["Spain"], "UK": ["United Kingdom"],
    "NL": ["Netherlands"], "BE": ["Belgium"], "PL": ["Poland"], "CZ": ["Czechia", "Czech Republic"],
    "HU": ["Hungary"], "NO": ["Norway"], "SE": ["Sweden"], "FI": ["Finland"], "DK": ["Denmark"],
    "PT": ["Portugal"], "SI": ["Slovenia"], "SK": ["Slovakia"], "HR": ["Croatia"],
    "EE": ["Estonia"], "LV": ["Latvia"], "LT": ["Lithuania"],
}

# Rough resident population, millions (Eurostat / ONS, 1 Jan 2024) — only a sanity
# check on the totals, never used in the weights.
REFERENCE_MILLIONS = {
    "DE": 83.4, "FR": 68.4, "IT": 58.9, "ES": 48.6, "UK": 68.3, "NL": 17.9, "BE": 11.8, "PL": 36.6,
    "CZ": 10.9, "HU": 9.6, "NO": 5.5, "SE": 10.5, "FI": 5.6, "DK": 5.9, "PT": 10.6, "SI": 2.1,
    "SK": 5.4, "HR": 3.9, "EE": 1.4, "LV": 1.9, "LT": 2.9,
}


def _window_for_edges(src) -> tuple[Window, "rasterio.Affine"]:
    """The raster window whose edges are the coarse grid's outer edges.

    Fails if the raster is not the 1/120° EPSG:4326 grid. The GHS-POP origin is
    not exactly on a whole degree (−180.0079, 89.0996), so the fine cell edges
    miss the .25 / .75 lines by up to half a fine cell — ~450 m at the coarse
    cell boundary, nothing at 0.5°. The window is rounded to the nearest fine
    cell and the misalignment is reported.
    """
    t = src.transform
    if src.crs is None or src.crs.to_epsg() != 4326:
        sys.exit(f"raster CRS is {src.crs}, expected EPSG:4326")
    if not (abs(t.a - 1 / 120) < 1e-9 and abs(-t.e - 1 / 120) < 1e-9):
        sys.exit(f"raster resolution is {t.a} x {-t.e}, expected 1/120 degree")
    col0, row0 = (LON_EDGE_W - t.c) / t.a, (LAT_EDGE_N - t.f) / t.e
    misalign_deg = max(abs(col0 - round(col0)), abs(row0 - round(row0))) / 120
    print(f"  raster edges miss the coarse cell edges by {misalign_deg:.4f} deg (~{misalign_deg * 111:.2f} km)")
    win = Window(int(round(col0)), int(round(row0)),
                 len(LON_POINTS) * FINE_PER_COARSE, len(LAT_POINTS) * FINE_PER_COARSE)
    return win, src.window_transform(win)


def _block_sum(fine: np.ndarray) -> np.ndarray:
    h, w = fine.shape
    return fine.reshape(h // FINE_PER_COARSE, FINE_PER_COARSE, w // FINE_PER_COARSE, FINE_PER_COARSE).sum(axis=(1, 3))


def build(tif: Path, shp: Path) -> pd.DataFrame:
    import geopandas as gpd

    t0 = time.time()
    with rasterio.open(tif) as src:
        win, tr = _window_for_edges(src)
        print(f"reading {win.width} x {win.height} cells of {tif.name} …", flush=True)
        pop = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float64)
        if src.nodata is not None:
            pop[pop == src.nodata] = 0.0
    pop[~np.isfinite(pop) | (pop < 0)] = 0.0
    print(f"  {pop.sum() / 1e6:,.1f} M persons in the window ({time.time() - t0:.0f}s)")

    gdf = gpd.read_file(shp)
    if "COUNTRY" not in gdf.columns:
        sys.exit(f"{shp.name} has no COUNTRY column (columns: {list(gdf.columns)})")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    names_in_file = set(gdf["COUNTRY"].astype(str))

    rows = []
    for code, names in COUNTRIES.items():
        hit = [n for n in names if n in names_in_file]
        if not hit:
            print(f"  !! {code}: none of {names} in the shapefile — skipped")
            continue
        geom = unary_union(gdf.loc[gdf["COUNTRY"].isin(hit), "geometry"])
        # cell centres inside the polygon, on the fine grid — ~1 km, so the
        # country line is drawn where it is, not at the coarse cell
        inside = geometry_mask([geom], transform=tr, invert=True, out_shape=pop.shape)
        coarse = _block_sum(np.where(inside, pop, 0.0))
        ii, jj = np.nonzero(coarse > 0.5)                       # at least one person
        rows.append(pd.DataFrame({"area": code, "latitude": LAT_POINTS[ii], "longitude": LON_POINTS[jj],
                                  "population": np.round(coarse[ii, jj]).astype(np.int64)}))
        tot = coarse.sum() / 1e6
        ref = REFERENCE_MILLIONS.get(code)
        flag = "" if ref is None or 0.85 <= tot / ref <= 1.15 else "   <-- check"
        print(f"  {code}  {hit[0]:<16s} {tot:7.2f} M in {len(ii):4d} cells"
              + (f"   (ref {ref:5.1f} M, x{tot / ref:.2f})" if ref else "") + flag, flush=True)

    out = pd.concat(rows, ignore_index=True).sort_values(["area", "latitude", "longitude"]).reset_index(drop=True)
    print(f"done in {time.time() - t0:.0f}s — {len(out):,} rows, {out['area'].nunique()} countries")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tif", type=Path, default=DEFAULT_TIF)
    ap.add_argument("--shp", type=Path, default=DEFAULT_SHP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    for p in (a.tif, a.shp):
        if not p.exists():
            sys.exit(f"missing input: {p}")
    df = build(a.tif, a.shp)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({a.out.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
