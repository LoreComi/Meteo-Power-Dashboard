"""Build the wind-capacity weights for capacity-weighted wind speed per region (the AI models' wind block).

    python build_wind_weights.py --gpkg <wind_power_europe.gpkg>
    python build_wind_weights.py --gpkg <...> --out <csv>

Input   wind_power_europe.gpkg — one POINT per wind farm (layer `wind_power`: geom, capacity_mw,
        country as an English name; EPSG:4326). Read with sqlite3 and struct only — a GeoPackage is a
        SQLite file and a point is 16 bytes of WKB behind a short header — so no geo library is needed.
Output  weather-power-desk-app/wind_weights_0p5.csv with columns
        area, latitude, longitude, capacity_mw — one row per 0.5° grid point per country: the MW of
        that country's farms standing in the 0.5° cell centred on the point. The layout of
        pop_weights_0p5.csv, so the same join serves both: a region's wind speed is
        SUM(value * capacity_mw) / SUM(capacity_mw) over the cells of its countries — the wind where the
        turbines are, offshore included, not the mean over a country's area.

        Read by power_desk_refresh.py (cell 1a → {SBX}.wind_weights, cell 7 → the Meteomatics models'
        'ws' rows of morning_daily) and by pipeline/google_weathernext.py (the WeatherNext 'ws' rows).

Grid    Points at multiples of 0.5°, extent 34–71.5 N, −12–36 E — the Meteomatics silver grid and the
        extent of the population weights. Farms outside it (the Canaries, Madeira, the Azores, the
        Faroes, Russia east of 36 E) are not counted; neither are countries the app has no code for.
"""
from __future__ import annotations

import argparse
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "weather-power-desk-app" / "wind_weights_0p5.csv"
RES = 0.5
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 34.0, 71.5, -12.0, 36.0      # as build_pop_weights.py

# The file's country names → the app's area codes (ISO-2, the UK as 'UK'). Åland is Finland's.
COUNTRY_CODES = {
    "Germany": "DE", "United Kingdom": "UK", "Spain": "ES", "France": "FR", "Sweden": "SE", "Netherlands": "NL",
    "Poland": "PL", "Italy": "IT", "Finland": "FI", "Åland Islands": "FI", "Greece": "GR", "Romania": "RO",
    "Norway": "NO", "Portugal": "PT", "Denmark": "DK", "Ireland": "IE", "Belgium": "BE", "Austria": "AT",
    "Lithuania": "LT", "Croatia": "HR", "Estonia": "EE", "Serbia": "RS", "Bulgaria": "BG", "North Macedonia": "MK",
    "Bosnia and Herzegovina": "BA", "Hungary": "HU", "Montenegro": "ME", "Albania": "AL", "Czech Republic": "CZ",
    "Kosovo": "XK", "Luxembourg": "LU", "Latvia": "LV", "Switzerland": "CH", "Slovenia": "SI", "Slovakia": "SK",
}
_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}             # GeoPackage header flags → envelope size


def read_points(gpkg: Path) -> pd.DataFrame:
    """fid, country, lon, lat, capacity_mw of every farm in the layer."""
    con = sqlite3.connect(f"file:{gpkg.as_posix()}?mode=ro", uri=True)
    rows = []
    try:
        for fid, blob, cap, country in con.execute("SELECT fid, geom, capacity_mw, country FROM wind_power"):
            lon = lat = None
            if blob is not None and blob[:2] == b"GP":
                wkb = blob[8 + _ENVELOPE_BYTES[(blob[3] >> 1) & 7]:]
                order = "<" if wkb[0] == 1 else ">"
                if struct.unpack(order + "I", wkb[1:5])[0] % 1000 == 1:          # a point (with or without Z / M)
                    lon, lat = struct.unpack(order + "dd", wkb[5:21])
            rows.append((fid, country, lon, lat, cap))
    finally:
        con.close()
    return pd.DataFrame(rows, columns=["fid", "country", "lon", "lat", "capacity_mw"])


def build(points: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p = points.copy()
    p["area"] = p["country"].map(COUNTRY_CODES)
    ok = p["lon"].notna() & p["capacity_mw"].gt(0)
    inside = p["lat"].between(LAT_MIN - RES / 2, LAT_MAX + RES / 2) & p["lon"].between(LON_MIN - RES / 2, LON_MAX + RES / 2)
    stats = {"farms": len(p), "gw": p["capacity_mw"].sum() / 1000,
             "no_geometry_or_capacity": int((~ok).sum()),
             "unmapped": p.loc[ok & p["area"].isna()].groupby("country")["capacity_mw"].sum().div(1000).round(2).to_dict(),
             "outside": p.loc[ok & p["area"].notna() & ~inside].groupby("area")["capacity_mw"].sum().div(1000).round(2).to_dict()}
    p = p[ok & p["area"].notna() & inside]
    p["latitude"] = np.clip((p["lat"] / RES).round() * RES, LAT_MIN, LAT_MAX)
    p["longitude"] = np.clip((p["lon"] / RES).round() * RES, LON_MIN, LON_MAX)
    w = (p.groupby(["area", "latitude", "longitude"], as_index=False)["capacity_mw"].sum()
         .sort_values(["area", "latitude", "longitude"]).reset_index(drop=True))
    w["capacity_mw"] = w["capacity_mw"].round(1)
    stats["kept_gw"] = w["capacity_mw"].sum() / 1000
    return w, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpkg", type=Path, required=True, help="wind_power_europe.gpkg")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    w, st = build(read_points(a.gpkg))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(a.out, index=False)
    print(f"{st['farms']:,} farms, {st['gw']:.1f} GW in the file → {len(w):,} cells, {w['area'].nunique()} countries, "
          f"{st['kept_gw']:.1f} GW kept → {a.out}")
    print(f"  not counted — outside the grid (GW): {st['outside']}; no area code (GW): {st['unmapped']}; "
          f"no geometry / capacity: {st['no_geometry_or_capacity']}")
    by = (w.groupby("area")["capacity_mw"].sum() / 1000).round(1).sort_values(ascending=False)
    print("  GW per country: " + ", ".join(f"{a_} {v}" for a_, v in by.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
