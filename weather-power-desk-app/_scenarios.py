"""Scenario engine — cluster ensemble members into weather scenarios on their Z500 field.

Method (trading-week version)
-----------------------------
1. Feature matrix: one row per ensemble member, one column per (trading week ×
   grid point) WEEKLY-MEAN 500 hPa geopotential height departure from the
   ensemble mean, on the 2° grid over Europe and the near Atlantic of
   SPATIAL_BBOX (the tables fcst_member_spatial_ecens — EC-ENS, from the
   Meteomatics API via the pipeline — and fcst_member_spatial — AIFS-ENS, from
   the silver layer via the notebook). Weeks are real Monday–Sunday weeks — the
   products the desk trades — not rolling lead-day windows. The ~830 points ×
   weeks are reduced with PCA to SPATIAL_N_PCA scores; the scores are NOT
   standardised, because their variance ordering is the signal.
2. K-means (n_init=20) with a FIXED k — 2 scenarios by default, the way the
   Morning Report frames "the scenario" and "the alternative scenario". No
   silhouette-driven k selection. Clusters are relabelled by size, largest
   first, so Scenario 1 is always the consensus one; clusters smaller than
   SCENARIO_MIN_MEMBERS fold into "Other".
3. Each scenario is NAMED from its centroid map (its members' mean Z500
   departure from the ensemble mean, averaged over the clustered weeks): where
   its ridge and its trough sit (NAMED_REGIONS) and, when it resembles one of
   the 7 regimes of the Weather Regimes tab (cos-lat weighted pattern
   correlation on the common grid points ≥ SCENARIO_NAME_REGIME_MIN_R), that
   regime — "Ridge over Scandinavia, trough over Iberia ≈ Scandinavian Blocking".
4. The member → scenario map is then applied to the LAND variables of the same
   member IDs — temperature, wind, solar per country (the Volue EC-ENS members
   for EC-ENS, the AIFS-ENS member rows the pipeline writes for EC-AI) — to
   read each scenario's weekly values. Everything downstream of
   cluster_members() is provider-agnostic.

Member ID normalisation: 'M07' / '07' / 7 / 'ens07' / 'pf07' / 'm7' / 'e07' -> 7, then
the provider's MEMBER_ID_OFFSET (Volue: −1, its '01' being the control run, '02'…'51'
the perturbations m1…m50 — see _config).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from _config import (
    SCENARIO_K_DEFAULT, SCENARIO_MIN_MEMBERS, SCENARIO_MIN_WEEK_DAYS, SPATIAL_N_PCA,
    SCENARIO_NAME_MIN_AMPLITUDE_M, SCENARIO_NAME_REGIME_MIN_R, WR_REGIMES, WR_REGIME_LONG, WR_REGIME_SHORT,
    MEMBER_ID_OFFSET, SCENARIO_NWE_AREAS, SCENARIO_READ_TEMP_C, SCENARIO_READ_WIND_PCT,
)

OTHER_LABEL = 0   # scenario id reserved for the "Other" fold
WR_PATTERNS_FILE = Path(__file__).resolve().parent / "wr_patterns.npz"

# Where a ridge or a trough is said to sit: (name, lat_min, lat_max, lon_min, lon_max),
# scanned in this order so the specific regions come before the broad ones.
NAMED_REGIONS: list[tuple[str, float, float, float, float]] = [
    ("Iceland",                       62, 68, -26, -12),
    ("the Norwegian Sea",             62, 74, -12, 4),
    ("the Barents Sea",               68, 74, 15, 40),
    ("the British Isles",             49.5, 61, -11, 2),
    ("the North Sea",                 53, 62, 2, 8),
    ("Scandinavia",                   58, 72, 4, 32),
    ("the Baltic",                    53, 58, 8, 30),
    ("the Alps",                      45, 48, 6, 16),
    ("France",                        43, 51, -5, 8),
    ("Iberia",                        36, 44, -10, 4),
    ("Italy",                         37, 45, 7, 19),
    ("Central Europe",                47, 55, 5, 19),
    ("the Balkans",                   39, 47, 18, 29),
    ("the Black Sea",                 40, 47, 27, 42),
    ("Turkey",                        36, 42, 26, 42),
    ("the eastern Mediterranean",     30, 40, 22, 36),
    ("the central Mediterranean",     32, 40, 8, 22),
    ("the western Mediterranean",     35, 43, -2, 10),
    ("Eastern Europe",                45, 60, 19, 40),
    ("North Africa",                  30, 36, -10, 30),
    ("the Atlantic west of Iberia",   30, 45, -30, -10),
    ("the Atlantic west of Ireland",  45, 62, -30, -11),
    ("the North Atlantic",            30, 74, -30, 40),
]


# ─── member ids ─────────────────────────────────────────────────────────────────

def normalise_member(m) -> int | None:
    """'M07' / 'ens07' / 'm7' / 'e07' / '7' / 7.0 -> 7. Control ('Ctrl', 'M00', 0) -> 0. None if unparseable."""
    if m is None:
        return None
    if isinstance(m, (int, float, np.integer, np.floating)):
        return None if (isinstance(m, float) and np.isnan(m)) else int(m)
    s = str(m).strip().lower()
    try:
        return int(float(s))
    except ValueError:
        pass
    if s in ("avg", "mean", "ensmean"):
        return None
    if s in ("ctrl", "control", "cf", "c"):
        return 0
    digits = re.findall(r"\d+", s)
    return int(digits[-1]) if digits else None


def _with_ids(members: pd.DataFrame) -> pd.DataFrame:
    """member -> member_id: the parsed number plus the provider's MEMBER_ID_OFFSET
    (read off the `provider` column; 0 when the frame has none)."""
    df = members.copy()
    df["member_id"] = df["member"].map(normalise_member)
    df = df.dropna(subset=["member_id"])
    if "provider" in df.columns:
        df["member_id"] = df["member_id"].astype(int) + df["provider"].map(MEMBER_ID_OFFSET).fillna(0).astype(int)
    df["member_id"] = df["member_id"].astype(int)
    return df


# ─── weeks ──────────────────────────────────────────────────────────────────────

def add_week(df: pd.DataFrame, day_col: str = "day") -> pd.DataFrame:
    """ISO week start (Monday) and a 'W38' style label."""
    out = df.copy()
    d = pd.to_datetime(out[day_col]).dt.normalize()
    out["week_start"] = d - pd.to_timedelta(d.dt.weekday, unit="D")
    out["week_label"] = ["W" + str(int(w)) for w in out["week_start"].dt.isocalendar().week]
    return out


def available_weeks(days: pd.DataFrame, min_days: int = SCENARIO_MIN_WEEK_DAYS) -> pd.DataFrame:
    """Weeks a forecast covers, with how many days of each are available. `days` is any
    frame with a `day` column (member rows, or the day range of a spatial table).

    Returns DataFrame(week_start, week_label, n_days, first_day, last_day, full)
    sorted by week_start; `usable` = n_days >= min_days.
    """
    if days.empty:
        return pd.DataFrame(columns=["week_start", "week_label", "n_days", "first_day", "last_day", "full", "usable"])
    d = add_week(days[["day"]].drop_duplicates())
    g = d.groupby(["week_start", "week_label"])["day"].agg(n_days="nunique", first_day="min", last_day="max").reset_index()
    g["full"] = g["n_days"] >= 7
    g["usable"] = g["n_days"] >= min_days
    return g.sort_values("week_start").reset_index(drop=True)


def default_weeks(usable: pd.DataFrame, n: int = 2, today: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    """The next n usable trading weeks after the current one (the current week itself
    when nothing lies beyond it)."""
    today = (today or pd.Timestamp.today()).normalize()
    this_week = today - pd.Timedelta(days=today.weekday())
    nxt = [w for w in usable["week_start"] if w > this_week][:n]
    return nxt or list(usable["week_start"][:n])


# ─── the spatial feature matrix ─────────────────────────────────────────────────

def spatial_member_matrix(grid_by_week: dict, n_components: int = SPATIAL_N_PCA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PCA-reduced member × feature matrix from the gridded member fields, per trading week.

    `grid_by_week` maps week_start -> DataFrame(member, latitude, longitude, anomaly)
    already averaged over that week's days (load_member_spatial_days). Each week's
    grid is a block of columns; the blocks are concatenated so a member's row is its
    weekly field for every clustered week, then PCA keeps the dominant spatial
    modes (~830 grid points × weeks would otherwise swamp k-means).

    Returns (scores, grid): scores = DataFrame(member_id × PC1..PCn) with attrs
    explained_variance_pct / n_grid_points; grid = the un-reduced member × (week_start,
    latitude, longitude) matrix the centroid maps are drawn from. Both empty when
    fewer than 4 complete members exist.
    """
    from sklearn.decomposition import PCA

    blocks = []
    for ws, g in grid_by_week.items():
        if g is None or g.empty:
            continue
        d = _with_ids(g)
        d["week_start"] = pd.Timestamp(ws).normalize()
        wide = d.pivot_table(index="member_id", columns=["week_start", "latitude", "longitude"], values="anomaly")
        blocks.append(wide.dropna(axis=1, how="all"))
    if not blocks:
        return pd.DataFrame(), pd.DataFrame()
    grid = pd.concat(blocks, axis=1, join="inner").dropna(axis=0, how="any")
    if grid.empty or grid.shape[0] < 4:
        return pd.DataFrame(), pd.DataFrame()
    grid.index = grid.index.astype(int)
    grid.index.name = "member_id"
    grid.columns.names = ["week_start", "latitude", "longitude"]
    n_comp = int(min(n_components, grid.shape[0] - 1, grid.shape[1]))
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(grid.values)
    scores = pd.DataFrame(pcs, index=grid.index, columns=[f"PC{i + 1}" for i in range(pcs.shape[1])])
    scores.attrs["explained_variance_pct"] = float(pca.explained_variance_ratio_.sum() * 100)
    scores.attrs["n_grid_points"] = int(grid.shape[1] // max(1, len(blocks)))
    return scores, grid


# ─── clustering ─────────────────────────────────────────────────────────────────

def cluster_members(features: pd.DataFrame, k: int = SCENARIO_K_DEFAULT, seed: int = 42,
                    min_members: int = SCENARIO_MIN_MEMBERS, scale: bool = False) -> dict:
    """K-means with a fixed k on the member feature matrix.

    `scale=False` (the default, for PCA scores) keeps the columns as they are: their
    variance ordering IS the signal, and standardising would give the noise components
    the same weight as the leading mode. `scale=True` standardises each column.

    Returns dict with:
      assignments  DataFrame(member_id, scenario, raw_label)
      k, silhouette (reported, not used to choose k)
      centroids    DataFrame(scenario × feature)
      sizes        {scenario: n_members}
    Scenario ids are 1..k ordered by size (largest first); 0 = 'Other'.
    """
    if features.empty or features.shape[0] < 4:
        return {}
    X = StandardScaler().fit_transform(features.values) if scale else features.values.astype(float)
    k = int(max(2, min(k, features.shape[0] - 1)))
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    labels = km.labels_
    sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else np.nan

    counts = pd.Series(labels).value_counts()
    remap: dict[int, int] = {}
    nxt = 1
    for raw in counts.index:
        if counts[raw] < min_members:
            remap[raw] = OTHER_LABEL
        else:
            remap[raw] = nxt
            nxt += 1
    scen = np.array([remap[l] for l in labels])

    assignments = pd.DataFrame({"member_id": features.index.values, "raw_label": labels, "scenario": scen})
    centroids = features.groupby(pd.Series(scen, index=features.index, name="scenario")).mean()
    return {
        "assignments": assignments, "k": k, "silhouette": sil,
        "centroids": centroids, "sizes": assignments["scenario"].value_counts().to_dict(),
        "n_members": int(features.shape[0]),
    }


def apply_scenarios(members: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Attach the scenario id to another provider's member rows via the normalised member id."""
    if members.empty or assignments is None or assignments.empty:
        return pd.DataFrame()
    df = _with_ids(members)
    return add_week(df.merge(assignments[["member_id", "scenario"]], on="member_id", how="inner"))


# ─── centroid maps and names ────────────────────────────────────────────────────

def centroid_maps(grid: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Per scenario, its members' mean field averaged over the clustered weeks:
    DataFrame(scenario, latitude, longitude, anomaly) — the Z500 departure from the
    ensemble mean (m) that names and draws the scenario."""
    if grid.empty or assignments is None or assignments.empty:
        return pd.DataFrame(columns=["scenario", "latitude", "longitude", "anomaly"])
    out = []
    for s, members in assignments.groupby("scenario")["member_id"]:
        ids = [m for m in members if m in grid.index]
        if not ids:
            continue
        mean = grid.loc[ids].mean(axis=0)            # MultiIndex (week_start, latitude, longitude)
        m = mean.reset_index()
        m.columns = ["week_start", "latitude", "longitude", "anomaly"]
        m = m.groupby(["latitude", "longitude"], as_index=False)["anomaly"].mean()
        m.insert(0, "scenario", int(s))
        out.append(m)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["scenario", "latitude", "longitude", "anomaly"])


def region_name(lat: float, lon: float) -> str:
    for name, la0, la1, lo0, lo1 in NAMED_REGIONS:
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return name
    return f"{abs(lat):.0f}°{'N' if lat >= 0 else 'S'} {abs(lon):.0f}°{'E' if lon >= 0 else 'W'}"


_PATTERN_CACHE: dict[tuple, pd.DataFrame | None] = {}


def regime_patterns_on(points: pd.DataFrame) -> pd.DataFrame | None:
    """The 7 fixed regime patterns (wr_patterns.npz, 0.5° on 30–90 N / 80 W–40 E) on
    the grid points given (latitude, longitude): DataFrame(latitude, longitude,
    w_<regime>…). None when the file is missing or the grids do not meet."""
    key = (round(float(points["latitude"].min()), 2), round(float(points["latitude"].max()), 2),
           round(float(points["longitude"].min()), 2), round(float(points["longitude"].max()), 2), len(points))
    if key in _PATTERN_CACHE:
        return _PATTERN_CACHE[key]
    result = None
    try:
        z = np.load(WR_PATTERNS_FILE)
        pat, lats, lons = z["patterns"], z["lats"], z["lons"]
        rows = {"latitude": np.repeat(lats, len(lons)), "longitude": np.tile(lons, len(lats))}
        for k, name in enumerate(WR_REGIMES):
            rows[f"w_{name}"] = pat[k].ravel()
        full = pd.DataFrame(rows)
        full["latitude"] = full["latitude"].round(2)
        full["longitude"] = full["longitude"].round(2)
        pts = points[["latitude", "longitude"]].drop_duplicates().copy()
        pts["latitude"] = pts["latitude"].astype(float).round(2)
        pts["longitude"] = pts["longitude"].astype(float).round(2)
        merged = pts.merge(full, on=["latitude", "longitude"], how="inner")
        if len(merged) >= 0.8 * len(pts) and len(merged) >= 50:
            result = merged
    except Exception:
        result = None
    _PATTERN_CACHE[key] = result
    return result


def regime_similarity(cmap: pd.DataFrame) -> dict[str, float]:
    """cos(lat)-weighted pattern correlation of one centroid map with each regime."""
    pats = regime_patterns_on(cmap)
    if pats is None:
        return {}
    d = cmap[["latitude", "longitude", "anomaly"]].copy()
    d["latitude"] = d["latitude"].astype(float).round(2)
    d["longitude"] = d["longitude"].astype(float).round(2)
    d = d.merge(pats, on=["latitude", "longitude"], how="inner")
    if len(d) < 50:
        return {}
    w = np.cos(np.deg2rad(d["latitude"].values))
    x = d["anomaly"].values.astype(float)
    xm = np.average(x, weights=w)
    sx = np.sqrt(np.average((x - xm) ** 2, weights=w))
    out = {}
    for name in WR_REGIMES:
        p = d[f"w_{name}"].values.astype(float)
        pm = np.average(p, weights=w)
        sp = np.sqrt(np.average((p - pm) ** 2, weights=w))
        if sx > 0 and sp > 0:
            out[name] = float(np.average((x - xm) * (p - pm), weights=w) / (sx * sp))
    return out


def name_cluster(cmap: pd.DataFrame, min_amplitude: float = SCENARIO_NAME_MIN_AMPLITUDE_M,
                 min_r: float = SCENARIO_NAME_REGIME_MIN_R) -> dict:
    """A name for one scenario from its centroid map.

    Returns dict(name, ridge, trough, regime, regime_label, r, amplitude): `name` is
    "Ridge over X, trough over Y" (one side alone when the other is weak, "Close to
    the ensemble mean" when both are), `regime` the most similar regime when its
    pattern correlation reaches min_r, else None.
    """
    if cmap is None or cmap.empty:
        return {"name": "", "ridge": None, "trough": None, "regime": None, "regime_label": "", "r": np.nan, "amplitude": np.nan}
    d = cmap.dropna(subset=["anomaly"])
    hi, lo = d.loc[d["anomaly"].idxmax()], d.loc[d["anomaly"].idxmin()]
    ridge = {"region": region_name(float(hi["latitude"]), float(hi["longitude"])), "value": float(hi["anomaly"]),
             "latitude": float(hi["latitude"]), "longitude": float(hi["longitude"])}
    trough = {"region": region_name(float(lo["latitude"]), float(lo["longitude"])), "value": float(lo["anomaly"]),
              "latitude": float(lo["latitude"]), "longitude": float(lo["longitude"])}
    parts = []
    if ridge["value"] >= min_amplitude:
        parts.append(f"ridge over {ridge['region']}")
    if -trough["value"] >= min_amplitude:
        parts.append(f"trough over {trough['region']}")
    name = ", ".join(parts) if parts else "close to the ensemble mean"
    name = name[0].upper() + name[1:]
    sim = regime_similarity(d)
    regime, r = None, np.nan
    if sim:
        best = max(sim, key=sim.get)
        if sim[best] >= min_r:
            regime, r = best, sim[best]
        else:
            r = sim[best]
    return {"name": name, "ridge": ridge, "trough": trough, "regime": regime,
            "regime_label": WR_REGIME_LONG.get(regime, "") if regime else "", "r": r,
            "amplitude": float(max(ridge["value"], -trough["value"])), "similarity": sim}


def short_name(name: dict | None) -> str:
    """The label the desk reads: the short regime name when the centroid matches one
    ('Scandi Block'), else the ridge / trough description."""
    if not name:
        return ""
    if name.get("regime"):
        return WR_REGIME_SHORT.get(name["regime"], name.get("regime_label", name["regime"]))
    return name.get("name", "")


def scenario_name(s: int, names: dict | None = None) -> str:
    """'Scenario 2' — or 'Scenario 2 · Scandi Block' when the cluster names are given."""
    s = int(s)
    if s == OTHER_LABEL:
        return "Other"
    base = f"Scenario {s}"
    label = short_name((names or {}).get(s))
    return f"{base} · {label}" if label else base


def nwe_read(tagged_t: pd.DataFrame, tagged_w: pd.DataFrame, scenario: int, weeks: list[pd.Timestamp],
             areas: list[str] = SCENARIO_NWE_AREAS, temp_c: float = SCENARIO_READ_TEMP_C,
             wind_pct: float = SCENARIO_READ_WIND_PCT) -> dict:
    """How a scenario reads over North-West Europe against the full ensemble over the
    clustered weeks: temperature averaged over the countries, wind summed, each as
    the scenario's departure from the ensemble mean (per member first). Returns
    dict(d_temp, d_wind, d_wind_pct, temp, wind, line) — `line` is the one-liner:
    'Cold and low wind in NWE', 'Mild and windy in NWE', 'Flat in NWE', or a mix."""
    weeks = [pd.Timestamp(w).normalize() for w in weeks]

    def _dep(tagged: pd.DataFrame, how: str):
        if tagged is None or tagged.empty:
            return np.nan, np.nan
        df = tagged[tagged["area"].isin(areas) & tagged["week_start"].isin(weeks)]
        if df.empty:
            return np.nan, np.nan
        per_member = df.groupby(["member_id", "area"])["value"].mean().groupby("member_id").agg(how)
        ens = float(per_member.mean())
        ids = df.loc[df["scenario"] == scenario, "member_id"].unique()
        if len(ids) == 0:
            return np.nan, ens
        return float(per_member.loc[per_member.index.isin(ids)].mean() - ens), ens

    d_t, ens_t = _dep(tagged_t, "mean")
    d_w, ens_w = _dep(tagged_w, "sum")
    d_w_pct = (d_w / ens_w * 100.0) if (ens_w and not np.isnan(d_w) and ens_w != 0) else np.nan
    t_word = None if np.isnan(d_t) else ("cold" if d_t <= -temp_c else ("mild" if d_t >= temp_c else "flat"))
    w_word = None if np.isnan(d_w_pct) else ("low wind" if d_w_pct <= -wind_pct else ("windy" if d_w_pct >= wind_pct else "flat"))
    words = [w for w in (t_word, w_word) if w is not None]
    if not words:
        line = ""
    elif all(w == "flat" for w in words):
        line = "Flat in NWE"
    else:
        parts = []
        if t_word and t_word != "flat":
            parts.append(t_word)
        if w_word and w_word != "flat":
            parts.append(w_word)
        line = " and ".join(parts) + " in NWE"
        if t_word == "flat":
            line += ", temperature flat"
        elif w_word == "flat":
            line += ", wind flat"
        line = line[0].upper() + line[1:]
    return {"d_temp": d_t, "d_wind": d_w, "d_wind_pct": d_w_pct, "temp": t_word, "wind": w_word, "line": line}


def describe_scenario(read: dict) -> str:
    """The card's second line: the NWE one-liner with its numbers vs the ensemble."""
    if not read or not read.get("line"):
        return ""
    nums = []
    if not np.isnan(read.get("d_temp", np.nan)):
        nums.append(f"{read['d_temp']:+.1f} °C")
    if not np.isnan(read.get("d_wind_pct", np.nan)):
        nums.append(f"{read['d_wind_pct']:+.0f} % wind")
    return read["line"] + (f' <span class="mc-muted">({", ".join(nums)} vs the ensemble)</span>' if nums else "")


# ─── outputs ────────────────────────────────────────────────────────────────────

def scenario_daily_summary(tagged: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Per (scenario, area, day): mean, p25, p75 of member values, plus ensemble mean and normal."""
    if tagged.empty:
        return pd.DataFrame()
    g = tagged.groupby(["scenario", "area", "day"])[value_col]
    out = g.agg(mean="mean", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75), n="count").reset_index()
    ens = tagged.groupby(["area", "day"])[value_col].mean().rename("ens_mean").reset_index()
    out = out.merge(ens, on=["area", "day"], how="left")
    if "normal" in tagged.columns:
        nrm = tagged.groupby(["area", "day"])["normal"].mean().rename("normal").reset_index()
        out = out.merge(nrm, on=["area", "day"], how="left")
    return out


def scenario_weekly_values(tagged: pd.DataFrame, weeks: list[pd.Timestamp],
                           value_col: str = "value") -> pd.DataFrame:
    """Per (scenario, area, week_start): weekly mean of member values, the full-ensemble
    weekly mean, the normal, and the two deltas. This is the side-by-side input."""
    if tagged.empty:
        return pd.DataFrame()
    weeks = [pd.Timestamp(w).normalize() for w in weeks]
    df = tagged[tagged["week_start"].isin(weeks)]
    if df.empty:
        return pd.DataFrame()
    # weekly mean per member first, then across members — so a member missing
    # a day doesn't tilt the scenario mean
    pm = df.groupby(["scenario", "area", "week_start", "week_label", "member_id"])[value_col].mean().reset_index()
    out = pm.groupby(["scenario", "area", "week_start", "week_label"])[value_col].agg(mean="mean", n="count").reset_index()
    ens = df.groupby(["area", "week_start", "member_id"])[value_col].mean().groupby(["area", "week_start"]).mean()
    out = out.merge(ens.rename("ens_mean").reset_index(), on=["area", "week_start"], how="left")
    if "normal" in df.columns and df["normal"].notna().any():
        nrm = df.groupby(["area", "week_start"])["normal"].mean().rename("normal").reset_index()
        out = out.merge(nrm, on=["area", "week_start"], how="left")
    else:
        out["normal"] = np.nan
    out["d_ens"] = out["mean"] - out["ens_mean"]
    out["d_norm"] = out["mean"] - out["normal"]
    return out.sort_values(["area", "week_start", "scenario"])


def scenario_table(tagged: pd.DataFrame, weeks: list[pd.Timestamp], value_col: str = "anomaly",
                   names: dict | None = None) -> pd.DataFrame:
    """Scenario × country table of the mean value over the selected weeks."""
    if tagged.empty:
        return pd.DataFrame()
    weeks = [pd.Timestamp(w).normalize() for w in weeks]
    df = tagged[tagged["week_start"].isin(weeks)]
    col = value_col if value_col in df.columns and df[value_col].notna().any() else "value"
    t = df.groupby(["scenario", "area"])[col].mean().unstack("area")
    t.index = [scenario_name(s, names) for s in t.index]
    return t


def silhouette_for_k(features: pd.DataFrame, ks=(2, 3, 4), seed: int = 42, scale: bool = False) -> dict[int, float]:
    """Diagnostic only — how separable the members are at each k. Not used to pick k."""
    if features.empty or features.shape[0] < 5:
        return {}
    X = StandardScaler().fit_transform(features.values) if scale else features.values.astype(float)
    out = {}
    for k in ks:
        if k >= features.shape[0]:
            continue
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        if len(set(labels)) > 1:
            out[k] = float(silhouette_score(X, labels))
    return out
