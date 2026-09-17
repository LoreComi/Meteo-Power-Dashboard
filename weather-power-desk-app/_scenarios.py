"""Scenario engine — cluster ensemble members into weather scenarios.

Method
------
1. Feature matrix: one row per ensemble member, columns = (area × lead day)
   temperature anomaly (optionally + wind anomaly), for the chosen countries
   and horizon. Standardised per column so a windy Nordic day and a warm
   Iberian day weigh the same.
2. K-means (n_init=20) with k chosen by silhouette score in SCENARIO_K_RANGE,
   or fixed by the user. Clusters are re-labelled by size, largest first, so
   "Scenario 1" is always the consensus/most-populated one.
3. Clusters below SCENARIO_MIN_MEMBERS are folded into an "Other" bucket.
4. The member -> scenario map is then applied to any other provider's members
   with the same member IDs (Volue wind/solar/temperature, Meteologica) to
   read the power-relevant values of each scenario. All ECMWF-ENS-derived
   products (Meteomatics, Volue, Meteologica) share the same 50 perturbed
   members, so member "M07" is the same atmosphere in each.

Member ID normalisation
-----------------------
Providers label members differently ('M07', '07', 7, 'ens07', 'pf07'). We map
each to the integer member number so IDs line up across providers.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from _config import SCENARIO_K_RANGE, SCENARIO_MIN_MEMBERS, SPATIAL_N_PCA

OTHER_LABEL = 0   # scenario id reserved for the "Other" fold


def normalise_member(m) -> int | None:
    """'M07' / 'ens07' / '7' / 7.0 -> 7. Control member ('Ctrl', 'M00', 0) -> 0. None if unparseable."""
    if m is None:
        return None
    if isinstance(m, (int, float, np.integer, np.floating)):
        return None if (isinstance(m, float) and np.isnan(m)) else int(m)
    s = str(m).strip().lower()
    try:                       # '7', '7.0'
        return int(float(s))
    except ValueError:
        pass
    if s in ("avg", "mean", "ensmean"):
        return None
    if s in ("ctrl", "control", "cf", "c"):
        return 0
    digits = re.findall(r"\d+", s)
    if not digits:
        return None
    return int(digits[-1])


def member_matrix(members: pd.DataFrame, areas: list[str], lead_days: tuple[int, int],
                  value_col: str = "anomaly") -> pd.DataFrame:
    """Pivot (member × [area, lead_day]) -> wide feature matrix, one row per member.

    Rows with any missing feature are dropped (a member missing one country
    can't be placed). Uses anomaly when available so that the clustering is
    about *departure from normal* rather than the seasonal cycle.
    """
    if members.empty:
        return pd.DataFrame()
    df = members.copy()
    df["member_id"] = df["member"].map(normalise_member)
    df = df.dropna(subset=["member_id"])
    df = df[df["area"].isin(areas) & df["lead_day"].between(lead_days[0], lead_days[1])]
    col = value_col if value_col in df.columns and df[value_col].notna().any() else "value"
    wide = df.pivot_table(index="member_id", columns=["area", "lead_day"], values=col, aggfunc="mean")
    wide = wide.dropna(axis=1, how="all").dropna(axis=0, how="any")
    wide.index = wide.index.astype(int)
    return wide


def spatial_member_matrix(grid_df: pd.DataFrame,
                          n_components: int = SPATIAL_N_PCA) -> pd.DataFrame:
    """Build a PCA-reduced feature matrix from gridded member anomaly data.

    The input *grid_df* is already averaged across days by the SQL loader
    (``load_member_spatial``), so each row is one (member, lat, lon) with a
    single anomaly value.

    Steps
    -----
    1. Pivot to (member × grid_points).
    2. PCA to *n_components* — the 50 members live in a very high-dimensional
       space (~1 800 grid points at 1°); PCA keeps the dominant modes of
       spatial variability while discarding noise.
    3. Return DataFrame with member_id index and PC columns, ready for
       ``cluster_members()``.
    """
    from sklearn.decomposition import PCA

    if grid_df.empty:
        return pd.DataFrame()
    df = grid_df.copy()
    df["member_id"] = df["member"].map(normalise_member)
    df = df.dropna(subset=["member_id"])

    # Pivot to wide: member × grid_points (already day-averaged by SQL)
    df["gp"] = df["latitude"].astype(str) + "_" + df["longitude"].astype(str)
    wide = df.pivot_table(index="member_id", columns="gp", values="anomaly")
    wide = wide.dropna(axis=1, how="all").dropna(axis=0, how="any")

    if wide.empty or wide.shape[0] < 4:
        return pd.DataFrame()

    n_comp = min(n_components, wide.shape[0] - 1, wide.shape[1])
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(wide.values)

    result = pd.DataFrame(
        pcs,
        index=wide.index.astype(int),
        columns=[f"PC{i+1}" for i in range(pcs.shape[1])],
    )
    result.index.name = "member_id"
    # Stash explained variance for captions downstream
    result.attrs["explained_variance_pct"] = float(pca.explained_variance_ratio_.sum() * 100)
    result.attrs["n_grid_points"] = int(wide.shape[1])
    return result


def choose_k(X: np.ndarray, k_range: tuple[int, int] = SCENARIO_K_RANGE, seed: int = 42) -> tuple[int, dict[int, float]]:
    """Silhouette-selected k. Returns (best_k, {k: silhouette})."""
    n = X.shape[0]
    scores: dict[int, float] = {}
    for k in range(k_range[0], min(k_range[1], n - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        scores[k] = float(silhouette_score(X, labels))
    if not scores:
        return 2, scores
    return max(scores, key=scores.get), scores


def cluster_members(features: pd.DataFrame, k: int | None = None, seed: int = 42,
                    min_members: int = SCENARIO_MIN_MEMBERS) -> dict:
    """Run k-means on the member feature matrix.

    Returns dict with:
      assignments  DataFrame(member_id, scenario, raw_label)
      k, silhouette, silhouette_by_k
      centroids    DataFrame(scenario × feature) in original (anomaly) units
      sizes        {scenario: n_members}
    Scenario ids are 1..k ordered by size (largest first); 0 = 'Other'.
    """
    if features.empty or features.shape[0] < 4:
        return {}
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    sil_by_k: dict[int, float] = {}
    if k is None:
        k, sil_by_k = choose_k(X)
    k = int(max(2, min(k, features.shape[0] - 1)))
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    labels = km.labels_
    sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else np.nan

    # relabel by size, fold tiny clusters
    counts = pd.Series(labels).value_counts()
    order = list(counts.index)
    remap: dict[int, int] = {}
    nxt = 1
    for raw in order:
        if counts[raw] < min_members:
            remap[raw] = OTHER_LABEL
        else:
            remap[raw] = nxt
            nxt += 1
    scen = np.array([remap[l] for l in labels])

    assignments = pd.DataFrame({"member_id": features.index.values, "raw_label": labels, "scenario": scen})
    centroids = (features.assign(scenario=scen).groupby("scenario").mean())
    sizes = assignments["scenario"].value_counts().to_dict()
    return {
        "assignments": assignments, "k": k, "silhouette": sil, "silhouette_by_k": sil_by_k,
        "centroids": centroids, "sizes": sizes, "n_members": int(features.shape[0]),
    }


def apply_scenarios(members: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Attach the scenario id to another provider's member rows via the normalised member id."""
    if members.empty or assignments is None or assignments.empty:
        return pd.DataFrame()
    df = members.copy()
    df["member_id"] = df["member"].map(normalise_member)
    df = df.dropna(subset=["member_id"])
    df["member_id"] = df["member_id"].astype(int)
    return df.merge(assignments[["member_id", "scenario"]], on="member_id", how="inner")


def scenario_daily_summary(tagged: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Per (scenario, area, day): mean, p25, p75 of member values, plus ensemble mean for reference."""
    if tagged.empty:
        return pd.DataFrame()
    g = tagged.groupby(["scenario", "area", "day"])[value_col]
    out = g.agg(mean="mean", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75),
                n="count").reset_index()
    ens = tagged.groupby(["area", "day"])[value_col].mean().rename("ens_mean").reset_index()
    out = out.merge(ens, on=["area", "day"], how="left")
    if "normal" in tagged.columns:
        nrm = tagged.groupby(["area", "day"])["normal"].mean().rename("normal").reset_index()
        out = out.merge(nrm, on=["area", "day"], how="left")
    return out


def scenario_table(tagged: pd.DataFrame, value_col: str = "anomaly", label: str = "T anomaly") -> pd.DataFrame:
    """Scenario × area table of horizon-mean values (what each scenario means per country)."""
    if tagged.empty:
        return pd.DataFrame()
    col = value_col if value_col in tagged.columns and tagged[value_col].notna().any() else "value"
    t = tagged.groupby(["scenario", "area"])[col].mean().unstack("area")
    t.index = [scenario_name(s) for s in t.index]
    t.columns.name = label
    return t


def scenario_name(s: int) -> str:
    return "Other" if int(s) == OTHER_LABEL else f"Scenario {int(s)}"


def describe_scenario(centroid_row: pd.Series, sizes_share: float) -> str:
    """One-line plain-English description of a cluster centroid (temperature anomaly features)."""
    vals = centroid_row.dropna()
    if vals.empty:
        return ""
    mean_anom = float(vals.mean())
    by_area = vals.groupby(level=0).mean().sort_values()
    coldest, warmest = by_area.index[0], by_area.index[-1]
    tone = "warm" if mean_anom > 0.75 else ("cold" if mean_anom < -0.75 else "near-normal")
    return (f"{sizes_share:.0%} of members · {tone} on average ({mean_anom:+.1f}°C) · "
            f"coldest {coldest} ({by_area.iloc[0]:+.1f}), warmest {warmest} ({by_area.iloc[-1]:+.1f})")
