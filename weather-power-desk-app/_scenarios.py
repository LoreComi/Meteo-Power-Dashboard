"""Scenario engine — cluster ensemble members into weather scenarios.

Method (trading-week version)
-----------------------------
1. Feature matrix: one row per ensemble member, one column per (country × ISO
   week) WEEKLY-MEAN temperature anomaly. Weeks are real Monday–Sunday weeks —
   the products the desk trades — not rolling lead-day windows. A week enters
   the feature space only if the forecast covers >= SCENARIO_MIN_WEEK_DAYS of
   it. Columns are standardised so every country/week weighs the same.
2. K-means (n_init=20) with a FIXED k — 2 scenarios by default, the way the
   Morning Report frames "the scenario" and "the alternative scenario". No
   silhouette-driven k selection. Clusters are relabelled by size, largest
   first, so Scenario 1 is always the consensus one; clusters smaller than
   SCENARIO_MIN_MEMBERS fold into "Other".
3. The member → scenario map is then applied to any provider's members with
   the same member IDs (Volue wind / solar / temperature, Meteologica) to read
   each scenario's power-relevant weekly values. All ECMWF-ENS-derived
   products share the same 50 perturbed members, so member 07 is the same
   atmosphere in each.

The proper input is 500 hPa geopotential members (regime clustering). That
field is not in Databricks yet; when it lands, build its member × feature
matrix and call cluster_members() unchanged — everything downstream is
provider-agnostic.

Member ID normalisation: 'M07' / '07' / 7 / 'ens07' / 'pf07' -> 7.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from _config import SCENARIO_K_DEFAULT, SCENARIO_MIN_MEMBERS, SCENARIO_MIN_WEEK_DAYS

OTHER_LABEL = 0   # scenario id reserved for the "Other" fold


# ─── member ids ─────────────────────────────────────────────────────────────────

def normalise_member(m) -> int | None:
    """'M07' / 'ens07' / '7' / 7.0 -> 7. Control ('Ctrl', 'M00', 0) -> 0. None if unparseable."""
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
    df = members.copy()
    df["member_id"] = df["member"].map(normalise_member)
    df = df.dropna(subset=["member_id"])
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


def available_weeks(members: pd.DataFrame, min_days: int = SCENARIO_MIN_WEEK_DAYS) -> pd.DataFrame:
    """Weeks the forecast covers, with how many days of each are available.

    Returns DataFrame(week_start, week_label, n_days, first_day, last_day, full)
    sorted by week_start; `usable` = n_days >= min_days.
    """
    if members.empty:
        return pd.DataFrame(columns=["week_start", "week_label", "n_days", "first_day", "last_day", "full", "usable"])
    d = add_week(members[["day"]].drop_duplicates())
    g = d.groupby(["week_start", "week_label"])["day"].agg(n_days="nunique", first_day="min", last_day="max").reset_index()
    g["full"] = g["n_days"] >= 7
    g["usable"] = g["n_days"] >= min_days
    return g.sort_values("week_start").reset_index(drop=True)


def weekly_member_matrix(members: pd.DataFrame, areas: list[str], weeks: list[pd.Timestamp],
                         value_col: str = "anomaly", min_days: int = SCENARIO_MIN_WEEK_DAYS) -> pd.DataFrame:
    """Pivot members -> one row per member, one column per (area, week_start) weekly mean.

    Uses `anomaly` when present so the clustering is about departure from
    normal, not the seasonal cycle. Members missing any feature are dropped.
    """
    if members.empty or not weeks:
        return pd.DataFrame()
    df = add_week(_with_ids(members))
    weeks = [pd.Timestamp(w).normalize() for w in weeks]
    df = df[df["area"].isin(areas) & df["week_start"].isin(weeks)]
    col = value_col if value_col in df.columns and df[value_col].notna().any() else "value"
    g = df.groupby(["member_id", "area", "week_start"])[col].agg(["mean", "count"]).reset_index()
    g = g[g["count"] >= min_days]
    wide = g.pivot_table(index="member_id", columns=["area", "week_start"], values="mean")
    wide = wide.dropna(axis=1, how="all").dropna(axis=0, how="any")
    wide.index = wide.index.astype(int)
    return wide


# ─── clustering ─────────────────────────────────────────────────────────────────

def cluster_members(features: pd.DataFrame, k: int = SCENARIO_K_DEFAULT, seed: int = 42,
                    min_members: int = SCENARIO_MIN_MEMBERS) -> dict:
    """K-means with a fixed k on the member feature matrix.

    Returns dict with:
      assignments  DataFrame(member_id, scenario, raw_label)
      k, silhouette (reported, not used to choose k)
      centroids    DataFrame(scenario × feature) in original (anomaly) units
      sizes        {scenario: n_members}
    Scenario ids are 1..k ordered by size (largest first); 0 = 'Other'.
    """
    if features.empty or features.shape[0] < 4:
        return {}
    X = StandardScaler().fit_transform(features.values)
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
    # groupby on a plain array keeps the (area, week_start) MultiIndex columns intact
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


def scenario_table(tagged: pd.DataFrame, weeks: list[pd.Timestamp], value_col: str = "anomaly") -> pd.DataFrame:
    """Scenario × country table of the mean value over the selected weeks."""
    if tagged.empty:
        return pd.DataFrame()
    weeks = [pd.Timestamp(w).normalize() for w in weeks]
    df = tagged[tagged["week_start"].isin(weeks)]
    col = value_col if value_col in df.columns and df[value_col].notna().any() else "value"
    t = df.groupby(["scenario", "area"])[col].mean().unstack("area")
    t.index = [scenario_name(s) for s in t.index]
    return t


def scenario_name(s: int) -> str:
    return "Other" if int(s) == OTHER_LABEL else f"Scenario {int(s)}"


def describe_scenario(centroid_row: pd.Series, share: float) -> str:
    """One-line description of a cluster centroid whose index is (area, week_start)."""
    vals = centroid_row.dropna()
    if vals.empty:
        return ""
    mean_anom = float(vals.mean())
    by_area = vals.groupby(level=0).mean().sort_values()
    tone = "warm" if mean_anom > 0.75 else ("cold" if mean_anom < -0.75 else "near-normal")
    return (f"{share:.0%} of members · {tone} on average ({mean_anom:+.1f}°C) · "
            f"coldest {by_area.index[0]} ({by_area.iloc[0]:+.1f}), warmest {by_area.index[-1]} ({by_area.iloc[-1]:+.1f})")


def silhouette_for_k(features: pd.DataFrame, ks=(2, 3, 4), seed: int = 42) -> dict[int, float]:
    """Diagnostic only — how separable the members are at each k. Not used to pick k."""
    if features.empty or features.shape[0] < 5:
        return {}
    X = StandardScaler().fit_transform(features.values)
    out = {}
    for k in ks:
        if k >= features.shape[0]:
            continue
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        if len(set(labels)) > 1:
            out[k] = float(silhouette_score(X, labels))
    return out
