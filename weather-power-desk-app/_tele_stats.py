"""Teleconnection statistics — pure pandas / scipy, no Streamlit, no Databricks.

The Teleconnections tab (_teleconnections.py) is a thin layer over these functions.

Everything works on one calendar period at a time — a month (1-12) or a calendar
quarter (1-4) — across years: one value of the index per year, one value of the
target's anomaly per year, and the question is how the two move together.

Time is carried as `t = year * 12 + month - 1` (months since year 0) so that a lead of
N months is a plain shift. A quarter is the mean of its three months; a month or a
quarter needs near-complete daily coverage or it is dropped, so a half-observed month
never passes for a month.

Standardisation: an index is converted to z-scores per calendar period (mean and
standard deviation over every year it exists, from 1979), so "+1" means one standard
deviation above that month's own history and indices on different scales (SOI in
standard deviations, NINO34 in °C, GLAAM in whatever it is) can be compared and
blended. A blend of several is the mean of the signed z-scores, standardised again.

Detrending: the temperature series carry the warming trend and Volue's wind and solar
the growing fleet, while indices such as AMO, PDO or the NINO regions drift too;
correlating two trending series finds the trend, not the link. With `detrend=True`
both series lose their linear trend in time before every statistic.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
from scipy import stats

MIN_YEARS = 10             # fewer paired years than this and no statistic is reported
MIN_COVER_INDEX = 0.8      # share of a month's days an index must have to count as observed
MIN_COVER_TARGET = 0.9     # same for a target's daily rows

RESOLUTIONS = ("Monthly", "Quarterly")
QUARTER_LABELS = ["Q1 (Jan–Mar)", "Q2 (Apr–Jun)", "Q3 (Jul–Sep)", "Q4 (Oct–Dec)"]


def n_periods(resolution: str) -> int:
    return 12 if resolution == "Monthly" else 4


def _days_in_month(year: pd.Series, month: pd.Series) -> np.ndarray:
    return np.array([calendar.monthrange(int(y), int(m))[1] for y, m in zip(year, month)])


# ══════════════════════════════════════════════════════════════════════════════
# INDICES
# ══════════════════════════════════════════════════════════════════════════════

def index_series(raw: pd.DataFrame) -> dict[str, pd.Series]:
    """Monthly index values from the app's monthly aggregate (index_name, year, month,
    value, n_days): one Series per index, indexed by t, complete months only."""
    out: dict[str, pd.Series] = {}
    if raw.empty:
        return out
    d = raw.dropna(subset=["value"]).copy()
    d = d[d["n_days"] >= MIN_COVER_INDEX * _days_in_month(d["year"], d["month"])]
    d["t"] = d["year"].astype(int) * 12 + d["month"].astype(int) - 1
    for name, g in d.groupby("index_name"):
        out[name] = g.set_index("t")["value"].sort_index()
    return out


def at_resolution(monthly: pd.Series, resolution: str, lead: int = 0) -> pd.DataFrame:
    """One value per (year, period) from a monthly series (indexed by t). `lead` shifts
    the index back: the row for year Y / period P holds the index as it was `lead`
    months before that period (for a quarter: the quarter's three months, each `lead`
    months earlier). Columns: year, period, value."""
    s = monthly.copy()
    s.index = s.index + int(lead)                 # value observed at t-lead is now filed under t
    if resolution == "Monthly":
        out = pd.DataFrame({"year": s.index // 12, "period": s.index % 12 + 1, "value": s.values})
    else:
        key = pd.DataFrame({"year": s.index // 12, "period": (s.index % 12) // 3 + 1, "value": s.values})
        g = key.groupby(["year", "period"])["value"].agg(["mean", "count"]).reset_index()
        out = g[g["count"] == 3].rename(columns={"mean": "value"})[["year", "period", "value"]]
    return out.reset_index(drop=True).astype({"year": int, "period": int})


def zscores(frame: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Add `z`: the value standardised per calendar period over all the years in `frame`."""
    f = frame.copy()
    g = f.groupby("period")[value_col]
    n, mu, sd = g.transform("count"), g.transform("mean"), g.transform("std")
    f["z"] = np.where((n >= MIN_YEARS) & (sd > 0), (f[value_col] - mu) / sd, np.nan)
    return f


def blend(frames: dict[str, pd.DataFrame], signs: dict[str, int]) -> pd.DataFrame:
    """Combine standardised indices into one: the mean of sign × z over the years every
    component exists, standardised again per period so that ±1 means one standard
    deviation of the blend. Columns: year, period, z."""
    cols = []
    for name, f in frames.items():
        c = f.set_index(["year", "period"])["z"].dropna() * signs.get(name, 1)
        cols.append(c.rename(name))
    if not cols:
        return pd.DataFrame(columns=["year", "period", "z"])
    wide = pd.concat(cols, axis=1, join="inner")
    comb = wide.mean(axis=1).rename("value").reset_index()
    return zscores(comb)[["year", "period", "z"]].dropna()


def blend_label(names: list[str], signs: dict[str, int]) -> str:
    out = ""
    for i, n in enumerate(names):
        neg = signs.get(n, 1) < 0
        out += (f"−{n}" if neg else n) if i == 0 else f" {'−' if neg else '+'} {n}"
    return out


# ══════════════════════════════════════════════════════════════════════════════
# TARGETS
# ══════════════════════════════════════════════════════════════════════════════

def reanalysis_monthly(daily_monthly: pd.DataFrame, base_years: tuple[int, int], scale: float = 0.001) -> pd.DataFrame:
    """Monthly actual / normal for the reanalysis from its monthly means (area, year,
    month, value_mw, n_days): actual = mean generation in GWh/h (× scale), normal = the
    mean of the same calendar month over `base_years` — complete months only, so a
    half-observed month cannot pull the climatology."""
    if daily_monthly.empty:
        return pd.DataFrame(columns=["area", "year", "month", "actual", "normal", "n_days"])
    d = daily_monthly.dropna(subset=["value_mw"]).copy()
    d["dim"] = _days_in_month(d["year"], d["month"])
    d = d[d["n_days"] >= MIN_COVER_TARGET * d["dim"]]
    d["actual"] = d["value_mw"] * scale
    lo, hi = base_years
    base = d[(d["year"] >= lo) & (d["year"] <= hi)].groupby(["area", "month"])["actual"].mean().rename("normal")
    d = d.join(base, on=["area", "month"])
    return d[["area", "year", "month", "actual", "normal", "n_days"]].reset_index(drop=True)


def targets_at_resolution(monthly: pd.DataFrame, resolution: str, anomaly_kind: str) -> pd.DataFrame:
    """Monthly (area, source, year, month, actual, normal, n_days) → one anomaly per
    (area, source, year, period). A quarter needs all three months, each weighted by
    its days. `anomaly_kind` 'diff' = actual − normal, 'pct' = % of normal. Columns:
    area, source, year, period, actual, normal, anomaly."""
    cols = ["area", "source", "year", "period", "actual", "normal", "anomaly"]
    if monthly.empty:
        return pd.DataFrame(columns=cols)
    d = monthly.dropna(subset=["actual", "normal"]).copy()
    d = d[d["n_days"] >= MIN_COVER_TARGET * _days_in_month(d["year"], d["month"])]
    if resolution == "Monthly":
        d["period"] = d["month"]
        g = d[["area", "source", "year", "period", "actual", "normal"]]
    else:
        d["period"] = (d["month"] - 1) // 3 + 1
        d["wa"], d["wn"] = d["actual"] * d["n_days"], d["normal"] * d["n_days"]
        g = d.groupby(["area", "source", "year", "period"]).agg(
            wa=("wa", "sum"), wn=("wn", "sum"), days=("n_days", "sum"), months=("month", "count")).reset_index()
        g = g[g["months"] == 3]
        g["actual"], g["normal"] = g["wa"] / g["days"], g["wn"] / g["days"]
        g = g[["area", "source", "year", "period", "actual", "normal"]]
    g = g.copy()
    if anomaly_kind == "diff":
        g["anomaly"] = g["actual"] - g["normal"]
    else:
        g["anomaly"] = np.where(g["normal"].abs() > 0, (g["actual"] / g["normal"] - 1) * 100, np.nan)
    return g[cols].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def detrended(s: pd.Series) -> pd.Series:
    """Residual of a linear fit of `s` against its (year) index."""
    s = s.dropna()
    if len(s) < 3:
        return s
    x = np.asarray(s.index, dtype=float)
    slope, icpt = np.polyfit(x, s.values, 1)
    return s - (slope * x + icpt)


def _paired(x: pd.Series, y: pd.Series, detrend: bool, years: tuple[int, int] | None) -> pd.DataFrame:
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner").dropna()
    if years is not None:
        df = df[(df.index >= years[0]) & (df.index <= years[1])]
    if detrend and len(df) >= 3:
        df = pd.concat([detrended(df["x"]).rename("x"), detrended(df["y"]).rename("y")], axis=1)
    return df


def pearson(x: pd.Series, y: pd.Series, detrend: bool = True, years: tuple[int, int] | None = None) -> tuple[float, float, int]:
    """(r, p, n) for two year-indexed series; NaN when fewer than MIN_YEARS pairs or no variance."""
    df = _paired(x, y, detrend, years)
    n = len(df)
    if n < MIN_YEARS or df["x"].std() == 0 or df["y"].std() == 0:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(df["x"], df["y"])
    return float(r), float(p), n


def _by_year(frame: pd.DataFrame, period: int, col: str) -> pd.Series:
    f = frame[frame["period"] == period]
    return f.set_index("year")[col].sort_index()


def corr_grid(index_frames: dict[str, pd.DataFrame], targets: pd.DataFrame, period: int,
              detrend: bool, years: tuple[int, int]) -> pd.DataFrame:
    """Correlation of every index (z, for `period`) with every (area, source)'s anomaly in
    the same period. Long frame: index, area, source, r, p, n."""
    rows = []
    tg = targets[targets["period"] == period]
    y_by = {k: g.set_index("year")["anomaly"].sort_index() for k, g in tg.groupby(["area", "source"])}
    for name, f in index_frames.items():
        x = _by_year(f, period, "z")
        for (area, source), y in y_by.items():
            r, p, n = pearson(x, y, detrend, years)
            rows.append((name, area, source, r, p, n))
    return pd.DataFrame(rows, columns=["index", "area", "source", "r", "p", "n"])


def profile_grid(index_frame: pd.DataFrame, targets: pd.DataFrame, resolution: str,
                 detrend: bool, years: tuple[int, int]) -> pd.DataFrame:
    """One index (or blend) against every (area, source) in every period of the year.
    Long frame: period, area, source, r, p, n."""
    rows = []
    for (area, source), g in targets.groupby(["area", "source"]):
        for period in range(1, n_periods(resolution) + 1):
            x = _by_year(index_frame, period, "z")
            y = _by_year(g, period, "anomaly")
            r, p, n = pearson(x, y, detrend, years)
            rows.append((period, area, source, r, p, n))
    return pd.DataFrame(rows, columns=["period", "area", "source", "r", "p", "n"])


def classify(z: pd.Series, thr: float) -> pd.Series:
    """'High' (z ≥ +thr), 'Low' (z ≤ −thr) or 'Neutral', per year."""
    return pd.Series(np.where(z >= thr, "High", np.where(z <= -thr, "Low", "Neutral")), index=z.index)


def regime_years(zs: dict[str, pd.Series], cond: dict[str, str], thr: float) -> pd.Series:
    """Years in which every condition holds at once: 'Positive' = z ≥ +thr, 'Negative' =
    z ≤ −thr, anything else is ignored. Returns a year-indexed 'Regime' / 'Rest' series
    over the years all the constrained indices exist."""
    used = [n for n, c in cond.items() if c in ("Positive", "Negative") and n in zs]
    if not used:
        return pd.Series(dtype=object)
    wide = pd.concat([zs[n].rename(n) for n in used], axis=1, join="inner").dropna()
    ok = pd.Series(True, index=wide.index)
    for n in used:
        ok &= (wide[n] >= thr) if cond[n] == "Positive" else (wide[n] <= -thr)
    return pd.Series(np.where(ok, "Regime", "Rest"), index=wide.index)


def composite(targets: pd.DataFrame, period: int, groups: pd.Series, order: list[str],
              detrend: bool, years: tuple[int, int]) -> pd.DataFrame:
    """Mean anomaly of each (area, source) in the years of each group, with the number of
    years and the Welch t-test p-value of the group against all the other years.
    Long frame: area, source, group, mean, n, n_rest, p."""
    rows = []
    groups = groups[(groups.index >= years[0]) & (groups.index <= years[1])]
    tg = targets[targets["period"] == period]
    for (area, source), g in tg.groupby(["area", "source"]):
        y = g.set_index("year")["anomaly"].sort_index().dropna()
        y = y[(y.index >= years[0]) & (y.index <= years[1])]
        if detrend and len(y) >= 3:
            y = detrended(y)
        y = y[y.index.isin(groups.index)]
        for grp in order:
            sel = y[groups.reindex(y.index) == grp]
            rest = y[groups.reindex(y.index) != grp]
            p = np.nan
            if len(sel) >= 2 and len(rest) >= 2:
                p = float(stats.ttest_ind(sel, rest, equal_var=False).pvalue)
            rows.append((area, source, grp, float(sel.mean()) if len(sel) else np.nan, len(sel), len(rest), p))
    return pd.DataFrame(rows, columns=["area", "source", "group", "mean", "n", "n_rest", "p"])


def ols(X: pd.DataFrame, y: pd.Series, detrend: bool = True, years: tuple[int, int] | None = None) -> dict | None:
    """OLS of the target on the predictors (all year-indexed; y and every column of X are
    standardised, so coefficients are standardised betas comparable across indices).
    Returns n, r2, adj_r2, loo_r2 (leave-one-year-out, from the hat matrix — the honest
    out-of-sample figure), per-predictor beta / p, and the fitted / observed series in
    the target's units. None when there are too few years for the number of predictors."""
    df = pd.concat([X, y.rename("__y")], axis=1, join="inner").dropna()
    if years is not None:
        df = df[(df.index >= years[0]) & (df.index <= years[1])]
    k = X.shape[1]
    if len(df) < max(MIN_YEARS, k + 4):
        return None
    if detrend:
        df = pd.concat([detrended(df[c]).rename(c) for c in df.columns], axis=1)
    sd = df.std()
    if (sd == 0).any():
        return None
    obs = df["__y"]
    Z = (df - df.mean()) / sd
    A = np.column_stack([np.ones(len(Z)), Z[X.columns].values])
    yz = Z["__y"].values
    AtA_inv = np.linalg.pinv(A.T @ A)
    beta = AtA_inv @ A.T @ yz
    fit = A @ beta
    resid = yz - fit
    n, p = A.shape
    sst = float(((yz - yz.mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / sst
    adj = 1 - (1 - r2) * (n - 1) / (n - p)
    h = np.einsum("ij,jk,ik->i", A, AtA_inv, A)
    loo = 1 - float(((resid / (1 - h)) ** 2).sum()) / sst
    s2 = float((resid ** 2).sum()) / (n - p)
    se = np.sqrt(np.diag(AtA_inv) * s2)
    pv = 2 * stats.t.sf(np.abs(beta / se), n - p)
    mu, s_y = obs.mean(), obs.std()
    return {
        "n": n, "r2": r2, "adj_r2": adj, "loo_r2": loo,
        "beta": dict(zip(X.columns, beta[1:])), "p": dict(zip(X.columns, pv[1:])),
        "years": list(df.index), "observed": obs.values, "fitted": fit * s_y + mu,
    }


def nearest_years(z: pd.Series, target: float, n: int, exclude: set[int] | None = None) -> pd.Series:
    """The `n` years whose z is closest to `target`, closest first (value = their z)."""
    z = z.dropna()
    if exclude:
        z = z[~z.index.isin(exclude)]
    return z.iloc[(z - target).abs().argsort()[:n]]
