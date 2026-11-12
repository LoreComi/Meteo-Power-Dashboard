"""Temperature-response curves for power demand (load) — fitting and prediction.

The Gas Demand curves (_gas_demand_model.py, fitted offline on Bloomberg gas
demand) carried over to electricity load, with what load needs on top:

    load(t) = intercept
              + daytype effect            (saturday / sunday / public holiday)
              + season(t)                 (annual harmonics — see SEASON)
              + b0*T + b_cold*relu(k_cold - T) + b_hot*relu(T - k_hot)

- The temperature part is the gas model's hinge curve: a heating branch below
  k_cold, a cooling branch above k_hot, either of them optional ('linear',
  'single', 'bathtub'), the structure and the knots chosen by the data.
- `T` is the gas model's effective temperature: an exponentially-weighted mean
  of the day and the previous MAX_LAG_DAYS days (thermal inertia), with the
  half-life picked per country by the backtest; 0.0 = the day's own reading.
- Recency: weighted least squares with a WEIGHT_HALFLIFE_YEARS half-life for
  the shape, then the level re-anchored on the last 365 days, as the gas model.

SEASON
------
Load has a yearly cycle that is not temperature: lighting follows daylight,
industry and holidays follow the calendar. Cold days are also dark days, so a
curve without a seasonal term books part of that cycle as temperature response
and overstates what one degree does. The sensitivity question — "next week
comes in 1 °C colder than forecast, same week, same daylight" — needs the
response at a fixed time of year, so the full fit carries N_HARMONICS annual
sine / cosine pairs. The explorer's per-year curves (fit_subset) are the plain
relation the scatter shows and carry none: one year cannot tell season from
temperature.

Unlike the gas curves nothing is shipped as a file: the daily history sits in
the sandbox (pipeline/eq_load.py), a fit is a few hundred small least-squares
problems, and the app fits per country on demand and caches the result until
the table is rewritten — the curves are never older than the data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from _gas_demand_model import MAX_LAG_DAYS, easter_date, smooth_temperature

WEIGHT_HALFLIFE_YEARS = 1.5
HOLDOUT_DAYS = 365
N_HARMONICS = 2
TEMP_HALFLIFE_GRID = [0.0, 0.5, 1.0, 1.5, 2.5]      # days; 0.0 = raw same-day temperature
SINGLE_KNOT_GRID = np.arange(0.0, 27.0, 1.0)
COLD_KNOT_GRID = np.arange(4.0, 19.0, 1.0)
HOT_KNOT_GRID = np.arange(15.0, 27.0, 1.0)
MIN_KNOT_GAP = 3.0
MIN_SUPPORT = 20                                    # observations needed on each side of a knot
SUBSET_SUPPORT_SHARE = 0.08                         # fit_subset: … and at least this share of the selection
MIN_FIT_DAYS = 400                                  # a full fit needs the holdout year plus something to train on
DAYTYPES = ["weekday", "sat", "sun", "holiday"]

# Public holidays that visibly cut load, on top of the gas model's common calendar
# (New Year, Good Friday, Easter Monday, 1 May, Christmas, Boxing Day). Fixed dates as
# (month, day); 'easter' offsets in days from Easter Sunday (39 = Ascension, 50 = Whit
# Monday, 60 = Corpus Christi). Not a legal calendar — regional holidays are left out.
_FIXED = {
    "DE": [(10, 3)], "FR": [(5, 8), (7, 14), (8, 15), (11, 1), (11, 11)],
    "IT": [(1, 6), (4, 25), (6, 2), (8, 15), (11, 1), (12, 8)],
    "ES": [(1, 6), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8)],
    "NL": [(4, 27)], "BE": [(7, 21), (8, 15), (11, 1), (11, 11)],
    "PL": [(1, 6), (5, 3), (8, 15), (11, 1), (11, 11)], "AT": [(1, 6), (8, 15), (10, 26), (11, 1), (12, 8)],
    "CH": [(8, 1)], "CZ": [(5, 8), (7, 5), (7, 6), (9, 28), (10, 28), (11, 17)],
    "HU": [(3, 15), (8, 20), (10, 23), (11, 1)], "PT": [(4, 25), (6, 10), (8, 15), (10, 5), (11, 1), (12, 1), (12, 8)],
    "SK": [(1, 6), (5, 8), (7, 5), (8, 29), (9, 1), (9, 15), (11, 1), (11, 17)],
    "SI": [(2, 8), (4, 27), (6, 25), (8, 15), (10, 31), (11, 1)], "HR": [(1, 6), (5, 30), (6, 22), (8, 5), (8, 15), (11, 1), (11, 18)],
    "NO": [(5, 17)], "SE": [(1, 6), (6, 6)], "FI": [(1, 6), (12, 6)], "DK": [(6, 5)],
    "EE": [(2, 24), (6, 23), (6, 24), (8, 20)], "LV": [(5, 4), (6, 23), (6, 24), (11, 18)],
    "LT": [(2, 16), (3, 11), (6, 24), (7, 6), (8, 15), (11, 1)],
}
_EASTER = {
    "DE": [39, 50], "FR": [39, 50], "NL": [39, 50], "BE": [39, 50], "AT": [39, 50, 60], "CH": [39, 50],
    "PL": [60], "HU": [50], "PT": [60], "HR": [60], "NO": [-3, 39, 50], "SE": [39], "FI": [39], "DK": [-3, 39, 50],
}
YEAR_END = [(12, 24), (12, 27), (12, 28), (12, 29), (12, 30), (12, 31)]     # the Christmas–New Year lull


def _uk_bank_holidays(year: int) -> list[pd.Timestamp]:
    """Early May, Spring and Summer bank holidays (first / last Monday of May, last of August)."""
    def monday(month: int, last: bool) -> pd.Timestamp:
        days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
        mondays = [d for d in days if d.month == month and d.dayofweek == 0]
        return mondays[-1] if last else mondays[0]
    return [monday(5, False), monday(5, True), monday(8, True)]


def holidays(area: str, years) -> set:
    out = set()
    for y in years:
        e = easter_date(y)
        out.update({pd.Timestamp(y, 1, 1), e - pd.Timedelta(days=2), e + pd.Timedelta(days=1),
                    pd.Timestamp(y, 5, 1), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26)})
        out.update(pd.Timestamp(y, m, d) for m, d in _FIXED.get(area, []) + YEAR_END)
        out.update(e + pd.Timedelta(days=k) for k in _EASTER.get(area, []))
        if area == "UK":
            out.update(_uk_bank_holidays(y))
    return out


def assign_daytype(dates, area: str) -> np.ndarray:
    """'weekday' / 'sat' / 'sun' / 'holiday' per date. Unlike the gas model a holiday
    wins over the weekend: the year-end lull is one regime whatever the weekday."""
    d = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    hol = d.isin(list(holidays(area, range(d.year.min(), d.year.max() + 1))))
    return np.select([hol, d.dayofweek == 5, d.dayofweek == 6], ["holiday", "sat", "sun"], default="weekday")


def _design(temp: np.ndarray, daytype: np.ndarray, doy: np.ndarray, k_cold, k_hot, n_harmonics: int):
    n = len(temp)
    cols = {"intercept": np.ones(n), "is_sat": (daytype == "sat").astype(float),
            "is_sun": (daytype == "sun").astype(float), "is_holiday": (daytype == "holiday").astype(float)}
    for h in range(1, n_harmonics + 1):
        ang = 2 * np.pi * h * doy / 365.25
        cols[f"sin{h}"], cols[f"cos{h}"] = np.sin(ang), np.cos(ang)
    cols["T"] = temp
    if k_cold is not None:
        cols["cold"] = np.maximum(k_cold - temp, 0.0)
    if k_hot is not None:
        cols["hot"] = np.maximum(temp - k_hot, 0.0)
    return np.column_stack(list(cols.values())), list(cols)


@dataclass
class LoadCurve:
    area: str
    structure: str
    k_cold: float | None
    k_hot: float | None
    names: list
    coef: list
    half_life_days: float = 0.0
    n_harmonics: int = 0
    bias_correction: float = 0.0
    r2: float = np.nan
    resid_std: float = np.nan              # MW, weighted, in sample
    backtest_mae: float = np.nan           # MW on the held-out last year (full fits only)
    backtest_mae_naive: float = np.nan     # same, daytype + linear raw temperature
    n_obs: int = 0
    mean_load: float = np.nan              # MW over the fitted days (last 365 for a full fit)
    t_min: float = np.nan                  # temperature range the fit saw
    t_max: float = np.nan

    def _c(self, name: str) -> float:
        return float(self.coef[self.names.index(name)]) if name in self.names else 0.0

    def predict(self, temp, dates) -> np.ndarray:
        """Load in MW. `temp` is the effective temperature when half_life_days > 0."""
        temp = np.asarray(temp, dtype=float)
        d = pd.DatetimeIndex(pd.to_datetime(dates))
        X, _ = _design(temp, assign_daytype(d, self.area), d.dayofyear.to_numpy(float),
                       self.k_cold, self.k_hot, self.n_harmonics)
        return X @ np.asarray(self.coef, dtype=float) + self.bias_correction

    def response(self, temp) -> np.ndarray:
        """The temperature part alone (MW, arbitrary level) — differences of it are
        load changes at a fixed day type and time of year."""
        temp = np.asarray(temp, dtype=float)
        out = self._c("T") * temp
        if self.k_cold is not None:
            out = out + self._c("cold") * np.maximum(self.k_cold - temp, 0.0)
        if self.k_hot is not None:
            out = out + self._c("hot") * np.maximum(temp - self.k_hot, 0.0)
        return out

    def slope(self, temp) -> np.ndarray:
        """d load / d T in MW per °C at `temp` (negative = heating regime)."""
        temp = np.asarray(temp, dtype=float)
        out = np.full(temp.shape, self._c("T"))
        if self.k_cold is not None:
            out = out - self._c("cold") * (temp < self.k_cold)
        if self.k_hot is not None:
            out = out + self._c("hot") * (temp > self.k_hot)
        return out

    def curve(self, temps, date=None) -> np.ndarray:
        """The fitted line for plotting: a working day, at `date`'s time of year when
        the curve has a season (else the season term is absent anyway)."""
        temps = np.asarray(temps, dtype=float)
        base = self._c("intercept") + self.bias_correction + self.response(temps)
        if self.n_harmonics and date is not None:
            doy = float(pd.Timestamp(date).dayofyear)
            for h in range(1, self.n_harmonics + 1):
                ang = 2 * np.pi * h * doy / 365.25
                base = base + self._c(f"sin{h}") * np.sin(ang) + self._c(f"cos{h}") * np.cos(ang)
        return base

    def segments(self) -> list[tuple[str, float]]:
        """[(temperature range, MW/°C)] — the slopes the curve actually has, coldest first."""
        knots = sorted(k for k in (self.k_cold, self.k_hot) if k is not None)
        if not knots:
            return [("all temperatures", float(self.slope(np.array([10.0]))[0]))]
        edges = [-np.inf] + knots + [np.inf]
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mid = (knots[0] - 1) if np.isinf(lo) else ((knots[-1] + 1) if np.isinf(hi) else (lo + hi) / 2)
            label = f"below {hi:.0f} °C" if np.isinf(lo) else (f"above {lo:.0f} °C" if np.isinf(hi) else f"{lo:.0f}–{hi:.0f} °C")
            out.append((label, float(self.slope(np.array([mid]))[0])))
        return out


def _wls(X, y, w):
    # solved on the p x p normal equations rather than the n x p weighted design: the same
    # curves (checked to 1e-7 MW on every default country) at about half the cost, and
    # lstsq on the small system still gives an all-zero column (a daytype a selection
    # does not have) a zero coefficient
    Xw = X * np.clip(w, 1e-12, None)[:, None]
    coef, *_ = np.linalg.lstsq(Xw.T @ X, Xw.T @ y, rcond=None)
    return coef, float(np.sum(w * (y - X @ coef) ** 2))


def _candidates(T: np.ndarray, structure: str, support: int) -> list[tuple]:
    def ok(k):
        return (T < k).sum() >= support and (T >= k).sum() >= support
    if structure == "linear":
        return [(None, None)]
    if structure == "single":
        # one free kink, written as a cold-side term: with the knot free this covers a
        # heating hinge and a cooling hinge alike (the gas model's parameterisation)
        return [(k, None) for k in SINGLE_KNOT_GRID if ok(k)]
    return [(kc, kh) for kc in COLD_KNOT_GRID if ok(kc) for kh in HOT_KNOT_GRID if ok(kh) and kh >= kc + MIN_KNOT_GAP]


def _best(T, y, daytype, doy, w, structure: str, n_harmonics: int, support: int = MIN_SUPPORT) -> dict | None:
    best = None
    for kc, kh in _candidates(T, structure, support):
        X, names = _design(T, daytype, doy, kc, kh, n_harmonics)
        coef, wsse = _wls(X, y, w)
        if best is None or wsse < best["wsse"]:
            best = {"k_cold": kc, "k_hot": kh, "coef": coef, "names": names, "wsse": wsse}
    return best


def _prepare(df: pd.DataFrame, area: str) -> pd.DataFrame:
    """Daily frame (day, load, temp) → gap-free, with every candidate effective temperature,
    daytype and day of year. Days without load stay in (the smoothing needs their
    temperature) and are dropped by the callers."""
    d = df[["day", "load", "temp"]].copy()
    d["day"] = pd.to_datetime(d["day"]).dt.normalize()
    d = d.drop_duplicates("day").set_index("day").sort_index()
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    d["temp"] = d["temp"].interpolate(limit=3)
    for hl in TEMP_HALFLIFE_GRID:
        d[f"t{hl}"] = smooth_temperature(d["temp"], hl)
    d["daytype"] = assign_daytype(d.index, area)
    d["doy"] = d.index.dayofyear.astype(float)
    return d


def effective_temperature(temp: pd.Series, half_life_days: float) -> pd.Series:
    """The curve's `T` from a gap-free daily temperature path (NaN for the first MAX_LAG_DAYS days)."""
    return smooth_temperature(temp, half_life_days, max_lag=MAX_LAG_DAYS)


def fit_curve(df: pd.DataFrame, area: str, n_harmonics: int = N_HARMONICS,
              half_life_years: float = WEIGHT_HALFLIFE_YEARS, holdout_days: int = HOLDOUT_DAYS) -> LoadCurve | None:
    """The country's curve from its whole history (frame: day, load MW, temp °C).

    As the gas model's fit_curve: (temperature half-life, structure) chosen on the
    held-out last year, the winner refitted on everything with the recency weights,
    the level re-anchored on the last year. None when the history is too short."""
    d = _prepare(df, area)
    d = d.dropna(subset=["load"] + [f"t{hl}" for hl in TEMP_HALFLIFE_GRID])
    if len(d) < MIN_FIT_DAYS:
        return None
    last = d.index.max()
    hold = d.index > last - pd.Timedelta(days=holdout_days)
    train, test = d[~hold], d[hold]

    def weights(idx):
        return 0.5 ** ((idx.max() - idx).days.to_numpy() / 365.25 / half_life_years)

    def mae(fit, hl, frame):
        X, _ = _design(frame[f"t{hl}"].to_numpy(), frame["daytype"].to_numpy(), frame["doy"].to_numpy(),
                       fit["k_cold"], fit["k_hot"], n_harmonics)
        return float(np.mean(np.abs(frame["load"].to_numpy() - X @ fit["coef"])))

    y_tr, dt_tr, doy_tr, w_tr = train["load"].to_numpy(), train["daytype"].to_numpy(), train["doy"].to_numpy(), weights(train.index)
    scored = []
    for hl in TEMP_HALFLIFE_GRID:
        for structure in ("linear", "single", "bathtub"):
            fit = _best(train[f"t{hl}"].to_numpy(), y_tr, dt_tr, doy_tr, w_tr, structure, n_harmonics)
            if fit is not None:
                # the holdout is scored after removing its mean error: the level is
                # re-anchored below anyway, the choice here is about the shape
                X, _ = _design(test[f"t{hl}"].to_numpy(), test["daytype"].to_numpy(), test["doy"].to_numpy(),
                               fit["k_cold"], fit["k_hot"], n_harmonics)
                err = test["load"].to_numpy() - X @ fit["coef"]
                scored.append((float(np.mean(np.abs(err - err.mean()))), hl, structure))
    if not scored:
        return None
    backtest_mae, hl, structure = min(scored)
    X0, _ = _design(train["t0.0"].to_numpy(), dt_tr, doy_tr, None, None, 0)
    c0, _ = _wls(X0, y_tr, w_tr)
    Xt0, _ = _design(test["t0.0"].to_numpy(), test["daytype"].to_numpy(), test["doy"].to_numpy(), None, None, 0)
    e0 = test["load"].to_numpy() - Xt0 @ c0
    naive_mae = float(np.mean(np.abs(e0 - e0.mean())))

    T, y, w = d[f"t{hl}"].to_numpy(), d["load"].to_numpy(), weights(d.index)
    fit = _best(T, y, d["daytype"].to_numpy(), d["doy"].to_numpy(), w, structure, n_harmonics)
    X, names = _design(T, d["daytype"].to_numpy(), d["doy"].to_numpy(), fit["k_cold"], fit["k_hot"], n_harmonics)
    fitted = X @ fit["coef"]
    bias = float((y[hold] - fitted[hold]).mean())
    resid = y - fitted - bias
    wm = np.average(y, weights=w)
    return LoadCurve(area=area, structure=structure, k_cold=fit["k_cold"], k_hot=fit["k_hot"], names=names,
                     coef=[float(c) for c in fit["coef"]], half_life_days=float(hl), n_harmonics=n_harmonics,
                     bias_correction=bias, r2=float(1 - np.sum(w * resid ** 2) / np.sum(w * (y - wm) ** 2)),
                     resid_std=float(np.sqrt(np.average(resid ** 2, weights=w))), backtest_mae=backtest_mae,
                     backtest_mae_naive=naive_mae, n_obs=int(len(d)), mean_load=float(y[hold].mean()),
                     t_min=float(np.nanmin(T)), t_max=float(np.nanmax(T)))


def fit_subset(days, load, temp, area: str, min_days: int = 30) -> LoadCurve | None:
    """The plain hinge curve of a selection of days (one year, a few months): daytype
    effects, no season, no recency weights; the structure by BIC, since there is no
    year to hold out. `temp` is whatever the scatter's x axis is. None under min_days."""
    T, y = np.asarray(temp, dtype=float), np.asarray(load, dtype=float)
    keep = ~(np.isnan(T) | np.isnan(y))
    T, y, d = T[keep], y[keep], pd.DatetimeIndex(pd.to_datetime(days))[keep]
    n = len(y)
    if n < min_days:
        return None
    daytype, doy, w = assign_daytype(d, area), d.dayofyear.to_numpy(float), np.ones(n)
    best = None
    for structure in ("linear", "single", "bathtub"):
        fit = _best(T, y, daytype, doy, w, structure, 0, support=max(MIN_SUPPORT, int(SUBSET_SUPPORT_SHARE * n)))
        if fit is None:
            continue
        # a knot is searched over a grid, which costs more than one fitted coefficient: count it twice
        k = len(fit["names"]) + 2 * ((fit["k_cold"] is not None) + (fit["k_hot"] is not None))
        bic = n * np.log(max(fit["wsse"], 1e-9) / n) + k * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, structure, fit)
    _, structure, fit = best
    # a daytype that does not occur in the selection has an all-zero column; lstsq gives it 0
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return LoadCurve(area=area, structure=structure, k_cold=fit["k_cold"], k_hot=fit["k_hot"], names=fit["names"],
                     coef=[float(c) for c in fit["coef"]], r2=float(1 - fit["wsse"] / ss_tot) if ss_tot > 0 else np.nan,
                     resid_std=float(np.sqrt(fit["wsse"] / n)), n_obs=n, mean_load=float(y.mean()),
                     t_min=float(T.min()), t_max=float(T.max()))


def linear_slope(temp, load) -> tuple[float, float, int]:
    """(MW per °C, R², n) of a straight line through a small cell (one month of one year)."""
    T, y = np.asarray(temp, dtype=float), np.asarray(load, dtype=float)
    keep = ~(np.isnan(T) | np.isnan(y))
    T, y = T[keep], y[keep]
    if len(y) < 8 or np.std(T) < 0.5:
        return np.nan, np.nan, int(len(y))
    b, a = np.polyfit(T, y, 1)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum((y - (a + b * T)) ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return float(b), float(r2), int(len(y))
