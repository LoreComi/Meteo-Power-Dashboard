"""Temperature-response curves for EU gas demand — prediction side only.

Port of P:/QFA/TonyWeather/Lorenzo_Trainee/EU-gas-demand/gas_demand_model.py,
trimmed to what the app needs: the fitted curves are read from
`curve_models.json` (produced there by fit_demand_curves.py) and evaluated.
Nothing here fits anything — no Bloomberg demand history, no knot search, no
backtest. Keep this file in sync with gas_demand_model.py whenever the curve
format changes; the fitting script stays the source of truth.

THE MODEL (as fitted)
---------------------
    demand(t) = intercept
                + weekday_effect[daytype(t)]      (weekday / sat / sun / holiday)
                + b0*T + b_cold*relu(k_cold - T) + b_hot*relu(T - k_hot)

- `b_cold*relu(k_cold - T)` is the heating response: it switches on below
  k_cold and lifts demand as temperature falls. This is what drives LDZ.
- `b_hot*relu(T - k_hot)` is the cooling response (air-conditioning load met
  by gas-fired generation) — the "bathtub" that shapes GTP.
- Either term can be absent: `structure` is 'linear' (no knot), 'single'
  (one knot) or 'bathtub' (both).

THERMAL INERTIA
---------------
`T` above is not the day's temperature but an exponentially-weighted
"effective temperature" over the day itself and the previous MAX_LAG_DAYS
days, with the half-life the backtest picked per (country, series) and stored
on the curve as `half_life_days` (0.0 = raw same-day reading). Buildings
retain heat, so one mild day after a cold week does not erase the heating
load. Consequence for a forecast: you need a temperature *path* with at least
MAX_LAG_DAYS days of history before the first day you want to predict, not
one day's number — see _gas.py, which seeds the path with Volue actuals.

HOLIDAYS
--------
Public holidays behave like a reduced-activity day but not identically to a
Sunday, so they get their own daytype. A common EU calendar (New Year, Good
Friday, Easter Monday, 1 May, Christmas, Boxing Day) is applied to all
countries — an approximation, but it covers the low-activity periods that
would otherwise read as unexplained residual.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = ["de", "be", "uk", "fr", "it", "nl"]
SERIES = ["ldz", "ind", "gtp"]

# Smoothing window of the effective temperature. The fitted half-life lives on
# each curve; this is the number of trailing days the weights span.
MAX_LAG_DAYS = 6
HOLIDAY_DAYTYPE = "holiday"


# --------------------------------------------------------------------------
# Holiday calendar (common EU fixed + Easter-based holidays, all countries)
# --------------------------------------------------------------------------
def easter_date(year: int) -> pd.Timestamp:
    """Anonymous Gregorian algorithm for the date of Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return pd.Timestamp(year=year, month=month, day=day)


def build_holidays(years) -> set:
    holidays = set()
    for year in years:
        easter = easter_date(year)
        holidays.update({
            pd.Timestamp(year=year, month=1, day=1),      # New Year's Day
            easter - pd.Timedelta(days=2),                 # Good Friday
            easter + pd.Timedelta(days=1),                 # Easter Monday
            pd.Timestamp(year=year, month=5, day=1),       # Labour Day
            pd.Timestamp(year=year, month=12, day=25),     # Christmas Day
            pd.Timestamp(year=year, month=12, day=26),     # Boxing Day
        })
    return holidays


def assign_daytype(dates: pd.Series) -> np.ndarray:
    years = range(dates.dt.year.min(), dates.dt.year.max() + 1)
    holidays = build_holidays(years)
    dow = dates.dt.dayofweek
    is_holiday = dates.isin(holidays)
    return np.select(
        [dow == 5, dow == 6, is_holiday],
        ["sat", "sun", HOLIDAY_DAYTYPE],
        default="weekday",
    )


# --------------------------------------------------------------------------
# Thermal inertia: exponentially-weighted "effective temperature"
# --------------------------------------------------------------------------
def smooth_temperature(temp: pd.Series, half_life_days: float, max_lag: int = MAX_LAG_DAYS) -> pd.Series:
    """`temp` must be a continuous daily series (no gaps), sorted by date.
    half_life_days=0.0 returns the raw series unchanged."""
    if half_life_days <= 0:
        return temp.copy()
    weights = np.array([0.5 ** (lag / half_life_days) for lag in range(max_lag + 1)])
    weights /= weights.sum()
    smoothed = sum(w * temp.shift(lag) for lag, w in enumerate(weights))
    return smoothed


# --------------------------------------------------------------------------
# The fitted curve
# --------------------------------------------------------------------------
@dataclass
class CurveModel:
    country: str
    series: str
    structure: str
    k_cold: float | None
    k_hot: float | None
    names: list
    coef: list
    bias_correction: float
    backtest_mae: float
    backtest_mae_naive: float
    in_sample_r2: float
    last_year_actual_mean: float
    last_year_fitted_mean: float
    n_obs: int
    half_life_days: float = 0.0

    def predict(self, temp: np.ndarray, daytype: np.ndarray) -> np.ndarray:
        """`temp` must already be the effective (smoothed) temperature if
        `self.half_life_days > 0` — see `smooth_temperature`.
        `daytype` values: 'weekday', 'sat', 'sun', or 'holiday'."""
        temp = np.asarray(temp, dtype=float)
        daytype = np.asarray(daytype)
        n = len(temp)
        cols = {
            "intercept": np.ones(n),
            "is_sat": (daytype == "sat").astype(float),
            "is_sun": (daytype == "sun").astype(float),
            "is_holiday": (daytype == HOLIDAY_DAYTYPE).astype(float),
            "T": temp,
        }
        if self.k_cold is not None:
            cols["cold"] = np.maximum(self.k_cold - temp, 0.0)
        if self.k_hot is not None:
            cols["hot"] = np.maximum(temp - self.k_hot, 0.0)
        X = np.column_stack([cols[k] for k in self.names])
        return X @ np.array(self.coef) + self.bias_correction


def load_models(path: str | Path) -> dict[tuple[str, str], CurveModel]:
    """Read curve_models.json → {(series, country): CurveModel}."""
    raw = json.loads(Path(path).read_text())
    models = {}
    for key, params in raw.items():
        series, country = key.split("_", 1)
        models[(series, country)] = CurveModel(**params)
    return models


def local_slope(model: CurveModel, T: float, eps: float = 0.01) -> float:
    """Sensitivity of demand to temperature at T, in demand units per °C."""
    hi = model.predict(np.array([T + eps]), np.array(["weekday"]))[0]
    lo = model.predict(np.array([T - eps]), np.array(["weekday"]))[0]
    return (hi - lo) / (2 * eps)


def describe_shape(model: CurveModel) -> str:
    """Human-readable description of the fitted curve, derived from the actual
    local slopes rather than which basis parameterisation won the knot search
    (a single kink can be written with either a cold- or hot-side ReLU term
    and both are the same shape)."""
    if model.structure == "linear":
        s = local_slope(model, 10)
        shape = f"linear, slope {s:+.1f}/°C"
    elif model.structure == "single":
        k = model.k_cold
        s_below = local_slope(model, k - 2)
        s_above = local_slope(model, k + 2)
        kind = "heating" if s_below < s_above else "cooling"
        shape = f"{kind} hinge @ {k:.0f}°C (slope {s_below:+.1f} → {s_above:+.1f}/°C)"
    elif model.structure == "bathtub":
        kc, kh = model.k_cold, model.k_hot
        s_cold = local_slope(model, kc - 2)
        s_mid = local_slope(model, (kc + kh) / 2)
        s_hot = local_slope(model, kh + 2)
        shape = (f"3-slope: <{kc:.0f}°C ({s_cold:+.1f}/°C), "
                 f"{kc:.0f}–{kh:.0f}°C ({s_mid:+.1f}/°C), "
                 f">{kh:.0f}°C ({s_hot:+.1f}/°C)")
    else:
        shape = model.structure
    hl = model.half_life_days
    smoothing = "raw temp" if hl <= 0 else f"{hl:.1f} d thermal-inertia smoothing"
    return f"{shape} [{smoothing}]"
