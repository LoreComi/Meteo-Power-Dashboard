"""European Weather Regimes — Forecast section, fourth tab.

Port of Franziska_Intern/corso_model_wr/working_wr_tool (uber_main.py →
run_WR_tool.py → max_proj_members.py), reading Databricks instead of the
daily NetCDF drops on the P: drive.

What the tool does and this reproduces
--------------------------------------
Every ensemble member, on every forecast day, is projected onto the 7 fixed
North-Atlantic / European weather regimes of Michel & Rivière (2011) — ScTr,
GL, EuBl, AR, AT, ScBl, ZO — and assigned to whichever it resembles most, if
it resembles it strongly enough. That gives three things a desk can trade on:

  the IWR curves        how strongly the ensemble mean projects onto each
                        regime through the forecast, with the lead-decaying
                        threshold and grey shading where nothing clears it
  the member spread     what share of members sits in each regime each day —
                        the honest measure of how settled the pattern is
  the climatology       what all of that looks like against 1979-present
                        reanalysis: is this run's pattern normal for the
                        season, does it persist longer or shorter than usual,
                        and are the transitions the usual ones

The projection itself runs in power_desk_refresh.py (it is an inner product
over 22 M gridded values, done as a Spark join); this module reads the ~900
rows per run it writes and does the assembly, episode and climatology maths.

Differences from the tool, and why
----------------------------------
- Source: dna_prod_silver.meteomatics.geopotential_height_forecast rather than
  the DSL download. The silver layer carries per-member 500 hPa geopotential
  for ECMWF AIFS-ENS (50 members) and NCEP GFS-ENS (30), not the IFS ENS the
  tool uses, so AIFS — ECMWF's own ensemble — is the default and GFS is
  offered next to it. Member shares are therefore out of 50 or 30, not 51.
- Horizon defaults to 15 days because the threshold's lead decay was fitted
  over leads 0-14; the table carries 17 and the extra days can be shown, with
  the threshold extrapolated and flagged.
- The member scoring and the k-means scenario clustering from the tool are not
  here: the scoring needs the 12 general_scoring_*.xlsx sheets and only feeds a
  score-weighted mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    WR_REGIMES, WR_NO_REGIME, WR_ALL_LABELS, WR_REGIME_LONG, WR_MODELS, WR_DEFAULT_MODEL,
    WR_THRESHOLD, WR_LEAD_FACTOR, WR_DEFAULT_HORIZON_DAYS, WR_MAX_HORIZON_DAYS,
    WR_CLIM_DOY_WINDOW, WR_CLIM_MIN_YEARS, WR_COLORS, SBX_SCHEMA,
    WR_IMPACT_TEMP_REGIONS, WR_IMPACT_WIND_REGIONS, WR_IMPACT_SEASON_DAYS,
)
from _charts import (
    make_wr_iwr_chart, make_wr_clim_compare, make_wr_seasonal_clim,
    make_wr_persistence, make_wr_transition_heatmap, make_wr_forecast_change_chart,
    make_wr_impact_chart,
)
from _data import load_wr_forecast_members, load_wr_reanalysis, load_wr_impact_daily, _region_key
from _ui import kpi_card, kpi_row, status_banner, view_switch

IWR_COLS = [f"iwr_{r}" for r in WR_REGIMES]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED MATHS
# ══════════════════════════════════════════════════════════════════════════════

WR_TRANS_MIN_DAYS = 4   # a transition counts only if the regime it lands in lasts this long


def doy_365(days: pd.Series) -> pd.Series:
    """Day of year on a fixed 365-day calendar — 29 Feb folds onto 28 Feb, so a
    day-of-year climatology has the same number of samples everywhere."""
    return days.dt.dayofyear - ((days.dt.month > 2) & days.dt.is_leap_year).astype(int)


def clean_singles(seq: list) -> list:
    """Isolated one-day flips become 'no regime'.

    clean_singles_2() in the tool: a regime that differs from both of its
    neighbours is a single-day wobble in the projection, not a regime, and
    would otherwise inflate the transition count and shorten every episode.
    """
    out = list(seq)
    for k in range(1, len(out) - 1):
        if out[k] != out[k - 1] and out[k] != out[k + 1]:
            out[k] = WR_NO_REGIME
    return out


def episodes(seq: list, days: list | None = None) -> list[tuple]:
    """Maximal runs of the same regime → [(regime, duration, start_index), ...]."""
    if not seq:
        return []
    out, cur, n, start = [], seq[0], 1, 0
    for i, v in enumerate(seq[1:], start=1):
        if v == cur:
            n += 1
        else:
            out.append((cur, n, start))
            cur, n, start = v, 1, i
    out.append((cur, n, start))
    return out


def threshold_for_lead(lead: np.ndarray | pd.Series) -> np.ndarray:
    """The tool's lead-decaying assignment threshold."""
    return WR_THRESHOLD + np.asarray(lead, dtype=float) * WR_LEAD_FACTOR


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST SIDE
# ══════════════════════════════════════════════════════════════════════════════

def member_sequences(run: pd.DataFrame, clean: bool = True) -> dict[str, list]:
    """{member: [regime per lead day]} for one run, in lead order."""
    out = {}
    for member, g in run.sort_values("lead_day").groupby("member"):
        seq = list(g["regime"])
        out[member] = clean_singles(seq) if clean else seq
    return out


def member_percentages(run: pd.DataFrame, clean: bool = True) -> pd.DataFrame:
    """Share of members in each regime per day — the headline of the tool.

    Computed from the per-member sequences so the single-day cleaning is
    reflected, and normalised by the members actually present rather than an
    assumed 51, because AIFS carries 50 and GFS 30.
    """
    if run.empty:
        return pd.DataFrame()
    seqs = member_sequences(run, clean=clean)
    days = sorted(run["day"].unique())
    n = len(seqs)
    if not n:
        return pd.DataFrame()
    rows = []
    for i, day in enumerate(days):
        counts = {lab: 0 for lab in WR_ALL_LABELS}
        for seq in seqs.values():
            if i < len(seq):
                counts[seq[i]] = counts.get(seq[i], 0) + 1
        rows.append({lab: counts[lab] / n * 100 for lab in WR_ALL_LABELS})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(days))


def ensemble_iwr(run: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict, list]:
    """(mean IWR per regime per day, threshold per day, P25-P75 band, no-regime days)."""
    if run.empty:
        return pd.DataFrame(), pd.Series(dtype=float), {}, []
    g = run.groupby("day")
    mean = g[IWR_COLS].mean()
    mean.columns = WR_REGIMES
    lo, hi = g[IWR_COLS].quantile(0.25), g[IWR_COLS].quantile(0.75)
    lo.columns = hi.columns = WR_REGIMES
    band = {r: (lo[r], hi[r]) for r in WR_REGIMES}
    lead = g["lead_day"].first()
    thr = pd.Series(threshold_for_lead(lead.to_numpy()), index=mean.index)
    no_days = [d for d in mean.index if mean.loc[d].max() < thr.loc[d]]
    return mean, thr, band, no_days


def dominant_regime(perc: pd.DataFrame) -> pd.Series:
    """Per day, the regime the most members sit in (ties → the earlier regime)."""
    if perc.empty:
        return pd.Series(dtype=object)
    return perc.idxmax(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# CLIMATOLOGY SIDE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def climatology(rean: pd.DataFrame, window: int = WR_CLIM_DOY_WINDOW,
                season_doy: int | None = None, season_days: int = 30) -> dict:
    """Everything the reanalysis says is normal.

    Returns
    -------
    doy_freq   365 x 8 — % of days in each regime, pooling +/- `window` days
               around each day of year so a single calendar day is not read off
               ~45 samples alone
    persistence  per regime: episode count, mean, p25, p75 (episodes >= 5 days,
               the filter the tool's histograms use)
    transitions  8 x 8 — % of exits from each regime into the next. Only
               transitions whose destination (a regime or "no regime") then lasts
               at least WR_TRANS_MIN_DAYS days are counted.
    years      how many years the record spans

    season_doy  if given, persistence and transitions only use episodes that
               start within +/- `season_days` of this day of year, in every year
               of the record (episodes are built on the full series first, so
               their durations are not cut at the window edge).
    """
    if rean.empty:
        return {}
    df = rean.copy()
    df["regime"] = clean_singles(list(df.sort_values("day")["regime"]))
    df["doy"] = doy_365(df["day"])
    df = df[df["doy"].between(1, 365)]

    # seasonal frequency
    freq = np.zeros((365, len(WR_ALL_LABELS)))
    by_doy = {d: g["regime"].tolist() for d, g in df.groupby("doy")}
    for target in range(1, 366):
        offs = [((target - 1 + o) % 365) + 1 for o in range(-window, window + 1)]
        pool = [r for o in offs for r in by_doy.get(o, [])]
        if not pool:
            continue
        vc = pd.Series(pool).value_counts(normalize=True) * 100
        for k, lab in enumerate(WR_ALL_LABELS):
            freq[target - 1, k] = float(vc.get(lab, 0.0))
    doy_freq = pd.DataFrame(freq, index=range(1, 366), columns=WR_ALL_LABELS)

    # persistence and transitions, from maximal runs
    ds = df.sort_values("day")
    eps = episodes(list(ds["regime"]))
    start_doy = doy_365(ds["day"].reset_index(drop=True))
    ep = pd.DataFrame([(r, n, int(start_doy.iloc[st])) for r, n, st in eps],
                      columns=["regime", "duration", "doy"])
    if season_doy is not None:
        gap = (ep["doy"] - season_doy).abs()
        ep = ep[np.minimum(gap, 365 - gap) <= season_days]
    long_ep = ep[(ep.regime != WR_NO_REGIME) & (ep.duration >= 5)]
    pers = {}
    for r in WR_REGIMES:
        d = long_ep[long_ep.regime == r]["duration"]
        pers[r] = {"n": int(len(d)),
                   "mean": float(d.mean()) if len(d) else np.nan,
                   "p25": float(d.quantile(0.25)) if len(d) else np.nan,
                   "p75": float(d.quantile(0.75)) if len(d) else np.nan}

    # consecutive episodes in the (possibly seasonal) set; the pair must be adjacent
    # in the full series, so work from the unfiltered list and filter by the
    # destination's start date
    tm = pd.DataFrame(0.0, index=WR_ALL_LABELS, columns=WR_ALL_LABELS)
    for (a, _, _), (b, nb, sb) in zip(eps[:-1], eps[1:]):
        if nb < WR_TRANS_MIN_DAYS:
            continue
        if season_doy is not None:
            g = abs(int(start_doy.iloc[sb]) - season_doy)
            if min(g, 365 - g) > season_days:
                continue
        tm.loc[a, b] += 1
    tm = tm.div(tm.sum(axis=1).replace(0, np.nan), axis=0) * 100

    years = df["day"].dt.year.nunique()
    return {"doy_freq": doy_freq, "persistence": pers, "transitions": tm.fillna(0),
            "years": years, "first": df["day"].min(), "last": df["day"].max()}


def window_climatology(clim: dict, days: pd.DatetimeIndex) -> pd.Series:
    """Climatological regime share averaged over the forecast's calendar days —
    the like-for-like comparison against this run's member shares."""
    if not clim or clim.get("doy_freq") is None or len(days) == 0:
        return pd.Series(dtype=float)
    d = doy_365(pd.Series(days))
    rows = clim["doy_freq"].reindex([x for x in d if 1 <= x <= 365])
    return rows.mean() if len(rows) else pd.Series(dtype=float)


@st.cache_data(ttl=86400, show_spinner=False)
def reanalysis_episodes(rean: pd.DataFrame) -> pd.DataFrame:
    """The reanalysis, one-day flips cleaned, as consecutive episodes:
    regime, duration (days), start day-of-year. Consecutive rows never share a regime."""
    if rean.empty:
        return pd.DataFrame(columns=["regime", "duration", "doy"])
    ds = rean.sort_values("day").reset_index(drop=True)
    doy = doy_365(ds["day"])
    return pd.DataFrame([(r, n, int(doy.iloc[i])) for r, n, i in episodes(clean_singles(list(ds["regime"])))],
                        columns=["regime", "duration", "doy"])


def path_outcomes(eps: pd.DataFrame, path: list[str], min_days: int = 1,
                  season_doy: int | None = None, season_days: int = 30) -> pd.DataFrame:
    """What history did next after a given sequence of episodes.

    `path` = [start, step 1, ...]. Every place in the record where those regimes
    follow one another as consecutive episodes is found and the episode after them
    is tallied. Every step after the start, and the outcome itself, must last at
    least `min_days`. With `season_doy`, only sequences whose start episode begins
    within +/- `season_days` of that calendar day (any year) are kept.

    Returns one row per outcome: n, share (%), median_days — most frequent first.
    """
    empty = pd.DataFrame(columns=["n", "share", "median_days"])
    k = len(path)
    reg, dur, doy = (eps[c].to_numpy() for c in ("regime", "duration", "doy"))
    m = len(reg) - k
    if m <= 0:
        return empty
    mask = np.ones(m, dtype=bool)
    for j, lab in enumerate(path):
        mask &= reg[j:j + m] == lab
        if j > 0:
            mask &= dur[j:j + m] >= min_days
    mask &= dur[k:k + m] >= min_days
    if season_doy is not None:
        gap = np.abs(doy[:m].astype(int) - season_doy)
        mask &= np.minimum(gap, 365 - gap) <= season_days
    idx = np.nonzero(mask)[0] + k
    if not len(idx):
        return empty
    out = pd.DataFrame({"state": reg[idx], "days": dur[idx]}).groupby("state")["days"]
    res = pd.DataFrame({"n": out.size(), "median_days": out.median()})
    res["share"] = res["n"] / res["n"].sum() * 100
    return res.sort_values(["n", "median_days"], ascending=False)


@st.cache_data(ttl=86400, show_spinner=False)
def impact_frame(rean: pd.DataFrame, impact: pd.DataFrame) -> pd.DataFrame:
    """Reanalysis regime (one-day flips cleaned, as everywhere else) joined to the
    regional daily anomalies; adds the wind anomaly in % of normal per region."""
    if rean.empty or impact.empty:
        return pd.DataFrame()
    r = rean.sort_values("day")[["day"]].copy()
    r["regime"] = clean_singles(list(rean.sort_values("day")["regime"]))
    r["doy"] = doy_365(r["day"])
    df = r.merge(impact, on="day", how="inner")
    for region in WR_IMPACT_WIND_REGIONS:
        k = _region_key(region)
        df[f"wp_{k}"] = df[f"w_{k}"] / df[f"wn_{k}"].where(df[f"wn_{k}"] > 0) * 100
    return df


def regime_impact(df: pd.DataFrame, regime: str, center_doy: int,
                  season_days: int = WR_IMPACT_SEASON_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(temperature, wind) composites for one regime — per region: mean, p25, p75,
    share of days below normal, n days in the seasonal window, and the regime's
    all-year mean. Wind is in % of normal: the mean is the ratio of the mean GW
    anomaly to the mean normal, so a windy day weighs by its GW, not its percentage."""
    gap = (df["doy"] - center_doy).abs()
    in_season = np.minimum(gap, 365 - gap) <= season_days
    reg = df["regime"] == regime
    s, a = df[reg & in_season], df[reg]
    trows, wrows = [], []
    for region in WR_IMPACT_TEMP_REGIONS:
        c = f"t_{_region_key(region)}"
        v = s[c].dropna()
        trows.append({"region": region, "mean": v.mean(), "p25": v.quantile(0.25), "p75": v.quantile(0.75),
                      "below": (v < 0).mean() * 100 if len(v) else np.nan, "n": len(v),
                      "all_mean": a[c].mean()})
    for region in WR_IMPACT_WIND_REGIONS:
        k = _region_key(region)
        sv = s[[f"w_{k}", f"wn_{k}", f"wp_{k}"]].dropna()
        av = a[[f"w_{k}", f"wn_{k}"]].dropna()
        mean_gw = sv[f"w_{k}"].mean()
        wrows.append({"region": region,
                      "mean": mean_gw / sv[f"wn_{k}"].mean() * 100 if len(sv) else np.nan,
                      "p25": sv[f"wp_{k}"].quantile(0.25), "p75": sv[f"wp_{k}"].quantile(0.75),
                      "below": (sv[f"w_{k}"] < 0).mean() * 100 if len(sv) else np.nan, "n": len(sv),
                      "all_mean": av[f"w_{k}"].mean() / av[f"wn_{k}"].mean() * 100 if len(av) else np.nan,
                      "extra": f"<br>mean {mean_gw:+.1f} GW on a normal of {sv[f'wn_{k}'].mean():.1f} GW"
                               if len(sv) else ""})
    return pd.DataFrame(trows).set_index("region"), pd.DataFrame(wrows).set_index("region")


# ══════════════════════════════════════════════════════════════════════════════
# RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def _regime_badge(label: str, share: float, sub: str) -> str:
    color = WR_COLORS.get(label, "#7f7f7f")
    return (f'<div class="kpi-card" style="border-top:2px solid {color};">'
            f'<div class="kpi-label">{label} — {WR_REGIME_LONG.get(label, "")}</div>'
            f'<div class="kpi-value" style="color:{color};">{share:.0f}%</div>'
            f'<div class="kpi-delta">{sub}</div></div>')


def _render_headline(perc: pd.DataFrame, clim_share: pd.Series, n_members: int, days) -> None:
    if perc.empty:
        return
    mean_share = perc.mean()
    top = mean_share.sort_values(ascending=False).head(4)
    cards = []
    for label, share in top.items():
        if clim_share is not None and len(clim_share) and label in clim_share:
            delta = share - float(clim_share[label])
            sub = f"{delta:+.0f} pp vs climatology"
        else:
            sub = "climatology unavailable"
        cards.append(_regime_badge(label, share, sub))
    kpi_row(cards, max_cols=4)


def _compare_run(sub: pd.DataFrame, run_ts, clean: bool, days: pd.DatetimeIndex):
    """Pick the run to compare against (default: the one just before) and return
    (its member shares on this run's days, its label) — or (None, None)."""
    earlier = sorted({pd.Timestamp(t) for t in sub["reference_date"].unique()
                      if pd.Timestamp(t) < pd.Timestamp(run_ts)}, reverse=True)
    if not earlier:
        st.caption("No earlier run of this model in the table — the change panel needs one.")
        return None, None
    key = "wr_cmp_run"
    if st.session_state.get(key) not in earlier:
        st.session_state[key] = earlier[0]
    prev_ts = st.selectbox("Change against run", earlier, key=key,
                           format_func=lambda t: t.strftime("%a %d %b %H:%Mz"),
                           help="The bottom panel is this run's member share minus that run's, "
                                "per regime and day, in percentage points.")
    prev = sub[sub["reference_date"].map(pd.Timestamp) == prev_ts]
    prev_perc = member_percentages(prev, clean=clean).reindex(days).dropna(how="all")
    return prev_perc, prev_ts.strftime("%Hz %a %d %b")


def _render_forecast(run: pd.DataFrame, perc: pd.DataFrame, clim: dict, model_label: str, n_members: int,
                     extended: bool, sub: pd.DataFrame, run_ts, clean: bool) -> None:
    mean, thr, band, no_days = ensemble_iwr(run)
    if perc.empty:
        status_banner("This run has no member rows to summarise.", "warning")
        return

    days = perc.index
    clim_share = window_climatology(clim, days)
    _render_headline(perc, clim_share, n_members, days)

    cc, _ = st.columns([1.6, 4])
    with cc:
        prev_perc, prev_label = _compare_run(sub, run_ts, clean, perc.index)
    change = None
    if prev_perc is not None and not prev_perc.empty:
        change = (perc.reindex(prev_perc.index) - prev_perc).reindex(columns=WR_ALL_LABELS)
    run_label = pd.Timestamp(run_ts).strftime("%Hz %Y-%m-%d")
    fig = make_wr_forecast_change_chart(
        perc, change, clim_share,
        f"Weather regime forecast — {model_label} {run_label}",
        f"Change from the {prev_label} run (pp)" if prev_label else "", n_members)
    try:
        event = st.plotly_chart(fig, use_container_width=True, key="wr_fc_chart",
                                on_select="rerun", selection_mode="points")
    except TypeError:                     # a Streamlit without chart selections
        st.plotly_chart(fig, use_container_width=True)
        event = None
    _take_chart_click(event, perc.index)
    if change is not None and len(change) < len(perc):
        st.caption(f"The comparison run does not reach the last {len(perc) - len(change)} day(s), "
                   "so those days have no change bar.")

    show_band = st.checkbox("Show the P25–P75 member band behind each mean line",
                            value=False, key="wr_band")
    st.plotly_chart(make_wr_iwr_chart(mean, thr, no_days,
                                      f"Ensemble-mean IWR per regime — {model_label}",
                                      spread=band if show_band else None),
                    use_container_width=True)
    if no_days:
        st.caption(f"No regime clears the threshold on: {', '.join(d.strftime('%a %d %b') for d in no_days)}.")
    if extended:
        status_banner(f"Showing beyond day {WR_DEFAULT_HORIZON_DAYS - 1}: the threshold's lead decay "
                      "was fitted over leads 0–14, so it is extrapolated on the last days.", "warning")

    _render_impact(perc)

    dom = dominant_regime(perc)
    tbl = perc.copy()
    tbl.insert(0, "Dominant", dom)
    tbl.index = [d.strftime("%a %d %b") for d in tbl.index]
    st.markdown("###### Members per regime, day by day")
    st.dataframe(tbl.style.format({c: "{:.0f}%" for c in WR_ALL_LABELS})
                 .background_gradient(cmap="Blues", subset=WR_ALL_LABELS, vmin=0, vmax=100),
                 use_container_width=True)
    st.download_button("Download member shares (CSV)", perc.to_csv().encode(),
                       "weather_regimes.csv", "text/csv", key="wr_dl")


def _render_climatology(run: pd.DataFrame, perc: pd.DataFrame, clim: dict, clean: bool) -> None:
    if not clim:
        status_banner(f"{SBX_SCHEMA}.wr_reanalysis_daily is empty — the refresh job classifies the "
                      "ERA5 record on its first run. Until then the forecast is shown without a "
                      "climatological reference.", "warning")
        return
    if clim["years"] < WR_CLIM_MIN_YEARS:
        status_banner(f"The classified reanalysis spans only {clim['years']} years "
                      f"({clim['first']:%Y}–{clim['last']:%Y}); at least {WR_CLIM_MIN_YEARS} are "
                      "wanted before these frequencies mean much.", "warning")
    else:
        pass

    days = perc.index
    clim_share = window_climatology(clim, days)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_wr_clim_compare(perc.mean().reindex(WR_ALL_LABELS).fillna(0),
                                             clim_share.reindex(WR_ALL_LABELS).fillna(0),
                                             "This run vs climatology, over the forecast window"),
                        use_container_width=True)
    with c2:
        d = doy_365(pd.Series(days))
        st.plotly_chart(make_wr_seasonal_clim(clim["doy_freq"][WR_REGIMES],
                                              (int(d.min()), int(d.max())) if len(d) else None,
                                              "Climatological frequency through the year"),
                        use_container_width=True)

    seasonal = st.checkbox("Climatology: only ±30 days around today's date, all years",
                           value=False, key="wr_season",
                           help="Restricts the climatological persistence and transitions to "
                                "episodes starting within 30 days of today's calendar date in "
                                "any year of the record, so the statistic reflects the current season.")
    st.caption(f"A transition is counted only when the regime it turns into — including "
               f"\"no regime\" — lasts at least {WR_TRANS_MIN_DAYS} days.")
    if seasonal:
        today_doy = int(doy_365(pd.Series([pd.Timestamp.today().normalize()])).iloc[0])
        clim_t = climatology(st.session_state["_wr_rean"], season_doy=today_doy)
        clim_title = f"Climatology, ±30 days around {pd.Timestamp.today():%d %b}"
    else:
        clim_t, clim_title = clim, "Climatology — % of exits into each regime"
    st.markdown("###### Persistence — forecast episodes against the climatological spread")
    seqs = member_sequences(run, clean=clean)
    per_regime: dict[str, list] = {r: [] for r in WR_REGIMES}
    for seq in seqs.values():
        for regime, n, _ in episodes(seq):
            if regime in per_regime:
                per_regime[regime].append(n)
    shown = [(r, v) for r, v in per_regime.items() if len(v) >= 3]
    shown.sort(key=lambda kv: -len(kv[1]))
    if not shown:
        st.caption("No regime is held by enough members to form a distribution in this run.")
    else:
        cols = st.columns(min(3, len(shown)))
        for i, (regime, durations) in enumerate(shown[:6]):
            p = clim_t["persistence"].get(regime, {})
            with cols[i % len(cols)]:
                st.plotly_chart(
                    make_wr_persistence(durations, p.get("p25"), p.get("mean"), p.get("p75"),
                                        regime, f"{regime} — {len(durations)} member episodes"),
                    use_container_width=True)

    st.markdown("###### Transitions")
    t1, t2 = st.columns(2)
    fc_tm = pd.DataFrame(0.0, index=WR_ALL_LABELS, columns=WR_ALL_LABELS)
    for seq in seqs.values():
        eps = episodes(seq)
        for (a, _, _), (b, nb, _) in zip(eps[:-1], eps[1:]):
            if nb >= WR_TRANS_MIN_DAYS:
                fc_tm.loc[a, b] += 1
    total = fc_tm.to_numpy().sum()
    with t1:
        st.plotly_chart(make_wr_transition_heatmap(
            fc_tm / total * 100 if total else fc_tm,
            f"This run — {int(total)} member transitions"), use_container_width=True)
    with t2:
        st.plotly_chart(make_wr_transition_heatmap(
            clim_t["transitions"], clim_title),
            use_container_width=True)


def _take_chart_click(event, days: pd.DatetimeIndex) -> None:
    """A click on a regime's bar in the forecast chart selects that regime and day
    for the impact panel. Only a new click is applied, so the selectors below the
    chart can still be changed by hand afterwards."""
    try:
        points = event.selection.points if event is not None else []
    except AttributeError:
        points = (event or {}).get("selection", {}).get("points", [])
    for p in points or []:
        regime, cd = p.get("legendgroup"), p.get("customdata")
        if regime not in WR_ALL_LABELS or not isinstance(cd, (list, tuple)) or len(cd) < 3:
            continue
        day = pd.Timestamp(cd[2])
        sig = f"{regime}|{day:%Y-%m-%d}"
        if sig != st.session_state.get("wr_imp_last_click") and day in days:
            st.session_state["wr_imp_last_click"] = sig
            st.session_state["wr_imp_regime"] = regime
            st.session_state["wr_imp_day"] = day
        return


def _render_impact(perc: pd.DataFrame) -> None:
    """What the selected regime has usually meant for temperature and wind."""
    st.markdown("###### What the regime usually brings — click a regime in the chart")
    try:
        rean, impact = load_wr_reanalysis(), load_wr_impact_daily()
    except Exception as e:
        st.caption(f"Regime impact unavailable: cannot read the historical anomalies ({e}).")
        return
    df = impact_frame(rean, impact) if not rean.empty and not impact.empty else pd.DataFrame()
    if df.empty:
        st.caption("Regime impact unavailable: the classified reanalysis or the historical "
                   "temperature / wind anomalies are not loaded.")
        return

    days = list(perc.index)
    dom = dominant_regime(perc)
    if st.session_state.get("wr_imp_day") not in days:
        st.session_state["wr_imp_day"] = days[0]
    if st.session_state.get("wr_imp_regime") not in WR_ALL_LABELS:
        st.session_state["wr_imp_regime"] = dom.loc[st.session_state["wr_imp_day"]]
    c1, c2, c3 = st.columns([1.6, 1.4, 3])
    with c1:
        regime = st.selectbox("Regime", WR_ALL_LABELS, key="wr_imp_regime",
                              format_func=lambda r: f"{r} — {WR_REGIME_LONG.get(r, r)}")
    with c2:
        day = st.selectbox("Season of", days, key="wr_imp_day",
                           format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b"),
                           help=f"The composite pools the regime's days within ±{WR_IMPACT_SEASON_DAYS} "
                                "days of this date, in every year — a regime's effect depends "
                                "strongly on the season.")
    share = float(perc.loc[day, regime]) if regime in perc.columns else 0.0
    with c3:
        st.markdown(f"<div style='height:1.9rem'></div><span style='color:{WR_COLORS.get(regime)}'>■</span> "
                    f"<b>{regime}</b> holds <b>{share:.0f}%</b> of members on {pd.Timestamp(day):%a %d %b}.",
                    unsafe_allow_html=True)

    center = int(doy_365(pd.Series([pd.Timestamp(day)])).iloc[0])
    temp, wind = regime_impact(df, regime, center)
    lo = pd.Timestamp(day) - pd.Timedelta(days=WR_IMPACT_SEASON_DAYS)
    hi = pd.Timestamp(day) + pd.Timedelta(days=WR_IMPACT_SEASON_DAYS)
    window = f"{lo:%d %b}–{hi:%d %b}"
    n = int(temp["n"].max()) if len(temp) else 0
    if n == 0:
        st.caption(f"{regime} never occurred between {window} in the reanalysis.")
        return
    t1, t2 = st.columns(2)
    with t1:
        st.plotly_chart(make_wr_impact_chart(temp, f"Temperature anomaly under {regime}, {window}", "°C",
                                             True, "colder than normal"), use_container_width=True)
    with t2:
        st.plotly_chart(make_wr_impact_chart(wind, f"Wind generation anomaly under {regime}, {window}",
                                             "% of normal", False, "below the normal"),
                        use_container_width=True)
    years = f"{df['day'].min():%Y}–{df['day'].max():%Y}"
    st.caption(f"{n} {regime} days between {window} in {years}"
               + (" — few enough that the averages are rough" if n < 60 else "")
               + ". Bars: mean anomaly on those days; whiskers: P25–P75 of the days; ◇: the regime's "
               "all-year mean. Temperature: ERA5, population-weighted (Iberia = ES + PT, Nordics = "
               "SE + DK + FI + NO). Wind: today's fleet on past weather, summed per region, % of its normal.")


def _render_explorer(rean: pd.DataFrame, perc: pd.DataFrame) -> None:
    """Pick a regime, then the next one, then the one after: at each step, what the
    reanalysis did next — over the whole year and within +/- 30 days of today."""
    eps = reanalysis_episodes(rean) if not rean.empty else pd.DataFrame()
    if eps.empty:
        status_banner(f"{SBX_SCHEMA}.wr_reanalysis_daily is empty — the explorer needs the "
                      "classified reanalysis.", "warning")
        return
    today = pd.Timestamp.today().normalize()
    tdoy = int(doy_365(pd.Series([today])).iloc[0])
    season = f"±30 d of {today:%d %b}"

    c1, c2 = st.columns([1.6, 3])
    with c1:
        key = "wr_x_start"
        if st.session_state.get(key) not in WR_ALL_LABELS:
            first = dominant_regime(perc).iloc[0] if not perc.empty else WR_REGIMES[0]
            st.session_state[key] = first if first in WR_ALL_LABELS else WR_REGIMES[0]
        start = st.selectbox("Start from", WR_ALL_LABELS, key=key,
                             format_func=lambda r: f"{r} — {WR_REGIME_LONG.get(r, r)}")
    with c2:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        strict = st.checkbox(f"Every step must last at least {WR_TRANS_MIN_DAYS} days", value=False,
                             key="wr_x_strict",
                             help="Counts a sequence only if each regime after the start — including "
                                  "'no regime' — lasts at least that long. Samples shrink fast, "
                                  "especially within ±30 days.")
    min_days = WR_TRANS_MIN_DAYS if strict else 1
    durs = eps.loc[eps["regime"] == start, "duration"]
    st.caption(f"Reanalysis {rean['day'].min():%Y}–{rean['day'].max():%Y}, one-day flips removed. "
               f"{start} episodes: {len(durs)}, median length {durs.median():.0f} days. Each step "
               f"is the next episode — a change of regime — so staying in {start} is not an option; "
               "'Median days' is how long that next state then lasts.")

    path = [start]
    cols = st.columns(3)
    for step in range(3):
        with cols[step]:
            st.markdown(f"###### Step {step + 1} — after {' → '.join(path)}")
            allyr = path_outcomes(eps, path, min_days)
            seas = path_outcomes(eps, path, min_days, season_doy=tdoy)
            if allyr.empty:
                st.caption("This sequence never occurs in the record.")
                break
            tbl = pd.DataFrame({
                "All year": allyr["share"].astype(float),
                "n": allyr["n"].astype(int),
                season: seas["share"].reindex(allyr.index).astype(float).fillna(0.0),
                "n ": seas["n"].reindex(allyr.index).astype(float).fillna(0).astype(int),
                "Median days": allyr["median_days"].astype(float),
            })
            st.dataframe(tbl.style.format({"All year": "{:.0f}%", season: "{:.0f}%",
                                           "Median days": "{:.0f}"})
                         .background_gradient(cmap="Blues", subset=["All year", season], vmin=0, vmax=100),
                         use_container_width=True)
            st.caption(f"{int(allyr['n'].sum())} cases all year · {int(seas['n'].sum())} within {season}"
                       + (" — too few to read much into" if seas["n"].sum() < 10 else ""))
            if step < 2:
                key = f"wr_x_step{step + 1}"
                options = list(allyr.index)
                if st.session_state.get(key) not in options:
                    st.session_state[key] = options[0]
                path.append(st.selectbox(f"Step {step + 1} goes to", options, key=key,
                                         format_func=lambda r: f"{r} — {WR_REGIME_LONG.get(r, r)}"))


def render_regimes():
    """Forecast → Weather Regimes."""
    st.markdown("##### European Weather Regimes")

    try:
        df = load_wr_forecast_members()
    except Exception as e:
        st.error(f"Cannot read wr_forecast_members ({e}). Has power_desk_refresh.py run since the "
                 "Weather Regimes cell was added?")
        return
    if df.empty:
        status_banner(f"{SBX_SCHEMA}.wr_forecast_members is empty — run the refresh job, and check "
                      "that wr_patterns.npz was uploaded to WR_PATTERNS_PATH.", "warning")
        return

    c1, c2, c3, c4 = st.columns([1.6, 1.6, 1.4, 1.6])
    with c1:
        available = [k for k, v in WR_MODELS.items() if v["model"] in set(df["model"])]
        if not available:
            status_banner(f"No configured model is present in the table (found: "
                          f"{', '.join(sorted(set(df['model'])))}).", "critical")
            return
        model_label = st.selectbox("Model", available,
                                   index=available.index(WR_DEFAULT_MODEL)
                                   if WR_DEFAULT_MODEL in available else 0, key="wr_model")
    sub = df[df["model"] == WR_MODELS[model_label]["model"]]
    runs = sorted(sub["reference_date"].unique(), reverse=True)
    with c2:
        run_ts = st.selectbox("Run", runs, format_func=lambda t: pd.Timestamp(t).strftime("%a %d %b %H:%Mz"),
                              key="wr_run")
    with c3:
        horizon = st.number_input("Days", min_value=5, max_value=WR_MAX_HORIZON_DAYS,
                                  value=WR_DEFAULT_HORIZON_DAYS, key="wr_horizon")
    with c4:
        clean = st.checkbox("Drop one-day flips", value=True, key="wr_clean",
                            help="A regime differing from both its neighbours is a single-day "
                                 "wobble, not a regime. The tool removes these before counting "
                                 "episodes and transitions.")

    run = sub[(sub["reference_date"] == run_ts) & (sub["lead_day"] < int(horizon))].copy()
    if run.empty:
        status_banner("That run has no rows in the selected horizon.", "warning")
        return
    n_members = run["member"].nunique()
    expected = WR_MODELS[model_label]["n_members"]
    if n_members < expected:
        status_banner(f"{n_members} of the expected {expected} members are present in this run — "
                      "shares are taken over the members that are there.", "warning")

    try:
        rean = load_wr_reanalysis()
    except Exception:
        rean = pd.DataFrame()
    clim = climatology(rean) if not rean.empty else {}
    st.session_state["_wr_rean"] = rean

    perc = member_percentages(run, clean=clean)
    view = view_switch("Regime view", ["Forecast", "Climatology", "Explorer"], "wr_view",
                       {"Forecast": ("wr_band", "wr_cmp"), "Climatology": ("wr_season",),
                        "Explorer": ("wr_x_",)})
    if view == "Forecast":
        _render_forecast(run, perc, clim, model_label, n_members,
                         extended=int(horizon) > WR_DEFAULT_HORIZON_DAYS, sub=sub, run_ts=run_ts, clean=clean)
    elif view == "Explorer":
        _render_explorer(rean, perc)
    elif not perc.empty:
        _render_climatology(run, perc, clim, clean)

    with st.expander("Method"):
        st.markdown(f"""
For each member and each forecast day the 500 hPa height anomaly over the fixed domain
(lat 30–90 N, lon 80 W–40 E, 0.5° = 121 × 241 points) is divided by that day-of-year's
domain-average amplitude, projected onto the 7 fixed regime patterns with cos(latitude)
weighting, and standardised into an index (IWR) with each regime's own long-term mean and
spread. The member is assigned the highest-scoring regime if it clears
**{WR_THRESHOLD:.2f} + lead × {WR_LEAD_FACTOR:.5f}** — {WR_THRESHOLD:.2f} on day 0 falling to
{threshold_for_lead(14):.2f} on day 14 — and "no regime" otherwise.

The 7 patterns and their normalisation constants are the fixed output of the original
1979–2019 k-means study and ship with the app in `wr_patterns.npz`; they reproduce the tool's
own classified reanalysis exactly. Anomalies are formed against an ERA5 day-of-year mean
computed in the refresh job, which was verified to give the same regime assignments as the
tool's original climatology.

| Regime | |
|---|---|
""" + "\n".join(f"| **{r}** | {WR_REGIME_LONG[r]} |" for r in WR_REGIMES))
