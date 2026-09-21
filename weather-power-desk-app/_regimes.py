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
)
from _charts import (
    make_wr_iwr_chart, make_wr_percent_chart, make_wr_clim_compare, make_wr_seasonal_clim,
    make_wr_persistence, make_wr_transition_heatmap,
)
from _data import load_wr_forecast_members, load_wr_reanalysis
from _ui import kpi_card, kpi_row, status_banner

IWR_COLS = [f"iwr_{r}" for r in WR_REGIMES]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED MATHS
# ══════════════════════════════════════════════════════════════════════════════

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
def climatology(rean: pd.DataFrame, window: int = WR_CLIM_DOY_WINDOW) -> dict:
    """Everything the reanalysis says is normal.

    Returns
    -------
    doy_freq   365 x 8 — % of days in each regime, pooling +/- `window` days
               around each day of year so a single calendar day is not read off
               ~45 samples alone
    persistence  per regime: episode count, mean, p25, p75 (episodes >= 5 days,
               the filter the tool's histograms use)
    transitions  8 x 8 — % of exits from each regime into the next
    years      how many years the record spans
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
    eps = episodes(list(df.sort_values("day")["regime"]))
    ep = pd.DataFrame([(r, n) for r, n, _ in eps], columns=["regime", "duration"])
    long_ep = ep[(ep.regime != WR_NO_REGIME) & (ep.duration >= 5)]
    pers = {}
    for r in WR_REGIMES:
        d = long_ep[long_ep.regime == r]["duration"]
        pers[r] = {"n": int(len(d)),
                   "mean": float(d.mean()) if len(d) else np.nan,
                   "p25": float(d.quantile(0.25)) if len(d) else np.nan,
                   "p75": float(d.quantile(0.75)) if len(d) else np.nan}

    seq = ep["regime"].tolist()
    tm = pd.DataFrame(0.0, index=WR_ALL_LABELS, columns=WR_ALL_LABELS)
    for a, b in zip(seq[:-1], seq[1:]):
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
    st.caption(f"Share of the {n_members} members averaged over {len(days)} forecast days "
               f"({days[0]:%d %b} – {days[-1]:%d %b}), against what the reanalysis says is "
               "normal for the same calendar days.")


def _render_forecast(run: pd.DataFrame, clim: dict, model_label: str, n_members: int,
                     clean: bool, extended: bool) -> pd.DataFrame:
    mean, thr, band, no_days = ensemble_iwr(run)
    perc = member_percentages(run, clean=clean)
    if perc.empty:
        status_banner("This run has no member rows to summarise.", "warning")
        return perc

    days = perc.index
    clim_share = window_climatology(clim, days)
    _render_headline(perc, clim_share, n_members, days)

    st.plotly_chart(make_wr_percent_chart(perc[[c for c in WR_ALL_LABELS if perc[c].sum() > 0]],
                                          f"Share of members per regime — {model_label}", n_members),
                    use_container_width=True)

    show_band = st.checkbox("Show the P25–P75 member band behind each mean line",
                            value=False, key="wr_band")
    st.plotly_chart(make_wr_iwr_chart(mean, thr, no_days,
                                      f"Ensemble-mean IWR per regime — {model_label}",
                                      spread=band if show_band else None),
                    use_container_width=True)
    if no_days:
        st.caption(f"Grey: {len(no_days)} day(s) where no regime clears the threshold — "
                   f"{', '.join(d.strftime('%a %d %b') for d in no_days)}. The threshold falls from "
                   f"{WR_THRESHOLD:.2f} at day 0 to {threshold_for_lead(14):.2f} at day 14, because "
                   "a regime is harder to identify further out.")
    if extended:
        status_banner(f"Showing beyond day {WR_DEFAULT_HORIZON_DAYS - 1}: the threshold's lead decay "
                      "was fitted over leads 0–14, so it is extrapolated on the last days.", "warning")

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
    return perc


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
        st.caption(f"Reanalysis {clim['first']:%b %Y} – {clim['last']:%b %Y} "
                   f"({clim['years']} years), classified through the same projection.")

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

    st.markdown("###### Persistence — forecast episodes against the climatological spread")
    st.caption("One histogram per regime the members actually settle into: how long each member "
               "holds it, against the reanalysis quartiles for episodes of 5 days or more (the "
               "filter the original tool's histograms use).")
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
            p = clim["persistence"].get(regime, {})
            with cols[i % len(cols)]:
                st.plotly_chart(
                    make_wr_persistence(durations, p.get("p25"), p.get("mean"), p.get("p75"),
                                        regime, f"{regime} — {len(durations)} member episodes"),
                    use_container_width=True)

    st.markdown("###### Transitions")
    t1, t2 = st.columns(2)
    fc_tm = pd.DataFrame(0.0, index=WR_ALL_LABELS, columns=WR_ALL_LABELS)
    for seq in seqs.values():
        eps = [r for r, _, _ in episodes(seq)]
        for a, b in zip(eps[:-1], eps[1:]):
            fc_tm.loc[a, b] += 1
    total = fc_tm.to_numpy().sum()
    with t1:
        st.plotly_chart(make_wr_transition_heatmap(
            fc_tm / total * 100 if total else fc_tm,
            f"This run — {int(total)} member transitions"), use_container_width=True)
    with t2:
        st.plotly_chart(make_wr_transition_heatmap(
            clim["transitions"], "Climatology — % of exits into each regime"),
            use_container_width=True)
    st.caption("Left: where the members go, as a share of all transitions in this run. Right: the "
               "reanalysis, as a share of exits from each regime — read it row by row.")


def render_regimes():
    """Forecast → Weather Regimes."""
    st.markdown("##### European Weather Regimes")
    st.caption("Each ensemble member's 500 hPa geopotential field projected onto the 7 "
               "North-Atlantic / European regimes of Michel & Rivière (2011) — the "
               "corso_model_wr tool, live off the Meteomatics silver layer. The projection runs "
               "in the refresh job; this page reads the per-member result.")

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

    tabs = st.tabs(["Forecast", "Climatology"])
    with tabs[0]:
        perc = _render_forecast(run, clim, model_label, n_members, clean,
                                extended=int(horizon) > WR_DEFAULT_HORIZON_DAYS)
    with tabs[1]:
        if perc is not None and not perc.empty:
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
