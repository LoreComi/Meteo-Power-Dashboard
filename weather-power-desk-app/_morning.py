"""Morning Call — the Morning Report table, live from the Volue tables.

Port of P:/QFA/TonyWeather/Lorenzo_Trainee/Morning_Report/import_00z_add_solar_np_tot.py
(which wrote table_00z_add_solar_np_tot.xlsx through wapi). Same content, same layout:

  Block 1 / Block 2   weekday-dependent windows
        Mon, Tue      this week (Mon–Sun)              | next week
        Wed, Thu      weekend (Sat–Sun)                | next week
        Fri–Sun       next week                        | week after
  Rows                Temperatures [°C]  FRA DE UK ITA HUN Nordic Iberia
                      Wind [GW]          DE UK FRA ITA SEE Nordic Iberia
                      Solar PV [GW]      DE FRA ITA SEE Nordic Iberia
                      Precip [TWh]       Alps(cwe+it-nord) Nordic SEE Iberia   (sum of the coming 2 weeks, block 1 only)
  Columns             abs. value | Δ vs previous 00z (Δ -24h, Δ -72h on Monday) | Δ norm
  Block 3             week after — free-text pattern / temperature / wind-precip-solar commentary

On top of the Excel: the same table for the other runs (EC 12z, GFS 00z,
EC-Extended, Meteomatics EC-ENS / AIFS-ENS means) side by side with EC 00z —
"confront the different models".

Data: {SBX_SCHEMA}.morning_daily, written by power_desk_refresh.py from the
exact wapi curve names, daily CET means ('Avg' tag) per run, with the normal.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    MORNING_BLOCKS, MORNING_MODELS, MORNING_DEFAULT_MODEL, MORNING_MIN_DAY_COVERAGE, SBX_SCHEMA,
)
from _data import load_morning_daily
from _charts import make_model_compare_chart
from _style import CATEGORICAL, PROVIDER_COLORS
from _ui import status_banner

MODEL_COLORS = {
    "EC-ENS 00z": CATEGORICAL[0], "EC-ENS 12z": CATEGORICAL[4], "GFS-ENS 00z": CATEGORICAL[1],
    "EC-Extended": CATEGORICAL[2], "Meteomatics EC-ENS": PROVIDER_COLORS["Meteomatics EC-ENS"],
    "Meteomatics AIFS-ENS": PROVIDER_COLORS["Meteomatics AIFS-ENS"],
}
MM_PATTERNS = {"Meteomatics EC-ENS": "ecmwf-ens", "Meteomatics AIFS-ENS": "ecmwf-aifs-ens"}


# ─── weekday logic (verbatim from the report) ──────────────────────────────────

def morning_windows(time_ref: pd.Timestamp) -> dict:
    """The two weekly windows, their titles, the Δ label and the previous-run offset.

    Returns dict(w1=(start, end), w2=(start, end), t1, t2, t3, delta_label, prev_days).
    Weekend days behave like Friday (both windows ahead).
    """
    d = time_ref.normalize()
    wd = d.weekday()
    wk = int(d.isocalendar().week)
    if wd < 2:                                  # Mon, Tue
        w1s = d - pd.Timedelta(days=wd)
        w1e = w1s + pd.Timedelta(days=6)
        w2s, w2e = w1e + pd.Timedelta(days=1), w1e + pd.Timedelta(days=7)
        t1, t2, t3 = f"Week {wk}", f"Week {wk + 1} (next week)", f"Week {wk + 2}"
        prev_days = 3 if wd == 0 else 1
        delta_label = "Δ -72h" if wd == 0 else "Δ -24h"
        kind1 = "week"
    elif wd in (2, 3):                          # Wed, Thu
        w1s = d + pd.Timedelta(days=5 - wd)     # Saturday
        w1e = w1s + pd.Timedelta(days=1)        # Sunday
        w2s, w2e = w1s + pd.Timedelta(days=2), w1s + pd.Timedelta(days=8)
        t1, t2, t3 = f"Weekend ({wk})", f"Week {wk + 1} (next week)", f"Week {wk + 2}"
        prev_days, delta_label, kind1 = 1, "Δ -24h", "weekend"
    else:                                       # Fri, Sat, Sun
        w1s = d + pd.Timedelta(days=7 - wd)     # next Monday
        w1e = w1s + pd.Timedelta(days=6)
        w2s, w2e = w1e + pd.Timedelta(days=1), w1e + pd.Timedelta(days=7)
        t1, t2, t3 = f"Week {wk + 1} (next week)", f"Week {wk + 2}", f"Week {wk + 3}"
        prev_days, delta_label, kind1 = (1 if wd == 4 else wd - 4), "Δ -24h" if wd == 4 else f"Δ -{24 * (wd - 4)}h", "week"
    return {"w1": (w1s, w1e), "w2": (w2s, w2e), "t1": t1, "t2": t2, "t3": t3,
            "delta_label": delta_label, "prev_days": prev_days, "kind1": kind1}


# ─── run selection ──────────────────────────────────────────────────────────────

def _drop_partial_days(df: pd.DataFrame) -> pd.DataFrame:
    """The report drops the trailing half day of the TT instance; here any day with
    fewer than MORNING_MIN_DAY_COVERAGE of the family's full point count goes."""
    if df.empty:
        return df
    full = df.groupby("family")["n_points"].transform("max")
    return df[df["n_points"] >= MORNING_MIN_DAY_COVERAGE * full]


def _pick_run(df: pd.DataFrame, pattern: str, target_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Rows of the run initialised on target_date (00z-snapped); else the latest run before it."""
    p = df[df["pattern"] == pattern]
    if p.empty:
        return p, None
    exact = p[p["init_date"] == target_date]
    if not exact.empty:
        return exact, target_date
    earlier = p[p["init_date"] < target_date]
    if earlier.empty:
        return earlier, None
    latest = earlier["init_date"].max()
    return p[p["init_date"] == latest], latest


def _series(run_df: pd.DataFrame, family: str, region) -> tuple[pd.Series, pd.Series]:
    """(forecast, normal) daily series for a family / region (or list of regions summed, e.g. Alps)."""
    regions = region if isinstance(region, list) else [region]
    if run_df is None or run_df.empty or "family" not in run_df.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    r = run_df[(run_df["family"] == family) & (run_df["region"].isin(regions))]
    if r.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    f = r.groupby("day")["value"].sum(min_count=len(regions))
    n = r.groupby("day")["normal"].sum(min_count=len(regions))
    return f.sort_index(), n.sort_index()


def block_values(cur: pd.DataFrame, prev: pd.DataFrame, block: dict, window: tuple | None) -> list[dict]:
    """One row per region: abs value, Δ vs previous run, Δ norm — the report's arithmetic.

    mean-type blocks: window mean of the forecast, of (forecast - previous forecast), of (forecast - normal)
    sum-type (precip): sum over the whole forecast range; deltas summed over overlapping days
    """
    out = []
    for label, region in block["rows"]:
        f, n = _series(cur, block["family"], region)
        fp, _ = _series(prev, block["family"], region)
        row = {"region": label, "value": np.nan, "d_run": np.nan, "d_norm": np.nan, "n_days": 0}
        if not f.empty:
            f = f * block["scale"]; n = n * block["scale"]; fp = fp * block["scale"]
            if block["agg"] == "mean" and window is not None:
                s, e = window
                fw = f[(f.index >= s) & (f.index <= e)]
                row["n_days"] = int(len(fw))
                if len(fw):
                    row["value"] = float(fw.mean())
                    dr = (f - fp).dropna(); dr = dr[(dr.index >= s) & (dr.index <= e)]
                    dn = (f - n).dropna(); dn = dn[(dn.index >= s) & (dn.index <= e)]
                    row["d_run"] = float(dr.mean()) if len(dr) else np.nan
                    row["d_norm"] = float(dn.mean()) if len(dn) else np.nan
            elif block["agg"] == "sum":
                row["n_days"] = int(len(f))
                row["value"] = float(f.sum())
                dr = (f - fp).dropna(); dn = (f - n).dropna()
                row["d_run"] = float(dr.sum()) if len(dr) else np.nan
                row["d_norm"] = float(dn.sum()) if len(dn) else np.nan
        out.append(row)
    return out


# ─── rendering ─────────────────────────────────────────────────────────────────

def _fmt_delta(v: float, fmt: str) -> str:
    if v is None or np.isnan(v):
        return '<span class="mc-muted">n/a</span>'
    r = round(v, 1)
    txt = ("+" if r > 0 else "") + fmt.format(v)
    cls = "mc-pos" if r > 0 else ("mc-neg" if r < 0 else "mc-zero")
    return f'<span class="{cls}">{txt}</span>'


def _fmt_val(v: float, fmt: str) -> str:
    return '<span class="mc-muted">n/a</span>' if v is None or np.isnan(v) else fmt.format(v)


def _block_html(title: str, unit: str, rows: list[dict], delta_label: str, fmt: str, extra_cols: dict | None = None) -> str:
    head = f"<tr><th>{title} [{unit}]</th><th>abs. value</th><th>{delta_label}</th><th>Δ norm</th>"
    if extra_cols:
        head += "".join(f"<th>{c}</th>" for c in extra_cols)
    head += "</tr>"
    body = ""
    for r in rows:
        body += (f'<tr><td class="mc-region">{r["region"]}</td><td>{_fmt_val(r["value"], fmt)}</td>'
                 f'<td>{_fmt_delta(r["d_run"], fmt)}</td><td>{_fmt_delta(r["d_norm"], fmt)}</td>')
        if extra_cols:
            for c, vals in extra_cols.items():
                v = vals.get(r["region"], np.nan)
                body += f"<td>{_fmt_val(v, fmt)}</td>"
        body += "</tr>"
    return f'<div class="mc-block"><table class="mc-table">{head}{body}</table></div>'


def _render_block(title: str, window: tuple | None, cur: pd.DataFrame, prev: pd.DataFrame, delta_label: str,
                  include_precip: bool, key: str, sub: str):
    st.markdown(f'<div class="mc-week">{title}</div><div class="mc-sub">{sub}</div>', unsafe_allow_html=True)
    st.text_area("Pattern", key=f"mc_pattern_{key}", placeholder="Pattern — e.g. High pressure in southern Europe, "
                 "low over the Nordics.", height=68, label_visibility="collapsed")
    for name, block in MORNING_BLOCKS.items():
        if block["agg"] == "sum" and not include_precip:
            continue
        rows = block_values(cur, prev, block, None if block["agg"] == "sum" else window)
        st.markdown(_block_html(name, block["unit"], rows, delta_label, block["fmt"]), unsafe_allow_html=True)
    st.text_area("Comment / view on alternative scenarios", key=f"mc_comment_{key}",
                 placeholder="Comment / view on alternative scenarios", height=68, label_visibility="collapsed")


def _download_table(cur: pd.DataFrame, prev: pd.DataFrame, win: dict, delta_label: str) -> bytes:
    recs = []
    for bt, window in (("block1", win["w1"]), ("block2", win["w2"])):
        for name, block in MORNING_BLOCKS.items():
            if block["agg"] == "sum" and bt == "block2":
                continue
            for r in block_values(cur, prev, block, None if block["agg"] == "sum" else window):
                recs.append({"block": win["t1"] if bt == "block1" else win["t2"], "table": name, "unit": block["unit"],
                             **r, "delta_label": delta_label})
    return pd.DataFrame(recs).to_csv(index=False).encode()


def _render_model_comparison(df: pd.DataFrame, win: dict, today: pd.Timestamp):
    st.markdown("##### Confront the models — same windows, latest run of each")
    st.caption("Absolute value per model for each window (weekly mean; precipitation = 2-week sum) with the "
               "difference to EC-ENS 00z in brackets. Meteomatics rows are country means of the gridded ensemble "
               "mean mapped onto the report regions (temperature only).")
    models = {**MORNING_MODELS, **MM_PATTERNS}
    picks = st.multiselect("Models", list(models.keys()), [m for m in models if m != "EC-Extended"], key="mc_models")
    if not picks:
        return
    for bt, window, title in (("b1", win["w1"], win["t1"]), ("b2", win["w2"], win["t2"])):
        st.markdown(f'<div class="mc-week">{title}</div>', unsafe_allow_html=True)
        for name, block in MORNING_BLOCKS.items():
            if block["agg"] == "sum" and bt == "b2":
                continue
            per_model: dict[str, dict] = {}
            run_info = []
            for mlabel in picks:
                pat = models[mlabel]
                run_df, init = _pick_run(df, pat, today)
                if run_df.empty:
                    continue
                rows = block_values(run_df, pd.DataFrame(), block, None if block["agg"] == "sum" else window)
                vals = {r["region"]: r["value"] for r in rows}
                if all(np.isnan(v) for v in vals.values()):
                    continue
                per_model[mlabel] = vals
                run_info.append(f"{mlabel} {pd.Timestamp(init):%d %b %Hz}" if init is not None else mlabel)
            if not per_model:
                continue
            base = per_model.get("EC-ENS 00z") or next(iter(per_model.values()))
            regions = [lbl for lbl, _ in block["rows"]]
            head = f"<tr><th>{name} [{block['unit']}]</th>" + "".join(f"<th>{m}</th>" for m in per_model) + "</tr>"
            body = ""
            for reg in regions:
                body += f'<tr><td class="mc-region">{reg}</td>'
                for m, vals in per_model.items():
                    v = vals.get(reg, np.nan)
                    if m == "EC-ENS 00z" or np.isnan(v) or np.isnan(base.get(reg, np.nan)):
                        body += f"<td>{_fmt_val(v, block['fmt'])}</td>"
                    else:
                        body += f"<td>{_fmt_val(v, block['fmt'])} <span class='mc-muted'>({_fmt_delta(v - base[reg], block['fmt'])})</span></td>"
                body += "</tr>"
            st.markdown(f'<div class="mc-block"><table class="mc-table">{head}{body}</table>'
                        f'<div class="mc-muted" style="padding:6px 8px 4px;">runs: {" · ".join(run_info)}</div></div>',
                        unsafe_allow_html=True)
            if name == "Temperatures":
                long = pd.DataFrame([{"model": m, "region": r, "value": v} for m, vals in per_model.items()
                                     for r, v in vals.items() if not np.isnan(v)])
                st.plotly_chart(make_model_compare_chart(long, block["unit"], f"{name} — {title}", MODEL_COLORS),
                                use_container_width=True)


def render_morning_call():
    st.markdown("#### MORNING CALL")
    st.caption("The Morning Report table, computed live from the Volue tables (EC-ENS 00z 'Avg' curves, daily CET "
               "means, Volue 30-year normal). Same rows, same windows, same deltas as "
               "import_00z_add_solar_np_tot.py — no Excel, refreshed with the sandbox job.")
    try:
        df = load_morning_daily()
    except Exception as e:
        st.error(f"Cannot read morning_daily ({e}). Has power_desk_refresh.py run?")
        return
    if df.empty:
        status_banner(f"{SBX_SCHEMA}.morning_daily is empty — run the refresh job after the 00z run has landed.", "warning")
        return
    df = _drop_partial_days(df)

    c1, c2, c3 = st.columns([1.6, 1.4, 4])
    with c1:
        model = st.selectbox("Report run", list(MORNING_MODELS.keys()),
                             index=list(MORNING_MODELS).index(MORNING_DEFAULT_MODEL), key="mc_model")
    with c2:
        ref_day = st.date_input("Report date", value=dt.date.today(), key="mc_date",
                                help="The report's 'today'. Windows and the previous-run offset follow the weekday logic.")
    today = pd.Timestamp(ref_day)
    win = morning_windows(today)
    pattern = MORNING_MODELS[model]

    cur, cur_init = _pick_run(df, pattern, today)
    prev, prev_init = _pick_run(df, pattern, today - pd.Timedelta(days=win["prev_days"]))
    if cur.empty:
        status_banner(f"No {model} run found on or before {today:%d %b}.", "critical")
        return
    if cur_init != today:
        status_banner(f"Today's {model} run has not landed yet — showing the run initialised {cur_init:%a %d %b}.", "warning")
    if prev.empty:
        status_banner("No previous run available — Δ vs previous run shows n/a.", "warning")
    delta_label = win["delta_label"] if prev_init is not None and cur_init is not None and \
        (cur_init - prev_init).days == win["prev_days"] else \
        (f"Δ vs {prev_init:%d %b}" if prev_init is not None else win["delta_label"])
    with c3:
        if prev_init is not None:
            sub = (f"{model} mean weekly average (absolute value, change {cur_init:%d.%m.%Y} vs "
                   f"{prev_init:%d.%m.%Y}, dev norm) · run initialised {cur_init:%d.%m.%Y}")
        else:
            sub = f"run initialised {cur_init:%d.%m.%Y}"
        st.markdown(f'<div class="mc-sub" style="margin-top:28px;">{sub}</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        _render_block(win["t1"], win["w1"], cur, prev, delta_label, include_precip=True, key="b1",
                      sub=f"{win['w1'][0]:%a %d %b} – {win['w1'][1]:%a %d %b} · "
                          f"{'weekend' if win['kind1'] == 'weekend' else 'weekly'} average · precip = sum of coming 2 weeks")
    with right:
        _render_block(win["t2"], win["w2"], cur, prev, delta_label, include_precip=False, key="b2",
                      sub=f"{win['w2'][0]:%a %d %b} – {win['w2'][1]:%a %d %b} · weekly average")

    st.markdown(f'<div class="mc-week">{win["t3"]} — EC weekly</div>'
                f'<div class="mc-sub">run initialised {cur_init:%d.%m.%Y} · qualitative outlook, as in the report</div>',
                unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.text_area("Pattern", key="mc_w3_pattern", placeholder="Pattern", height=90)
    with k2:
        st.text_area("Temperatures", key="mc_w3_temp", placeholder="Temperatures", height=90)
    with k3:
        st.text_area("Wind / Precip / Solar", key="mc_w3_wps", placeholder="Wind / Precip / Solar", height=90)
    st.download_button("Download table (CSV)", _download_table(cur, prev, win, delta_label),
                       f"morning_call_{today:%Y%m%d}.csv", "text/csv", key="mc_dl")

    st.divider()
    _render_model_comparison(df, win, today)
