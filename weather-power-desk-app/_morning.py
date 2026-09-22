"""Morning Call — the Morning Report table, live from the Volue tables.

Port of P:/QFA/TonyWeather/Lorenzo_Trainee/Morning_Report/import_00z_add_solar_np_tot.py
(which wrote table_00z_add_solar_np_tot.xlsx through wapi). Same arithmetic, laid
out as one grid, so that the run-over-run change and the agreement between the
models are read on the same rows:

  Windows             weekday-dependent, taken from the reference run's init day
        Mon, Tue      this week (Mon–Sun)              | next week
        Wed, Thu      weekend (Sat–Sun)                | next week
        Fri–Sun       next week                        | week after
  Rows                Temperatures [°C]  FRA DE UK ITA HUN Nordic Iberia
                      Wind [GW]          DE UK FRA ITA SEE Nordic Iberia
                      Solar PV [GW]      DE FRA ITA SEE Nordic Iberia
                      Precip [TWh]       Alps(cwe+it-nord) Nordic SEE Iberia   (15-day sum, first group only)
  Columns             window → run. The reference run first (EC-ENS 00z, the report's
                      run), then any other runs: the latest GFS 00z, EC 12z, Meteomatics
                      EC-ENS / AIFS-ENS, or an earlier run of the same model. Every run
                      carries the report's three numbers — abs. value, Δ vs its own
                      previous run (Δ -24h; Δ -72h on a Monday), Δ norm — plus Δ vs the
                      reference and, per window, the spread between the runs shown.
  Cells show          "Detail" puts the Excel triple in every cell; the other views put
                      one quantity per cell and shade it on the diverging pair, for a
                      one-glance read across runs.

The Excel's free-text boxes — Pattern and Comment per window, and the Week 3
commentary block — are gone. In their place two agent families read the
reference run's numbers and write the commentary (see _ai_brief.py). Week 3 is
still reached: it is handed to the agents whenever the reference run has enough
days in it, which is normal for EC-Extended and rare for EC-ENS.

Data: {SBX_SCHEMA}.morning_daily, written by power_desk_refresh.py from the
exact wapi curve names, daily CET means ('Avg' tag) per run, with the normal;
plus Meteomatics population-weighted country means (temperature only) on the report regions.
"""
from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    MORNING_BLOCKS, MORNING_MIN_DAY_COVERAGE, MORNING_MODEL_LABELS, MORNING_DEFAULT_REFERENCE_PATTERN,
    MORNING_DEFAULT_COMPARE_PATTERNS, MORNING_PREV_RULES, MORNING_PRECIP_SUM_DAYS, EXPECTED_HORIZON,
    AI_BRIEF_MODEL, AI_BRIEF_MAX_TOKENS, AI_BRIEF_SYNTHESIS_MAX_TOKENS, AI_POWER_AGENTS, AI_GAS_AGENTS,
    SCENARIO_MIN_WEEK_DAYS,
)
from _data import load_morning_daily
from _style import BRIEF_CSS, INK_MUTED, CAT_BLUE, STATUS_CRITICAL, DIV_NEG, DIV_MID, DIV_POS
from _ui import status_banner

# What a cell shows. None = the report's triple; otherwise the grid column to put in the cell.
VIEWS: dict[str, str | None] = {
    "Detail": None, "Δ norm": "d_norm", "Δ prev run": "d_run", "Δ vs ref": "d_ref", "Value": "value",
}
_HEAT_VIEWS = ("d_norm", "d_run", "d_ref")


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
    # Week 3 is the report's qualitative block: the week after w2, usually past
    # the EC-ENS 15-day horizon. It is not tabulated, but it is handed to the
    # agent families whenever the chosen run actually reaches into it.
    w3s, w3e = w2e + pd.Timedelta(days=1), w2e + pd.Timedelta(days=7)
    return {"w1": (w1s, w1e), "w2": (w2s, w2e), "w3": (w3s, w3e), "t1": t1, "t2": t2, "t3": t3,
            "delta_label": delta_label, "prev_days": prev_days, "kind1": kind1}


# ─── the report's arithmetic ────────────────────────────────────────────────────

def _drop_partial_days(df: pd.DataFrame) -> pd.DataFrame:
    """The report drops the trailing half day of the TT instance; here any day with
    fewer than MORNING_MIN_DAY_COVERAGE of the family's full point count goes."""
    if df.empty:
        return df
    full = df.groupby("family")["n_points"].transform("max")
    return df[df["n_points"] >= MORNING_MIN_DAY_COVERAGE * full]


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
    sum-type (precip): sum over the run's first MORNING_PRECIP_SUM_DAYS days (the whole EC-ENS
                       range, as the report does); deltas summed over the overlapping days
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
                f = f[f.index <= f.index.min() + pd.Timedelta(days=MORNING_PRECIP_SUM_DAYS - 1)]
                row["n_days"] = int(len(f))
                row["value"] = float(f.sum())
                dr = (f - fp).dropna(); dn = (f - n).dropna()
                row["d_run"] = float(dr.sum()) if len(dr) else np.nan
                row["d_norm"] = float(dn.sum()) if len(dn) else np.nan
        out.append(row)
    return out


# ─── runs and columns ───────────────────────────────────────────────────────────

def _run_label(pattern: str, init: pd.Timestamp) -> str:
    return f"{MORNING_MODEL_LABELS.get(pattern, pattern)} {init:%H}z · {init:%a %d %b}"


def _list_runs(df: pd.DataFrame) -> list[tuple[str, pd.Timestamp]]:
    """Every (pattern, init time) in the table, newest first; ties in the config's model order."""
    sub = df.loc[df["init_time"].notna(), ["pattern", "init_time"]].drop_duplicates()
    order = {p: i for i, p in enumerate(MORNING_MODEL_LABELS)}
    runs = [(str(p), pd.Timestamp(t)) for p, t in zip(sub["pattern"], sub["init_time"])]
    runs.sort(key=lambda r: (-r[1].value, order.get(r[0], 99)))
    return runs


def _run_rows(df: pd.DataFrame, pattern: str, init: pd.Timestamp | None) -> pd.DataFrame:
    if init is None:
        return pd.DataFrame()
    return df[(df["pattern"] == pattern) & (df["init_time"] == init)]


def _previous_run(df: pd.DataFrame, pattern: str, init: pd.Timestamp, rule: str | int,
                  report_prev_days: int) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """(init of the run compared against, init that the rule asked for).

    The two differ when the wanted run is missing from the table and the latest
    earlier one is taken instead — the header and the footer say so. Under the
    'previous' rule there is no wanted run, so both are the same.
    """
    times = pd.DatetimeIndex(sorted(df.loc[df["pattern"] == pattern, "init_time"].dropna().unique()))
    if len(times) == 0:
        return None, None
    if rule == "previous":
        earlier = times[times < init]
        prev = earlier.max() if len(earlier) else None
        return prev, prev
    days = report_prev_days if rule == "report" else int(rule)
    wanted = init - pd.Timedelta(days=days)
    if wanted in times:
        return wanted, wanted
    earlier = times[times < wanted]
    return (earlier.max() if len(earlier) else None), wanted


@dataclass
class RunCol:
    """One column group of the grid: a run, and the run its Δ is taken against."""
    pattern: str
    init: pd.Timestamp
    rows: pd.DataFrame
    prev_init: pd.Timestamp | None
    prev_wanted: pd.Timestamp | None
    prev_rows: pd.DataFrame
    is_ref: bool = False

    @property
    def model(self) -> str:
        return f"{MORNING_MODEL_LABELS.get(self.pattern, self.pattern)} {self.init:%H}z"

    @property
    def label(self) -> str:
        return _run_label(self.pattern, self.init)

    @property
    def prev_label(self) -> str:
        return "n/a" if self.prev_init is None else f"{self.prev_init:%a %d %b %H}z"

    @property
    def prev_exact(self) -> bool:
        return self.prev_init is None or self.prev_wanted is None or self.prev_init == self.prev_wanted

    @property
    def temperature_only(self) -> bool:
        return not self.rows.empty and set(self.rows["family"].unique()) <= {"tt"}

    def coverage(self) -> tuple[int, int]:
        return (int(self.rows["day"].nunique()) if not self.rows.empty else 0), EXPECTED_HORIZON.get(self.pattern, 15)


def _make_col(df: pd.DataFrame, pattern: str, init: pd.Timestamp, rule: str | int, win: dict,
              is_ref: bool) -> RunCol:
    prev_init, wanted = _previous_run(df, pattern, init, rule, win["prev_days"])
    return RunCol(pattern, init, _run_rows(df, pattern, init), prev_init, wanted,
                  _run_rows(df, pattern, prev_init), is_ref)


def compute_grid(cols: list[RunCol], win: dict) -> pd.DataFrame:
    """The whole page as one tidy frame: a row per (window, block, region, column).

    value / d_run / d_norm are the report's triple for that run; d_ref is the
    difference to the reference (cols[0]); spread is max − min of the value
    across the columns shown, per window and region. Precipitation is tabulated
    in the first group only — its sum spans the forecast range, not a window.
    """
    recs = []
    for wkey, tkey in (("w1", "t1"), ("w2", "t2")):
        window = win[wkey]
        for name, block in MORNING_BLOCKS.items():
            if block["agg"] == "sum" and wkey == "w2":
                continue
            per_col = [
                {r["region"]: r for r in block_values(c.rows, c.prev_rows, block,
                                                      None if block["agg"] == "sum" else window)}
                for c in cols
            ]
            for region, _ in block["rows"]:
                vals = [pc[region]["value"] for pc in per_col]
                finite = [v for v in vals if not np.isnan(v)]
                spread = (max(finite) - min(finite)) if len(finite) >= 2 else np.nan
                ref_v = per_col[0][region]["value"]
                for ci, c in enumerate(cols):
                    r = per_col[ci][region]
                    d_ref = np.nan
                    if ci and not np.isnan(ref_v) and not np.isnan(r["value"]):
                        d_ref = r["value"] - ref_v
                    recs.append({"window": wkey, "window_title": win[tkey], "block": name, "unit": block["unit"],
                                 "region": region, "col": ci, "run": c.label,
                                 "run_init": c.init, "prev_init": c.prev_init,
                                 "value": r["value"], "d_run": r["d_run"], "d_norm": r["d_norm"],
                                 "d_ref": d_ref, "n_days": r["n_days"], "spread": spread})
    return pd.DataFrame(recs)


# ─── rendering ─────────────────────────────────────────────────────────────────

def _signed(v: float, fmt: str) -> str:
    r = round(v, 1)
    return ("+" if r > 0 else "") + fmt.format(0.0 if r == 0 else v)


def _fmt_delta(v: float, fmt: str) -> str:
    if v is None or np.isnan(v):
        return '<span class="mc-muted">n/a</span>'
    r = round(v, 1)
    cls = "mc-pos" if r > 0 else ("mc-neg" if r < 0 else "mc-zero")
    return f'<span class="{cls}">{_signed(v, fmt)}</span>'


def _fmt_val(v: float, fmt: str) -> str:
    return '<span class="mc-muted">n/a</span>' if v is None or np.isnan(v) else fmt.format(v)


def _blend(a_hex: str, b_hex: str, t: float) -> str:
    a = [int(a_hex[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(b_hex[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def _heat_bg(v: float, scale: float, warm_is_positive: bool) -> str:
    """Cell shade on the diverging pair: neutral at 0, saturating at ±scale, ink stays readable."""
    if scale <= 0 or np.isnan(v) or round(v, 1) == 0:
        return ""
    t = min(abs(v) / scale, 1.0) ** 0.7 * 0.8
    pole = DIV_POS if (v > 0) == warm_is_positive else DIV_NEG
    return f"background:{_blend(DIV_MID, pole, t)};"


def _tip(lines: list[str]) -> str:
    return html.escape("\n".join(lines), quote=True).replace("\n", "&#10;")


def _cell_tip(c: RunCol, region: str, wtitle: str, r: pd.Series, block: dict, span: int | None) -> str:
    fmt, unit = block["fmt"], block["unit"]

    def sgn(x):
        return "n/a" if np.isnan(x) else _signed(float(x), fmt)

    lines = [f"{c.label} · {region} · {wtitle}", f"abs. value {fmt.format(r['value'])} {unit}",
             f"Δ vs {c.prev_label}: {sgn(r['d_run'])}", f"Δ norm: {sgn(r['d_norm'])}"]
    if not c.is_ref:
        lines.append(f"Δ vs reference: {sgn(r['d_ref'])}")
    if span and 0 < r["n_days"] < span:
        lines.append(f"run covers {int(r['n_days'])} of {span} days")
    return _tip(lines)


def _col_tip(c: RunCol, delta_label: str) -> str:
    lines = [c.label + (" · reference" if c.is_ref else ""), f"{delta_label}: vs {c.prev_label}"]
    if not c.prev_exact:
        lines.append(f"({c.prev_wanted:%a %d %b %H}z is not in the table; nearest earlier run taken)")
    if c.temperature_only:
        lines.append("temperature only")
    return _tip(lines)


def _cell_html(r: pd.Series, c: RunCol, region: str, wtitle: str, block: dict, span: int | None,
               view_key: str | None, cls: str) -> str:
    fmt = block["fmt"]
    v = float(r["value"])
    if np.isnan(v):
        return f'<td class="{cls}"><span class="mc-muted">·</span></td>'
    n_days = int(r["n_days"])
    sup = f'<sup class="mcg-n">{n_days}d</sup>' if span and 0 < n_days < span else ""
    tip = _cell_tip(c, region, wtitle, r, block, span)
    if view_key is None:
        return (f'<td class="{cls}" title="{tip}"><div class="mcg-v">{fmt.format(v)}{sup}</div>'
                f'<div class="mcg-d">{_fmt_delta(r["d_run"], fmt)}<span class="mcg-sep">·</span>'
                f'{_fmt_delta(r["d_norm"], fmt)}</div></td>')
    if view_key == "d_ref" and c.is_ref:
        return f'<td class="{cls} mcg-heat" title="{tip}"><span class="mc-muted">ref</span></td>'
    q = float(r[view_key])
    if np.isnan(q):
        return f'<td class="{cls} mcg-heat" title="{tip}"><span class="mc-muted">n/a</span></td>'
    if view_key == "value":
        return f'<td class="{cls} mcg-heat" title="{tip}">{fmt.format(q)}{sup}</td>'
    bg = _heat_bg(q, block["heat_scale"], block["warm_is_positive"])
    return f'<td class="{cls} mcg-heat" style="{bg}" title="{tip}">{_signed(q, fmt)}{sup}</td>'


def _grid_block_html(cells: pd.DataFrame, name: str, block: dict, cols: list[RunCol], win: dict,
                     view_key: str | None, delta_label: str) -> str:
    """One block of the report as a table: rows = regions, columns = window → run (+ spread)."""
    fmt = block["fmt"]
    is_sum = block["agg"] == "sum"
    n = len(cols)
    show_spread = n >= 2
    per_win = n + (1 if show_spread else 0)
    windows = [("w1", win["t1"], win["w1"]), ("w2", win["t2"], win["w2"])]

    colgroup = '<colgroup><col class="mcg-c0">' + "<col>" * (2 * per_win) + "</colgroup>"
    corner_sub = {None: f"abs · {delta_label} · Δ norm", "d_norm": "Δ norm", "d_run": delta_label,
                  "d_ref": "Δ vs reference run", "value": "abs. value"}[view_key]
    short = name.split(" (")[0]          # "Precip (sum of coming 2 weeks)" → "Precip"; the group header says the rest
    head1 = (f'<tr><th class="mcg-corner" rowspan="2"><div><span class="mcg-corner-name">{short}</span> '
             f'<span class="mcg-corner-unit">[{block["unit"]}]</span></div>'
             f'<div class="mcg-corner-sub">{corner_sub}</div></th>')
    for wkey, title, (s, e) in windows:
        if is_sum:
            txt = (f"sum over the first {MORNING_PRECIP_SUM_DAYS} forecast days" if wkey == "w1"
                   else '<span class="mcg-range">the sum spans both windows — not tabulated per week</span>')
        else:
            txt = f'{title} <span class="mcg-range">{s:%a %d %b} – {e:%a %d %b}</span>'
        head1 += f'<th class="mcg-win" colspan="{per_win}">{txt}</th>'
    head1 += "</tr>"

    head2 = "<tr>"
    for wkey, _, _ in windows:
        blank = is_sum and wkey == "w2"
        for i, c in enumerate(cols):
            cls = "mcg-col" + (" mcg-first" if i == 0 else "") + (" mcg-ref" if c.is_ref and not blank else "")
            if blank:
                head2 += f'<th class="{cls}"></th>'
            else:
                head2 += (f'<th class="{cls}" title="{_col_tip(c, delta_label)}">'
                          f'<span class="mcg-model">{c.model}</span><span class="mcg-date">{c.init:%a %d %b}</span></th>')
        if show_spread:
            head2 += f'<th class="mcg-col mcg-spread">{"" if blank else "spread"}</th>'
    head2 += "</tr>"

    # The reference column is tinted only where cells are not shaded themselves —
    # in the shaded views a faint blue tint would read as a small cold anomaly.
    tint_ref = view_key not in _HEAT_VIEWS
    body = ""
    for region, _ in block["rows"]:
        body += f'<tr><td class="mc-region">{region}</td>'
        for wkey, title, (s, e) in windows:
            blank = is_sum and wkey == "w2"
            span = None if is_sum else (e - s).days + 1
            sel = None if blank else cells[(cells["window"] == wkey) & (cells["region"] == region)].set_index("col")
            for i, c in enumerate(cols):
                cls = ("mcg-cell" + (" mcg-first" if i == 0 else "")
                       + (" mcg-refcell" if c.is_ref and tint_ref and not blank else ""))
                if blank or i not in sel.index:
                    body += f'<td class="{cls}"></td>'
                    continue
                body += _cell_html(sel.loc[i], c, region, title, block, span, view_key, cls)
            if show_spread:
                sp = np.nan if (blank or sel.empty) else float(sel["spread"].iloc[0])
                body += f'<td class="mcg-cell mcg-spread">{"" if blank else _fmt_val(sp, fmt)}</td>'
        body += "</tr>"
    return f'<div class="mc-block"><table class="mc-table mcg-table">{colgroup}{head1}{head2}{body}</table></div>'


def _grid_footer(cols: list[RunCol], delta_label: str, view_key: str | None) -> str:
    """Legend for the shaded views, then one line per column: which run, paired with which."""
    out = ""
    if view_key in _HEAT_VIEWS:
        scales = " · ".join(f"±{b['heat_scale']:g} {b['unit']} {n.split(' ')[0].lower()}"
                            for n, b in MORNING_BLOCKS.items())
        out += (f'<div class="mcg-legend"><span>colder · more wind, solar, precipitation'
                f'<span class="mcg-swatch" style="background:linear-gradient(90deg,{DIV_NEG},{DIV_MID},{DIV_POS});"></span>'
                f'warmer · less</span><span>shade saturates at {scales}</span></div>')
    lines = []
    for c in cols:
        n_days, expected = c.coverage()
        cov = "" if n_days >= 0.9 * expected else f' · <span class="mc-neg">{n_days}/{expected} forecast days</span>'
        pairing = f"{delta_label} vs {c.prev_label}" + ("" if c.prev_exact else " (nearest earlier run)")
        extra = " · temperature only" if c.temperature_only else ""
        lines.append(f"<b>{c.model}</b> {c.init:%a %d %b}{' · reference' if c.is_ref else ''} — {pairing}{extra}{cov}")
    return out + '<div class="mcg-runs">' + "<br>".join(lines) + "</div>"


def _grid_csv(cells: pd.DataFrame, delta_label: str) -> bytes:
    out = cells.rename(columns={"window_title": "window_label", "d_run": "delta_run", "d_norm": "delta_norm",
                                "d_ref": "delta_vs_reference", "spread": "spread_across_runs"})
    out = out.drop(columns=["window", "col"])
    out.insert(len(out.columns), "delta_run_label", delta_label)
    return out.to_csv(index=False).encode()


# ─── AI commentary — two agent families ────────────────────────────────────────

def build_brief_context(cur: pd.DataFrame, prev: pd.DataFrame, win: dict, model: str,
                        cur_init, prev_init, delta_label: str, today: pd.Timestamp) -> dict:
    """The on-screen numbers, as a structure the agents' context documents are
    rendered from — so what the commentary reads is exactly what the tables show.

    Week 3 is included only when the run genuinely reaches into it, using the
    same minimum the scenario engine applies to a forecast week
    (SCENARIO_MIN_WEEK_DAYS) — normal for EC-Extended, rare for EC-ENS. Any
    window the run covers only partly is labelled with its day count, so an
    average over four days is never read as a full week.
    """
    windows = []
    specs = [("w1", "t1", True), ("w2", "t2", False), ("w3", "t3", False)]
    for wkey, tkey, include_precip in specs:
        window = win[wkey]
        span = (window[1] - window[0]).days + 1
        blocks, has_data, covered = {}, False, 0
        for name, block in MORNING_BLOCKS.items():
            if block["agg"] == "sum" and not include_precip:
                continue
            rows = block_values(cur, prev, block, None if block["agg"] == "sum" else window)
            if any(not np.isnan(r["value"]) for r in rows):
                has_data = True
            if block["agg"] == "mean":
                covered = max(covered, max((r["n_days"] for r in rows), default=0))
            blocks[name] = {"unit": block["unit"], "rows": rows}
        if not blocks or not has_data:
            continue
        if wkey == "w3" and covered < SCENARIO_MIN_WEEK_DAYS:
            continue
        rng = f"{window[0]:%a %d %b} – {window[1]:%a %d %b}"
        if 0 < covered < span:
            rng += f"; the run covers only {covered} of these {span} days"
        windows.append({"title": win[tkey], "range": rng, "blocks": blocks})
    return {"run": model, "today": today, "delta_label": delta_label,
            "run_init": f"{cur_init:%a %d %b}" if cur_init is not None else "n/a",
            "prev_init": f"{prev_init:%a %d %b}" if prev_init is not None else "n/a",
            "windows": windows}


_SIG_COLOR = {"BULLISH": STATUS_CRITICAL, "BEARISH": CAT_BLUE, "NEUTRAL": INK_MUTED}
_SIG_ARROW = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "●"}


def _signal_card(label: str, subtitle: str, signal: str) -> str:
    color = _SIG_COLOR.get(signal, INK_MUTED)
    arrow = _SIG_ARROW.get(signal, "●")
    return (f'<div class="agent-card" style="border-top-color:{color};">'
            f'<div class="agent-card-label">{label}</div>'
            f'<div class="agent-card-signal" style="color:{color};">{arrow} {signal}</div>'
            f'<div class="agent-card-sub">{subtitle}</div></div>')


def _render_family(result: dict, agents_meta: list[tuple[str, str, str]]) -> None:
    from _ai_brief import FAMILIES, prose_to_html

    signals, briefs = result["signals"], result["briefs"]
    cards = [_signal_card(label, sub, signals.get(key, "NEUTRAL")) for key, label, sub in agents_meta]
    cols = st.columns(len(cards))
    for col, html_ in zip(cols, cards):
        with col:
            st.markdown(html_, unsafe_allow_html=True)

    overall = signals.get("synthesis", "NEUTRAL")
    color = _SIG_COLOR.get(overall, INK_MUTED)
    st.markdown(f'<div class="agent-overall" style="background:{color};">'
                f'<span class="agent-overall-label">{result["commodity"]} — net weather signal</span>'
                f'<span class="agent-overall-value">{_SIG_ARROW.get(overall, "●")} {overall}</span></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="brief-box">{prose_to_html(briefs.get("synthesis", ""))}</div>',
                unsafe_allow_html=True)

    agent_labels = {k: v[0] for k, v in FAMILIES[result["family"]]["agents"].items()}
    with st.expander(f"{result['label']} — the three specialist reads"):
        for key, label in agent_labels.items():
            st.markdown(f"**{label}** · {signals.get(key, 'NEUTRAL')}")
            st.markdown(briefs.get(key, "_no output_"))
    with st.expander(f"{result['label']} — exactly what each agent was shown"):
        for key, doc in result["docs"].items():
            st.markdown(f"**{agent_labels.get(key, 'synthesis')}**")
            st.code(doc)
    st.caption(f"Generated {result['generated_at']} · {result['label']} family")


def _render_ai_brief(ctx: dict, today: pd.Timestamp) -> None:
    """Two families of agents commenting on the reference run — power and gas."""
    from _ai_brief import FAMILIES, get_az_credentials, generate_family_brief

    st.markdown("##### Market read — agent families")
    st.caption("Two families comment on the reference run's numbers and nothing else: one on the power "
               "market (temperature → load, wind & solar → residual load, precipitation → hydro), one on gas "
               "(temperature → LDZ heating demand, wind & solar → gas-for-power). Each specialist reads "
               "only its own rows; a synthesis agent per family nets them into a few sentences. The gas "
               "family is additionally given the Gas Demand section's LDZ and displaced-gas figures for "
               "these same runs, so the words and the GWh cannot drift apart.")
    st.markdown(BRIEF_CSS, unsafe_allow_html=True)

    if not ctx.get("windows"):
        status_banner("No window in this run has data to comment on.", "warning")
        return

    tenant_id, client_id, client_secret = get_az_credentials()
    if not all([tenant_id, client_id, client_secret]):
        status_banner("Azure OpenAI credentials not found — the agent families are unavailable. Set "
                      "azure_tenant_id / azure_client_id / azure_client_secret in the Databricks secret "
                      "scope `axpo`, in st.secrets, or as AZURE_* environment variables (app.yaml wires "
                      "the secret scope for a deployed app).", "warning")
        return

    c1, c2, c3 = st.columns([1.3, 1.3, 4])
    with c1:
        run_power = st.button("Run power brief", type="primary", key="mc_brief_power")
    with c2:
        run_gas = st.button("Run gas brief", type="primary", key="mc_brief_gas")

    snapshot = {}
    if run_gas:
        try:
            from _gas import gas_demand_snapshot
            snapshot = gas_demand_snapshot(today.date())
        except Exception:
            snapshot = {}

    progress = st.empty()

    def _prog(msg: str):
        progress.caption(f"⟳  {msg}")

    for family, run_it in (("power", run_power), ("gas", run_gas)):
        state_key = f"mc_brief_{family}_result"
        if run_it:
            try:
                st.session_state[state_key] = generate_family_brief(
                    family, ctx, tenant_id, client_id, client_secret,
                    gas_snapshot=snapshot if family == "gas" else None,
                    model=AI_BRIEF_MODEL, max_tokens=AI_BRIEF_MAX_TOKENS,
                    synthesis_max_tokens=AI_BRIEF_SYNTHESIS_MAX_TOKENS, progress_cb=_prog)
                if family == "gas" and not snapshot:
                    st.session_state[f"{state_key}_no_model"] = True
                else:
                    st.session_state.pop(f"{state_key}_no_model", None)
            except Exception as exc:
                progress.empty()
                st.error(f"{FAMILIES[family]['label']} brief failed: {exc}")
                with st.expander("Error details"):
                    import traceback
                    st.code(traceback.format_exc())
    progress.empty()

    n_calls = {f: len(FAMILIES[f]["agents"]) + 1 for f in FAMILIES}
    if not any(st.session_state.get(f"mc_brief_{f}_result") for f in FAMILIES):
        st.info(f"Run a family to get the commentary. The power family makes {n_calls['power']} Azure "
                f"OpenAI calls, the gas family {n_calls['gas']} — about 10-20 seconds each.")
        return

    for family in FAMILIES:
        result = st.session_state.get(f"mc_brief_{family}_result")
        if not result:
            continue
        st.markdown(f"**{result['label']}**")
        if st.session_state.get(f"mc_brief_{family}_result_no_model"):
            status_banner("The Gas Demand section's figures were unavailable, so the gas family worked "
                          "from the Morning Call table alone.", "warning")
        meta = AI_POWER_AGENTS if family == "power" else AI_GAS_AGENTS
        _render_family(result, meta)
        st.markdown("")


# ─── page ──────────────────────────────────────────────────────────────────────

def render_morning_call():
    st.markdown("#### MORNING CALL")
    st.caption("The Morning Report table, live from the sandbox (morning_daily): EC-ENS 00z 'Avg' curves, "
               "daily CET means, Volue 30-year normal — same rows, windows and deltas as "
               "import_00z_add_solar_np_tot.py. The other models' runs sit next to it in the same grid, "
               "each with its own change vs its previous run and its difference to the reference.")
    try:
        df = load_morning_daily()
    except Exception as e:
        st.error(f"Cannot read from the sandbox tables (morning_daily): {e}")
        return
    if df.empty:
        status_banner("No data found in morning_daily.", "warning")
        return
    df = _drop_partial_days(df)

    runs = _list_runs(df)
    if not runs:
        status_banner("morning_daily has rows but no run could be identified.", "critical")
        return
    run_map = {_run_label(p, t): (p, t) for p, t in runs}
    keys = list(run_map)
    latest_by_pattern: dict[str, str] = {}
    for k in keys:
        latest_by_pattern.setdefault(run_map[k][0], k)
    ref_default = latest_by_pattern.get(MORNING_DEFAULT_REFERENCE_PATTERN, keys[0])

    # --- controls: reference run · compare columns · Δ pairing · what the cells show ---
    c1, c2, c3, c4 = st.columns([1.9, 3.1, 1.9, 3.1])
    with c1:
        ref_key = st.selectbox("Reference run", keys, index=keys.index(ref_default), key="mcg_ref",
                               help="The report's run. Its init day sets the two windows (weekday rule) and "
                                    "the Δ pairing; the commentary and the CSV are built on it. EC-ENS 00z "
                                    "is the Morning Report's.")
    ref_pat, ref_init = run_map[ref_key]
    cmp_default = [latest_by_pattern[p] for p in MORNING_DEFAULT_COMPARE_PATTERNS
                   if p in latest_by_pattern and p != ref_pat]
    with c2:
        cmp_keys = st.multiselect("Compare columns", keys, default=cmp_default, key="mcg_cols",
                                  help="Any runs — other models' latest, or earlier runs of the same model. "
                                       "Each gets the same three numbers as the reference, plus its Δ vs the "
                                       "reference. Meteomatics rows carry temperature only.")
    with c3:
        rule_key = st.selectbox("Δ run vs", list(MORNING_PREV_RULES), key="mcg_prev",
                                help="The earlier run every column's Δ run is taken against — the same "
                                     "interval for all of them, so the models' moves are comparable.")
    with c4:
        view = st.radio("Cells show", list(VIEWS), horizontal=True, key="mcg_view",
                        help="Detail: abs. value with Δ run · Δ norm underneath, in every cell. The other "
                             "views put one quantity per cell and shade it, for a one-glance read across runs.")
    rule = MORNING_PREV_RULES[rule_key]
    view_key = VIEWS[view]

    today = ref_init.normalize()
    win = morning_windows(today)
    if rule == "report":
        delta_label = win["delta_label"]
    elif rule == "previous":
        delta_label = "Δ prev. run"
    else:
        delta_label = f"Δ -{24 * int(rule)}h"

    cols = [_make_col(df, ref_pat, ref_init, rule, win, is_ref=True)]
    for k in cmp_keys:
        p, t = run_map[k]
        if (p, t) != (ref_pat, ref_init):
            cols.append(_make_col(df, p, t, rule, win, is_ref=False))
    ref = cols[0]

    # --- what the reader must know before the numbers ---
    n_days, expected = ref.coverage()
    if n_days < 0.9 * expected:
        status_banner(f"{ref.label} has {n_days}/{expected} forecast days — the run may still be loading.", "warning")
    if ref.prev_init is None:
        status_banner("No earlier run of the reference model in the table — its Δ run shows n/a.", "warning")
    elif not ref.prev_exact:
        status_banner(f"{ref.prev_wanted:%a %d %b %H}z is not in the table — the reference's Δ run is taken "
                      f"against {ref.prev_label} instead.", "warning")

    span1 = "weekend" if win["kind1"] == "weekend" else "weekly"
    st.markdown(f'<div class="mc-sub">Windows follow the reference run\'s init day ({today:%A %d %b}): '
                f'<b>{win["t1"]}</b> {win["w1"][0]:%d %b} – {win["w1"][1]:%d %b} ({span1} average) · '
                f'<b>{win["t2"]}</b> {win["w2"][0]:%d %b} – {win["w2"][1]:%d %b} (weekly average) · '
                f'precipitation = sum of the first {MORNING_PRECIP_SUM_DAYS} forecast days · '
                f'{delta_label} = change vs each model\'s own earlier run · superscript = days of the window '
                f'the run covers, when not all · hover a cell for every number.</div>', unsafe_allow_html=True)

    # --- the grid ---
    cells = compute_grid(cols, win)
    for name, block in MORNING_BLOCKS.items():
        st.markdown(_grid_block_html(cells[cells["block"] == name], name, block, cols, win, view_key, delta_label),
                    unsafe_allow_html=True)
    st.markdown(_grid_footer(cols, delta_label, view_key), unsafe_allow_html=True)
    st.download_button("Download grid (CSV)", _grid_csv(cells, delta_label),
                       f"morning_call_{today:%Y%m%d}.csv", "text/csv", key="mcg_dl",
                       help="Every cell of every column: value, Δ run, Δ norm, Δ vs reference, spread.")

    # --- commentary on the reference run ---
    st.divider()
    ctx = build_brief_context(ref.rows, ref.prev_rows, win, ref.model, ref.init, ref.prev_init, delta_label, today)
    _render_ai_brief(ctx, today)
