"""Morning Call — the Morning Report table, live, built around what changed.

Port of P:/QFA/TonyWeather/Lorenzo_Trainee/Morning_Report/import_00z_add_solar_np_tot.py
(which wrote table_00z_add_solar_np_tot.xlsx through wapi). Same arithmetic, laid
out so that the two things the desk looks for — the run-over-run change and the
departure from normal — are read first, and the absolute values second:

  At a glance         the reference run's biggest moves since its previous run and its
                      largest anomalies, as chips ranked across the blocks on each block's
                      own scale (a 1 °C move counts like a 1 GW wind move or 0.5 GW of
                      solar), plus where the models shown disagree most.
  Arrow chart         per block and window, every region's anomaly on one axis: the
                      filled dot is the reference run (red = warmer / less wind, solar,
                      precipitation; blue = colder / more), the hollow dot where its
                      previous run stood, the bar between them the change; the other
                      runs as small diamonds.
  Grid                rows = the report's regions per block, columns = window → run. By
                      default every cell shows the two deltas as shaded pills (Δ run |
                      Δ norm) with the absolute value small beneath; the other views
                      put the report's triple, or a single shaded quantity, in the cell.
  Windows             weekday-dependent, taken from the reference run's init day
        Mon, Tue      this week (Mon–Sun)              | next week
        Wed, Thu      weekend (Sat–Sun)                | next week
        Fri–Sun       next week                        | week after
  Rows                Temperatures [°C]  FRA DE UK ITA HUN Nordic Iberia
                      Wind [GW]          DE UK FRA ITA SEE Nordic Iberia
                      Solar PV [GW]      DE FRA ITA SEE Nordic Iberia
                      Precip [TWh]       Alps(cwe+it-nord) Nordic SEE Iberia   (15-day sum, first group only)
  Columns             the reference run first (the latest issue of EC-ENS by default),
                      then any other runs — the latest GFS-ENS, AIFS-ENS, the operational
                      models, or an earlier cycle of the same model. Every run carries
                      the report's three numbers — abs. value, Δ vs its own previous run
                      (Δ -24h; Δ -72h on a Monday), Δ norm — plus Δ vs the reference and,
                      per window, the spread between the runs shown.

The Excel's free-text boxes — Pattern and Comment per window, and the Week 3
commentary block — are gone. In their place two agent families read the
reference run's numbers and write the commentary (see _ai_brief.py); the gas
family is also handed the Gas Demand section's GWh for the very runs the grid
shows. Week 3 is still reached: it is handed to the agents whenever the
reference run has enough days in it, which is normal for EC-Extended and rare
for EC-ENS.

Data (MORNING_SOURCES): by default {SBX_SCHEMA}.morning_daily_eq — Energy
Quantified, every model EQ has and every cycle (00/06/12/18), with the EQ
normal, refreshed every couple of hours by Power_dashboard/pipeline/run_morning.bat,
so the grid opens on the latest issue and the agents brief about it; or
{SBX_SCHEMA}.morning_daily — the notebook's Volue 'Avg' curves (00z/12z) with
the Volue normal plus Meteomatics means. Layout: PC (everything) or Phone (one
window, the reference run plus one compare column, big type).
"""
from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    MORNING_BLOCKS, MORNING_MIN_DAY_COVERAGE, MORNING_MODEL_LABELS, MORNING_PREV_RULES, MORNING_PRECIP_SUM_DAYS,
    EXPECTED_HORIZON, MORNING_SOURCES, enabled_sources, MORNING_DEFAULT_SOURCE, MORNING_PHONE_MAX_COMPARE,
    AI_BRIEF_MODEL, AI_BRIEF_MAX_TOKENS, AI_BRIEF_SYNTHESIS_MAX_TOKENS, AI_POWER_AGENTS, AI_GAS_AGENTS,
    SCENARIO_MIN_WEEK_DAYS,
)
from _charts import make_morning_anomaly_chart
from _data import load_morning_daily
from _style import BRIEF_CSS, PHONE_CSS, INK_MUTED, CAT_BLUE, STATUS_CRITICAL, DIV_NEG, DIV_MID, DIV_POS
from _ui import status_banner

# What a cell shows. "pair" = the two deltas as shaded pills with the value beneath (default);
# None = the report's triple; otherwise the grid column to put in the cell, shaded.
VIEWS: dict[str, str | None] = {
    "Change · anomaly": "pair", "Detail": None, "Δ norm": "d_norm", "Δ prev run": "d_run",
    "Δ vs ref": "d_ref", "Value": "value",
}
_HEAT_VIEWS = ("d_norm", "d_run", "d_ref")
_SHADED_VIEWS = _HEAT_VIEWS + ("pair",)
GLANCE_CHIPS = 5                 # chips per "at a glance" line
GLANCE_MIN_SCORE = 0.15          # |Δ| / heat_scale below this is not worth a chip


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


def _short_title(title: str) -> str:
    """'Week 41 (next week)' → 'Week 41'; 'Weekend (40)' → 'Weekend 40'."""
    return title.replace(" (next week)", "").replace("(", "").replace(")", "")


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


def _run_label_short(pattern: str, init: pd.Timestamp) -> str:
    return f"{MORNING_MODEL_LABELS.get(pattern, pattern)} {init:%H}z"


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


# ─── rendering helpers ─────────────────────────────────────────────────────────

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


def _heat_bg(v: float, scale: float, warm_is_positive: bool, strength: float = 0.8) -> str:
    """Cell shade on the diverging pair: neutral at 0, saturating at ±scale, ink stays readable."""
    if scale <= 0 or v is None or np.isnan(v) or round(v, 1) == 0:
        return ""
    t = min(abs(v) / scale, 1.0) ** 0.7 * strength
    pole = DIV_POS if (v > 0) == warm_is_positive else DIV_NEG
    return f"background:{_blend(DIV_MID, pole, t)};"


def _pill(v: float, block: dict) -> str:
    """One shaded pill for a delta — the building block of the default cell."""
    if v is None or np.isnan(v):
        return '<span class="mcg-pill mcg-pill-na">n/a</span>'
    bg = _heat_bg(v, block["heat_scale"], block["warm_is_positive"])
    return f'<span class="mcg-pill" style="{bg}">{_signed(v, block["fmt"])}</span>'


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


# ─── at a glance ───────────────────────────────────────────────────────────────

def _glance_rows(cells: pd.DataFrame, windows: tuple[str, ...]) -> pd.DataFrame:
    """The reference run's cells with each delta scored on its block's scale."""
    ref = cells[(cells["col"] == 0) & (cells["window"].isin(windows))].copy()
    if ref.empty:
        return ref
    ref["heat_scale"] = [MORNING_BLOCKS[b]["heat_scale"] for b in ref["block"]]
    ref["score_run"] = ref["d_run"].abs() / ref["heat_scale"]
    ref["score_norm"] = ref["d_norm"].abs() / ref["heat_scale"]
    ref["score_spread"] = ref["spread"].abs() / ref["heat_scale"]
    return ref


def _chip(row: pd.Series, quantity: str, win: dict, n_windows: int) -> str:
    block = MORNING_BLOCKS[row["block"]]
    v = float(row[quantity])
    short = row["block"].split(" (")[0]
    if block["agg"] == "sum":
        wtxt = f"{MORNING_PRECIP_SUM_DAYS}-day sum"
    else:
        wtxt = _short_title(win["t1"] if row["window"] == "w1" else win["t2"])
    if quantity == "spread":
        bg = _heat_bg(v, block["heat_scale"], True, strength=0.0)
        q = f"{block['fmt'].format(v)} {block['unit']} apart"
    else:
        bg = _heat_bg(v, block["heat_scale"], block["warm_is_positive"], strength=0.85)
        q = f"{_signed(v, block['fmt'])} {block['unit']}"
    where = f'<span class="mc-chip-w">{wtxt}</span>' if (n_windows > 1 or block["agg"] == "sum") else ""
    tip = _tip([f"{row['region']} · {short} · {wtxt}", f"value {block['fmt'].format(float(row['value']))} {block['unit']}",
                f"Δ vs previous run {_signed(float(row['d_run']), block['fmt']) if not np.isnan(row['d_run']) else 'n/a'}",
                f"Δ norm {_signed(float(row['d_norm']), block['fmt']) if not np.isnan(row['d_norm']) else 'n/a'}"])
    return (f'<span class="mc-chip" style="{bg}" title="{tip}"><b>{row["region"]}</b> {short.lower()} '
            f'<span class="mc-chip-q">{q}</span>{where}</span>')


def _glance_html(cells: pd.DataFrame, ref: RunCol, win: dict, delta_label: str, windows: tuple[str, ...],
                 n_cols: int) -> str:
    """Three lines of chips: biggest moves since the previous run, largest anomalies,
    and — with two or more runs shown — where the models disagree most."""
    rows = _glance_rows(cells, windows)
    if rows.empty:
        return ""
    n_w = len(windows)
    lines = []

    def line(label: str, score: str, quantity: str) -> str:
        sel = rows[rows[score].notna() & (rows[score] >= GLANCE_MIN_SCORE)].sort_values(score, ascending=False)
        sel = sel.head(GLANCE_CHIPS)
        chips = "".join(_chip(r, quantity, win, n_w) for _, r in sel.iterrows()) if not sel.empty else \
            '<span class="mc-chip-empty">nothing notable</span>'
        return f'<div class="mc-glance"><span class="mc-glance-label">{label}</span>{chips}</div>'

    lines.append(line(f"Biggest moves · {delta_label} vs {ref.prev_label}" if ref.prev_init is not None
                      else "Biggest moves · no earlier run", "score_run", "d_run"))
    lines.append(line("Furthest from normal", "score_norm", "d_norm"))
    if n_cols >= 2:
        lines.append(line("Models disagree most", "score_spread", "spread"))
    return "".join(lines)


# ─── grid ──────────────────────────────────────────────────────────────────────

def _cell_html(r: pd.Series, c: RunCol, region: str, wtitle: str, block: dict, span: int | None,
               view_key: str | None, cls: str) -> str:
    fmt = block["fmt"]
    v = float(r["value"])
    if np.isnan(v):
        return f'<td class="{cls}"><span class="mc-muted">·</span></td>'
    n_days = int(r["n_days"])
    sup = f'<sup class="mcg-n">{n_days}d</sup>' if span and 0 < n_days < span else ""
    tip = _cell_tip(c, region, wtitle, r, block, span)
    if view_key == "pair":
        return (f'<td class="{cls}" title="{tip}"><div class="mcg-pair">{_pill(float(r["d_run"]), block)}'
                f'{_pill(float(r["d_norm"]), block)}</div><div class="mcg-abs">{fmt.format(v)}{sup}</div></td>')
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
    corner_sub = {"pair": f"{delta_label} | Δ norm · abs", None: f"abs · {delta_label} · Δ norm", "d_norm": "Δ norm",
                  "d_run": delta_label, "d_ref": "Δ vs reference run", "value": "abs. value"}[view_key]
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
    tint_ref = view_key not in _SHADED_VIEWS
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


def _grid_block_html_phone(cells: pd.DataFrame, name: str, block: dict, cols: list[RunCol], win: dict,
                           wkey: str, delta_label: str) -> str:
    """One block for a phone: one window, regions as rows, the reference run's
    triple spread over three columns (Δ run · Δ norm shaded, then the value) and,
    for each compare run, its value with its Δ vs the reference underneath."""
    fmt, unit = block["fmt"], block["unit"]
    is_sum = block["agg"] == "sum"
    use = "w1" if is_sum else wkey                 # precipitation is the 15-day sum, whatever the window
    s, e = win[use]
    ref = cols[0]
    short = name.split(" (")[0]
    sub = (f"sum of the first {MORNING_PRECIP_SUM_DAYS} forecast days" if is_sum
           else f"{win['t1'] if use == 'w1' else win['t2']} · {s:%d %b} – {e:%d %b}")
    head = (f"<tr><th>[{unit}]</th><th>{delta_label}<span class='mcg-date'>vs {ref.prev_label}</span></th>"
            f"<th>Δ norm</th><th>{ref.model}<span class='mcg-date'>{ref.init:%a %d %b} · value</span></th>")
    for c in cols[1:]:
        head += f"<th>{c.model}<span class='mcg-date'>{c.init:%a %d %b}<br>value · Δ vs ref</span></th>"
    head += "</tr>"
    span = None if is_sum else (e - s).days + 1
    body = ""
    for region, _ in block["rows"]:
        sel = cells[(cells["window"] == use) & (cells["region"] == region)].set_index("col")
        body += f"<tr><td class='mc-region'>{region}</td>"
        if 0 not in sel.index or np.isnan(float(sel.loc[0, "value"])):
            body += "<td></td><td></td><td><span class='mc-muted'>·</span></td>" + "<td></td>" * (len(cols) - 1) + "</tr>"
            continue
        r0 = sel.loc[0]
        n_days = int(r0["n_days"])
        sup = f"<span class='mcp-sub'>{n_days}/{span} days</span>" if span and 0 < n_days < span else ""
        d_run, d_norm = float(r0["d_run"]), float(r0["d_norm"])
        body += (f"<td class='mcp-heat' style='{_heat_bg(d_run, block['heat_scale'], block['warm_is_positive'])}'>"
                 f"{_signed(d_run, fmt) if not np.isnan(d_run) else '<span class=mc-muted>n/a</span>'}</td>"
                 f"<td class='mcp-heat' style='{_heat_bg(d_norm, block['heat_scale'], block['warm_is_positive'])}'>"
                 f"{_signed(d_norm, fmt) if not np.isnan(d_norm) else '<span class=mc-muted>n/a</span>'}</td>"
                 f"<td>{fmt.format(float(r0['value']))}{sup}</td>")
        for i, c in enumerate(cols[1:], start=1):
            if i not in sel.index or np.isnan(float(sel.loc[i, "value"])):
                body += "<td><span class='mc-muted'>·</span></td>"
                continue
            r = sel.loc[i]
            body += f"<td>{fmt.format(float(r['value']))}<span class='mcp-sub'>{_fmt_delta(r['d_ref'], fmt)}</span></td>"
        body += "</tr>"
    return (f"<div class='mc-block'><div class='mcp-title'>{short} <span>{sub}</span></div>"
            f"<table class='mcp-table'>{head}{body}</table></div>")


def _legend_html(delta_label: str, view_key: str | None) -> str:
    scales = " · ".join(f"±{b['heat_scale']:g} {b['unit']} {n.split(' ')[0].lower()}"
                        for n, b in MORNING_BLOCKS.items())
    what = {"pair": f"left pill = {delta_label} (change vs the model's earlier run) · right pill = Δ norm · "
                    "abs. value beneath",
            "d_norm": "Δ norm", "d_run": f"{delta_label}", "d_ref": "Δ vs the reference run"}.get(view_key, "")
    return (f'<div class="mcg-legend"><span>{what}</span><span>colder · more wind, solar, precipitation'
            f'<span class="mcg-swatch" style="background:linear-gradient(90deg,{DIV_NEG},{DIV_MID},{DIV_POS});"></span>'
            f'warmer · less</span><span>shade saturates at {scales}</span></div>')


def _grid_footer(cols: list[RunCol], delta_label: str, view_key: str | None) -> str:
    """Legend for the shaded views, then one line per column: which run, paired with which."""
    out = _legend_html(delta_label, view_key) if view_key in _SHADED_VIEWS else ""
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


def _render_family(result: dict, agents_meta: list[tuple[str, str, str]], phone: bool = False) -> None:
    from _ai_brief import FAMILIES, prose_to_html

    signals, briefs = result["signals"], result["briefs"]
    cards = [_signal_card(label, sub, signals.get(key, "NEUTRAL")) for key, label, sub in agents_meta]
    if phone:
        for html_ in cards:                      # stacked, one per row, on a narrow screen
            st.markdown(html_, unsafe_allow_html=True)
    else:
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


def _render_ai_brief(ctx: dict, today: pd.Timestamp, phone: bool = False, source: str = MORNING_DEFAULT_SOURCE,
                     gas_runs: tuple[tuple[str, str], ...] = ()) -> None:
    """Two families of agents commenting on the reference run — power and gas. The
    gas family is also given the Gas Demand section's figures for `gas_runs`, the
    (pattern, init) pairs of the grid's columns, read from the same source."""
    from _ai_brief import FAMILIES, get_az_credentials, generate_family_brief

    st.markdown("##### Market read — agent families")
    if phone:
        st.caption(f"Commentary on the reference run — {ctx.get('run', '')}, initialised {ctx.get('run_init', '')} — "
                   "by the power and gas agent families.")
    else:
        st.caption("Two families comment on the reference run's numbers and nothing else — the latest issue of the "
                   "reference model by default, so every new cycle gets its own read: one on the power "
                   "market (temperature → load, wind & solar → residual load, precipitation → hydro), one on gas "
                   "(temperature → LDZ heating demand, wind & solar → gas-for-power). Each specialist reads "
                   "only its own rows; a synthesis agent per family nets them into a few sentences. The gas "
                   "family is additionally given the Gas Demand section's LDZ and displaced-gas figures for "
                   "the runs in this grid, from the same source, so the words and the GWh cannot drift apart.")
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

    if phone:
        run_power = st.button("Run power brief", type="primary", key="mc_brief_power", use_container_width=True)
        run_gas = st.button("Run gas brief", type="primary", key="mc_brief_gas", use_container_width=True)
    else:
        c1, c2, c3 = st.columns([1.3, 1.3, 4])
        with c1:
            run_power = st.button("Run power brief", type="primary", key="mc_brief_power")
        with c2:
            run_gas = st.button("Run gas brief", type="primary", key="mc_brief_gas")

    snapshot = {}
    if run_gas:
        try:
            from _gas import gas_demand_snapshot
            snapshot = gas_demand_snapshot(source, gas_runs or None)
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
        _render_family(result, meta, phone)
        st.markdown("")


# ─── page ──────────────────────────────────────────────────────────────────────

def _layout_and_source() -> tuple[str, bool]:
    """The two switches at the top of the page: data source and PC / Phone layout.
    `?layout=phone` in the URL opens the phone layout (bookmark it on the phone)."""
    names = enabled_sources(MORNING_SOURCES)
    if st.session_state.get("mc_source") not in names:
        st.session_state["mc_source"] = MORNING_DEFAULT_SOURCE if MORNING_DEFAULT_SOURCE in names else names[0]
    if "mc_layout" not in st.session_state:
        try:
            qp = str(st.query_params.get("layout", "")).lower()
        except Exception:
            qp = ""
        st.session_state["mc_layout"] = "Phone" if qp == "phone" else "PC"
    t1, t2, t3 = st.columns([1.4, 1.4, 4.2])
    with t1:
        if len(names) == 1:
            source = names[0]
        else:
            source = st.radio("Source", names, horizontal=True, key="mc_source",
                              help="\n".join(f"{k}: {MORNING_SOURCES[k]['desc']}" for k in names))
    with t2:
        layout = st.radio("Layout", ["PC", "Phone"], horizontal=True, key="mc_layout",
                          help="Phone: one window at a time, one compare column, big type, stacked cards. "
                               "Open the app with ?layout=phone to start in it.")
    return source, layout == "Phone"


def _window_title(win: dict, wkey: str) -> str:
    s, e = win[wkey]
    return f"{_short_title(win['t1'] if wkey == 'w1' else win['t2'])} · {s:%d %b} – {e:%d %b}"


def render_morning_call():
    st.markdown("#### MORNING CALL")
    source, phone = _layout_and_source()
    src = MORNING_SOURCES[source]
    if phone:
        st.markdown(PHONE_CSS, unsafe_allow_html=True)
    else:
        st.caption(f"The Morning Report, live — {src['desc']}. Same rows, windows and deltas as "
                   "import_00z_add_solar_np_tot.py, read change-first: the biggest moves and anomalies of the "
                   "reference run up front, the arrow chart per block, then the grid with the other models' runs "
                   "as columns — each with its own change vs its previous run and its difference to the reference. "
                   "The grid opens on the latest issue of the reference model.")
    try:
        df = load_morning_daily(source)
    except Exception as e:
        st.error(f"Cannot read from the sandbox table {src['table']}: {e}")
        return
    if df.empty:
        status_banner(f"No rows in {src['table']} yet — it is written by Power_dashboard/pipeline "
                      "(run_morning.bat, run_daily.py --only volue).", "warning")
        return
    df = _drop_partial_days(df)

    runs = _list_runs(df)
    if not runs:
        status_banner(f"{src['table']} has rows but no run could be identified.", "critical")
        return
    run_map = {_run_label(p, t): (p, t) for p, t in runs}
    keys = list(run_map)
    latest_by_pattern: dict[str, str] = {}
    for k in keys:
        latest_by_pattern.setdefault(run_map[k][0], k)
    ref_default = latest_by_pattern.get(src["reference"], keys[0])
    if st.session_state.get("mc_source_last") != source:
        # a new source has other run keys: forget the previous selections
        for k in ("mcg_ref", "mcg_cols", "mcp_ref", "mcp_cmp"):
            st.session_state.pop(k, None)
        st.session_state["mc_source_last"] = source

    # --- controls: reference run · compare columns · Δ pairing · what the cells show · chart ---
    if phone:
        with st.expander("Settings", expanded=False):
            ref_key = st.selectbox("Reference run", keys, index=keys.index(ref_default), key="mcp_ref")
            ref_pat, ref_init = run_map[ref_key]
            cmp_default = [latest_by_pattern[p] for p in src["compare"]
                           if p in latest_by_pattern and p != ref_pat][:MORNING_PHONE_MAX_COMPARE]
            cmp_keys = st.multiselect(f"Compare (max {MORNING_PHONE_MAX_COMPARE})", keys, default=cmp_default,
                                      key="mcp_cmp", max_selections=MORNING_PHONE_MAX_COMPARE)
            rule_key = st.selectbox("Δ run vs", list(MORNING_PREV_RULES), key="mcg_prev")
            show_chart = st.toggle("Arrow chart", value=True, key="mcp_chart")
        view_key = "pair"
    else:
        c1, c2, c3, c4, c5 = st.columns([1.9, 2.9, 1.8, 2.9, 0.9])
        with c1:
            ref_key = st.selectbox("Reference run", keys, index=keys.index(ref_default), key="mcg_ref",
                                   help="The report's run. Its init day sets the two windows (weekday rule) and "
                                        "the Δ pairing; the chips, the chart, the commentary and the CSV are built "
                                        "on it. Defaults to the latest issue of the reference model.")
        ref_pat, ref_init = run_map[ref_key]
        cmp_default = [latest_by_pattern[p] for p in src["compare"] if p in latest_by_pattern and p != ref_pat]
        with c2:
            cmp_keys = st.multiselect("Compare columns", keys, default=cmp_default, key="mcg_cols",
                                      help="Any runs — other models' latest, or earlier cycles of the same model. "
                                           "Each gets the same three numbers as the reference, plus its Δ vs the "
                                           "reference; in the chart they are the diamonds. Meteomatics rows carry "
                                           "temperature only.")
        with c3:
            rule_key = st.selectbox("Δ run vs", list(MORNING_PREV_RULES), key="mcg_prev",
                                    help="The earlier run every column's Δ run is taken against — the same "
                                         "interval for all of them, so the models' moves are comparable. "
                                         "'Previous run of the same model' is the previous cycle (6 or 12 h).")
        with c4:
            view = st.radio("Cells show", list(VIEWS), horizontal=True, key="mcg_view",
                            help="Change · anomaly: the two deltas as shaded pills, the value small beneath. "
                                 "Detail: the report's triple. The other views put one shaded quantity per cell.")
        with c5:
            show_chart = st.toggle("Chart", value=True, key="mcg_chart", help="The arrow chart above the grid.")
        view_key = VIEWS[view]
    rule = MORNING_PREV_RULES[rule_key]

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
    col_labels = [c.label for c in cols]

    # --- what the reader must know before the numbers ---
    n_days, expected = ref.coverage()
    if n_days < 0.9 * expected:
        status_banner(f"{ref.label} has {n_days}/{expected} forecast days — the run may still be loading.", "warning")
    if ref.prev_init is None:
        status_banner("No earlier run of the reference model in the table — its Δ run shows n/a.", "warning")
    elif not ref.prev_exact:
        status_banner(f"{ref.prev_wanted:%a %d %b %H}z is not in the table — the reference's Δ run is taken "
                      f"against {ref.prev_label} instead.", "warning")

    cells = compute_grid(cols, win)

    if phone:
        # --- phone: the reference run's issue up front, one window at a time ---
        st.markdown(f"<div class='mc-sub'><b>{ref.model}</b> {ref.init:%a %d %b} · {delta_label} vs {ref.prev_label}"
                    f"{'' if ref.prev_exact else ' (nearest earlier run)'}</div>", unsafe_allow_html=True)
        wkey = st.radio("Window", ["w1", "w2"], horizontal=True, key="mcp_win", label_visibility="collapsed",
                        format_func=lambda k: _window_title(win, k))
        st.markdown(_glance_html(cells, ref, win, delta_label, (wkey,), len(cols)), unsafe_allow_html=True)
        if show_chart:
            st.plotly_chart(make_morning_anomaly_chart(cells, MORNING_BLOCKS, [(wkey, _window_title(win, wkey))],
                                                       col_labels, compact=True),
                            use_container_width=True, config={"displayModeBar": False})
        for name, block in MORNING_BLOCKS.items():
            st.markdown(_grid_block_html_phone(cells[cells["block"] == name], name, block, cols, win, wkey, delta_label),
                        unsafe_allow_html=True)
        st.caption(f"{source} · shaded cells: red = warmer / less wind, solar, precipitation, blue = colder / more · "
                   f"Δ norm = vs the {source} normal · {len(runs)} runs in the table, latest "
                   f"{runs[0][1]:%a %d %b %H}z ({MORNING_MODEL_LABELS.get(runs[0][0], runs[0][0])}).")
    else:
        span1 = "weekend" if win["kind1"] == "weekend" else "weekly"
        st.markdown(f'<div class="mc-sub">Windows follow the reference run\'s init day ({today:%A %d %b}): '
                    f'<b>{win["t1"]}</b> {win["w1"][0]:%d %b} – {win["w1"][1]:%d %b} ({span1} average) · '
                    f'<b>{win["t2"]}</b> {win["w2"][0]:%d %b} – {win["w2"][1]:%d %b} (weekly average) · '
                    f'precipitation = sum of the first {MORNING_PRECIP_SUM_DAYS} forecast days · '
                    f'{delta_label} = change vs each model\'s own earlier run · superscript = days of the window '
                    f'the run covers, when not all · hover a cell or a chip for every number.</div>', unsafe_allow_html=True)
        st.markdown(_glance_html(cells, ref, win, delta_label, ("w1", "w2"), len(cols)), unsafe_allow_html=True)
        if show_chart:
            st.plotly_chart(make_morning_anomaly_chart(cells, MORNING_BLOCKS,
                                                       [("w1", _window_title(win, "w1")), ("w2", _window_title(win, "w2"))],
                                                       col_labels),
                            use_container_width=True, config={"displayModeBar": False})
        for name, block in MORNING_BLOCKS.items():
            st.markdown(_grid_block_html(cells[cells["block"] == name], name, block, cols, win, view_key, delta_label),
                        unsafe_allow_html=True)
        st.markdown(_grid_footer(cols, delta_label, view_key), unsafe_allow_html=True)
    st.download_button("Download grid (CSV)", _grid_csv(cells, delta_label),
                       f"morning_call_{source.lower()}_{today:%Y%m%d}.csv", "text/csv", key="mcg_dl",
                       help="Every cell of every column: value, Δ run, Δ norm, Δ vs reference, spread.")

    # --- commentary on the reference run (its latest issue by default) ---
    st.divider()
    ctx = build_brief_context(ref.rows, ref.prev_rows, win, ref.model, ref.init, ref.prev_init, delta_label, today)
    gas_runs = tuple((c.pattern, c.init.isoformat()) for c in cols)
    _render_ai_brief(ctx, today, phone, source, gas_runs)
