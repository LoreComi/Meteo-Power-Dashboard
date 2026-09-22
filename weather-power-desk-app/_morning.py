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

The Excel's free-text boxes — Pattern and Comment per window, and the Week 3
commentary block — are gone. In their place two agent families read these same
numbers and write the commentary: one on the power market, one on gas (see
_ai_brief.py). Week 3 is still reached: it is handed to the agents whenever the
selected run has at least three days in it, which is normal for EC-Extended and
rare for EC-ENS.

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
    AI_BRIEF_MODEL, AI_BRIEF_MAX_TOKENS, AI_BRIEF_SYNTHESIS_MAX_TOKENS, AI_POWER_AGENTS, AI_GAS_AGENTS,
    SCENARIO_MIN_WEEK_DAYS, MORNING_FAMILIES, MORNING_DEFAULT_FAMILY,
)
from _data import (load_morning_daily, pick_run as _pick_run, list_available_runs, run_completeness,
                   format_run, list_family_runs, format_family_run, average_multi_runs)
from _charts import make_model_compare_chart
from _style import CATEGORICAL, PROVIDER_COLORS, BRIEF_CSS, INK_MUTED, CAT_BLUE, STATUS_CRITICAL
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
    # Week 3 is the report's qualitative block: the week after w2, usually past
    # the EC-ENS 15-day horizon. It is not tabulated, but it is handed to the
    # agent families whenever the chosen run actually reaches into it.
    w3s, w3e = w2e + pd.Timedelta(days=1), w2e + pd.Timedelta(days=7)
    return {"w1": (w1s, w1e), "w2": (w2s, w2e), "w3": (w3s, w3e), "t1": t1, "t2": t2, "t3": t3,
            "delta_label": delta_label, "prev_days": prev_days, "kind1": kind1}


# ─── run selection ──────────────────────────────────────────────────────────────

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
    """One weekly window: the report's four tables. The free-text Pattern and
    Comment boxes the Excel sheet had are gone — the agent families below the
    tables write that commentary from these same numbers."""
    st.markdown(f'<div class="mc-week">{title}</div><div class="mc-sub">{sub}</div>', unsafe_allow_html=True)
    for name, block in MORNING_BLOCKS.items():
        if block["agg"] == "sum" and not include_precip:
            continue
        rows = block_values(cur, prev, block, None if block["agg"] == "sum" else window)
        st.markdown(_block_html(name, block["unit"], rows, delta_label, block["fmt"]), unsafe_allow_html=True)


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
    for col, html in zip(cols, cards):
        with col:
            st.markdown(html, unsafe_allow_html=True)

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
    """Two families of agents commenting on the table above — power and gas."""
    from _ai_brief import FAMILIES, get_az_credentials, generate_family_brief

    st.markdown("##### Market read — agent families")
    st.caption("Two families comment on the table above and nothing else: one on the power market "
               "(temperature → load, wind & solar → residual load, precipitation → hydro), one on gas "
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
    st.caption("The Morning Report table from the **sandbox** (morning_daily) — EC-ENS 00z 'Avg' "
               "curves, daily CET means, Volue 30-year normal. Same rows, windows and deltas as "
               "import_00z_add_solar_np_tot.py. Pick a model family and runs below.")
    try:
        df = load_morning_daily()
    except Exception as e:
        st.error(f"Cannot read from the sandbox tables (morning_daily): {e}")
        return
    if df.empty:
        status_banner("No data found in morning_daily.", "warning")
        return
    df = _drop_partial_days(df)

    # --- Model family and run selection -----------------------------------------
    c1, c2, c3, c4 = st.columns([1.4, 2.0, 2.2, 2.4])
    with c1:
        families = st.multiselect("Model family", list(MORNING_FAMILIES.keys()),
                                  default=[MORNING_DEFAULT_FAMILY], key="mc_families")
    if not families:
        status_banner("Select at least one model family.", "warning")
        return
    family_patterns = [p for f in families for p in MORNING_FAMILIES[f]]
    avail = list_family_runs(df, family_patterns)
    if not avail:
        status_banner("No runs available for the selected families.", "critical")
        return
    run_map = {format_family_run(dt, pat): (dt, pat) for dt, pat in avail}
    run_keys = list(run_map.keys())

    with c2:
        cur_sel = st.multiselect("Current run(s)", run_keys, default=[run_keys[0]], key="mc_cur_runs",
                                 help="Select one or more runs.")
    if not cur_sel:
        status_banner("Select at least one current run.", "warning")
        return
    cur_runs = [run_map[l] for l in cur_sel]
    cur_init = cur_runs[0][0]
    # Default compare: first run NOT in the current selection
    cur_set = {(dt.normalize(), pat) for dt, pat in cur_runs}
    cmp_defaults = [k for k in run_keys if (run_map[k][0].normalize(), run_map[k][1]) not in cur_set]
    with c3:
        mode = st.radio("Multi-run mode", ["Average", "Compare"], key="mc_mode", horizontal=True,
                        help="Average: merge selected runs into one table. Compare: one tab per run.")
        cmp_sel = st.multiselect("Compare with", run_keys, default=cmp_defaults[:1], key="mc_cmp_runs",
                                 help="Δ columns show (current − comparison). Multiple are averaged.")
    cmp_runs = [run_map[l] for l in cmp_sel]
    cmp_init = cmp_runs[0][0] if cmp_runs else None

    # Completeness banners
    for _dt, _pat in cur_runs:
        comp = run_completeness(df, _pat, _dt)
        if not comp["complete"]:
            status_banner(f"⚡ {format_family_run(_dt, _pat)} has {comp['n_days']}/{comp['expected']} "
                          f"forecast days ({comp['pct']:.0f}%).", "warning")
    for _dt, _pat in cmp_runs:
        comp = run_completeness(df, _pat, _dt)
        if not comp["complete"]:
            status_banner(f"⚡ Compare: {format_family_run(_dt, _pat)} — {comp['n_days']}/{comp['expected']} days.",
                          "warning")

    cmp_label = " + ".join(format_family_run(dt, pat) for dt, pat in cmp_runs) if cmp_runs else "none"
    prev = average_multi_runs(df, cmp_runs) if cmp_runs else pd.DataFrame()
    prev_init = cmp_init

    if mode == "Compare" and len(cur_runs) > 1:
        # --- COMPARE MODE: all runs stacked, visible at once ---
        with c4:
            cur_label = " vs ".join(format_family_run(dt, pat) for dt, pat in cur_runs)
            sub = f"Comparing {cur_label}"
            if cmp_runs:
                sub += f" · Δ vs {cmp_label}"
            st.markdown(f'<div class="mc-sub" style="margin-top:28px;">{sub}</div>', unsafe_allow_html=True)

        if prev.empty:
            status_banner("No comparison run selected — Δ vs previous run shows n/a.", "warning")

        for i, (init_dt, pattern) in enumerate(cur_runs):
            run_label = format_family_run(init_dt, pattern)
            st.markdown(f"##### {run_label}")
            single = average_multi_runs(df, [(init_dt, pattern)])
            today_i = init_dt.normalize()
            win_i = morning_windows(today_i)
            delta_label_i = f"Δ vs {cmp_label}" if cmp_runs else win_i["delta_label"]
            left, right = st.columns(2)
            with left:
                _render_block(win_i["t1"], win_i["w1"], single, prev, delta_label_i,
                              include_precip=True, key=f"b1_{i}",
                              sub=f"{win_i['w1'][0]:%a %d %b} – {win_i['w1'][1]:%a %d %b} · "
                                  f"{'weekend' if win_i['kind1'] == 'weekend' else 'weekly'} average · "
                                  f"precip = sum of coming 2 weeks")
            with right:
                _render_block(win_i["t2"], win_i["w2"], single, prev, delta_label_i,
                              include_precip=False, key=f"b2_{i}",
                              sub=f"{win_i['w2'][0]:%a %d %b} – {win_i['w2'][1]:%a %d %b} · weekly average")
            if i < len(cur_runs) - 1:
                st.divider()

        # AI brief and download use the first run
        today = cur_init.normalize()
        win = morning_windows(today)
        cur = average_multi_runs(df, [cur_runs[0]])
        model = families[0]
        pattern = family_patterns[0]
        delta_label = f"Δ vs {cmp_label}" if cmp_runs else win["delta_label"]
        st.download_button("Download table (CSV)", _download_table(cur, prev, win, delta_label),
                           f"morning_call_{today:%Y%m%d}.csv", "text/csv", key="mc_dl")

    else:
        # --- AVERAGE MODE (or single run): merge all selected runs ---
        today = cur_init.normalize()
        win = morning_windows(today)
        cur = average_multi_runs(df, cur_runs)
        model = families[0]
        pattern = family_patterns[0]
        delta_label = f"Δ vs {cmp_label}" if cmp_runs else win["delta_label"]

        with c4:
            cur_label = " + ".join(format_family_run(dt, pat) for dt, pat in cur_runs)
            sub = f"{'Ensemble mean' if len(cur_runs) > 1 else families[0]} · {cur_label}"
            if cmp_runs:
                sub += f" vs {cmp_label}"
            st.markdown(f'<div class="mc-sub" style="margin-top:28px;">{sub}</div>', unsafe_allow_html=True)

        if prev.empty:
            status_banner("No comparison run selected — Δ vs previous run shows n/a.", "warning")

        left, right = st.columns(2)
        with left:
            _render_block(win["t1"], win["w1"], cur, prev, delta_label, include_precip=True, key="b1",
                          sub=f"{win['w1'][0]:%a %d %b} – {win['w1'][1]:%a %d %b} · "
                              f"{'weekend' if win['kind1'] == 'weekend' else 'weekly'} average · precip = sum of coming 2 weeks")
        with right:
            _render_block(win["t2"], win["w2"], cur, prev, delta_label, include_precip=False, key="b2",
                          sub=f"{win['w2'][0]:%a %d %b} – {win['w2'][1]:%a %d %b} · weekly average")

        st.download_button("Download table (CSV)", _download_table(cur, prev, win, delta_label),
                           f"morning_call_{today:%Y%m%d}.csv", "text/csv", key="mc_dl")

    st.divider()
    ctx = build_brief_context(cur, prev, win, model, cur_init, prev_init, delta_label, today)
    _render_ai_brief(ctx, today)

    st.divider()
    _render_model_comparison(df, win, today)
