"""Morning Call commentary — two tailored agent families.

Replaces the Morning Report's free-text Pattern / Comment boxes. Each family
reads the numbers already on the Morning Call screen — nothing else, no web,
no news — and writes a few sentences on what this run means for its own
market.

  Power family   temperature -> load · wind & solar -> residual load and the
                 merit order · precipitation -> hydro inflows. Then a
                 synthesis across the three.
  Gas family     temperature -> LDZ heating demand · wind & solar -> the
                 gas-fired generation they displace. Then a synthesis. When
                 the Gas Demand section's figures are available they are handed
                 to this family as quantified evidence, so the commentary and
                 the LDZ / displaced-gas deltas cannot drift apart.

Structure follows Coal_Dashboard/weather-coal-desk-app/_ai_coal_brief.py:
Python builds compact context documents, one independent LLM call per
specialist, one synthesis call per family that only sees the specialists'
findings. Each agent ends on a single SIGNAL line, which is stripped from the
prose and rendered as a card.

The columns every agent is given, and how they are told to weigh them:
  value   the window average (precipitation: the 2-week sum)
  Δrun    the change against the run being compared — what re-prices today
  Δnorm   the deviation from the 30-year normal — where the level sits
"""
from __future__ import annotations

import os
import re
from datetime import datetime

import numpy as np

# ── Shared framing ────────────────────────────────────────────────────────────

_COLUMNS_NOTE = """\
Columns per region: `value` = the window average (precipitation = the sum of
the coming two weeks), `Δrun` = the change against the run this one is compared
with, `Δnorm` = the deviation from the 30-year normal.

How to weigh them: Δrun is what re-prices the curve today — it is the new
information. Δnorm tells you where the level sits and therefore how much a
further move would matter. Lead with Δrun, frame it with Δnorm.
"""

_STYLE_NOTE = """\
Write 2-3 sentences of flowing prose. No bullets, no headers, no preamble, no
restating the question. Quote the specific regions and magnitudes that carry
your argument and name the units. Say plainly when the move is too small to
trade — a quiet run is a legitimate finding, do not manufacture a signal.
"""

_SIGNAL_NOTE = """\
Last line only, exactly one of:
SIGNAL: BULLISH
SIGNAL: BEARISH
SIGNAL: NEUTRAL
"""


# ── Power family system prompts ───────────────────────────────────────────────

_SYS_PW_TEMP = f"""\
You are a European power analyst reading the morning weather table for the
prompt and front weekly contracts.

Temperature drives load, and the sign of that depends on the season:
  Winter / shoulder: colder = more heating load = higher demand = bullish
    power. France is the most temperature-sensitive market in Europe because
    of resistive electric heating (roughly 2 GW per degree below normal);
    Germany and the UK respond more through gas- and heat-pump heating.
  Summer: hotter = more air-conditioning load = bullish, strongest in Italy
    and Iberia, and hot-and-still weather also derates thermal and nuclear
    plant (river cooling limits in France) which tightens supply further.
  Nordic: temperature moves both consumption and snowmelt timing, so it hits
    the system price through demand and hydro availability at once.

{_COLUMNS_NOTE}
Assess this run's temperature change: which regions moved, whether it adds or
removes load given the season, and whether the level versus normal makes that
move matter. {_STYLE_NOTE}
{_SIGNAL_NOTE}
(BULLISH = this temperature change tightens the power balance and supports
prices; BEARISH = it loosens it; NEUTRAL = too small or offsetting.)
"""

_SYS_PW_WIND_SOLAR = f"""\
You are a European power analyst covering renewable supply and the merit order.

Wind and solar are price-setting because they enter at zero marginal cost and
push the residual load down the stack:
  More wind displaces gas and coal at the margin, so the clean spark and dark
    spreads and the prompt fall. Germany and the UK carry the largest wind
    fleets, so a GW there moves more than a GW elsewhere; heavy German wind
    also spills into the French, Dutch and Nordic prices through the
    interconnectors.
  More solar compresses the midday block specifically, deepening the duck
    curve, lifting the evening peak-to-baseload spread and, in Germany, Iberia
    and Italy at high load factors, raising the risk of negative midday prices.
  Wind and solar arriving together on a low-demand weekend is the classic
    negative-price setup; the two arriving against high load is simply
    bearish baseload.

Values are in GW of average generation over the window.
{_COLUMNS_NOTE}
Assess this run's renewable change: which regions and which technology moved,
what that does to residual load and the merit order, and which part of the
curve — baseload, peak or the midday block — takes it. {_STYLE_NOTE}
{_SIGNAL_NOTE}
(BULLISH = less renewable supply, higher residual load, supports power prices;
BEARISH = more renewable supply pushing prices down; NEUTRAL = marginal.)
"""

_SYS_PW_PRECIP = f"""\
You are a European power analyst covering hydro.

The figures are precipitation energy — the electricity the forecast rainfall
and snow is worth to the hydro fleet — summed over the coming two weeks, in
TWh, per catchment region:
  Alps (Swiss, Austrian and north-Italian reservoirs) feeds storage hydro that
    sets the Italian and Swiss peak and the flexibility premium across CWE.
  Nordic feeds the Nordic reservoir system, so it moves the system price and
    the NO/SE area spreads, and with a lag also the CWE import balance.
  SEE and Iberia are more run-of-river and reservoir mixed; Iberian hydro
    competes directly with gas-fired generation in the Spanish stack.
Above-normal precipitation means more inflow, fuller reservoirs and cheaper
hydro, so it is bearish — and it is a persistent bearishness, because water
stays in the reservoir. Below-normal precipitation tightens the balance over
weeks, not days, and shows up first as a wider forward spread than as a prompt
move. Season matters: in winter, Nordic precipitation falling as snow is stored
for the spring melt rather than available now.

{_COLUMNS_NOTE}
Assess this run's precipitation change by catchment, what it does to hydro
availability and over what horizon it bites. {_STYLE_NOTE}
{_SIGNAL_NOTE}
(BULLISH = drier, less hydro, tighter balance; BEARISH = wetter, more hydro;
NEUTRAL = near normal or offsetting across catchments.)
"""

_SYS_PW_SYNTHESIS = f"""\
You are the head power trader writing the weather part of the morning call.
Three specialists have reported: temperature and load, wind and solar and the
merit order, precipitation and hydro.

Write 3-4 sentences. Lead with whichever driver is actually moving the market
this morning and say why it dominates the others. Say explicitly where the
three reinforce or offset each other — a cold run with heavy wind is not the
same trade as a cold run with still weather. Name the part of the curve the
view applies to (prompt, front week, front month) and close with the one thing
that would flip it. No bullets, no headers, no preamble.

{_SIGNAL_NOTE}
(Net direction for European power prices over the next two weeks.)
"""


# ── Gas family system prompts ─────────────────────────────────────────────────

_SYS_GAS_TEMP = f"""\
You are a European gas analyst covering TTF and the front of the curve.

Temperature drives the largest single swing in European gas demand, through
local distribution zone (LDZ) heating load:
  Colder than normal = more residential and commercial heating = higher LDZ
    demand = bullish TTF, and in winter it also draws harder on storage, which
    is what turns a weather move into a curve move.
  The response is non-linear: below roughly 11-15°C, depending on the country,
    each further degree of cooling adds materially more demand than the degree
    before it, and above about 17°C the heating response is exhausted so a warm
    run barely matters.
  Buildings have thermal inertia, so a single mild day inside a cold spell does
    not erase the load — what counts is the multi-day path.
  Germany, the UK, France, Italy and the Netherlands carry nearly all of the
    NW European LDZ demand.

{_COLUMNS_NOTE}
If a fitted LDZ block is supplied below, it is this desk's own
temperature-response model applied to exactly these runs — it converts the
temperature change into GWh of gas demand. Treat it as the quantitative answer
and use the table for the regional colour behind it.

Assess this run's temperature change for heating gas demand: which countries,
how far into the non-linear heating range the level sits, and whether the
change is big enough to price. {_STYLE_NOTE}
{_SIGNAL_NOTE}
(BULLISH = more heating gas demand; BEARISH = less; NEUTRAL = marginal.)
"""

_SYS_GAS_WIND_SOLAR = f"""\
You are a European gas analyst covering gas-for-power demand.

Wind and solar compete directly with gas-fired generation: every GW of
renewable output displaces the gas plant at the margin, so renewables are a
demand-side driver of TTF, not a supply-side one.
  More wind and solar = less gas burn = bearish TTF. The effect is largest in
    Germany, the UK, the Netherlands and Iberia, where gas sits at the margin
    most hours.
  Roughly, a GW of renewable generation held for a day displaces about 40 GWh
    of gas at CCGT efficiency, though only part of that swing clears against
    gas rather than against coal, hydro or exports.
  Less wind and solar is the mirror image: gas plant is called back, gas-for-
    power demand rises, bullish TTF — and in a tight market a still, dark
    fortnight in NW Europe is a bigger TTF story than a mild one.

Values are in GW of average generation over the window.
{_COLUMNS_NOTE}
If a displaced-gas block is supplied below, it is this desk's own conversion of
exactly these runs into GWh of gas displaced. Treat it as the quantitative
answer and use the table for the regional colour behind it. Note its sign
convention: a positive delta means MORE gas displaced, which is BEARISH gas.

Assess this run's renewable change for gas-for-power demand: which regions
moved, how much burn that adds or removes, and whether it offsets or compounds
the heating story. {_STYLE_NOTE}
{_SIGNAL_NOTE}
(BULLISH = less renewable output, more gas burn; BEARISH = more renewable
output displacing gas; NEUTRAL = marginal.)
"""

_SYS_GAS_SYNTHESIS = f"""\
You are the head gas trader writing the weather part of the morning call. Two
specialists have reported: temperature and LDZ heating demand, and wind and
solar and gas-for-power demand.

Write 3-4 sentences. Net the two demand legs against each other explicitly —
they routinely point opposite ways, and a cold but windy run can be a smaller
TTF story than either leg alone suggests. Where the desk's own LDZ and
displaced-gas figures were supplied, quote the net in GWh. Say which part of
the curve the view applies to and close with the one thing that would flip it.
No bullets, no headers, no preamble.

{_SIGNAL_NOTE}
(Net direction for TTF over the next two weeks.)
"""


# ── Agent families ────────────────────────────────────────────────────────────
# key -> (label, system prompt, which Morning Call blocks it is shown)
POWER_FAMILY: dict[str, tuple[str, str, list[str]]] = {
    "pw_temp":       ("Temperature → load",              _SYS_PW_TEMP,       ["Temperatures"]),
    "pw_wind_solar": ("Wind & solar → residual load",    _SYS_PW_WIND_SOLAR, ["Wind", "Solar PV"]),
    "pw_precip":     ("Precipitation → hydro",           _SYS_PW_PRECIP,     ["Precip (sum of coming 2 weeks)"]),
}
GAS_FAMILY: dict[str, tuple[str, str, list[str]]] = {
    "gas_temp":       ("Temperature → LDZ demand",       _SYS_GAS_TEMP,       ["Temperatures"]),
    "gas_wind_solar": ("Wind & solar → gas-for-power",   _SYS_GAS_WIND_SOLAR, ["Wind", "Solar PV"]),
}
FAMILIES = {
    "power": {"label": "Power market", "agents": POWER_FAMILY, "synthesis": _SYS_PW_SYNTHESIS,
              "commodity": "European power"},
    "gas":   {"label": "Gas market",   "agents": GAS_FAMILY,   "synthesis": _SYS_GAS_SYNTHESIS,
              "commodity": "TTF gas"},
}


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT DOCUMENTS (pure Python, no LLM)
# ══════════════════════════════════════════════════════════════════════════════

_MONTH_SEASON = {
    12: "mid-winter, peak heating season", 1: "mid-winter, peak heating season",
    2: "late winter, heating season", 3: "early spring shoulder", 4: "spring shoulder",
    5: "late spring shoulder", 6: "early summer, cooling season starting",
    7: "mid-summer, peak cooling season", 8: "mid-summer, peak cooling season",
    9: "early autumn shoulder, storage injection ending", 10: "autumn shoulder, heating season starting",
    11: "early winter, heating season",
}


def _fmt_num(v, fmt: str = "{:.1f}") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return fmt.format(v)


def _fmt_signed(v, fmt: str = "{:+.1f}") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return fmt.format(v)


def _block_table(name: str, unit: str, rows: list[dict]) -> str:
    """One Morning Call block as fixed-width text."""
    lines = [f"  {name} [{unit}]",
             f"    {'Region':<12}{'value':>9}{'Δrun':>9}{'Δnorm':>9}"]
    for r in rows:
        lines.append(f"    {r['region']:<12}{_fmt_num(r.get('value')):>9}"
                     f"{_fmt_signed(r.get('d_run')):>9}{_fmt_signed(r.get('d_norm')):>9}")
    return "\n".join(lines)


def format_morning_doc(ctx: dict, block_names: list[str]) -> str:
    """Context document for one specialist: only the blocks it is responsible for.

    `ctx` is what _morning.build_brief_context() produces:
      run, run_init, prev_init, delta_label, today,
      windows = [{title, range, blocks: {name: {unit, rows}}}]
    """
    today = ctx.get("today")
    month = today.month if today is not None else datetime.utcnow().month
    header = [
        f"Morning Call — {ctx.get('run', 'EC-ENS 00z')}, run initialised {ctx.get('run_init', 'n/a')}, "
        f"compared with the run of {ctx.get('prev_init', 'n/a')} ({ctx.get('delta_label', 'Δ')}).",
        f"Date: {today:%d %b %Y}" if today is not None else "",
        f"Season: {_MONTH_SEASON.get(month, '')}.",
        "",
    ]
    body = []
    for win in ctx.get("windows", []):
        blocks = {k: v for k, v in win.get("blocks", {}).items() if k in block_names}
        if not blocks:
            continue
        present = [b for b in blocks.values() if any(
            not (r.get("value") is None or (isinstance(r.get("value"), float) and np.isnan(r["value"])))
            for r in b["rows"])]
        if not present:
            continue
        body.append(f"{win['title']} ({win['range']})")
        for bname, b in blocks.items():
            body.append(_block_table(bname, b["unit"], b["rows"]))
        body.append("")
    if not body:
        return "\n".join(h for h in header if h) + "\nNo data available for this block in any window."
    return "\n".join(h for h in header if h) + "\n" + "\n".join(body)


def format_gas_model_doc(snapshot: dict, leg: str) -> str:
    """The Gas Demand section's own figures for one leg, as evidence for the gas family.

    `snapshot` is _gas.gas_demand_snapshot(); `leg` is 'ldz' or 'rdl'.
    Returns "" when the section has no figures, in which case the agent works
    from the Morning Call table alone.
    """
    if not snapshot:
        return ""
    title = ("DESK LDZ MODEL — fitted temperature-response curves applied to these runs"
             if leg == "ldz" else
             "DESK DISPLACED-GAS MODEL — wind + solar converted to gas-fired generation displaced")
    sign = ("Positive delta = more heating gas demand = BULLISH gas."
            if leg == "ldz" else
            "Positive delta = more gas displaced = BEARISH gas.")
    lines = [f"=== {title} ===", sign, ""]
    any_rows = False
    for run_label, legs in snapshot.items():
        res = legs.get(leg)
        if not res:
            continue
        rows = [r for r in res.get("rows", []) if not (r.get("delta") is None or np.isnan(r["delta"]))]
        if not rows:
            continue
        any_rows = True
        lines.append(f"{run_label} — run {res.get('cur_init')} vs {res.get('prev_init')}, "
                     f"window {res['window'][0]} to {res['window'][1]}")
        lines.append(f"    {'Region':<12}{'Δ GWh':>10}   {'signal':<10}")
        for r in rows:
            lines.append(f"    {r['area']:<12}{r['delta']:>+10,.0f}   {r['read']:<10}")
        lines.append(f"    {'TOTAL':<12}{res['total_delta']:>+10,.0f}   "
                     f"{res['total_read']} ({res['total_trade']})")
        lines.append("")
    return "\n".join(lines) if any_rows else ""


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def prose_to_html(text: str) -> str:
    """Agent prose into the styled box: markdown bold kept, paragraphs preserved,
    and any stray bullet the model emitted flattened back into a sentence."""
    if not text:
        return ""
    out = []
    for para in re.split(r"\n\s*\n", text.strip()):
        lines = [re.sub(r"^[-*•]\s+", "", ln.strip()) for ln in para.split("\n") if ln.strip()]
        joined = " ".join(lines)
        joined = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", joined)
        out.append(f"<p style='margin:0 0 8px;'>{joined}</p>")
    return "".join(out)


def extract_signal(text: str) -> tuple[str, str]:
    """Strip the SIGNAL line(s) from an agent's output; return (signal, prose)."""
    signal = "NEUTRAL"
    clean = []
    for line in text.strip().split("\n"):
        m = re.match(r"^\**SIGNAL:\**\s*\**(BULLISH|BEARISH|NEUTRAL)\**\s*$", line.strip(), re.IGNORECASE)
        if m:
            signal = m.group(1).upper()
        else:
            clean.append(line)
    while clean and not clean[-1].strip():
        clean.pop()
    return signal, "\n".join(clean).strip()


# ══════════════════════════════════════════════════════════════════════════════
# AZURE OPENAI
# ══════════════════════════════════════════════════════════════════════════════

def get_az_credentials() -> tuple[str | None, str | None, str | None]:
    """Databricks secrets (scope 'axpo') -> Databricks SDK -> st.secrets -> env vars.

    Same order as the coal dashboard, so a workspace configured for one app
    needs no extra setup for this one.
    """
    try:
        import streamlit as st
        t = st.secrets["azure_tenant_id"]
        c = st.secrets["azure_client_id"]
        s = st.secrets["azure_client_secret"]
        if t and c and s:
            return t, c, s
    except Exception:
        pass
    try:
        from databricks.sdk.runtime import dbutils
        t = dbutils.secrets.get("axpo", "azure_tenant_id")
        c = dbutils.secrets.get("axpo", "azure_client_id")
        s = dbutils.secrets.get("axpo", "azure_client_secret")
        if t and c and s:
            return t, c, s
    except Exception:
        pass
    t = os.environ.get("AZURE_TENANT_ID")
    c = os.environ.get("AZURE_CLIENT_ID")
    s = os.environ.get("AZURE_CLIENT_SECRET")
    if t and c and s:
        return t, c, s
    return None, None, None


def build_client(tenant_id: str, client_id: str, client_secret: str):
    from azure.identity import ClientSecretCredential
    from openai import AzureOpenAI

    cred = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)
    return AzureOpenAI(
        azure_endpoint="https://azure-oai-prod.openai.azure.com/",
        api_version="2024-12-01-preview",
        azure_ad_token_provider=lambda: cred.get_token(
            "https://cognitiveservices.azure.com/.default").token,
    )


def _llm(client, system: str, user_doc: str, model: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user_doc}],
        temperature=0.3,
        max_completion_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_family_brief(family: str, ctx: dict, azure_tenant_id: str, azure_client_id: str,
                          azure_client_secret: str, gas_snapshot: dict | None = None,
                          model: str = "gpt-4o", max_tokens: int = 320,
                          synthesis_max_tokens: int = 420, progress_cb=None) -> dict:
    """Run one family: its specialists, then its synthesis.

    Parameters
    ----------
    family        'power' or 'gas'
    ctx           _morning.build_brief_context() output — the on-screen numbers
    gas_snapshot  _gas.gas_demand_snapshot() output; only the gas family uses it
    progress_cb   optional callable(str) for Streamlit progress messages

    Returns
    -------
    dict with one entry per agent key plus 'synthesis', a 'signals' map, the
    context documents that were sent ('docs', for the audit expander), the
    family label and generated_at.
    """
    cfg = FAMILIES[family]

    def _prog(msg: str):
        if progress_cb:
            progress_cb(msg)

    client = build_client(azure_tenant_id, azure_client_id, azure_client_secret)

    briefs, signals, docs = {}, {}, {}
    for key, (label, system, blocks) in cfg["agents"].items():
        _prog(f"{cfg['label']} · {label}…")
        doc = format_morning_doc(ctx, blocks)
        if family == "gas" and gas_snapshot:
            leg = "ldz" if key == "gas_temp" else "rdl"
            extra = format_gas_model_doc(gas_snapshot, leg)
            if extra:
                doc = f"{doc}\n\n{extra}"
        docs[key] = doc
        signals[key], briefs[key] = extract_signal(_llm(client, system, doc, model, max_tokens))

    _prog(f"{cfg['label']} · synthesis…")
    synthesis_input = "\n\n".join(
        f"=== {cfg['agents'][k][0].upper()} (signal: {signals[k]}) ===\n{briefs[k]}"
        for k in cfg["agents"]
    )
    if family == "gas" and gas_snapshot:
        net = "\n\n".join(filter(None, (format_gas_model_doc(gas_snapshot, "ldz"),
                                        format_gas_model_doc(gas_snapshot, "rdl"))))
        if net:
            synthesis_input = f"{synthesis_input}\n\n{net}"
    docs["synthesis"] = synthesis_input
    signals["synthesis"], briefs["synthesis"] = extract_signal(
        _llm(client, cfg["synthesis"], synthesis_input, model, synthesis_max_tokens))

    return {"family": family, "label": cfg["label"], "commodity": cfg["commodity"],
            "briefs": briefs, "signals": signals, "docs": docs,
            "generated_at": datetime.utcnow().strftime("%d %b %Y %H:%M UTC")}
