"""Power Desk Weather Dashboard — entry point (Databricks App, Streamlit).

Landing page with the Morning Call tile plus a grid of section tiles; the
sidebar mirrors them and adds a Home button. Sections:
  0. Morning Call           — the Morning Report table, live, with two agent families commenting,
                              and a country deep dive on the same runs (forecast · hydro · scenarios)
  1. Forecast               — Volue ensemble values, spread vs normal spread, member scenarios
  2. Historical & Analysis  — monthly / weekly history by country, anomalies, index analogues
  3. Hydro Monitoring       — Hydro Report quantify_* figures and stats, live
  4. Gas Demand             — LDZ heating demand and wind/solar as displaced gas, run over run
  5. Strategy               — locked, work in progress

Structure follows the LPG desk dashboard (weather-lpg-desk-app): _config /
_data / _charts / _style modules, one render function per section, sandbox
tables refreshed by a Databricks job (power_desk_refresh.py).
"""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from _config import SECTIONS
from _style import CUSTOM_CSS
from _morning import render_morning_call
from _forecast import render_forecast
from _historical import render_historical
from _hydro import render_hydro
from _gas import render_gas_demand
from _strategy import render_strategy

st.set_page_config(page_title="Power Desk — Weather Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

HOME = "Home"
RENDERERS = {
    "Morning Call": render_morning_call,
    "Forecast": render_forecast,
    "Historical & Analysis": render_historical,
    "Hydro Monitoring": render_hydro,
    "Gas Demand": render_gas_demand,
    "Strategy": render_strategy,
}

if "section" not in st.session_state:
    st.session_state["section"] = HOME

# PC or phone: asked once when the app is opened (?layout=phone / ?layout=pc skips the question).
DEVICES = ("PC", "Phone")
if "device" not in st.session_state:
    try:
        _qp = str(st.query_params.get("layout", "")).lower()
    except Exception:
        _qp = ""
    if _qp in ("pc", "phone"):
        st.session_state["device"] = "Phone" if _qp == "phone" else "PC"


def go(section: str) -> None:
    st.session_state["section"] = section
    st.session_state["nav_radio"] = section


# ─── Header ──────────────────────────────────────────────────────────────────────
_now = datetime.now(timezone.utc)
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
    <div>
        <span style="font-size:1.7rem; font-weight:800; letter-spacing:-0.03em; color:var(--text-primary);">Power Desk</span>
        <span style="font-size:1.7rem; font-weight:300; letter-spacing:-0.03em; color:var(--text-muted); margin-left:8px;">Weather Intelligence</span>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="display:inline-flex; align-items:center; gap:6px; padding:4px 12px; background:#e9f7e9;
            border:1px solid rgba(12,163,12,0.28); border-radius:999px; font-size:0.72rem; font-weight:600; color:#0f5c0f;">
            <span style="width:6px;height:6px;background:#0ca30c;border-radius:50%;display:inline-block;"></span> Live
        </span>
        <span style="font-size:0.72rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace;">
            {_now:%d %b %Y} &middot; {_now:%H:%M} UTC
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── First open: PC or phone ─────────────────────────────────────────────────────
def _set_device(device: str) -> None:
    st.session_state["device"] = device


if "device" not in st.session_state:
    st.markdown("#### HOW ARE YOU VIEWING THE DASHBOARD?")
    st.caption("Phone shows the Morning Call one window at a time, with one compare column and big type. "
               "You can switch later in the sidebar. Open the app with ?layout=phone to skip this question.")
    _c1, _c2, _ = st.columns([1, 1, 3])
    _c1.button("💻  PC", use_container_width=True, on_click=_set_device, args=("PC",), key="device_pc")
    _c2.button("📱  Phone", use_container_width=True, on_click=_set_device, args=("Phone",), key="device_phone")
    st.stop()


# ─── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Sections")
    options = [HOME] + list(SECTIONS.keys())
    if "nav_radio" not in st.session_state:
        st.session_state["nav_radio"] = st.session_state["section"]
    picked = st.radio("Sections", options, key="nav_radio", label_visibility="collapsed",
                      format_func=lambda s: s if s == HOME or not SECTIONS[s]["locked"] else f"{s}  🔒")
    if picked != st.session_state["section"]:
        st.session_state["section"] = picked
    st.divider()
    st.radio("Layout", DEVICES, horizontal=True, index=DEVICES.index(st.session_state["device"]),
             key="device_radio", on_change=lambda: _set_device(st.session_state["device_radio"]),
             help="Phone: the Morning Call one window at a time, one compare column, big type.")
    st.divider()
    st.caption("Data: Volue (sandbox) · Meteomatics · Hydro Report methodology. "
               "Morning Call and Gas Demand read the sandbox tables (5-min cache).")


# ─── Landing page ────────────────────────────────────────────────────────────────
def _tile(name: str, cfg: dict, min_height: int = 210) -> None:
    locked = cfg["locked"]
    lock_html = '<span class="wip-pill">🔒 Work in progress</span>' if locked else ""
    st.markdown(f"""
    <div class="section-tile {'section-tile-locked' if locked else ''}" style="min-height:{min_height}px;">
        <div class="section-tile-accent" style="background:{cfg['color']};"></div>
        <div class="section-tile-num">{'Daily' if cfg.get('wide') else 'Section ' + cfg['num']}</div>
        <div class="section-tile-title">{name}</div>
        <div class="section-tile-desc">{cfg['desc']}</div>
        <div style="margin-top:12px;">{lock_html}</div>
    </div>
    """, unsafe_allow_html=True)
    st.button(f"Open {name}" if not locked else f"{name} — locked", key=f"tile_{cfg['num']}",
              use_container_width=True, on_click=go, args=(name,))


def render_home():
    st.markdown("#### CHOOSE A SECTION")
    st.caption("Morning Call is the daily table; below it the analysis sections. "
               "Strategy is locked while under construction.")
    wide = {n: c for n, c in SECTIONS.items() if c.get("wide")}
    grid = {n: c for n, c in SECTIONS.items() if not c.get("wide")}
    for name, cfg in wide.items():
        _tile(name, cfg, min_height=120)
    cols = st.columns(2)
    for i, (name, cfg) in enumerate(grid.items()):
        with cols[i % 2]:
            _tile(name, cfg)


section = st.session_state["section"]
if section == HOME:
    render_home()
else:
    RENDERERS[section]()
