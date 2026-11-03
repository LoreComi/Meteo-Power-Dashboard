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
from _ui import keyed, icon_button
from _morning import render_morning_call
from _forecast import render_forecast
from _historical import render_historical
from _hydro import render_hydro
from _gas import render_gas_demand
from _strategy import render_strategy

st.set_page_config(page_title="Weather Power Desk", layout="wide", initial_sidebar_state="expanded")
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


# ─── Header ──────────────────────────────────────────────────────────────────────
_now = datetime.now(timezone.utc)
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
    <div></div>
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
    _l, _mid, _r = st.columns([1, 2.2, 1])
    with _mid, keyed("welcome"):
        st.markdown('<div class="wl-logo"><svg viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" '
                    'stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg></div>'
                    '<div class="wl-title">Welcome to the Weather Power Desk Dashboard</div>'
                    '<div class="wl-sub">How are you viewing the dashboard? Phone shows the Morning Call one window at a '
                    'time, with one compare column and big type. You can switch any time in the sidebar.</div>',
                    unsafe_allow_html=True)
        _c1, _c2 = st.columns(2)
        icon_button(_c1, "Desktop", ":material/computer:", use_container_width=True, on_click=_set_device,
                   args=("PC",), key="device_pc", type="primary")
        icon_button(_c2, "Phone", ":material/smartphone:", use_container_width=True, on_click=_set_device,
                   args=("Phone",), key="device_phone")
        st.markdown('<div class="wl-hint">Tip: open the app with ?layout=phone or ?layout=pc to skip this question.</div>',
                    unsafe_allow_html=True)
    st.stop()


# ─── Sidebar navigation ─────────────────────────────────────────────────────────
NAV_ICONS = {
    HOME: ":material/home:", "Morning Call": ":material/wb_sunny:", "Forecast": ":material/insights:",
    "Historical & Analysis": ":material/history:", "Hydro Monitoring": ":material/water_drop:",
    "Gas Demand": ":material/local_fire_department:", "Strategy": ":material/lock:",
}
_LOGO = ('<svg viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" '
         'stroke-linejoin="round"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg>')


with st.sidebar:
    st.markdown(f'<div class="pd-brand"><div class="pd-logo">{_LOGO}</div><div><div class="pd-brand-name">Weather Power Desk'
                f'</div><div class="pd-brand-sub">Dashboard</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Sections</div>', unsafe_allow_html=True)
    with keyed("nav"):
        for name in [HOME] + list(SECTIONS.keys()):
            icon_button(st, name, NAV_ICONS.get(name), key=f"nav_{name}",
                      type="primary" if st.session_state["section"] == name else "secondary",
                      use_container_width=True, on_click=go, args=(name,))
    st.markdown('<div class="nav-label">View</div>', unsafe_allow_html=True)
    with keyed("view"):
        _v1, _v2 = st.columns(2, gap="small")
        for _col, _dev, _ico in ((_v1, "PC", ":material/computer:"), (_v2, "Phone", ":material/smartphone:")):
            icon_button(_col, _dev, _ico, key=f"view_{_dev}", use_container_width=True, on_click=_set_device,
                        args=(_dev,), type="primary" if st.session_state["device"] == _dev else "secondary")


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
