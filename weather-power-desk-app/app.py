"""Power Desk Weather Dashboard — entry point (Databricks App, Streamlit).

Landing page with four big section tiles; the sidebar mirrors them and adds
a Home button. Sections:
  1. Forecast               — Volue ensemble values, spread vs normal spread, member scenarios
  2. Historical & Analysis  — monthly / weekly history by country, anomalies, index analogues
  3. Hydro Monitoring       — Hydro Report quantify_* figures and stats, live
  4. Strategy               — locked, work in progress

Structure follows the LPG desk dashboard (weather-lpg-desk-app): _config /
_data / _charts / _style modules, one render function per section, sandbox
tables refreshed by a Databricks job (power_desk_refresh.py).
"""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from _config import SECTIONS
from _style import CUSTOM_CSS
from _forecast import render_forecast
from _historical import render_historical
from _hydro import render_hydro
from _strategy import render_strategy

st.set_page_config(page_title="Power Desk — Weather Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

HOME = "Home"
RENDERERS = {
    "Forecast": render_forecast,
    "Historical & Analysis": render_historical,
    "Hydro Monitoring": render_hydro,
    "Strategy": render_strategy,
}

if "section" not in st.session_state:
    st.session_state["section"] = HOME


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
    st.caption("Data: Volue delta share · Meteomatics · Hydro Report methodology. "
               "Sandbox tables refreshed every 6 h by power_desk_refresh.py.")


# ─── Landing page ────────────────────────────────────────────────────────────────
def render_home():
    st.markdown("#### CHOOSE A SECTION")
    st.caption("Four entry points. Strategy is locked while under construction.")
    cols = st.columns(2)
    for i, (name, cfg) in enumerate(SECTIONS.items()):
        locked = cfg["locked"]
        with cols[i % 2]:
            lock_html = '<span class="wip-pill">🔒 Work in progress</span>' if locked else ""
            st.markdown(f"""
            <div class="section-tile {'section-tile-locked' if locked else ''}">
                <div class="section-tile-accent" style="background:{cfg['color']};"></div>
                <div class="section-tile-num">Section {cfg['num']}</div>
                <div class="section-tile-title">{name}</div>
                <div class="section-tile-desc">{cfg['desc']}</div>
                <div style="margin-top:12px;">{lock_html}</div>
            </div>
            """, unsafe_allow_html=True)
            st.button(f"Open {name}" if not locked else f"{name} — locked", key=f"tile_{cfg['num']}",
                      use_container_width=True, on_click=go, args=(name,))


section = st.session_state["section"]
if section == HOME:
    render_home()
else:
    RENDERERS[section]()
