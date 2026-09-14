"""Section 4 — Strategy. Locked: work in progress.

Kept as its own module so the landing tile, sidebar entry and routing already
exist; when the content is ready only this file changes.
"""
from __future__ import annotations

import streamlit as st

LOCKED = True
STRATEGY_PLACEHOLDER_ITEMS = [
    "Weather-driven positioning ideas by country and horizon",
    "Forecast-vs-market divergence monitor",
    "Scenario P&L sensitivities from the Forecast section clusters",
]


def render_strategy():
    st.markdown("#### STRATEGY")
    st.caption("Trading strategy views built on top of the Forecast, Historical and Hydro sections.")

    items = "".join(f"<li>{x}</li>" for x in STRATEGY_PLACEHOLDER_ITEMS)
    st.markdown(
        f"""
        <div class="locked-box">
            <div class="lock-glyph">&#128274;</div>
            <span class="wip-pill">Work in progress</span>
            <h2>This section is locked</h2>
            <p>The Strategy section is under construction and not yet available.
            Planned content:</p>
            <ul style="display:inline-block; text-align:left; color:var(--text-secondary);
                       font-size:0.9rem; margin-top:12px;">{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
