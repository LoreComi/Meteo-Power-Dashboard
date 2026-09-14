"""Small HTML building blocks shared by every section: KPI cards, status banners."""
from __future__ import annotations

import numpy as np
import streamlit as st

_STATUS_TAGS = {"critical": "Alert", "warning": "Notice", "good": "OK"}


def kpi_card(label: str, value_str: str, card_class: str = "kpi-card-neutral",
             delta_html: str = "", rank_html: str = "") -> str:
    return (f'<div class="kpi-card {card_class}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value_str}</div>{delta_html}{rank_html}</div>')


def anomaly_kpi(label: str, value, unit: str, anomaly, anomaly_unit: str = "",
                warm_is_positive: bool = True, fmt: str = "{:.1f}") -> str:
    """Value with its departure from normal. Red = above normal for temperature;
    for production metrics, above normal is blue (more supply)."""
    try:
        v = float(value)
        val_str = f"{fmt.format(v)} {unit}"
    except (TypeError, ValueError):
        val_str = "N/A"
    delta_html, cls = "", "kpi-card-neutral"
    try:
        d = float(anomaly)
        if not np.isnan(d):
            up = d > 0
            warm = up if warm_is_positive else (not up)
            cls = "kpi-card-warm" if warm else "kpi-card-cool"
            d_cls = "kpi-delta-up" if warm else "kpi-delta-down"
            arrow = "▲" if up else "▼"
            delta_html = f'<div class="kpi-delta {d_cls}">{arrow} {abs(d):.1f}{anomaly_unit} vs normal</div>'
    except (TypeError, ValueError):
        pass
    return kpi_card(label, val_str, cls, delta_html)


def status_banner(text: str, level: str = "warning") -> None:
    tag = _STATUS_TAGS.get(level, "Notice")
    st.markdown(f'<div class="status-banner status-banner-{level}"><span class="status-tag">{tag}</span>{text}</div>',
                unsafe_allow_html=True)


def kpi_row(cards: list[str], max_cols: int = 6) -> None:
    if not cards:
        return
    n = min(len(cards), max_cols)
    cols = st.columns(n)
    for i, html in enumerate(cards):
        with cols[i % n]:
            st.markdown(html, unsafe_allow_html=True)


def ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
