"""Small HTML building blocks shared by every section: KPI cards, status banners, and the
modern pieces (page header, section header, stat tiles) — their CSS is _style.PAGE_CSS."""
from __future__ import annotations

import numpy as np
import streamlit as st

from _style import DD_WARM, DD_COOL

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


# ─── modern pieces ─────────────────────────────────────────────────────────────

_SVG = ('<svg viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round">{p}</svg>')
_ICONS = {
    "tt": '<path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/>',
    "wnd": '<path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/>'
           '<path d="M12.6 19.4A2 2 0 1 0 14 16H2"/>',
    "spv": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41'
           'M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
    "rre": '<path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/>',
    "forecast": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "hydro": '<path d="M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5c2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>'
             '<path d="M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>'
             '<path d="M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>',
    "prod": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "scen": '<line x1="6" x2="6" y1="3" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/>'
            '<path d="M18 9a9 9 0 0 1-9 9"/>',
}
_ICON_TINT = {"tt": "#e5484d", "wnd": "#0e9384", "spv": "#e09100", "rre": "#2a78d6", "forecast": "#4a3aa7",
              "hydro": "#2a78d6", "prod": "#0e9384", "scen": "#c4407a"}


def icon_box(name: str, box_cls: str) -> str:
    c = _ICON_TINT.get(name, "#54534d")
    return (f'<div class="{box_cls}" style="background:{c}14;">'
            f'{_SVG.format(c=c, p=_ICONS.get(name, ""))}</div>')


def keyed(key: str, **kw):
    """st.container(key=...) — the key gives the container the `st-key-<key>` class the CSS
    styles. Streamlit before 1.42 has no container keys: a plain container then, unstyled but working."""
    try:
        return st.container(key=key, **kw)
    except TypeError:
        return st.container(**kw)


def icon_button(target, label: str, icon: str | None = None, **kw):
    """`target`.button with a Material icon; on a Streamlit without button icons, the label alone."""
    try:
        return target.button(label, icon=icon, **kw)
    except TypeError:
        return target.button(label, **kw)


def section(key: str):
    """A white card around one section (keyed so the CSS can find it)."""
    return keyed(key, border=True)


def sec_header(icon: str, title: str, sub: str = "") -> None:
    st.markdown(f'<div class="dd-sec">{icon_box(icon, "dd-sec-ico")}<div><div class="dd-sec-title">{title}</div>'
                + (f'<div class="dd-sec-sub">{sub}</div>' if sub else "") + "</div></div>", unsafe_allow_html=True)


def tile(icon: str, label: str, sub: str, value: str, unit: str, gauge: float | None, tone: str,
          pill: str, foot: str = "", foot_tone: str = "") -> str:
    """A stat tile: icon + label, big value, a centred gauge of the departure (−1…+1 = the
    block's full shade), a pill with the departure and a foot line with the move."""
    if gauge is not None and not np.isnan(gauge):
        x = float(np.clip(gauge, -1, 1)) * 50
        color = DD_WARM if tone == "warm" else (DD_COOL if tone == "cool" else "#a9a89f")
        left = 50 if x >= 0 else 50 + x
        g = (f'<div class="dd-gauge"><div class="dd-gauge-bar" style="left:{left:.1f}%;width:{abs(x):.1f}%;'
             f'background:{color};"></div><div class="dd-gauge-mid"></div></div>')
    else:
        g = '<div class="dd-gauge"><div class="dd-gauge-mid"></div></div>'
    return (f'<div class="dd-tile"><div class="dd-tile-head">{icon_box(icon, "dd-tile-ico")}<div>'
            f'<div class="dd-tile-label">{label}</div><div class="dd-tile-sub">{sub}</div></div></div>'
            f'<div class="dd-tile-val">{value}<span class="dd-tile-unit">{unit}</span></div>{g}'
            f'<div class="dd-tile-foot"><span class="dd-pill dd-pill-{tone or "flat"}">{pill}</span>'
            + (f'<span class="dd-move {("dd-move-" + foot_tone) if foot_tone else ""}">{foot}</span>' if foot else "")
            + "</div></div>")


def tiles(items: list[str]) -> None:
    if items:
        st.markdown(f'<div class="dd-tiles">{"".join(items)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="dd-note">{text}</div>', unsafe_allow_html=True)


_ICONS.update({
    "home": '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 22V12h6v10"/>',
    "spread": '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>',
    "table": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/>',
    "members": '<path d="M3 3v18h18"/><rect x="7" y="9" width="3" height="8"/><rect x="14" y="5" width="3" height="12"/>',
})
_ICON_TINT.update({"home": "#54534d", "spread": "#e09100", "table": "#54534d", "members": "#2a78d6"})


def page_header(icon: str, eyebrow: str, title: str, subtitle: str = "", color: str = "#2a78d6") -> None:
    """The top of a section page: a gradient icon tile, the eyebrow (section number), the
    title and one line on what the page is for."""
    svg = _SVG.format(c="#ffffff", p=_ICONS.get(icon, ""))
    st.markdown(f'<div class="pg-head"><div class="pg-ico" style="background:linear-gradient(135deg,{color},'
                f'{color}b3);">{svg}</div><div><div class="pg-eyebrow">{eyebrow}</div>'
                f'<div class="pg-title">{title}</div>'
                + (f'<div class="pg-sub">{subtitle}</div>' if subtitle else "") + "</div></div>",
                unsafe_allow_html=True)


def info_tip(text: str, label: str = "How to read this") -> None:
    """An explanation kept out of the page: a small muted label with a "?" that shows `text`
    on hover. On a Streamlit whose st.markdown has no `help`, nothing is drawn."""
    try:
        st.markdown(f'<div class="info-tip">{label}</div>', unsafe_allow_html=True, help=text)
    except TypeError:
        pass
