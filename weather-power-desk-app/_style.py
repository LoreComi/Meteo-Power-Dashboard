"""Shared visual styling — Power Desk Weather Dashboard.

Same light trading-desk theme as the LPG desk app, so the two dashboards read
as one family. Color is assigned by job — categorical for series identity,
sequential for magnitude, diverging for anomaly polarity, status for alert
state — using the validated 8-hue palette (dataviz skill, references/palette.md).
Never mix jobs: a category never borrows a status color and vice versa.

Additions over the LPG app:
  - landing-page section tiles (.section-tile, .section-tile-locked)
  - ensemble "fan" opacities (ENS_FAN_ALPHA) shared by every spread chart
  - scenario cluster colors (SCENARIO_COLORS) — fixed by cluster rank
"""
from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════

SURFACE = "#ffffff"
PAGE_PLANE = "#f4f5f4"
INK_PRIMARY = "#1a1a18"
INK_SECONDARY = "#54534d"
INK_MUTED = "#8b897f"
GRIDLINE = "#e7e6e0"
BASELINE = "#c9c8bd"
BORDER = "rgba(20,20,15,0.09)"
BORDER_BRIGHT = "rgba(20,20,15,0.20)"

# Categorical — fixed order, assign in sequence, never cycle past 8
CAT_BLUE = "#2a78d6"
CAT_GREEN = "#008300"
CAT_MAGENTA = "#e87ba4"
CAT_YELLOW = "#eda100"
CAT_AQUA = "#1baf7a"
CAT_ORANGE = "#eb6834"
CAT_VIOLET = "#4a3aa7"
CAT_RED = "#e34948"
CATEGORICAL = [CAT_BLUE, CAT_GREEN, CAT_MAGENTA, CAT_YELLOW, CAT_AQUA, CAT_ORANGE, CAT_VIOLET, CAT_RED]

# Sequential (magnitude) — single hue, light -> dark
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]

# Diverging (polarity) — blue (cold/wet/positive-supply) <-> red (warm/dry), neutral mid
DIV_NEG = "#2a78d6"       # below normal temperature = cold = blue
DIV_MID = "#f0efec"
DIV_POS = "#e34948"       # above normal temperature = warm = red

# Status — fixed, reserved, never reused for series identity
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#b8790a"
STATUS_SERIOUS = "#c85a30"
STATUS_CRITICAL = "#c22f2f"

# Ensemble fan: percentile bands drawn in one hue at stepped opacity. Outer
# band (P10-P90) lightest, inner (P25-P75) darker, median as a solid line.
ENS_FAN_ALPHA = {"outer": 0.14, "inner": 0.28}

# Provider colors — fixed per provider so the same source is always the same
# hue across every forecast chart (Volue = blue, Meteologica = green,
# Meteomatics EC-ENS = orange, Meteomatics ECAI/AIFS-ENS = violet).
PROVIDER_COLORS = {
    "Volue": CAT_BLUE,
    "Meteologica": CAT_GREEN,
    "Meteomatics EC-ENS": CAT_ORANGE,
    "Meteomatics AIFS-ENS": CAT_VIOLET,
}

# Scenario clusters — ordered by cluster size (largest first). Cluster 1 is
# the "consensus" scenario and always takes blue.
SCENARIO_COLORS = [CAT_BLUE, CAT_ORANGE, CAT_GREEN, CAT_VIOLET, CAT_MAGENTA, CAT_AQUA]

# Hydro climatology chart (ported from Hydro_Report/hydro_plot_style.py):
# grey spread of historical years, blue ramp for the last few years (oldest
# light -> newest dark), norm dashed ink, current year in red.
HYDRO_HIST_GREY = "#c9c8c2"
HYDRO_CURRENT_RED = "#d03b3b"
HYDRO_BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab"]


def hydro_recent_colours(n: int) -> list[str]:
    """n evenly spaced blue steps, oldest (light) to newest (dark)."""
    if n <= 0:
        return []
    if n == 1:
        return [HYDRO_BLUE_RAMP[3]]
    last = len(HYDRO_BLUE_RAMP) - 1
    return [HYDRO_BLUE_RAMP[round(i * last / (n - 1))] for i in range(n)]


def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert '#rrggbb' to 'rgba(r, g, b, alpha)' for Plotly fill colors."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --bg-app: {PAGE_PLANE};
    --bg-surface: {SURFACE};
    --bg-card: #fafaf8;
    --border: {BORDER};
    --border-bright: {BORDER_BRIGHT};
    --text-primary: {INK_PRIMARY};
    --text-secondary: {INK_SECONDARY};
    --text-muted: {INK_MUTED};
    --accent: {CAT_BLUE};
    --good: {STATUS_GOOD};
    --warning: {STATUS_WARNING};
    --serious: {STATUS_SERIOUS};
    --critical: {STATUS_CRITICAL};
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --shadow-card: 0 1px 6px rgba(20,20,15,0.05), 0 0 0 1px rgba(20,20,15,0.04);
    --shadow-pop: 0 6px 24px rgba(20,20,15,0.10);
}}

.stApp {{
    background: var(--bg-app) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}}
.stApp > header, header[data-testid="stHeader"] {{ background: transparent !important; box-shadow: none !important; }}
[data-testid="stMainBlockContainer"], .main .block-container {{ padding-top: 2.2rem !important; }}

h1 {{
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    font-size: 1.5rem !important;
    margin-bottom: 0 !important;
}}
h2, h3 {{ color: var(--text-primary) !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }}
h4 {{
    color: var(--text-secondary) !important;
    font-weight: 700 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-top: 4px !important;
    margin-bottom: 4px !important;
}}
p, .stMarkdown, span, label {{ color: var(--text-primary); }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: var(--text-muted) !important; font-size: 0.80rem !important; }}

.main .block-container {{
    background: var(--bg-surface);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-card);
    padding: 24px 28px 32px !important;
    margin-top: 16px !important;
}}

/* Sidebar page navigation */
[data-testid="stSidebar"] {{
    background: var(--bg-app);
    border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] h3 {{
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted) !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{ gap: 2px; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    padding: 9px 12px !important;
    border-radius: var(--radius-md);
    width: 100%;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background: rgba(20,20,15,0.04);
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {{
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
    background: var(--bg-surface);
    box-shadow: var(--shadow-card);
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {{
    color: var(--accent) !important;
    font-weight: 600 !important;
}}

/* Sidebar: brand, nav buttons, view switch */
.pd-brand {{ display: flex; align-items: center; gap: 11px; padding: 2px 6px 10px; }}
.pd-logo {{ width: 36px; height: 36px; border-radius: 11px; flex: none; display: flex; align-items: center;
    justify-content: center; background: linear-gradient(135deg, {CAT_BLUE}, {CAT_VIOLET});
    box-shadow: 0 4px 12px rgba(42,120,214,0.30); }}
.pd-logo svg {{ width: 19px; height: 19px; }}
.pd-brand-name {{ font-size: 1rem; font-weight: 800; letter-spacing: -0.02em; color: var(--text-primary) !important; line-height: 1.1; }}
.pd-brand-sub {{ font-size: 0.7rem; color: var(--text-muted) !important; }}
.nav-label {{ font-size: 0.62rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text-muted) !important; margin: 14px 8px 6px; }}
.nav-foot {{ font-size: 0.7rem; color: var(--text-muted) !important; line-height: 1.45; margin: 18px 8px 0; }}
.st-key-nav {{ gap: 2px !important; }}
.st-key-nav .stButton > button {{ justify-content: flex-start !important; border: none !important;
    background: transparent !important; box-shadow: none !important; padding: 7px 12px !important;
    border-radius: 10px !important; min-height: 38px; }}
.st-key-nav .stButton > button > div {{ justify-content: flex-start !important; gap: 10px; }}
.st-key-nav .stButton > button p {{ font-size: 0.9rem !important; font-weight: 500 !important; color: var(--text-secondary) !important; }}
.st-key-nav .stButton > button [data-testid="stIconMaterial"] {{ color: var(--text-muted) !important; font-size: 1.15rem; }}
.st-key-nav .stButton > button:hover {{ background: rgba(20,20,15,0.05) !important; }}
.st-key-nav .stButton > button[kind="primary"], .st-key-nav .stButton > button[data-testid="stBaseButton-primary"] {{
    background: rgba(42,120,214,0.11) !important; }}
.st-key-nav .stButton > button[kind="primary"] p, .st-key-nav .stButton > button[data-testid="stBaseButton-primary"] p,
.st-key-nav .stButton > button[kind="primary"] [data-testid="stIconMaterial"],
.st-key-nav .stButton > button[data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"] {{
    color: {CAT_BLUE} !important; font-weight: 650 !important; }}
.st-key-view {{ background: #e9ebee; padding: 3px; border-radius: 12px; margin: 0 4px; }}
.st-key-view [data-testid="stHorizontalBlock"] {{ gap: 3px !important; }}
.st-key-view .stButton > button {{ border: none !important; background: transparent !important; box-shadow: none !important;
    border-radius: 9px !important; min-height: 34px; padding: 4px 8px !important; }}
.st-key-view .stButton > button p {{ font-weight: 600 !important; font-size: 0.85rem !important; color: var(--text-secondary) !important; }}
.st-key-view .stButton > button [data-testid="stIconMaterial"] {{ color: var(--text-muted) !important; }}
.st-key-view .stButton > button[kind="primary"], .st-key-view .stButton > button[data-testid="stBaseButton-primary"] {{
    background: #ffffff !important; box-shadow: 0 1px 4px rgba(20,20,15,0.12) !important; }}
.st-key-view .stButton > button[kind="primary"] p, .st-key-view .stButton > button[data-testid="stBaseButton-primary"] p,
.st-key-view .stButton > button[kind="primary"] [data-testid="stIconMaterial"],
.st-key-view .stButton > button[data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"] {{ color: {CAT_BLUE} !important; }}

/* first open: welcome card */
.st-key-welcome {{ background: #ffffff; border-radius: 20px; padding: 36px 36px 26px; margin-top: 8vh;
    box-shadow: 0 12px 40px rgba(20,20,15,0.08); border: 1px solid rgba(20,20,15,0.06); text-align: center; }}
.wl-logo {{ width: 56px; height: 56px; border-radius: 16px; margin: 0 auto 16px; display: flex; align-items: center;
    justify-content: center; background: linear-gradient(135deg, {CAT_BLUE}, {CAT_VIOLET});
    box-shadow: 0 8px 22px rgba(42,120,214,0.35); }}
.wl-logo svg {{ width: 28px; height: 28px; }}
.wl-title {{ text-align: center; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.03em; color: var(--text-primary) !important; }}
.wl-sub {{ text-align: center; font-size: 0.9rem; color: var(--text-secondary) !important; margin: 8px auto 22px; max-width: 460px; line-height: 1.5; }}
.wl-hint {{ text-align: center; font-size: 0.74rem; color: var(--text-muted) !important; margin-top: 14px; }}
.st-key-welcome .stButton > button {{ min-height: 64px; border-radius: 14px !important; }}
.st-key-welcome .stButton > button p {{ font-size: 1rem !important; font-weight: 650 !important; }}
.st-key-welcome .stButton > button [data-testid="stIconMaterial"] {{ font-size: 1.4rem; }}
.st-key-welcome .stButton > button[kind="primary"] p, .st-key-welcome .stButton > button[data-testid="stBaseButton-primary"] p,
.st-key-welcome .stButton > button[kind="primary"] [data-testid="stIconMaterial"],
.st-key-welcome .stButton > button[data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"] {{ color: #ffffff !important; }}

/* Landing page — four big section tiles. The tile is a styled block; the
   real click target is the full-width st.button rendered right under it. */
.section-tile {{
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 26px 24px 20px;
    min-height: 210px;
    box-shadow: var(--shadow-card);
    margin-bottom: 6px;
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}}
.section-tile:hover {{ box-shadow: var(--shadow-pop); transform: translateY(-1px); }}
.section-tile-accent {{ height: 4px; border-radius: 999px; width: 44px; margin-bottom: 16px; }}
.section-tile-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}}
.section-tile-title {{
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 6px 0 10px;
}}
.section-tile-desc {{ font-size: 0.86rem; color: var(--text-secondary); line-height: 1.5; }}
.section-tile-locked {{ background: var(--bg-card); opacity: 0.75; }}
.section-tile-locked .section-tile-title {{ color: var(--text-muted); }}
.wip-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border-radius: 999px;
    background: #fbf1e2; color: #7a5a00; border: 1px solid rgba(184,121,10,0.35);
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
}}
.stButton > button[kind="secondary"] {{
    border-radius: var(--radius-md);
    border: 1px solid var(--border-bright);
    background: var(--bg-surface);
    color: var(--text-primary);
    font-weight: 600;
}}
.stButton > button[kind="secondary"]:hover {{ border-color: var(--accent); color: var(--accent); }}

/* Locked page */
.locked-box {{
    text-align: center;
    padding: 64px 24px;
    background: var(--bg-card);
    border: 1px dashed var(--border-bright);
    border-radius: var(--radius-lg);
    margin-top: 16px;
}}
.locked-box .lock-glyph {{ font-size: 3rem; line-height: 1; margin-bottom: 14px; }}
.locked-box h2 {{ margin-bottom: 8px !important; }}
.locked-box p {{ color: var(--text-secondary) !important; max-width: 520px; margin: 0 auto; }}

/* KPI cards */
.kpi-card {{
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px 18px;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: var(--shadow-card);
}}
.kpi-card-warm  {{ border-top: 2px solid {CAT_RED}; }}
.kpi-card-cool  {{ border-top: 2px solid {CAT_BLUE}; }}
.kpi-card-neutral {{ border-top: 2px solid var(--text-muted); }}
.kpi-card-good     {{ border-top: 2px solid var(--good); }}
.kpi-card-warning  {{ border-top: 2px solid var(--warning); }}
.kpi-card-critical {{ border-top: 2px solid var(--critical); }}
.kpi-label {{
    font-size: 0.62rem;
    font-weight: 700;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}}
.kpi-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--text-primary) !important;
    margin-top: 6px;
    font-variant-numeric: tabular-nums;
}}
.kpi-delta {{ font-size: 0.72rem; margin-top: 4px; font-weight: 500; }}
.kpi-delta-up    {{ color: {CAT_RED} !important; }}
.kpi-delta-down  {{ color: {CAT_BLUE} !important; }}
.kpi-delta-flat  {{ color: var(--text-muted) !important; }}
.kpi-rank {{
    display: inline-block;
    margin-top: 8px;
    padding: 2px 9px;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--text-secondary);
    background: rgba(20,20,15,0.05);
    border-radius: 999px;
}}

/* Status alert banners — bold text tag + label, never color alone */
.status-banner {{
    border-radius: var(--radius-md);
    padding: 8px 14px;
    font-weight: 500;
    font-size: 0.85rem;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0 10px 0;
}}
.status-tag {{
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
}}
.status-banner-critical {{ background: #fbeaea; color: #7a1f1f; border: 1px solid rgba(194,47,47,0.30); }}
.status-banner-warning  {{ background: #fbf1e2; color: #7a5a00; border: 1px solid rgba(184,121,10,0.35); }}
.status-banner-good     {{ background: #e9f7e9; color: #0f5c0f; border: 1px solid rgba(12,163,12,0.28); }}

[data-testid="stDataFrame"] {{
    border-radius: var(--radius-md) !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}}
.stPlotlyChart {{
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--shadow-card);
    border: 1px solid var(--border);
    background: var(--bg-surface);
}}
hr {{ border-color: var(--border) !important; }}

/* Inputs: white fields with a visible outline, so every control stands out from the grey page */
.stMultiSelect [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] > div,
.stNumberInput [data-baseweb="input"],
.stTextInput [data-baseweb="input"],
.stDateInput [data-baseweb="input"],
.stTextArea textarea {{
    background: #ffffff !important;
    border: 1px solid rgba(20,20,15,0.22) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 1px 2px rgba(20,20,15,0.06);
}}
.stNumberInput input, .stTextInput input, .stDateInput input {{ background: #ffffff !important; }}
.stNumberInput button {{ background: #ffffff !important; }}
.stMultiSelect [data-baseweb="select"] > div:hover,
.stSelectbox [data-baseweb="select"] > div:hover,
.stNumberInput [data-baseweb="input"]:hover,
.stTextInput [data-baseweb="input"]:hover,
.stDateInput [data-baseweb="input"]:hover {{ border-color: {CAT_BLUE} !important; }}
.stMultiSelect [data-baseweb="select"] > div:focus-within,
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stNumberInput [data-baseweb="input"]:focus-within,
.stTextInput [data-baseweb="input"]:focus-within,
.stDateInput [data-baseweb="input"]:focus-within {{
    border-color: {CAT_BLUE} !important; box-shadow: 0 0 0 3px rgba(42,120,214,0.15) !important; }}
/* expanders read as cards, not as faint outlines */
[data-testid="stExpander"] details {{ background: #ffffff; border: 1px solid rgba(20,20,15,0.16) !important;
    border-radius: var(--radius-md) !important; }}
/* radio and checkbox circles/boxes: a darker ring when off */
[data-testid="stRadio"] label > div:first-child, [data-testid="stCheckbox"] label > span:first-child {{
    border-color: rgba(20,20,15,0.35) !important; }}
/* softer multiselect chips, everywhere */
[data-baseweb="tag"] {{ background: rgba(42,120,214,0.10) !important; border-radius: 8px !important; }}
[data-baseweb="tag"] span {{ color: {CAT_BLUE} !important; font-weight: 600; }}
[data-baseweb="tag"] svg {{ fill: {CAT_BLUE} !important; }}
/* "How to read this ?" tips — explanations live only behind the "?" */
.info-tip {{ font-size: 0.72rem; font-weight: 600; color: var(--text-muted) !important; margin: -2px 0 6px; }}
/* Morning Call table — the Excel sheet, as HTML */
.mc-block {{
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 12px 16px 6px;
    margin-bottom: 14px;
    box-shadow: var(--shadow-card);
}}
.mc-title {{
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.10em;
    color: var(--text-secondary); margin-bottom: 6px;
}}
.mc-table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
.mc-table th {{
    text-align: right; font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-muted); padding: 4px 8px; border-bottom: 1px solid var(--border);
}}
.mc-table th:first-child, .mc-table td:first-child {{ text-align: left; }}
.mc-table td {{
    padding: 5px 8px; border-bottom: 1px solid var(--border); text-align: right;
    font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; color: var(--text-primary);
}}
.mc-table td.mc-region {{ font-family: 'Inter', sans-serif; font-weight: 600; }}
.mc-table tr:last-child td {{ border-bottom: none; }}
.mc-pos {{ color: {CAT_GREEN} !important; font-weight: 600; }}
.mc-neg {{ color: {CAT_RED} !important; font-weight: 600; }}
.mc-zero {{ color: var(--text-muted) !important; }}
.mc-muted {{ color: var(--text-muted) !important; font-size: 0.78rem; }}
.mc-week {{
    font-size: 1.05rem; font-weight: 800; letter-spacing: -0.01em; color: var(--text-primary);
}}
.mc-sub {{ font-size: 0.78rem; color: var(--text-muted); margin-bottom: 8px; }}

/* Morning Call grid — columns are window → run; every cell carries the
   report's triple (Detail) or one quantity, shaded on the diverging pair.
   Fixed layout + shared <colgroup> so the four block tables line up. */
.mcg-table {{ table-layout: fixed; }}
.mcg-table col.mcg-c0 {{ width: 104px; }}
.mcg-table th.mcg-corner {{
    text-align: left; vertical-align: bottom; color: var(--text-secondary); font-size: 0.70rem;
    text-transform: none; letter-spacing: 0; border-bottom: 1px solid var(--border-bright); padding-bottom: 5px;
}}
.mcg-corner-name {{ text-transform: uppercase; letter-spacing: 0.08em; }}
.mcg-corner-unit {{ font-weight: 500; color: var(--text-muted); }}
.mcg-corner-sub {{
    font-weight: 500; text-transform: none; letter-spacing: 0; color: var(--text-muted);
    font-size: 0.64rem; margin-top: 3px;
}}
.mcg-table th.mcg-win {{
    text-align: center; color: var(--text-primary); font-size: 0.74rem; letter-spacing: 0.01em;
    text-transform: none; padding: 3px 8px 3px; border-bottom: 1px solid var(--border);
    border-left: 2px solid var(--border-bright);
}}
.mcg-range {{ color: var(--text-muted); font-weight: 500; margin-left: 6px; }}
.mcg-table th.mcg-col {{
    text-align: center; text-transform: none; letter-spacing: 0; padding: 3px 4px 5px;
    border-bottom: 1px solid var(--border-bright); vertical-align: bottom;
}}
.mcg-model {{ display: block; font-weight: 700; font-size: 0.68rem; color: var(--text-secondary); }}
.mcg-date {{ display: block; font-weight: 500; font-size: 0.63rem; color: var(--text-muted); margin-top: 1px; }}
.mcg-table th.mcg-ref {{ box-shadow: inset 0 -2px 0 var(--accent); }}
.mcg-table th.mcg-ref .mcg-model {{ color: var(--accent); }}
.mcg-table .mcg-first {{ border-left: 2px solid var(--border-bright); }}
.mcg-table td.mcg-cell {{ text-align: center; padding: 4px 5px; }}
.mcg-table td.mcg-refcell {{ background: rgba(42,120,214,0.035); }}
.mcg-v {{ font-size: 0.9rem; font-weight: 600; line-height: 1.15; }}
.mcg-d {{ font-size: 0.69rem; line-height: 1.2; margin-top: 1px; white-space: nowrap; }}
.mcg-sep {{ color: var(--text-muted); margin: 0 3px; }}
.mcg-table td.mcg-heat {{ font-weight: 600; font-size: 0.88rem; }}
.mcg-table td.mcg-spread, .mcg-table th.mcg-spread {{
    color: var(--text-muted); font-weight: 500; font-size: 0.8rem; border-left: 1px dashed var(--border);
}}
sup.mcg-n {{
    font-size: 0.56rem; color: var(--text-muted); font-weight: 500; margin-left: 2px;
    font-family: 'Inter', sans-serif;
}}
/* "Change · anomaly" cells: the anomaly (Δ norm) as a shaded pill, the move since the
   previous run in brackets after it — arrow and number as large as the pill's — and the
   absolute value beneath in dark ink. */
.mcg-pair {{ display: flex; gap: 6px; justify-content: center; align-items: center; }}
.mcg-move {{ font-size: 0.82rem; font-weight: 700; line-height: 1.2; white-space: nowrap; }}
.mcg-move-a {{ font-size: 0.95rem; margin-right: 1px; }}
.mcg-move-v {{ margin-left: 1px; }}
.mcg-move-a, .mcg-move-v {{ color: inherit; }}          /* the global `span` colour would otherwise win over the cell's red / blue */
.mcg-br {{ color: var(--text-secondary); font-weight: 500; }}
.mcg-move-flat {{ color: var(--text-muted) !important; font-weight: 500; }}
.mcg-pill {{
    display: inline-block; min-width: 46px; padding: 2px 5px; border-radius: 5px;
    font-weight: 600; font-size: 0.8rem; line-height: 1.25; color: var(--text-primary);
}}
.mcg-pill-na {{ color: var(--text-muted) !important; font-weight: 500; }}
.mcg-abs {{ font-size: 0.74rem; color: var(--text-primary); font-weight: 600; margin-top: 3px; line-height: 1.1; }}

/* At a glance — the reference run's biggest moves and anomalies as chips, ranked
   across blocks on each block's own scale, shaded on the same diverging pair. */
.mc-glance {{ display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: center; margin: 0 0 8px; }}
.mc-glance-label {{
    font-size: 0.64rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-muted); min-width: 158px; line-height: 1.3;
}}
.mc-chip {{
    display: inline-flex; align-items: baseline; gap: 6px; padding: 5px 10px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--bg-surface); font-size: 0.8rem; line-height: 1.2;
    color: var(--text-primary);
}}
.mc-chip b {{ font-weight: 700; }}
.mc-chip .mc-chip-q {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.88rem; font-variant-numeric: tabular-nums; }}
.mc-chip .mc-chip-w {{ color: var(--text-secondary); font-size: 0.68rem; }}
.mc-chip-empty {{ color: var(--text-muted); font-size: 0.78rem; }}

.mcg-legend {{
    display: flex; flex-wrap: wrap; gap: 18px; align-items: center; font-size: 0.76rem;
    color: var(--text-muted); margin: -4px 0 8px 2px;
}}
.mcg-swatch {{ display: inline-block; width: 34px; height: 10px; border-radius: 3px; vertical-align: middle; margin: 0 6px; }}
.mcg-runs {{ font-size: 0.76rem; color: var(--text-muted); line-height: 1.6; margin: 0 0 8px 2px; }}
.mcg-runs b {{ color: var(--text-secondary); font-weight: 600; }}

/* Scenario side-by-side table */
.sc-table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
.sc-table th {{
    font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-muted); padding: 4px 8px; border-bottom: 1px solid var(--border); text-align: right;
}}
.sc-table th.sc-scen {{ color: var(--text-primary); font-size: 0.72rem; border-bottom: 2px solid; }}
.sc-table td {{
    padding: 5px 8px; border-bottom: 1px solid var(--border); text-align: right;
    font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums;
}}
.sc-table th:first-child, .sc-table td:first-child {{ text-align: left; font-family: 'Inter', sans-serif; font-weight: 600; }}
.sc-table tr:last-child td {{ border-bottom: none; }}

/* Hydro overview — regions × layers grid (reuses .mc-table) and criticality chips.
   Cell shading is a status job: red = critically low, amber = low / fast drawdown,
   blue = very high / fast refill — with the number and the word in the cell too. */
.hy-table th.hy-layer {{
    text-align: center; color: var(--text-primary); text-transform: none; letter-spacing: 0.01em;
    font-size: 0.72rem; border-left: 2px solid var(--border-bright); padding: 4px 8px 3px;
}}
.hy-table th.hy-sub {{ text-transform: none; letter-spacing: 0; padding-top: 2px; }}
.hy-table .hy-first {{ border-left: 2px solid var(--border-bright); }}
.hy-table td.hy-crit {{ background: #fbeaea; color: #7a1f1f !important; font-weight: 600; }}
.hy-table td.hy-low  {{ background: #fbf1e2; color: #7a5a00 !important; font-weight: 600; }}
.hy-table td.hy-high {{ background: #e3eefb; color: #184f95 !important; font-weight: 600; }}
.hy-table td.hy-na   {{ color: var(--text-muted) !important; }}
.hy-table td.hy-agg  {{ color: var(--text-muted) !important; font-size: 0.72rem; font-family: 'Inter', sans-serif; }}
.hy-flags {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 10px; }}
.hy-flag {{
    display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px;
    font-size: 0.8rem; border: 1px solid; line-height: 1.3;
}}
.hy-flag b {{ font-weight: 700; }}
.hy-flag-critical {{ background: #fbeaea; color: #7a1f1f; border-color: rgba(194,47,47,0.30); }}
.hy-flag-warning  {{ background: #fbf1e2; color: #7a5a00; border-color: rgba(184,121,10,0.35); }}
.hy-flag-high     {{ background: #e3eefb; color: #184f95; border-color: rgba(42,120,214,0.30); }}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: var(--radius-md) var(--radius-md) 0 0;
    padding: 8px 16px;
    font-weight: 600;
}}
</style>
"""

# Agent-family commentary (Morning Call). Signal colour is a status job, not a
# series identity: bullish = critical red (tightening), bearish = blue, neutral
# = muted ink — and the arrow and the word carry it too, never colour alone.
BRIEF_CSS = f"""
<style>
.brief-box {{
    background: var(--bg-card);
    border: 1px solid {BORDER};
    border-left: 3px solid {CAT_BLUE};
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0 14px;
    font-size: 0.91rem;
    line-height: 1.6;
    color: {INK_PRIMARY};
}}
.agent-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-top: 3px solid {INK_MUTED};
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
    box-shadow: 0 1px 6px rgba(20,20,15,0.05);
}}
.agent-card-label {{
    font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    color: {INK_MUTED};
}}
.agent-card-signal {{ font-size: 1.0rem; font-weight: 800; margin: 5px 0 3px; }}
.agent-card-sub {{ font-size: 0.7rem; color: {INK_MUTED}; }}
.agent-overall {{
    border-radius: 10px; padding: 10px 22px; text-align: center; margin: 10px 0 0;
    display: flex; align-items: center; justify-content: center; gap: 14px;
}}
.agent-overall-label {{
    font-size: 0.7rem; color: rgba(255,255,255,0.88); font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
}}
.agent-overall-value {{ font-size: 1.15rem; font-weight: 900; color: #ffffff; }}
</style>
"""

# Morning Call phone layout: the page narrows to one column, paddings shrink, the
# tables grow to a thumb-readable size. Injected only while "Phone" is selected.
PHONE_CSS = f"""
<style>
.main .block-container {{ padding: 10px 10px 24px !important; margin-top: 6px !important; border-radius: 8px; }}
[data-testid="stSidebar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: block; }}
h4 {{ font-size: 0.8rem !important; }}
.mc-block {{ padding: 8px 8px 4px; margin-bottom: 10px; }}
.mcp-table {{ width: 100%; border-collapse: collapse; font-size: 1.0rem; table-layout: fixed; }}
.mcp-table th {{
    font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted);
    padding: 4px 4px; border-bottom: 1px solid var(--border); text-align: right; line-height: 1.2;
}}
.mcp-table th:first-child, .mcp-table td:first-child {{ text-align: left; }}
.mcp-table th .mcg-date {{ display: block; font-weight: 500; text-transform: none; letter-spacing: 0; font-size: 0.62rem; }}
.mcp-table td {{
    padding: 9px 4px; border-bottom: 1px solid var(--border); text-align: right;
    font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; font-size: 1.0rem;
}}
.mcp-table td.mc-region {{ font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.95rem; }}
.mcp-table tr:last-child td {{ border-bottom: none; }}
.mcp-table td .mcp-sub {{ display: block; font-size: 0.74rem; color: var(--text-muted); }}
.mcp-table td.mcp-heat {{ font-weight: 600; }}
.mcp-title {{ font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-secondary); margin: 2px 0 4px; }}
.mcp-title span {{ font-weight: 500; text-transform: none; letter-spacing: 0; color: var(--text-muted); margin-left: 6px; }}
.mc-glance {{ gap: 6px; }}
.mc-glance-label {{ min-width: 100%; }}
.mc-chip {{ font-size: 0.9rem; padding: 7px 11px; }}
.mc-chip .mc-chip-q {{ font-size: 1.0rem; }}
.agent-card {{ margin-bottom: 8px; }}
.stButton > button {{ width: 100%; }}
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=SURFACE,
    font=dict(family="Inter, sans-serif", color=INK_PRIMARY, size=12),
    margin=dict(l=55, r=20, t=50, b=70),
    hovermode="x unified",
    legend=dict(
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor=GRIDLINE,
        borderwidth=1,
        font=dict(size=11, color=INK_SECONDARY),
        orientation="h",
        yanchor="top",
        y=-0.22,
        xanchor="center",
        x=0.5,
    ),
    xaxis=dict(
        gridcolor=GRIDLINE, zerolinecolor=BASELINE, linecolor=GRIDLINE,
        tickfont=dict(color=INK_MUTED, size=11), title_font=dict(color=INK_SECONDARY),
    ),
    yaxis=dict(
        gridcolor=GRIDLINE, zerolinecolor=BASELINE, linecolor=GRIDLINE,
        tickfont=dict(color=INK_MUTED, size=11), title_font=dict(color=INK_SECONDARY),
    ),
)


# ══════════════════════════════════════════════════════════════════════════════
# Morning Call → Country deep dive: hero with the outlook, section cards, stat tiles
# ══════════════════════════════════════════════════════════════════════════════
DD_WARM = "#e5484d"       # warm side: warmer, or less wind / solar / precipitation
DD_COOL = "#2a78d6"       # cool side: colder, or more
PAGE_CSS = f"""
<style>
/* page header */
.pg-head {{ display: flex; align-items: center; gap: 16px; margin: 4px 0 18px; }}
.pg-ico {{ width: 48px; height: 48px; border-radius: 14px; display: flex; align-items: center; justify-content: center;
    flex: none; box-shadow: 0 6px 16px rgba(42,120,214,0.25); }}
.pg-ico svg {{ width: 24px; height: 24px; }}
.pg-eyebrow {{ font-size: 0.64rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--text-muted) !important; }}
.pg-title {{ font-size: 1.75rem; font-weight: 800; letter-spacing: -0.035em; line-height: 1.1;
    color: var(--text-primary) !important; }}
.pg-sub {{ font-size: 0.82rem; color: var(--text-secondary) !important; margin-top: 3px; }}

/* toolbar: the controls of a tab grouped in one soft card */
div[class*="st-key-tb_"] {{ background: #f7f8fa; border: 1px solid rgba(20,20,15,0.06); border-radius: 14px;
    padding: 10px 16px 2px; margin-bottom: 14px; }}
div[class*="st-key-tb_"] label p {{ font-size: 0.72rem !important; font-weight: 650 !important;
    color: var(--text-muted) !important; text-transform: uppercase; letter-spacing: 0.06em; }}

/* pill tabs inside a page container */
div[class*="st-key-pg_"] [data-baseweb="tab-list"] {{ gap: 4px; background: #eef0f3; padding: 4px; border-radius: 12px;
    width: fit-content; margin-bottom: 14px; }}
div[class*="st-key-pg_"] [data-baseweb="tab-list"] button[role="tab"] {{ padding: 7px 18px; border-radius: 9px;
    height: auto; background: transparent; }}
div[class*="st-key-pg_"] [data-baseweb="tab-list"] button[role="tab"] p {{ font-size: 0.86rem !important;
    font-weight: 600 !important; color: var(--text-secondary) !important; }}
div[class*="st-key-pg_"] [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {{ background: #ffffff;
    box-shadow: 0 1px 4px rgba(20,20,15,0.10); }}
div[class*="st-key-pg_"] [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] p {{ color: var(--accent) !important; }}
div[class*="st-key-pg_"] [data-baseweb="tab-highlight"], div[class*="st-key-pg_"] [data-baseweb="tab-border"] {{ display: none; }}
.dd-hero {{
    position: relative; overflow: hidden;
    background: radial-gradient(120% 140% at 100% 0%, #2c4f86 0%, rgba(44,79,134,0) 55%),
                linear-gradient(125deg, #0f1726 0%, #172338 55%, #1c2f4f 100%);
    border-radius: 18px; padding: 22px 26px 18px; margin: 6px 0 18px;
    box-shadow: 0 10px 30px rgba(15,23,38,0.18);
}}
.dd-hero-top {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; flex-wrap: wrap; }}
.dd-eyebrow {{ font-size: 0.66rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
    color: rgba(255,255,255,0.55) !important; }}
.dd-country {{ font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1;
    color: #ffffff !important; margin-top: 4px; }}
.dd-meta {{ font-size: 0.78rem; color: rgba(255,255,255,0.62) !important; margin-top: 6px;
    font-family: 'JetBrains Mono', monospace; }}
.dd-out-box {{ text-align: right; min-width: 240px; }}
.dd-out-label {{ font-size: 0.66rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
    color: rgba(255,255,255,0.55) !important; }}
.dd-out {{ font-size: 2.3rem; font-weight: 800; letter-spacing: -0.035em; line-height: 1.05; margin-top: 4px; }}
.dd-tone-warm {{ color: #ff8f8a !important; }}
.dd-tone-cool {{ color: #8cc4ff !important; }}
.dd-tone-flat {{ color: #ffffff !important; }}
.dd-out-next {{ font-size: 0.8rem; color: rgba(255,255,255,0.7) !important; margin-top: 6px; }}
.dd-out-next b {{ color: #ffffff !important; font-weight: 700; }}
.dd-signals {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; padding-top: 14px;
    border-top: 1px solid rgba(255,255,255,0.10); }}
.dd-sig {{ display: inline-flex; align-items: center; gap: 7px; padding: 5px 11px; border-radius: 999px;
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.10);
    font-size: 0.76rem; color: rgba(255,255,255,0.85) !important; white-space: nowrap; }}
.dd-sig b {{ color: #ffffff !important; font-weight: 700; }}
.dd-dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex: none; }}

/* section cards (keyed st.container) */
div[class*="st-key-dd_sec_"] {{
    background: #ffffff; border-radius: 16px !important; padding: 6px 8px 4px;
    border: 1px solid rgba(20,20,15,0.07) !important; box-shadow: 0 2px 14px rgba(20,20,15,0.05);
    margin-bottom: 14px;
}}
.dd-sec {{ display: flex; align-items: center; gap: 12px; margin: 2px 0 12px; }}
.dd-sec-ico {{ width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center;
    justify-content: center; flex: none; }}
.dd-sec-ico svg {{ width: 19px; height: 19px; }}
.dd-sec-title {{ font-size: 1.05rem; font-weight: 750; letter-spacing: -0.01em; color: var(--text-primary) !important; }}
.dd-sec-sub {{ font-size: 0.76rem; color: var(--text-muted) !important; margin-top: 1px; }}

/* stat tiles */
.dd-tiles {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 4px 0 16px; }}
@media (max-width: 1000px) {{ .dd-tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 560px) {{ .dd-tiles {{ grid-template-columns: 1fr; }} }}
.dd-tile {{ background: #fbfbfa; border: 1px solid rgba(20,20,15,0.07); border-radius: 14px; padding: 14px 16px 13px; }}
.dd-tile-head {{ display: flex; align-items: center; gap: 9px; }}
.dd-tile-ico {{ width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; flex: none; }}
.dd-tile-ico svg {{ width: 16px; height: 16px; }}
.dd-tile-label {{ font-size: 0.8rem; font-weight: 650; color: var(--text-primary) !important; line-height: 1.15; }}
.dd-tile-sub {{ font-size: 0.68rem; color: var(--text-muted) !important; font-weight: 500; }}
.dd-tile-val {{ font-size: 1.85rem; font-weight: 750; letter-spacing: -0.03em; margin-top: 10px;
    color: var(--text-primary) !important; font-variant-numeric: tabular-nums; line-height: 1.1; }}
.dd-tile-unit {{ font-size: 0.85rem; font-weight: 600; color: var(--text-muted) !important; margin-left: 4px; letter-spacing: 0; }}
.dd-gauge {{ position: relative; height: 6px; border-radius: 99px; background: #ecebe7; margin: 10px 0 9px; overflow: hidden; }}
.dd-gauge-mid {{ position: absolute; left: 50%; top: -2px; bottom: -2px; width: 2px; background: #c9c8bd; }}
.dd-gauge-bar {{ position: absolute; top: 0; bottom: 0; border-radius: 99px; }}
.dd-tile-foot {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }}
.dd-pill {{ font-size: 0.72rem; font-weight: 700; padding: 2px 9px; border-radius: 999px; }}
.dd-pill-warm {{ background: rgba(229,72,77,0.10); color: {DD_WARM} !important; }}
.dd-pill-cool {{ background: rgba(42,120,214,0.10); color: {DD_COOL} !important; }}
.dd-pill-flat {{ background: rgba(20,20,15,0.05); color: var(--text-secondary) !important; }}
.dd-move {{ font-size: 0.72rem; font-weight: 600; color: var(--text-muted) !important; }}
.dd-move-warm {{ color: {DD_WARM} !important; }}
.dd-move-cool {{ color: {DD_COOL} !important; }}
.dd-wk {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap;
    margin: 10px 0 8px; padding: 8px 14px; border-radius: 10px; background: #f3f4f7;
    border-left: 4px solid #4a3aa7; }}
.dd-wk-title {{ font-size: 0.95rem; font-weight: 800; letter-spacing: -0.01em; color: var(--text-primary) !important; }}
.dd-wk-dates {{ font-size: 0.76rem; font-weight: 500; color: var(--text-muted) !important; margin-left: 10px; }}
.dd-wk-out {{ font-size: 0.95rem; font-weight: 800; letter-spacing: -0.01em; }}
.dd-wk-warm {{ color: {DD_WARM} !important; }}
.dd-wk-cool {{ color: {DD_COOL} !important; }}
.dd-wk-flat {{ color: var(--text-secondary) !important; }}
.dd-week {{ font-size: 0.82rem; font-weight: 700; color: var(--text-primary) !important; margin: 14px 0 8px;
    padding-left: 10px; border-left: 3px solid #c4407a; }}
.dd-note {{ font-size: 0.74rem; color: var(--text-muted) !important; margin: -4px 0 10px; }}

/* the shared KPI cards, softened inside the Morning Call page (hydro state, scenarios) */
.kpi-card {{ text-align: left; border-radius: 14px; background: #fbfbfa; box-shadow: none;
    border: 1px solid rgba(20,20,15,0.07); padding: 14px 16px; }}
.kpi-value {{ font-family: 'Inter', sans-serif; font-weight: 750; letter-spacing: -0.03em; }}
</style>
"""

DEEP_DIVE_CSS = PAGE_CSS          # the Morning Call deep dive and the section pages share one sheet
