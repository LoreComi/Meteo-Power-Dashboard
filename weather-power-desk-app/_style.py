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
.stApp > header {{ background: transparent !important; }}

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

.stMultiSelect [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] > div {{
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-md) !important;
}}
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

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: var(--radius-md) var(--radius-md) 0 0;
    padding: 8px 16px;
    font-weight: 600;
}}
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
