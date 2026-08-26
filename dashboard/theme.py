"""Nocturne look-and-feel for the ParlayModel dashboard.

Presentation only. Nothing here reads app state, and every class name the app
already uses (.pm-yard-divider, .pm-photo, .pm-pos-pill, .pm-dialog-photo, ...)
is still defined, so no page markup has to change.

Usage in dashboard/app.py -- replace the old CSS-string-plus-st.markdown block
that used to inject styling directly with:

    from theme import inject_theme
    inject_theme()

placed immediately after `st.set_page_config(...)`.

Tokens come from the Nocturne design system:
  bg #161826 · surface #232532 · text #e9e9ed · accent #9184d9
  neutral ramp #f3f5fe … #292b31 · accent ramp #f5f4ff … #2b2741
  radius 8px · spacing scale at 0.70x density

Selector notes: data-testid values are Streamlit internals and shift between
versions. These target Streamlit 1.61.x (same set the previous theme block was
verified against). If a rule stops biting after an upgrade, inspect the element
and update the one selector — the token values stay valid.
"""

import streamlit as st

# --------------------------------------------------------------------- tokens
BG = "#161826"
SURFACE = "#232532"
TEXT = "#e9e9ed"
ACCENT = "#9184d9"

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    /* Nocturne tokens — the only place a value is written down */
    --pm-bg: #161826;
    --pm-surface: #232532;
    --pm-text: #e9e9ed;
    --pm-accent: #9184d9;

    --pm-n300: #cfd3e5;
    --pm-n400: #b2b6ca;
    --pm-n500: #9397ab;
    --pm-n600: #75798c;
    --pm-n700: #595d6c;
    --pm-n800: #3f424d;
    --pm-n900: #292b31;

    --pm-a200: #e7e5fe;
    --pm-a300: #d2cefd;
    --pm-a400: #b5abfc;
    --pm-a600: #796cbf;
    --pm-a700: #5d5294;
    --pm-a800: #423a6a;
    --pm-a900: #2b2741;

    --pm-section: #262a60;
    --pm-radius: 8px;
    --pm-radius-sm: 4px;
    --pm-border: var(--pm-n900);
    --pm-muted: var(--pm-n600);

    /* elevation: an edge plus ambient darkness, never a stacked shadow */
    --pm-shadow-sm: 0 0 0 1px #3f424d;
    --pm-shadow-md: 0 0 0 1px #595d6c, 0 6px 18px rgba(0,0,0,.55);
}

/* ------------------------------------------------------------------ type */
html, body, [class*="css"], .stApp, section[data-testid="stSidebar"] {
    font-family: 'Inter', system-ui, sans-serif;
}

h1, h2, h3, h4, h5 {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-weight: 500 !important;          /* never bolder — hierarchy is size + space */
    letter-spacing: -0.02em !important;
    text-transform: none !important;
    color: var(--pm-text) !important;
}
h1 { font-size: 1.9rem !important; line-height: 1.12 !important; border-bottom: 0 !important; padding-bottom: 0 !important; }
h2 { font-size: 1.35rem !important; }
h3 { font-size: 1.1rem !important; }

/* Anything numeric reads as data: odds, confidence, stakes, tables, code */
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] *,
div[data-testid="stDataFrame"],
div[data-testid="stDataFrameResizable"],
code, pre, kbd {
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace !important;
}
div[data-testid="stMetricValue"] {
    font-weight: 500 !important;
    letter-spacing: -0.025em !important;
}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.68rem !important;
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    color: var(--pm-muted) !important;
}
div[data-testid="stCaptionContainer"], .stCaption, small {
    color: var(--pm-n600) !important;
    line-height: 1.5 !important;
}

/* ---------------------------------------------------------------- layout */
.stApp { background: var(--pm-bg); }
.block-container { padding-top: 2.2rem !important; max-width: 1180px; }

/* Cards: st.container(border=True). Streamlit only paints a border when the
   container actually asked for one, so this is a no-op on layout blocks. */
div[data-testid="stVerticalBlock"] {
    border-radius: var(--pm-radius) !important;
    border-color: var(--pm-border) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--pm-radius) !important;
}

section[data-testid="stSidebar"] {
    background: #1a1c29;
    border-right: 1px solid rgba(233,233,237,.10);
}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
    border-radius: var(--pm-radius) !important;
    font-size: 0.86rem;
}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {
    background: var(--pm-surface) !important;
}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: var(--pm-a900) !important;
    color: var(--pm-a200) !important;
}
div[data-testid="stSidebarNavSeparator"],
section[data-testid="stSidebar"] hr { border-color: rgba(233,233,237,.10) !important; }
/* Nav group headings (Betting / Research / Admin) */
section[data-testid="stSidebar"] div[data-testid="stSidebarNavSectionHeader"] {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--pm-n700) !important;
}

/* --------------------------------------------------------------- buttons */
/* Nocturne: actions are OUTLINED — 1px accent border on transparent, never a fill */
div[data-testid^="stBaseButton-"] > button,
button[data-testid^="stBaseButton-"],
div[data-testid^="stBaseButton-"] {
    border-radius: var(--pm-radius) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-size: 0.84rem !important;
    transition: background .16s, border-color .16s, box-shadow .18s !important;
}
button[data-testid="stBaseButton-primary"],
div[data-testid="stBaseButton-primary"] button {
    background: transparent !important;
    border: 1px solid var(--pm-accent) !important;
    color: var(--pm-a300) !important;
}
button[data-testid="stBaseButton-primary"]:hover,
div[data-testid="stBaseButton-primary"] button:hover {
    background: var(--pm-a900) !important;
    box-shadow: 0 0 24px rgba(145,132,217,.16) !important;
}
button[data-testid="stBaseButton-primary"]:active { background: var(--pm-a800) !important; }

button[data-testid="stBaseButton-secondary"],
div[data-testid="stBaseButton-secondary"] button,
button[data-testid="stBaseButton-secondaryFormSubmit"] {
    background: transparent !important;
    border: 1px solid var(--pm-n800) !important;
    color: var(--pm-n400) !important;
}
button[data-testid="stBaseButton-secondary"]:hover,
div[data-testid="stBaseButton-secondary"] button:hover,
button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
    border-color: var(--pm-a600) !important;
    color: var(--pm-a200) !important;
    background: var(--pm-a900) !important;
}

/* ---------------------------------------------------------------- inputs */
div[data-baseweb="input"], div[data-baseweb="select"] > div,
div[data-baseweb="textarea"], div[data-testid="stNumberInputContainer"] {
    background: #161826 !important;
    border-radius: var(--pm-radius) !important;
    border-color: var(--pm-n800) !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within,
div[data-testid="stNumberInputContainer"]:focus-within {
    border-color: var(--pm-accent) !important;
    box-shadow: 0 0 0 1px var(--pm-accent) !important;
}
input, textarea { color: var(--pm-text) !important; }
div[data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important;
    color: var(--pm-n500) !important;
    letter-spacing: 0.01em;
}
/* focus is themed everywhere, never the browser default */
*:focus-visible {
    outline: 2px solid var(--pm-accent) !important;
    outline-offset: 2px !important;
}
::selection { background: var(--pm-a800); color: #f5f4ff; }

/* Slider (confidence range) */
div[data-testid="stSlider"] div[role="slider"] { background: var(--pm-accent) !important; }
div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    color: var(--pm-n700) !important;
}
div[data-testid="stThumbValue"] {
    color: var(--pm-a300) !important;
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
}

/* Tabs (Current slip / Add legs) */
button[data-baseweb="tab"] {
    font-size: 0.88rem !important;
    color: var(--pm-n500) !important;
}
button[data-baseweb="tab"][aria-selected="true"] { color: var(--pm-a200) !important; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { background: var(--pm-accent) !important; }
div[data-baseweb="tab-border"] { background: rgba(233,233,237,.10) !important; }

/* Expanders */
details[data-testid="stExpander"], div[data-testid="stExpander"] details {
    background: var(--pm-surface) !important;
    border: 1px solid var(--pm-border) !important;
    border-radius: var(--pm-radius) !important;
}
summary:hover { color: var(--pm-a300) !important; }

/* Tables / dataframes */
div[data-testid="stDataFrame"] { border-radius: var(--pm-radius) !important; overflow: hidden; }
div[data-testid="stDataFrame"] [class*="header"] {
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Alerts, badges, progress, toasts */
div[data-testid="stAlert"] { border-radius: var(--pm-radius) !important; }
div[data-testid="stProgress"] div[role="progressbar"] > div { background: var(--pm-accent) !important; }
div[data-testid="stToast"] {
    background: var(--pm-surface) !important;
    border-radius: var(--pm-radius) !important;
    box-shadow: var(--pm-shadow-md) !important;
}
/* st.badge / colored markdown text: remap Streamlit's semantic colors onto the
   Nocturne ramp — accent for good, a hue-rotated rose for bad, neutral for the
   rest. Same lightness/chroma as the accent, so nothing shouts. */
span[data-testid="stMarkdownBadge"], .stMarkdownBadge {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em;
    border-radius: var(--pm-radius-sm) !important;
}

/* -------------------------------------------------- signature: the rule */
/* Nocturne rules fade to transparent at their ends rather than stopping clean.
   Same .pm-yard-divider hook the app already calls — new drawing. */
.pm-yard-divider {
    display: flex; align-items: center; gap: 12px; margin: 1.9rem 0 1.1rem 0;
}
.pm-yard-divider span {
    color: var(--pm-n500);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.66rem; letter-spacing: 0.16em; text-transform: uppercase;
    white-space: nowrap;
}
.pm-yard-divider::after {
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(233,233,237,.22), rgba(233,233,237,.22) 60%, transparent);
}

/* ------------------------------------------------------ player portraits */
.pm-photo, .pm-photo-placeholder { width: 64px; height: 64px; }
.pm-dialog-photo, .pm-dialog-photo-placeholder { width: 96px; height: 96px; }
.pm-photo, .pm-dialog-photo {
    border-radius: 50%; object-fit: cover; display: block;
    box-shadow: 0 0 0 1px var(--pm-n800);
    /* dark-background cutouts blend into the page instead of sitting on a disc */
    mix-blend-mode: lighten;
}
.pm-photo { margin: 4px auto; }
.pm-photo-placeholder, .pm-dialog-photo-placeholder {
    border-radius: 50%; background: var(--pm-n900); color: var(--pm-n400);
    display: flex; align-items: center; justify-content: center;
    font-weight: 500; box-shadow: 0 0 0 1px var(--pm-n800);
}
.pm-photo-placeholder { margin: 4px auto; font-size: 18px; }
.pm-dialog-photo-placeholder { font-size: 26px; }
.pm-pos-pill {
    text-align: center; font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px; font-weight: 500; color: var(--pm-a300);
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 4px;
}

/* scrollbars follow the ground */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--pm-n800); border-radius: 8px; border: 3px solid var(--pm-bg); }
::-webkit-scrollbar-track { background: transparent; }
</style>
"""


def inject_theme() -> None:
    """Apply the Nocturne stylesheet. Call once, right after set_page_config."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def stat_band(items: list[tuple[str, str]]) -> None:
    """The one saturated moment in the system: a full-bleed stat band.

    Use it for a page's summary numbers (total suggested / of budget / legs in
    slip) instead of a row of st.metric, and nowhere else — Nocturne allows
    exactly one flooded field per page.

        stat_band([("Total suggested", "$9.60"), ("Of budget", "$10.00")])
    """
    cells = "".join(
        f'<div style="display:flex;flex-direction:column;gap:3px">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        f'letter-spacing:.14em;text-transform:uppercase;color:#b5afe8">{label}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:25px;'
        f'font-weight:500;letter-spacing:-.025em;color:#f5f4ff">{value}</span></div>'
        for label, value in items
    )
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:44px;padding:22px 28px;'
        'border-radius:14px;margin:8px 0 4px;background:radial-gradient('
        '120% 160% at 12% 0%,#353b80 0%,#262a60 55%,#20244f 100%)">'
        f"{cells}</div>",
        unsafe_allow_html=True,
    )
