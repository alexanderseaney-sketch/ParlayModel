"""
Small display-only HTML pieces shared by the redesign pages (Nocturne tokens, see
theme.py and the design handoff). Interactive controls stay native Streamlit widgets;
these only render text, badges and table rows.
"""
import html

import streamlit as st

# Nocturne tokens (theme.py / design handoff)
ROW = "#20222f"
TABLE_ROW = "#1c1e2b"
SURFACE = "#232532"
EDGE = "#292b31"
TEXT = "#e9e9ed"
N1, N2, N3, N4, N5 = "#cfd3e5", "#b2b6ca", "#9397ab", "#75798c", "#595d6c"
BORDER = "#3f424d"
ACCENT = "#9184d9"
A_TEXT = "#d2cefd"
A_TINT = "#2b2741"
A_DEEP = "#423a6a"
A_MID = "#5d5294"
ROSE = "#d1798a"
ROSE_FILL = "#3d2530"
ROSE_BAR = "#a8566a"
# Unquoted on purpose: every helper drops this into a single-quoted style='...' attribute,
# where a quoted 'JetBrains Mono' would end the attribute and lose the rest of the style.
MONO = "JetBrains Mono,ui-monospace,monospace"

INJ_BADGE = {"—": (EDGE, N2), "TBD": (EDGE, N3), "Q": (A_DEEP, A_TEXT), "D": (ROSE_FILL, ROSE),
             "O": (ROSE_FILL, ROSE), "IR": (ROSE_FILL, ROSE)}
PRACTICE_COLOR = {"DNP": ROSE, "LP": N2, "FP": A_TEXT}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def badge(text, bg=A_TINT, fg=A_TEXT, border=None, size="10px") -> str:
    b = f"border:1px solid {border};" if border else ""
    return (f"<span style='font-size:{size};padding:2px 6px;border-radius:4px;background:{bg};color:{fg};{b}"
            f"font-family:{MONO};white-space:nowrap'>{esc(text)}</span>")


def mono(text, color=N4, size="10px") -> str:
    return f"<span style='font-size:{size};color:{color};font-family:{MONO}'>{esc(text)}</span>"


def section(label: str) -> None:
    """Small-caps label + fading rule (same look as app.yard_divider / .pm-yard-divider)."""
    st.markdown(f"<div class='pm-yard-divider'><span>{esc(label.upper())}</span></div>", unsafe_allow_html=True)


def table(headers: list[str], rows: list[list[str]], widths: list[str], min_width: int = 720) -> None:
    """Grid table: header strip on surface, rows on table-row ground, hairline dividers.
    Cell values are pre-rendered HTML (callers escape their text)."""
    cols = " ".join(widths)
    head = "".join(f"<span>{esc(h)}</span>" for h in headers)
    body = "".join(
        f"<div style='display:grid;grid-template-columns:{cols};gap:10px;padding:10px 16px;background:{TABLE_ROW};"
        f"border-top:1px solid rgba(233,233,237,.07);align-items:center;font-size:12px'>"
        + "".join(f"<span>{c}</span>" for c in r) + "</div>"
        for r in rows)
    st.markdown(
        f"<div style='border-radius:14px;overflow:auto;box-shadow:0 0 0 1px {EDGE}'><div style='min-width:{min_width}px'>"
        f"<div style='display:grid;grid-template-columns:{cols};gap:10px;padding:9px 16px;background:{SURFACE};"
        f"font-size:9.5px;letter-spacing:.14em;color:{N4};font-family:{MONO}'>{head}</div>{body}</div></div>",
        unsafe_allow_html=True)


def row_card(main: str, sub: str = "", right: str = "", extra: str = "") -> str:
    return (f"<div style='display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;background:{ROW};"
            f"box-shadow:0 0 0 1px {EDGE};margin-bottom:6px'><div style='flex:1;min-width:0;display:flex;"
            f"flex-direction:column;gap:2px'><span style='font-size:12.5px;font-weight:500'>{main}</span>"
            f"<span style='font-size:11px;color:{N3}'>{sub}</span>{extra}</div>{right}</div>")
