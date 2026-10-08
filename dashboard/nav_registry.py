"""
st.Page objects by key, filled in by app.py on every run before nav.run(), so page
modules can link to each other with st.switch_page() without importing app.py
(which would be circular).
"""
PAGES: dict = {}


def go(key: str) -> None:
    """st.switch_page to a registered page; no-op if it isn't registered."""
    import streamlit as st
    page = PAGES.get(key)
    if page is not None:
        st.switch_page(page)
