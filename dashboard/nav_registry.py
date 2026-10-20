"""
The app's five tabs and the views inside each one. app.py fills PAGES and HUBS on every
run before nav.run(), so page modules can link to each other with go() without importing
app.py (which would be circular).

Every older page lives on as a view inside one of the tabs (2026-10-20 cleanup), picked
from a row of pills at the top of the tab. LINKS maps the old page keys to their new home
so existing go("ev_finder")-style links still land on the right view.
"""
PAGES: dict = {}   # tab key -> st.Page
HUBS: dict = {}    # tab key -> list of view labels, first one is the default
LINKS: dict = {
    "ev_finder": ("picks", "+EV Finder"),
    "parlay_builder": ("build", "Advanced builder"),
    "bet_log": ("my_bets", "Full log"),
    "bankroll": ("my_bets", "Bankroll"),
    "player": ("research", "Players"),
    "game_center": ("research", "Games"),
    "matchups": ("research", "Matchups"),
    "usage": ("research", "Usage"),
    "injuries": ("news", "Injuries"),
}


def _goto_key(hub: str) -> str:
    return f"_hub_{hub}_goto"


def go(key: str) -> None:
    """Open a tab (or an old page's view inside its tab); no-op if it isn't registered."""
    import streamlit as st
    hub, view = LINKS.get(key, (key, None))
    page = PAGES.get(hub)
    if page is None:
        return
    view = view or (HUBS.get(hub) or [None])[0]
    if view:
        # Applied by show_hub on the next run, before the pills render -- a widget's
        # value can't be changed once it's drawn in the current run.
        st.session_state[_goto_key(hub)] = view
    st.switch_page(page)


def show_hub(hub: str, views: dict) -> None:
    """Draw the tab's view pills and run the chosen view. `views` maps label -> function."""
    import streamlit as st
    labels = list(views)
    HUBS[hub] = labels
    key, store = f"hub_{hub}", f"_hub_{hub}"
    goto = st.session_state.pop(_goto_key(hub), None)
    if goto in views:
        st.session_state[key] = goto
    elif st.session_state.get(key) not in views:
        st.session_state[key] = st.session_state.get(store) if st.session_state.get(store) in views else labels[0]

    def _keep_selected():
        # Tapping the selected pill again would clear it; keep the last view instead.
        if st.session_state.get(key) is None:
            st.session_state[key] = st.session_state.get(store, labels[0])

    if len(labels) > 1:
        st.pills("View", labels, key=key, label_visibility="collapsed", on_change=_keep_selected)
    choice = st.session_state.get(key) or labels[0]
    st.session_state[store] = choice
    views[choice]()
