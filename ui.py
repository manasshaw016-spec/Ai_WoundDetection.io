from __future__ import annotations

import streamlit as st


def apply_theme(theme: str) -> None:
    """Inject a light or dark theme stylesheet for the dashboard."""
    if theme == "dark":
        css = """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: linear-gradient(135deg, #0f172a 0%, #111827 100%); color: #f8fafc; }
        .block-container { padding-top: 1rem; }
        div[data-testid='stMetric'] { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 0.8rem; }
        </style>
        """
    else:
        css = """
        <style>
        .stApp { background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%); color: #111827; }
        .block-container { padding-top: 1rem; }
        div[data-testid='stMetric'] { background: rgba(255,255,255,0.9); border: 1px solid rgba(15,23,42,0.08); border-radius: 14px; padding: 0.8rem; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar(model_status: str) -> tuple[str, str]:
    """Render the dashboard sidebar and return the selected page and current theme."""
    with st.sidebar:
        st.header("🩹 Wound Support Dashboard")
        st.caption("Educational analysis only — not a diagnosis.")
        page = st.radio("Navigation", ["Dashboard", "History", "About"], horizontal=False)
        st.divider()
        st.subheader("Controls")
        theme = "dark" if st.toggle("Dark mode", value=st.session_state.get("theme") == "dark") else "light"
        st.session_state["theme"] = theme
        st.caption(f"Model status: {model_status}")
        st.divider()
        st.caption("This workspace uses a lightweight local model and image-analysis heuristics for educational support.")
    return page, theme
