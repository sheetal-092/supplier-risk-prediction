"""SUPPLIER RISK PREDICTION - Intelligent Supplier Risk Prediction and Analytics.

Streamlit entry point.  Run locally with::

    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from src import config, ui
from src.pages import analytics as page_analytics
from src.pages import anomaly as page_anomaly
from src.pages import live as page_live
from src.pages import overview as page_overview
from src.pages import prediction as page_prediction
from src.prediction import get_registry
from src.preprocessing import spark_available
from src.state import PAGES, advance_simulation, build_context, handle_upload, init_state, load_sample, render_filters

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": f"{config.APP_TITLE} - {config.APP_SUBTITLE}"},
)
ui.inject_css()
init_state()
ss = st.session_state
sim = ss["sim"]

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        '<div class="srp-brand"><div class="logo">SR</div><div><div class="title">SUPPLIER RISK SYSTEM</div>'
        '<div class="sub">Intelligent Risk Prediction &amp; Analytics</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="srp-side-label">Data Source</div>', unsafe_allow_html=True)
    st.radio("Data source", ["Dataset", "Live Data"], key="mode", horizontal=True, label_visibility="collapsed")
    if ss["mode"] == "Live Data":
        st.markdown('<div class="srp-mode live">DATA MODE: 🔴 Live Data Mode<small>SIMULATED / ARTIFICIAL DATA</small></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="srp-mode dataset">DATA MODE: 📁 Dataset Mode<small>Uploaded CSV / Excel dataset</small></div>', unsafe_allow_html=True)

    st.markdown('<div class="srp-side-label">Navigation</div>', unsafe_allow_html=True)
    st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")

    if ss["mode"] == "Dataset":
        st.markdown('<div class="srp-side-label">📁 Dataset Mode</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload Supplier Dataset", type=["csv", "xlsx", "xls", "txt"], help="CSV or Excel. Columns are matched automatically.")
        if uploaded is not None:
            with st.spinner("Reading, validating and scoring the dataset…"):
                handle_upload(uploaded)
        if ss.get("dataset_error"):
            st.error(ss["dataset_error"])
        ds = ss.get("dataset")
        if ds:
            st.markdown(
                f'<div class="srp-card" style="padding:10px 12px;margin-bottom:6px"><div style="font-size:12px;color:{config.MUTED}">Loaded</div>'
                f'<div style="font-weight:600;font-size:13px;word-break:break-all">{ds["name"]}</div>'
                f'<div style="font-size:12px;margin-top:4px">Records: <b>{ds["summary"].records:,}</b> · Columns: <b>{ds["summary"].columns}</b></div>'
                f'<div style="font-size:12px">{ds["schema"].profile_label}</div></div>',
                unsafe_allow_html=True,
            )
        with st.expander("Sample datasets", expanded=ds is None):
            if st.button("Financial sample (project dataset)", width="stretch"):
                load_sample("financial")
                st.rerun()
            if st.button("Operational sample (simulated)", width="stretch"):
                load_sample("operational")
                st.rerun()
            if config.SAMPLE_PROCESSED_XLSX.exists() and st.button("Processed dataset (with FinBERT)", width="stretch"):
                load_sample("processed")
                st.rerun()
        if ds and st.button("Clear dataset", width="stretch"):
            ss["dataset"], ss["dataset_error"], ss["dataset_file_id"] = None, None, None
            st.rerun()
    else:
        st.markdown('<div class="srp-side-label">🔴 Live Data Mode</div>', unsafe_allow_html=True)

        @st.fragment(run_every=1.0 if sim.running else None)
        def _live_status() -> None:
            s = ss["sim"]
            kind = "red" if s.running else "neutral"
            st.markdown(
                f'<div style="font-size:12px;color:{config.MUTED}">Simulation status</div>'
                f'<div style="margin:2px 0 6px 0">{ui.badge(("● " if s.running else "○ ") + s.status, kind)} '
                f'<span style="font-size:12px;color:{config.MUTED}">{s.total_generated:,} records · {s.rate}/sec</span></div>',
                unsafe_allow_html=True,
            )

        _live_status()
        page_live.controls(sim, "sb")
        rate = st.select_slider("Data generation rate (records/sec)", options=config.LIVE_RATE_OPTIONS, value=sim.rate if sim.rate in config.LIVE_RATE_OPTIONS else 5, key="live_rate")
        if rate != sim.rate:
            sim.rate = int(rate)
        n_sup = st.number_input("Simulated suppliers (applied on reset)", min_value=10, max_value=200, value=sim.n_suppliers, step=10, key="live_nsup")
        if int(n_sup) != sim.n_suppliers:
            sim.n_suppliers = int(n_sup)
        st.toggle("Auto-refresh while running", key="auto_refresh")
        st.select_slider("Refresh interval (seconds)", options=[1, 2, 3, 5, 10], key="refresh_seconds")

    with st.expander("Analysis settings"):
        from src.analytics import AGG_CHOICES

        agg_label = st.selectbox("Supplier risk aggregation", list(AGG_CHOICES), index=list(AGG_CHOICES.values()).index(ss.get("agg", "recent")))
        ss["agg"] = AGG_CHOICES[agg_label]
        ss["cluster_k"] = st.slider("K-Means clusters", 2, 6, int(ss.get("cluster_k", 4)))
        spark_ok = spark_available()
        st.toggle("Use PySpark engine for preprocessing" + ("" if spark_ok else " (not available here)"), key="use_spark", disabled=not spark_ok,
                  help="Runs the notebook's Spark de-duplication and Financial Stress Index step. Requires PySpark + Java.")

    # Filters need the context of the active data source
    _ctx_for_filters = build_context()
    with st.expander("Filters", expanded=False):
        if _ctx_for_filters.ready:
            render_filters(_ctx_for_filters)
        else:
            st.caption("Filters become available once data is loaded.")

    reg = get_registry()
    st.markdown(
        f'<div style="font-size:11px;color:{config.MUTED};margin-top:14px">Models: {len(reg.bundles)} loaded'
        + (" · <span style='color:#DC2626'>issues</span>" if reg.load_errors else " · XGBoost, Random Forest, K-Means, Isolation Forest, SHAP")
        + "</div>",
        unsafe_allow_html=True,
    )
    if reg.load_errors:
        for e in reg.load_errors:
            st.caption(f"⚠️ {e}")

# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
RENDERERS = {
    PAGES[0]: page_overview.render,
    PAGES[1]: page_analytics.render,
    PAGES[2]: page_prediction.render,
    PAGES[3]: page_anomaly.render,
    PAGES[4]: page_live.render,
}


def render_page() -> None:
    if ss["mode"] == "Live Data":
        advance_simulation()
    ctx = build_context()
    try:
        RENDERERS[ss["page"]](ctx)
    except Exception as exc:  # noqa: BLE001 - never show a raw traceback to the user
        if os.environ.get("SRP_DEBUG"):
            raise
        st.error(f"Something went wrong while rendering this page: {exc}")
        st.caption("Try adjusting the filters, reloading the dataset, or resetting the simulation.")


live_auto = ss["mode"] == "Live Data" and sim.running and ss.get("auto_refresh", True)
if live_auto:
    @st.fragment(run_every=float(ss.get("refresh_seconds", 2)))
    def _auto_page() -> None:
        render_page()

    _auto_page()
else:
    render_page()
