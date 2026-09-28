"""SUPPLIER RISK PREDICTION - Intelligent Supplier Risk Prediction and Analytics.

Streamlit entry point.  Run locally with::

    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from src import config, ui
from src.analytics import AGG_CHOICES
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
# Header + top bar (data source switch, horizontal navigation)
# --------------------------------------------------------------------------- #
ui.app_header(ss["mode"], sim.running)

top_left, top_right = st.columns([1.5, 3.5])
with top_left:
    st.markdown('<div class="srp-side-label">Data source</div>', unsafe_allow_html=True)
    with st.container(key="datasource"):
        st.segmented_control("Data source", ["Dataset", "Live Data"], key="mode", required=True,
                             format_func=lambda m: "📁 Dataset" if m == "Dataset" else "🔴 Live Data",
                             label_visibility="collapsed")
with top_right:
    st.markdown('<div class="srp-side-label">Navigation</div>', unsafe_allow_html=True)
    with st.container(key="navbar"):
        st.segmented_control("Navigation", PAGES, key="page", required=True, label_visibility="collapsed")

# --------------------------------------------------------------------------- #
# Sidebar: mode panel, analysis settings, filters
# --------------------------------------------------------------------------- #
with st.sidebar:
    if ss["mode"] == "Dataset":
        st.markdown('<div class="srp-side-label">📁 Dataset analysis</div>', unsafe_allow_html=True)
        ds = ss.get("dataset")
        if ds:
            s = ds["summary"]
            st.markdown(
                f'<div class="srp-card" style="padding:12px 14px;margin-bottom:8px">'
                f'<div style="font-weight:700;color:#86EFAC;font-size:13px">✓ Dataset Loaded</div>'
                f'<div style="font-size:12px;color:{config.MUTED};word-break:break-all;margin:4px 0 8px 0">{ds["name"]}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:12px">'
                f'<div><span style="color:{config.MUTED}">Records</span><br><b>{s.records:,}</b></div>'
                f'<div><span style="color:{config.MUTED}">Columns</span><br><b>{s.columns}</b></div>'
                f'<div><span style="color:{config.MUTED}">Missing values</span><br><b>{s.missing_cells:,}</b></div>'
                f'<div><span style="color:{config.MUTED}">Duplicates</span><br><b>{s.duplicate_rows:,}</b></div></div>'
                f'<div style="font-size:11px;color:{config.MUTED};margin-top:8px">{ds["schema"].profile_label} · engine {ds["report"].get("engine", "pandas")}</div></div>',
                unsafe_allow_html=True,
            )
            with st.expander("Replace dataset"):
                uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls", "txt"], key="sb_uploader", label_visibility="collapsed")
                if uploaded is not None:
                    with st.spinner("Processing dataset…"):
                        handle_upload(uploaded)
                    if ss.get("dataset_error"):
                        st.error(ss["dataset_error"])
                    else:
                        st.rerun()
                c1, c2 = st.columns(2)
                if c1.button("Financial sample", width="stretch", key="sb_fin"):
                    load_sample("financial")
                    st.rerun()
                if c2.button("Operational sample", width="stretch", key="sb_op"):
                    load_sample("operational")
                    st.rerun()
                if config.SAMPLE_PROCESSED_XLSX.exists() and st.button("Processed dataset (FinBERT)", width="stretch", key="sb_proc"):
                    load_sample("processed")
                    st.rerun()
            if st.button("Clear dataset", width="stretch", key="sb_clear"):
                ss["dataset"], ss["dataset_error"], ss["dataset_file_id"] = None, None, None
                st.rerun()
        else:
            with st.container(border=True):
                st.markdown('<div class="upload-card" style="padding-top:8px"><div class="ic">⬆</div><div class="h">Upload Supplier Dataset</div><div class="p">CSV · XLSX · XLS</div></div>', unsafe_allow_html=True)
                uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls", "txt"], key="sb_uploader", label_visibility="collapsed")
                if uploaded is not None:
                    with st.spinner("Reading, validating and scoring the dataset…"):
                        handle_upload(uploaded)
                    if not ss.get("dataset_error"):
                        st.rerun()
                if ss.get("dataset_error"):
                    st.error(ss["dataset_error"])
                st.caption("Or use a sample dataset")
                if st.button("Load Financial Sample", width="stretch", key="sb_fin"):
                    load_sample("financial")
                    st.rerun()
                if st.button("Load Operational Sample", width="stretch", key="sb_op"):
                    load_sample("operational")
                    st.rerun()
                if config.SAMPLE_PROCESSED_XLSX.exists() and st.button("Load Processed Dataset (FinBERT)", width="stretch", key="sb_proc"):
                    load_sample("processed")
                    st.rerun()
    else:
        st.markdown('<div class="srp-side-label">🔴 Live data controls</div>', unsafe_allow_html=True)
        with st.container(border=True):
            @st.fragment(run_every=1.0 if sim.running else None)
            def _live_status() -> None:
                s = ss["sim"]
                kind = "red" if s.running else "neutral"
                st.markdown(
                    f'<div class="kpi-label">Simulation status</div>'
                    f'<div style="margin:4px 0 8px 0">{ui.badge(("● " if s.running else "○ ") + s.status, kind)} '
                    f'<span style="font-size:12px;color:{config.MUTED}">{s.total_generated:,} records · {s.rate}/sec</span></div>',
                    unsafe_allow_html=True,
                )

            _live_status()
            page_live.controls(sim, "sb", compact=True)
            rate = st.select_slider("Data generation rate (records/sec)", options=config.LIVE_RATE_OPTIONS, value=sim.rate if sim.rate in config.LIVE_RATE_OPTIONS else 5, key="live_rate")
            if rate != sim.rate:
                sim.rate = int(rate)
            n_sup = st.number_input("Simulated suppliers (applied on reset)", min_value=10, max_value=200, value=sim.n_suppliers, step=10, key="live_nsup")
            if int(n_sup) != sim.n_suppliers:
                sim.n_suppliers = int(n_sup)
            ss["auto_refresh"] = st.toggle("Auto-refresh while running", value=bool(ss.get("auto_refresh", True)))
            ss["refresh_seconds"] = st.select_slider("Refresh interval (seconds)", options=[1, 2, 3, 5, 10], value=int(ss.get("refresh_seconds", 2)))
        ui.note("<b>SIMULATED / ARTIFICIAL DATA</b>", "sim")

    with st.expander("Analysis settings"):
        agg_label = st.selectbox("Supplier risk aggregation", list(AGG_CHOICES), index=list(AGG_CHOICES.values()).index(ss.get("agg", "recent")))
        ss["agg"] = AGG_CHOICES[agg_label]
        ss["cluster_k"] = st.slider("K-Means clusters", 2, 6, int(ss.get("cluster_k", 4)))
        spark_ok = spark_available()
        st.toggle("Use PySpark engine for preprocessing" + ("" if spark_ok else " (not available here)"), key="use_spark", disabled=not spark_ok,
                  help="Runs the notebook's Spark de-duplication and Financial Stress Index step. Requires PySpark + Java.")

    _ctx_for_filters = build_context()
    with st.expander("Filters · active" if _ctx_for_filters.ready and _ctx_for_filters.filters_active else "Filters", expanded=False):
        if _ctx_for_filters.ready:
            render_filters(_ctx_for_filters)
        else:
            st.caption("Filters become available once data is loaded.")

    reg = get_registry()
    st.markdown(
        f'<div style="font-size:11px;color:{config.MUTED};margin-top:14px">Models: {len(reg.bundles)} loaded'
        + (" · <span style='color:#FCA5A5'>issues</span>" if reg.load_errors else " · XGBoost · Random Forest · K-Means · Isolation Forest · SHAP")
        + "</div>",
        unsafe_allow_html=True,
    )
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
