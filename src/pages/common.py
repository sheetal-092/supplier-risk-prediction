"""Blocks shared by several pages (empty states, model info, risk trend)."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from src import config, ui
from src.analytics import risk_over_time, trend_status
from src.state import DataContext, load_sample

TREND_COLORS = {"Increasing": "red", "Stable": "blue", "Decreasing": "green"}
TREND_ICONS = {"Increasing": "▲", "Stable": "■", "Decreasing": "▼"}


def empty_state(ctx: DataContext) -> None:
    """Shown when no scored data is available for the current mode."""
    if ctx.is_live:
        ui.simulated_notice()
        st.info(ctx.empty_reason or "Start the simulation from the sidebar to generate live data.")
        if st.button("▶ Start simulation", type="primary"):
            ctx.sim.start()
            st.rerun()
        return
    if ctx.empty_reason and ctx.summary is not None:
        st.error(ctx.empty_reason)
    else:
        st.info(ctx.empty_reason or "Upload a dataset to begin.")
    st.markdown('<div class="srp-section">Get started</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="srp-card"><h4>Upload your data</h4>Use the <b>Upload Supplier Dataset</b> control in the sidebar. CSV and Excel files are supported. Columns are matched automatically to the financial or operational model.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="srp-card"><h4>Financial sample</h4>The project dataset: 250 suppliers, financial ratios, news sentiment and a 90-day disruption label.</div>', unsafe_allow_html=True)
        if st.button("Load financial sample", width="stretch", key="es_fin"):
            load_sample("financial")
            st.rerun()
    with c3:
        st.markdown('<div class="srp-card"><h4>Operational sample</h4>Delivery delay, defect rate, lead time, cost variation and fulfilment metrics (simulated example data).</div>', unsafe_allow_html=True)
        if st.button("Load operational sample", width="stretch", key="es_op"):
            load_sample("operational")
            st.rerun()
    if ctx.summary is not None and ctx.scoring is not None:
        with st.expander("Diagnostics", expanded=True):
            st.write("Detected profile:", ctx.summary.schema.profile_label)
            st.write("Coverage:", {k: f"{v:.0%}" for k, v in ctx.summary.schema.coverage.items()})
            for msg in ctx.scoring.errors:
                st.error(msg)
            for msg in ctx.scoring.warnings:
                st.warning(msg)


def model_banner(ctx: DataContext) -> None:
    """One-line description of how the risk scores were produced."""
    sc = ctx.scoring
    if sc is None:
        return
    if sc.method == "pretrained":
        b = sc.bundle
        metrics = b.metrics if b else {}
        auc = f" · ROC-AUC {metrics['roc_auc']:.3f}" if metrics.get("roc_auc") else ""
        news = f" · news fusion: {sc.news_source}" if sc.news_source else ""
        ui.note(f"<b>Model:</b> {b.algorithm if b else 'XGBoost'} · {config.PROFILES[sc.profile]['label']}{auc}{news}", "info")
    elif sc.method == "on_the_fly":
        ui.note("<b>Model:</b> " + " ".join(sc.messages), "info")
    elif sc.method == "unsupervised":
        st.warning(" ".join(sc.warnings))
    for w in sc.warnings if sc.method != "unsupervised" else []:
        st.warning(w)
    for e in sc.errors:
        st.error(e)


def risk_trend_block(ctx: DataContext, key: str, default_supplier: Optional[str] = None, show_supplier_picker: bool = True) -> None:
    """Risk score over time with supplier + date filter and a trend status badge."""
    recs = ctx.records
    if recs.empty:
        return
    ids = ctx.suppliers["Supplier_ID"].tolist()
    c1, c2 = st.columns([2, 3])
    supplier = default_supplier
    if show_supplier_picker:
        options = ["All suppliers (portfolio average)"] + ids
        idx = options.index(default_supplier) if default_supplier in options else 0
        choice = c1.selectbox("Supplier", options, index=idx, key=f"{key}_sup")
        supplier = None if choice.startswith("All") else choice
    d = recs if supplier is None else recs[recs["Supplier_ID"] == supplier]
    if ctx.has_date and d["Date"].notna().any() and not ctx.is_live:
        dmin, dmax = d["Date"].min().date(), d["Date"].max().date()
        if dmin < dmax:
            rng = c2.slider("Date window", min_value=dmin, max_value=dmax, value=(dmin, dmax), key=f"{key}_dates", format="YYYY-MM-DD")
            d = d[(d["Date"].dt.date >= rng[0]) & (d["Date"].dt.date <= rng[1])]
    trend = risk_over_time(d)
    if trend.empty:
        st.info("Not enough records for a trend.")
        return
    status, change = trend_status(trend["Risk_Score"])
    title = "Risk Score over Time" + (f" - {supplier}" if supplier else " - portfolio average")
    fig = ui.line(trend, "Period", "Risk_Score", title, colors=[config.PRIMARY], ylabel="Risk score", ylim=(0, 100))
    ui.risk_bands(fig)
    st.markdown(
        f"Trend: {ui.badge(TREND_ICONS[status] + ' ' + status.upper(), TREND_COLORS[status])} "
        f"<span style='color:{config.MUTED};font-size:12px'>change of {change:+.1f} points over the last {min(len(trend), 10)} periods "
        f"(slope of a linear fit; ±5 points = stable)</span>",
        unsafe_allow_html=True,
    )
    ui.show(fig, key=f"{key}_fig")
