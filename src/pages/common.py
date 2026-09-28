"""Blocks shared by several pages (empty states, data quality, model info, risk trend)."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from src import config, ui
from src.analytics import risk_over_time, trend_status
from src.state import DataContext, handle_upload, load_sample

TREND_COLORS = {"Increasing": "red", "Stable": "blue", "Decreasing": "green"}
TREND_ICONS = {"Increasing": "▲", "Stable": "■", "Decreasing": "▼"}


def empty_state(ctx: DataContext) -> None:
    """Shown when no scored data is available for the current mode."""
    if ctx.is_live:
        st.info(ctx.empty_reason or "Start the simulation to generate live data.")
        if st.button("▶ Start simulation", type="primary", key="es_start"):
            ctx.sim.start()
            st.rerun()
        return
    if ctx.empty_reason and ctx.summary is not None:
        st.error(ctx.empty_reason)
    c1, c2 = st.columns([3, 2])
    with c1:
        with st.container(border=True):
            st.markdown('<div class="upload-card"><div class="ic">⬆</div><div class="h">DATASET ANALYSIS · Upload Supplier Dataset</div>'
                        '<div class="p">Drop a CSV or Excel file. Columns are matched automatically to the financial or operational risk model, '
                        'then validation → preprocessing → feature engineering → risk prediction → analytics → anomaly detection run automatically.</div></div>',
                        unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls", "txt"], key="main_uploader", label_visibility="collapsed")
            if uploaded is not None:
                with st.spinner("Reading, validating and scoring the dataset…"):
                    handle_upload(uploaded)
                if st.session_state.get("dataset_error"):
                    st.error(st.session_state["dataset_error"])
                else:
                    st.rerun()
    with c2:
        with st.container(border=True):
            st.markdown('<div class="kpi-label" style="margin:6px 0 8px 0">Sample data</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:12.5px;color:{config.MUTED};margin-bottom:8px"><b style="color:{config.TEXT}">Financial sample</b> · the project dataset: 250 suppliers, financial ratios, news sentiment and a 90-day disruption label.</div>', unsafe_allow_html=True)
            if st.button("Load Financial Sample", width="stretch", key="es_fin", type="primary"):
                load_sample("financial")
                st.rerun()
            st.markdown(f'<div style="font-size:12.5px;color:{config.MUTED};margin:10px 0 8px 0"><b style="color:{config.TEXT}">Operational sample</b> · delivery delay, defect rate, lead time, cost variation and fulfilment metrics (simulated example).</div>', unsafe_allow_html=True)
            if st.button("Load Operational Sample", width="stretch", key="es_op"):
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


def data_quality(ctx: DataContext) -> None:
    """Compact data-quality strip with details in an expander (Dataset Mode)."""
    s = ctx.summary
    if s is None:
        return
    ui.stat_strip([
        ("Source", ctx.source_label), ("Records", f"{s.records:,}"), ("Columns", s.columns),
        ("Missing values", f"{s.missing_cells:,}"), ("Duplicate rows", f"{s.duplicate_rows:,}"),
        ("Numerical", len(s.numeric_columns)), ("Categorical", len(s.categorical_columns)),
        ("Profile", s.schema.profile_label), ("Engine", ctx.report.get("engine", "pandas")),
    ])
    with st.expander("Data quality, preview and preprocessing details"):
        t1, t2, t3 = st.tabs(["Preview (first 50 rows)", "Validation", "Preprocessing"])
        with t1:
            st.dataframe(st.session_state["dataset"]["raw"].head(50), width="stretch", hide_index=True)
        with t2:
            if s.missing_by_column:
                miss = pd.DataFrame({"Column": list(s.missing_by_column), "Missing": list(s.missing_by_column.values())})
                miss["Missing %"] = (miss["Missing"] / s.records * 100).round(2)
                st.dataframe(miss, width="stretch", hide_index=True)
            else:
                st.success("No missing values detected.")
            for w in s.warnings:
                st.warning(w)
            for n in s.schema.notes:
                st.info(n)
            mapping = {k: v for k, v in s.schema.rename_map.items() if k != v}
            if mapping:
                st.caption("Column mapping applied: " + ", ".join(f"{k} → {v}" for k, v in mapping.items()))
            st.caption("Numerical: " + ", ".join(s.numeric_columns[:30]) + (" …" if len(s.numeric_columns) > 30 else ""))
            st.caption("Categorical: " + ", ".join(s.categorical_columns[:30]) + (" …" if len(s.categorical_columns) > 30 else ""))
        with t3:
            for step in ctx.report.get("steps", []):
                st.write("•", step)
            if ctx.report.get("imputed"):
                st.caption("Median values used for imputation: " + ", ".join(f"{k}={v:.3g}" for k, v in list(ctx.report["imputed"].items())[:12]))


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


def risk_trend_block(ctx: DataContext, key: str, default_supplier: Optional[str] = None, show_supplier_picker: bool = True, height: int = 360) -> None:
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
    title = "Risk Score over Time" + (f" · {supplier}" if supplier else " · portfolio average")
    fig = ui.line(trend, "Period", "Risk_Score", title, colors=[config.PRIMARY_LIGHT], ylabel="Risk score", ylim=(0, 100), height=height, fill=True)
    ui.risk_bands(fig)
    st.markdown(
        f"Trend: {ui.badge(TREND_ICONS[status] + ' ' + status.upper(), TREND_COLORS[status])} "
        f"<span style='color:{config.MUTED};font-size:12px'>change of {change:+.1f} points over the last {min(len(trend), 10)} periods "
        f"(slope of a linear fit; ±5 points = stable)</span>",
        unsafe_allow_html=True,
    )
    ui.show(fig, key=f"{key}_fig")
