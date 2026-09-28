"""📊 Supplier Analytics page."""
from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config, ui
from src.analytics import correlation_matrix, metric_over_time, supplier_metric_ranking
from src.pages.common import empty_state
from src.schema import feature_label, feature_unit
from src.state import DataContext


def _section_kpis(ctx: DataContext, title: str, metrics: List[str]) -> None:
    recs = ctx.records
    items = []
    if title == "Delivery Performance" and "Delivery_Delay_Days" in recs.columns:
        on_time = float((recs["Delivery_Delay_Days"] <= 0).mean() * 100)
        delayed = int((recs["Delivery_Delay_Days"] > 0).sum())
        items += [
            {"label": "On-time delivery", "value": f"{on_time:.1f}%", "color": config.RISK_COLORS["LOW"] if on_time >= 80 else config.RISK_COLORS["MEDIUM"]},
            {"label": "Delayed deliveries", "value": f"{delayed:,}", "sub": f"of {len(recs):,} records"},
            {"label": "Average delay", "value": f"{recs['Delivery_Delay_Days'].mean():.2f} days"},
        ]
        if "Lead_Time_Days" in recs.columns:
            items.append({"label": "Average lead time", "value": f"{recs['Lead_Time_Days'].mean():.1f} days"})
    elif title == "Quality Performance" and "Defect_Rate" in recs.columns:
        items += [{"label": "Average defect rate", "value": f"{recs['Defect_Rate'].mean():.2f}%"}]
        if "Quality_Score" in recs.columns:
            items.append({"label": "Average quality score", "value": f"{recs['Quality_Score'].mean():.1f}"})
        rej = recs["Rejection_Rate"].mean() if "Rejection_Rate" in recs.columns else float((recs["Defect_Rate"] > recs["Defect_Rate"].quantile(0.9)).mean() * 100)
        items.append({"label": "Rejection rate" if "Rejection_Rate" in recs.columns else "Lots above P90 defects", "value": f"{rej:.1f}%"})
    else:
        for m in metrics:
            if m in recs.columns:
                items.append({"label": feature_label(m), "value": f"{ui.fmt_num(recs[m].mean(), 2)} {feature_unit(m)}".strip(), "sub": f"min {ui.fmt_num(recs[m].min(), 2)} · max {ui.fmt_num(recs[m].max(), 2)}"})
    if items:
        ui.kpi_row(items[:5])


def _metric_section(ctx: DataContext, title: str, metrics: List[str], key: str) -> None:
    avail = [m for m in metrics if m in ctx.records.columns]
    if not avail:
        st.info(f"No {title.lower()} columns were found in the current data.")
        return
    _section_kpis(ctx, title, avail)
    metric = st.selectbox("Metric", avail, format_func=feature_label, key=f"{key}_metric")
    c1, c2 = st.columns(2)
    with c1:
        worst = supplier_metric_ranking(ctx.suppliers, metric, n=15, worst=True)
        if not worst.empty:
            fig = ui.bar(worst.sort_values(metric), metric, "Supplier_ID", f"Suppliers with weakest {feature_label(metric).lower()}",
                         color="Risk_Level", color_map=config.RISK_COLORS, orientation="h", height=420, xlabel=f"{feature_label(metric)} ({feature_unit(metric)})", ylabel="Supplier")
            with ui.card():
                ui.show(fig, key=f"{key}_worst")
    with c2:
        trend = metric_over_time(ctx.records, metric)
        if not trend.empty and len(trend) > 1:
            fig = ui.line(trend, "Period", metric, f"{feature_label(metric)} trend", colors=[config.PRIMARY], height=420, ylabel=f"{feature_label(metric)} ({feature_unit(metric)})")
            with ui.card():
                ui.show(fig, key=f"{key}_trend")
        else:
            st.info("Not enough time points for a trend chart.")
    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(ctx.records, x="Risk_Level", y=metric, color="Risk_Level", color_discrete_map=config.RISK_COLORS,
                     category_orders={"Risk_Level": config.RISK_ORDER}, points=False)
        fig.update_layout(title=f"{feature_label(metric)} by risk level", xaxis_title="Risk level", yaxis_title=feature_label(metric), showlegend=False)
        with ui.card():
            ui.show(ui.style(fig, 360), key=f"{key}_box")
    with c4:
        other = [m for m in ctx.profile_cfg("features") if m in ctx.suppliers.columns and m != metric]
        if other:
            y = st.selectbox("Compare against", other, format_func=feature_label, key=f"{key}_y")
            hover = ["Supplier_Name", "Risk_Score"]
            fig = ui.scatter(ctx.suppliers, metric, y, "Risk_Level", f"{feature_label(metric)} vs {feature_label(y)} (supplier level)",
                             hover=hover, color_map=config.RISK_COLORS, height=330)
            with ui.card():
                ui.show(fig, key=f"{key}_scatter")


def _news_section(ctx: DataContext, key: str) -> None:
    recs = ctx.records
    if "Synthetic_News_Sentiment" not in recs.columns and "News_Risk_Probability" not in recs.columns:
        st.info("No news sentiment columns in the current data.")
        return
    if "Synthetic_News_Sentiment" in recs.columns:
        g = recs.groupby("Synthetic_News_Sentiment").agg(Records=("Risk_Score", "size"), Avg_Risk_Score=("Risk_Score", "mean")).reset_index()
        c1, c2 = st.columns(2)
        with c1:
            fig = ui.bar(g, "Synthetic_News_Sentiment", "Records", "News sentiment distribution", xlabel="Sentiment", ylabel="Records")
            with ui.card():
                ui.show(fig, key=f"{key}_dist")
        with c2:
            fig = ui.bar(g, "Synthetic_News_Sentiment", "Avg_Risk_Score", "Average risk score by sentiment", color="Avg_Risk_Score", text=True, xlabel="Sentiment", ylabel="Avg risk score")
            with ui.card():
                ui.show(fig, key=f"{key}_risk")
    if "News_Risk_Probability" in recs.columns:
        trend = metric_over_time(recs, "News_Risk_Probability")
        if len(trend) > 1:
            with ui.card():
                ui.show(ui.line(trend, "Period", "News_Risk_Probability", "News risk over time", colors=[config.CATEGORICAL[2]], ylabel="News risk probability"), key=f"{key}_trend")


def _segmentation(ctx: DataContext) -> None:
    seg = ctx.segmentation
    if seg is None:
        st.info("Segmentation needs at least two clustering features and a handful of suppliers.")
        return
    st.caption(seg.message + ". Labels are derived from each cluster's standardised centre (largest deviation in the risk direction).")
    sups = ctx.suppliers
    counts = sups["Cluster_Label"].value_counts().reset_index()
    counts.columns = ["Cluster_Label", "Suppliers"]
    ui.kpi_row([{"label": row["Cluster_Label"], "value": int(row["Suppliers"]), "color": config.CATEGORICAL[i % len(config.CATEGORICAL)]} for i, row in counts.iterrows()][:6])
    c1, c2 = st.columns([2, 3])
    with c1:
        fig = ui.bar(counts.sort_values("Suppliers"), "Suppliers", "Cluster_Label", "Cluster distribution", orientation="h", xlabel="Suppliers", ylabel="", height=380)
        with ui.card():
            ui.show(fig, key="seg_dist")
    with c2:
        feats = [f for f in seg.features if f in sups.columns]
        x = st.selectbox("X axis", feats, index=0, format_func=feature_label, key="seg_x")
        y = st.selectbox("Y axis", feats, index=min(1, len(feats) - 1), format_func=feature_label, key="seg_y")
        fig = ui.scatter(sups, x, y, "Cluster_Label", "Supplier segmentation (K-Means)", hover=["Supplier_ID", "Supplier_Name", "Risk_Score", "Risk_Level"], size="Risk_Score", height=440, legend_bottom=True)
        with ui.card():
            ui.show(fig, key="seg_scatter")
    ui.section("Cluster summary", "Mean feature values per cluster (supplier level)")
    summary = seg.summary.copy()
    summary = summary[["Cluster", "Cluster_Label", "Suppliers"] + [c for c in summary.columns if c not in ("Cluster", "Cluster_Label", "Suppliers")]]
    ui.dataframe(summary, key="seg_summary")


def render(ctx: DataContext) -> None:
    ui.page_title("📊 SUPPLIER ANALYTICS", "Performance · cost · segmentation", "Delivery, quality, lead-time, fulfilment and cost analytics with K-Means supplier segmentation")
    if ctx.is_live:
        ui.simulated_notice()
    if not ctx.ready:
        empty_state(ctx)
        return
    f = ctx.filters
    chips = []
    if f.get("levels") and set(f["levels"]) != set(config.RISK_ORDER):
        chips.append("risk level: " + ", ".join(f["levels"]))
    if f.get("score") and tuple(f["score"]) != (0, 100):
        chips.append(f"risk score {f['score'][0]}-{f['score'][1]}")
    if f.get("suppliers"):
        chips.append(f"{len(f['suppliers'])} supplier(s)")
    if f.get("clusters"):
        chips.append("cluster: " + ", ".join(f["clusters"]))
    st.caption(("Active filters · " + " · ".join(chips)) if chips else "Filters (supplier, risk level, location, cluster, date range) are in the sidebar and apply to every chart and table on this page.")
    sections = config.ANALYTICS_SECTIONS.get(ctx.profile or "", [])
    if not sections:
        numeric = ctx.profile_cfg("features")
        sections = [("Metrics", numeric)]
    tab_names = [s[0] for s in sections] + ["Segmentation", "Correlation"]
    tabs = st.tabs(tab_names)
    for tab, (title, metrics) in zip(tabs, sections):
        with tab:
            if title == "News Sentiment":
                _news_section(ctx, "news")
            else:
                _metric_section(ctx, title, metrics, key=title.replace(" ", "_").lower())
    with tabs[-2]:
        _segmentation(ctx)
    with tabs[-1]:
        corr = correlation_matrix(ctx.records, ctx.profile_cfg("features"))
        if corr.empty:
            st.info("Not enough numeric features for a correlation matrix.")
        else:
            with ui.card():
                ui.show(ui.heatmap(corr, "Feature correlation (including risk score)"), key="corr")
