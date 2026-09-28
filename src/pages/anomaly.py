"""🚨 Anomaly & Alerts page (Isolation Forest + early-warning system)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, ui
from src.analytics import early_warnings
from src.anomaly_detection import anomaly_counts_by_metric, build_alert_feed, detect_anomalies
from src.pages.common import empty_state, risk_trend_block
from src.schema import feature_label
from src.state import DataContext


def render(ctx: DataContext) -> None:
    ui.page_title("🚨 ANOMALY & ALERTS", "Isolation Forest · early warning", "Unusual delivery, quality, cost and order behaviour flagged against the portfolio's normal ranges")
    if ctx.is_live:
        ui.simulated_notice()
    if not ctx.ready:
        empty_state(ctx)
        return

    feats = [f for f in ctx.profile_cfg("anomaly_features") if f in ctx.records.columns]
    c1, c2 = st.columns([1, 3])
    contamination = c1.slider("Expected anomaly share", 1, 10, 3, format="%d%%", key="an_cont") / 100.0
    recs = ctx.records.tail(3000) if ctx.is_live else ctx.records
    res = detect_anomalies(recs, feats, contamination)
    warnings = early_warnings(ctx.records, ctx.profile_cfg("features"))
    if res is None:
        st.warning("Anomaly detection needs at least two numeric feature columns and 20 records. Nothing to analyse in the current data.")
        return
    c2.caption(f"{res.message} · {len(recs):,} records analysed · features: " + ", ".join(feature_label(f) for f in res.features))

    warn_n = int((res.alerts["Severity"] == "Warning").sum()) if not res.alerts.empty else 0
    n_normal = int(recs["Supplier_ID"].nunique() - (res.alerts["Supplier_ID"].nunique() if not res.alerts.empty else 0))
    ui.kpi_row([
        {"label": "Total Anomalies", "value": res.total, "sub": f"{res.total / max(len(recs), 1):.1%} of records"},
        {"label": "Critical Alerts", "value": res.critical, "color": config.SEVERITY_COLORS["Critical"]},
        {"label": "Warnings", "value": warn_n, "color": config.SEVERITY_COLORS["Warning"]},
        {"label": "Suppliers Under Observation", "value": res.suppliers_under_observation, "color": config.SEVERITY_COLORS["Attention"], "sub": "flagged without a critical alert"},
        {"label": "Risk Increasing", "value": int(len(warnings)), "color": config.RISK_COLORS["MEDIUM"], "sub": "early-warning suppliers"},
        {"label": "Normal", "value": n_normal, "color": config.SEVERITY_COLORS["Normal"], "sub": "suppliers with no anomaly"},
    ])

    # ---------------------------------------------------------------- feed
    if ctx.is_live and ctx.sim is not None:
        feed = ctx.sim.alerts_frame()
        feed_sub = "Generated from the simulation stream · HIGH RISK · RISK INCREASING · ANOMALY DETECTED"
    else:
        feed = build_alert_feed(ctx.suppliers, res.alerts, warnings, ctx.records, ctx.profile_cfg("features"))
        feed_sub = "Derived from model scores (HIGH RISK), the early-warning system (RISK INCREASING) and Isolation Forest (ANOMALY DETECTED)"
    ui.section("Latest Alerts", feed_sub)
    c1, c2 = st.columns([3, 2])
    with c1:
        ui.alert_feed(feed, limit=14, empty_text="No alerts for the current selection.")
    with c2:
        if not res.alerts.empty:
            counts = anomaly_counts_by_metric(res.alerts)
            fig = ui.bar(counts, "Count", "Alert_Type", "Anomalies by type and severity", color="Severity", color_map=config.SEVERITY_COLORS, orientation="h", xlabel="Alerts", ylabel="", height=380)
            fig.update_layout(barmode="stack", yaxis=dict(categoryorder="total ascending"))
            with ui.card():
                ui.show(fig, key="an_types")

    # ------------------------------------------------------- early warning
    ui.section("Early Warning System", "Suppliers whose risk rose by 10+ points between their previous and most recent records, with the metrics that moved most")
    if warnings.empty:
        st.markdown(f'<div class="srp-card" style="color:{config.MUTED}">No supplier shows a significant risk increase in the current data window.</div>', unsafe_allow_html=True)
    else:
        top = warnings.head(6)
        cols = st.columns(2)
        for i, (_, row) in enumerate(top.iterrows()):
            with cols[i % 2]:
                ui.warning_card(row)
        with st.expander(f"All risk-increasing suppliers ({len(warnings)})"):
            show = ["Supplier_ID", "Supplier_Name", "Previous_Risk", "Current_Risk", "Change", "Reasons"]
            if "Last_Seen" in warnings.columns:
                show.append("Last_Seen")
            ui.dataframe(warnings[show], height=300, key="an_warn_table")

    # ----------------------------------------------------------- anomalies
    ui.section("Anomaly Table", "Alert type = metric with the largest robust deviation · severity from deviation size (≥3σ warning, ≥4.5σ critical)")
    if res.alerts.empty:
        st.success("No anomalies detected at this sensitivity.")
    else:
        sev_filter = st.multiselect("Severity", ["Critical", "Warning", "Attention"], default=["Critical", "Warning", "Attention"], key="an_sev")
        alerts = res.alerts[res.alerts["Severity"].isin(sev_filter)].copy()
        if "Date" in alerts.columns and alerts["Date"].notna().any():
            alerts["Timestamp"] = pd.to_datetime(alerts["Date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        cols_ = ["Supplier_ID", "Supplier_Name", "Alert_Type", "Severity", "Detected", "Normal_Range"] + (["Timestamp"] if "Timestamp" in alerts.columns else []) + ["Status", "Risk_Level", "Risk_Score"]
        ui.dataframe(alerts[cols_], height=380, key="an_table")
        d1, d2 = st.columns([1, 3])
        d1.download_button("Download alerts (CSV)", alerts[cols_].to_csv(index=False).encode("utf-8"), "supplier_alerts.csv", "text/csv")
        per_sup = alerts.groupby(["Supplier_ID", "Severity"]).size().rename("Alerts").reset_index()
        top_ids = per_sup.groupby("Supplier_ID")["Alerts"].sum().nlargest(12).index
        per_sup = per_sup[per_sup["Supplier_ID"].isin(top_ids)]
        if not per_sup.empty:
            fig = ui.bar(per_sup, "Supplier_ID", "Alerts", "Suppliers with most anomalies", color="Severity", color_map=config.SEVERITY_COLORS, xlabel="Supplier", ylabel="Alerts", height=320)
            fig.update_layout(barmode="stack")
            with ui.card():
                ui.show(fig, key="an_sups")

    ui.section("Anomaly Map", "Records flagged by Isolation Forest against two chosen metrics")
    c1, c2 = st.columns(2)
    x = c1.selectbox("X axis", res.features, index=0, format_func=feature_label, key="an_x")
    y = c2.selectbox("Y axis", res.features, index=min(1, len(res.features) - 1), format_func=feature_label, key="an_y")
    plot = res.records.copy()
    plot["Status"] = plot["Is_Anomaly"].map({True: "Anomaly", False: "Normal"})
    if len(plot) > 4000:
        plot = pd.concat([plot[plot["Is_Anomaly"]], plot[~plot["Is_Anomaly"]].sample(4000 - int(plot["Is_Anomaly"].sum()), random_state=1)])
    fig = ui.scatter(plot, x, y, "Status", "Isolation Forest anomalies", hover=["Supplier_ID", "Risk_Score", "Risk_Level"],
                     color_map={"Anomaly": config.RISK_COLORS["HIGH"], "Normal": "#3B82F6"}, height=420)
    with ui.card():
        ui.show(fig, key="an_scatter")

    ui.section("Risk Trend", "Risk score over time with supplier and date filters")
    with ui.card():
        risk_trend_block(ctx, key="an_trend")
