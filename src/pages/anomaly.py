"""🚨 Anomaly & Alerts page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, ui
from src.anomaly_detection import anomaly_counts_by_metric, detect_anomalies
from src.pages.common import empty_state, risk_trend_block
from src.schema import feature_label
from src.state import DataContext


def render(ctx: DataContext) -> None:
    ui.page_title("🚨 Anomaly & Alerts", "Isolation Forest anomaly detection", "Unusual delivery, quality, cost and order behaviour flagged against the portfolio's normal ranges")
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
    if res is None:
        st.warning("Anomaly detection needs at least two numeric feature columns and 20 records. Nothing to analyse in the current data.")
        return
    c2.caption(f"{res.message} · {len(recs):,} records analysed · features: " + ", ".join(feature_label(f) for f in res.features))

    warn = int((res.alerts["Severity"] == "Warning").sum()) if not res.alerts.empty else 0
    ui.kpi_row([
        {"label": "Total Anomalies", "value": res.total, "sub": f"{res.total / max(len(recs), 1):.1%} of records"},
        {"label": "Critical Alerts", "value": res.critical, "color": config.SEVERITY_COLORS["Critical"]},
        {"label": "Warnings", "value": warn, "color": config.SEVERITY_COLORS["Warning"]},
        {"label": "Suppliers Under Observation", "value": res.suppliers_under_observation, "color": config.SEVERITY_COLORS["Attention"], "sub": "flagged without a critical alert"},
        {"label": "Normal", "value": int(recs["Supplier_ID"].nunique() - (res.alerts["Supplier_ID"].nunique() if not res.alerts.empty else 0)), "color": config.SEVERITY_COLORS["Normal"], "sub": "suppliers with no anomaly"},
    ])

    if res.alerts.empty:
        st.success("No anomalies detected at this sensitivity.")
    else:
        ui.section("Alerts", "Alert type = metric with the largest robust deviation · severity from deviation size (≥3σ warning, ≥4.5σ critical)")
        sev_filter = st.multiselect("Severity", ["Critical", "Warning", "Attention"], default=["Critical", "Warning", "Attention"], key="an_sev")
        alerts = res.alerts[res.alerts["Severity"].isin(sev_filter)]
        cols = ["Supplier_ID", "Supplier_Name", "Alert_Type", "Severity", "Detected", "Normal_Range", "Status", "Risk_Level", "Risk_Score"]
        if "Date" in alerts.columns and alerts["Date"].notna().any():
            cols.append("Date")
        ui.dataframe(alerts[cols], height=380, key="an_table")
        st.download_button("Download alerts (CSV)", alerts[cols].to_csv(index=False).encode("utf-8"), "supplier_alerts.csv", "text/csv")

        c1, c2 = st.columns(2)
        with c1:
            counts = anomaly_counts_by_metric(alerts)
            if not counts.empty:
                fig = ui.bar(counts, "Alert_Type", "Count", "Alerts by type and severity", color="Severity", color_map=config.SEVERITY_COLORS, xlabel="Alert type", ylabel="Alerts")
                fig.update_layout(barmode="stack")
                ui.show(fig, key="an_types")
        with c2:
            per_sup = alerts.groupby(["Supplier_ID", "Severity"]).size().rename("Alerts").reset_index()
            top = per_sup.groupby("Supplier_ID")["Alerts"].sum().nlargest(12).index
            per_sup = per_sup[per_sup["Supplier_ID"].isin(top)]
            if not per_sup.empty:
                fig = ui.bar(per_sup, "Supplier_ID", "Alerts", "Suppliers with most alerts", color="Severity", color_map=config.SEVERITY_COLORS, xlabel="Supplier", ylabel="Alerts")
                fig.update_layout(barmode="stack")
                ui.show(fig, key="an_sups")

    ui.section("Anomaly map", "Records flagged by Isolation Forest against two chosen metrics")
    c1, c2 = st.columns(2)
    x = c1.selectbox("X axis", res.features, index=0, format_func=feature_label, key="an_x")
    y = c2.selectbox("Y axis", res.features, index=min(1, len(res.features) - 1), format_func=feature_label, key="an_y")
    plot = res.records.copy()
    plot["Status"] = plot["Is_Anomaly"].map({True: "Anomaly", False: "Normal"})
    if len(plot) > 4000:
        plot = pd.concat([plot[plot["Is_Anomaly"]], plot[~plot["Is_Anomaly"]].sample(4000 - int(plot["Is_Anomaly"].sum()), random_state=1)])
    fig = ui.scatter(plot, x, y, "Status", "Isolation Forest anomalies", hover=["Supplier_ID", "Risk_Score", "Risk_Level"],
                     color_map={"Anomaly": config.RISK_COLORS["HIGH"], "Normal": "#94A3B8"}, height=420)
    ui.show(fig, key="an_scatter")

    if ctx.is_live and ctx.sim is not None:
        ui.section("Latest live alerts", "Generated from the simulation stream (HIGH RISK · RISK INCREASING · ANOMALY DETECTED)")
        ui.alert_feed(ctx.sim.alerts_frame(), limit=15)

    ui.section("Risk Trend", "Risk score over time with supplier and date filters")
    risk_trend_block(ctx, key="an_trend")
