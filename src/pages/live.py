"""📈 Live Monitoring page."""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, ui
from src.analytics import kpis
from src.state import DataContext


def controls(sim, key: str, compact: bool = False) -> None:
    """Start / Pause / Reset buttons. ``compact`` uses two rows (fits the sidebar)."""
    if compact:
        c1, c2 = st.columns(2)
        c3 = st
    else:
        c1, c2, c3 = st.columns(3)
    if c1.button("▶ START", width="stretch", key=f"{key}_start", type="primary", disabled=sim.running):
        sim.start()
        st.rerun()
    if c2.button("⏸ PAUSE", width="stretch", key=f"{key}_pause", disabled=not sim.running):
        sim.pause()
        st.rerun()
    if c3.button("↻ RESET", width="stretch", key=f"{key}_reset"):
        sim.reset()
        st.rerun()


def _active_alerts(alerts: pd.DataFrame, minutes: int = 2) -> pd.DataFrame:
    if alerts.empty or "Time" not in alerts.columns:
        return alerts
    cutoff = datetime.now() - timedelta(minutes=minutes)
    return alerts[pd.to_datetime(alerts["Time"]) >= cutoff]


def render(ctx: DataContext) -> None:
    ui.page_title("LIVE MONITORING", "Live supplier monitor", "Real-time supplier risk stream scored by the XGBoost operational model")
    ui.simulated_notice()
    if not ctx.is_live:
        st.info("Live Monitoring uses the built-in supplier simulator. Switch the data source to **🔴 Live Data** to start streaming.")
        if st.button("🔴 Switch to Live Data Mode", type="primary"):
            st.session_state["pending_mode"] = "Live Data"
            st.rerun()
        return

    sim = ctx.sim
    status_kind = "red" if sim.running else "neutral"
    up = int(time.time() - sim.started_at) if sim.started_at else 0
    uptime = f"{up // 60:02d}:{up % 60:02d}"
    alerts = sim.alerts_frame()
    active = _active_alerts(alerts)
    active_anom = int((active["Alert_Type"] == "ANOMALY DETECTED").sum()) if not active.empty else 0
    st.markdown(
        f'<div class="srp-card" style="display:flex;gap:28px;align-items:center;flex-wrap:wrap">'
        f'<div><div class="kpi-label">Simulation status</div><div style="margin-top:4px">{ui.badge(("● SIMULATION " if sim.running else "○ SIMULATION ") + sim.status, status_kind)}</div></div>'
        f'<div><div class="kpi-label">Data rate</div><div style="font-weight:700">{sim.rate} rec/sec</div></div>'
        f'<div><div class="kpi-label">Records generated</div><div style="font-weight:700">{sim.total_generated:,}</div></div>'
        f'<div><div class="kpi-label">Suppliers</div><div style="font-weight:700">{len(sim.suppliers)}</div></div>'
        f'<div><div class="kpi-label">Uptime</div><div style="font-weight:700">{uptime}</div></div>'
        f'<div><div class="kpi-label">Buffer</div><div style="font-weight:700">{len(sim.records):,} / {config.LIVE_MAX_RECORDS:,}</div></div>'
        f'<div><div class="kpi-label">Last update</div><div style="font-weight:700">{time.strftime("%H:%M:%S")}</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([2, 1.5])
    with c1:
        controls(sim, "lv")
    with c2:
        rate = st.segmented_control("Data rate (records/sec)", config.LIVE_RATE_OPTIONS, default=sim.rate if sim.rate in config.LIVE_RATE_OPTIONS else 5,
                                    format_func=lambda r: f"{r}/s", key="lv_rate", label_visibility="collapsed")
        st.caption("Data generation rate · records per second")
        if rate and rate != sim.rate:
            sim.rate = int(rate)
    if sim.last_error:
        st.error(sim.last_error)
    if not ctx.ready:
        st.info("Press ▶ START to begin generating simulated supplier records. Charts appear after the first records arrive.")
        return

    k = kpis(ctx.suppliers, ctx.records)
    ui.kpi_row([
        {"label": "Live Suppliers", "value": k["total"]},
        {"label": "Transactions", "value": f"{sim.total_generated:,}", "sub": f"{len(ctx.records):,} in window"},
        {"label": "High Risk", "value": k["high"], "color": config.RISK_COLORS["HIGH"]},
        {"label": "Medium Risk", "value": k["medium"], "color": config.RISK_COLORS["MEDIUM"]},
        {"label": "Low Risk", "value": k["low"], "color": config.RISK_COLORS["LOW"]},
        {"label": "Average Risk", "value": f"{k['avg_risk']:.1f}%"},
        {"label": "Active Alerts", "value": int(len(active)), "color": config.SEVERITY_COLORS["Warning"], "sub": f"{active_anom} anomalies · last 2 min"},
    ])

    hist = sim.history_frame()
    ui.section("Live Risk Trend", "Bounded history · portfolio average and per-batch average risk")
    c1, c2 = st.columns(2)
    with c1:
        with ui.card():
            if len(hist) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist["Time"], y=hist["Avg_Risk_Score"], name="Batch average", mode="lines", line=dict(color="#60A5FA", width=1.2, dash="dot")))
                fig.add_trace(go.Scatter(x=hist["Time"], y=hist["Portfolio_Avg_Risk"], name="Portfolio average", mode="lines+markers",
                                         line=dict(color=config.PRIMARY, width=2.6), marker=dict(size=4), fill="tozeroy", fillcolor="rgba(59,130,246,0.12)"))
                fig.update_layout(title="Risk Score over time", yaxis_title="Risk score", xaxis_title="")
                fig.update_yaxes(range=[0, 100])
                ui.risk_bands(fig)
                ui.show(ui.style(fig, 340), key="lv_risk_trend")
            else:
                st.info("Collecting the first data points…")
    with c2:
        with ui.card():
            if len(hist) > 1:
                fig = go.Figure()
                for col, lvl in (("High", "HIGH"), ("Medium", "MEDIUM"), ("Low", "LOW")):
                    fig.add_trace(go.Scatter(x=hist["Time"], y=hist[col], name=f"{col} risk", mode="lines", stackgroup="one",
                                             line=dict(color=config.RISK_COLORS[lvl], width=1.5), fillcolor=config.RISK_SOFT[lvl]))
                fig.update_layout(title="Suppliers by risk level over time", yaxis_title="Suppliers", xaxis_title="")
                ui.show(ui.style(fig, 340), key="lv_counts")
            else:
                st.info("Collecting the first data points…")

    c1, c2 = st.columns([3, 2])
    with c1:
        ui.section("Latest Events", "Generated dynamically from model scores and Isolation Forest on the stream")
        ui.alert_feed(alerts, limit=12)
    with c2:
        ui.section("Highest-risk suppliers now")
        top = ctx.suppliers.head(10).sort_values("Risk_Score")
        fig = ui.bar(top, "Risk_Score", "Supplier_ID", "", color="Risk_Level", color_map=config.RISK_COLORS, orientation="h", height=360, xlabel="Risk score", ylabel="")
        fig.update_layout(margin=dict(t=10))
        with ui.card():
            ui.show(fig, key="lv_top")

    ui.section("Recent Supplier Records", "Most recent simulated transactions with their model scores")
    cols = ["Date", "Supplier_ID", "Supplier_Name", "Risk_Score", "Risk_Level", "Delivery_Delay_Days", "Defect_Rate", "Lead_Time_Days", "Order_Quantity", "Order_Value", "Quality_Score", "Cost_Variation", "Order_Fulfillment_Rate", "Performance_Score"]
    recent = ctx.records.sort_values("Date", ascending=False).head(20)
    ui.dataframe(recent[[c for c in cols if c in recent.columns]], height=380, key="lv_recent")
