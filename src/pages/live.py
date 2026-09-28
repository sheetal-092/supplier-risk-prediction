"""📈 Live Monitoring page."""
from __future__ import annotations

import time

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
    if c1.button("▶ Start", width="stretch", key=f"{key}_start", type="primary", disabled=sim.running):
        sim.start()
        st.rerun()
    if c2.button("⏸ Pause", width="stretch", key=f"{key}_pause", disabled=not sim.running):
        sim.pause()
        st.rerun()
    if c3.button("↻ Reset", width="stretch", key=f"{key}_reset"):
        sim.reset()
        st.rerun()


def render(ctx: DataContext) -> None:
    ui.page_title("📈 Live Monitoring", "Real-time supplier risk stream")
    ui.simulated_notice()
    if not ctx.is_live:
        st.info("Live Monitoring uses the built-in supplier simulator. Switch the data source to **Live Data** to start streaming.")
        if st.button("🔴 Switch to Live Data Mode", type="primary"):
            st.session_state["pending_mode"] = "Live Data"
            st.rerun()
        return

    sim = ctx.sim
    status_kind = "red" if sim.running else "neutral"
    uptime = f"{int(time.time() - sim.started_at) // 60:02d}:{int(time.time() - sim.started_at) % 60:02d}" if sim.started_at else "00:00"
    st.markdown(
        f'<div class="srp-card" style="display:flex;gap:26px;align-items:center;flex-wrap:wrap">'
        f'<div><div class="kpi-label">Simulation status</div><div style="margin-top:4px">{ui.badge(("● " if sim.running else "○ ") + sim.status, status_kind)}</div></div>'
        f'<div><div class="kpi-label">Data rate</div><div style="font-weight:600">{sim.rate} records/sec</div></div>'
        f'<div><div class="kpi-label">Records generated</div><div style="font-weight:600">{sim.total_generated:,}</div></div>'
        f'<div><div class="kpi-label">Suppliers simulated</div><div style="font-weight:600">{len(sim.suppliers)}</div></div>'
        f'<div><div class="kpi-label">Uptime</div><div style="font-weight:600">{uptime}</div></div>'
        f'<div><div class="kpi-label">Buffer</div><div style="font-weight:600">{len(sim.records):,} / {config.LIVE_MAX_RECORDS:,}</div></div>'
        f'<div><div class="kpi-label">Last update</div><div style="font-weight:600">{time.strftime("%H:%M:%S")}</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    controls(sim, "lv")
    if sim.last_error:
        st.error(sim.last_error)
    if not ctx.ready:
        st.info("Press ▶ Start to begin generating simulated supplier records. Charts appear after the first records arrive.")
        return

    k = kpis(ctx.suppliers, ctx.records)
    ui.kpi_row([
        {"label": "Total Suppliers", "value": k["total"]},
        {"label": "Total Transactions", "value": f"{sim.total_generated:,}", "sub": f"{len(ctx.records):,} in window"},
        {"label": "High Risk", "value": k["high"], "color": config.RISK_COLORS["HIGH"]},
        {"label": "Medium Risk", "value": k["medium"], "color": config.RISK_COLORS["MEDIUM"]},
        {"label": "Low Risk", "value": k["low"], "color": config.RISK_COLORS["LOW"]},
        {"label": "Average Risk Score", "value": f"{k['avg_risk']:.1f}%"},
    ])

    hist = sim.history_frame()
    c1, c2 = st.columns(2)
    with c1:
        if len(hist) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["Time"], y=hist["Avg_Risk_Score"], name="Batch average", mode="lines", line=dict(color="#93C5FD", width=1.5)))
            fig.add_trace(go.Scatter(x=hist["Time"], y=hist["Portfolio_Avg_Risk"], name="Portfolio average", mode="lines+markers", line=dict(color=config.PRIMARY, width=2.5), marker=dict(size=4)))
            fig.update_layout(title="Risk Score over time", yaxis_title="Risk score", xaxis_title="")
            fig.update_yaxes(range=[0, 100])
            ui.risk_bands(fig)
            ui.show(ui.style(fig, 340), key="lv_risk_trend")
        else:
            st.info("Collecting the first data points…")
    with c2:
        if len(hist) > 1:
            fig = go.Figure()
            for col, lvl in (("High", "HIGH"), ("Medium", "MEDIUM"), ("Low", "LOW")):
                fig.add_trace(go.Scatter(x=hist["Time"], y=hist[col], name=f"{col} risk", mode="lines", stackgroup="one",
                                         line=dict(color=config.RISK_COLORS[lvl], width=1.5), fillcolor=config.RISK_SOFT[lvl]))
            fig.update_layout(title="Suppliers by risk level over time", yaxis_title="Suppliers", xaxis_title="")
            ui.show(ui.style(fig, 340), key="lv_counts")

    c1, c2 = st.columns([3, 2])
    with c1:
        ui.section("Latest alerts", "Generated dynamically from model scores and Isolation Forest on the stream")
        ui.alert_feed(sim.alerts_frame(), limit=12)
    with c2:
        ui.section("Highest-risk suppliers now")
        top = ctx.suppliers.head(10).sort_values("Risk_Score")
        fig = ui.bar(top, "Risk_Score", "Supplier_ID", "", color="Risk_Level", color_map=config.RISK_COLORS, orientation="h", height=360, xlabel="Risk score", ylabel="")
        fig.update_layout(margin=dict(t=10))
        ui.show(fig, key="lv_top")

    ui.section("Recent supplier records", "Most recent simulated transactions with their model scores")
    cols = ["Date", "Supplier_ID", "Supplier_Name", "Risk_Score", "Risk_Level", "Delivery_Delay_Days", "Defect_Rate", "Lead_Time_Days", "Order_Quantity", "Order_Value", "Quality_Score", "Cost_Variation", "Order_Fulfillment_Rate", "Performance_Score"]
    recent = ctx.records.sort_values("Date", ascending=False).head(20)
    ui.dataframe(recent[[c for c in cols if c in recent.columns]], height=380, key="lv_recent")
