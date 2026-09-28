"""🏠 Overview page."""
from __future__ import annotations

import streamlit as st

from src import config, ui
from src.analytics import group_risk, kpis
from src.pages.common import data_quality, empty_state, model_banner, risk_trend_block
from src.state import DataContext

OVERVIEW_COLUMNS = {
    "operational": ["On_Time_Delivery", "Delivery_Delay_Days", "Quality_Score", "Lead_Time_Days", "Defect_Rate"],
    "financial": ["Financial_Stress_Index", "Leverage", "Current_Ratio", "Profit_Margin", "Cash_Flow_Million_USD"],
}
STATUS = {"HIGH": "🔴 HIGH", "MEDIUM": "🟠 MEDIUM", "LOW": "🟢 LOW"}


def render(ctx: DataContext) -> None:
    ui.page_title("INTELLIGENT SUPPLIER RISK ANALYTICS", "Overview", "Monitor, predict and analyze supplier risk.")
    if ctx.is_live:
        ui.simulated_notice()
    else:
        data_quality(ctx)
    if not ctx.ready:
        empty_state(ctx)
        return
    model_banner(ctx)

    k = kpis(ctx.suppliers, ctx.records)
    ui.kpi_row([
        {"label": "Total Suppliers", "value": f"{k['total']:,}", "sub": f"{k['records']:,} records" + (" · filters active" if ctx.filters_active else "")},
        {"label": "High Risk", "value": k["high"], "color": config.RISK_COLORS["HIGH"], "sub": f"{k['high'] / max(k['total'], 1):.0%} of suppliers"},
        {"label": "Medium Risk", "value": k["medium"], "color": config.RISK_COLORS["MEDIUM"], "sub": f"{k['medium'] / max(k['total'], 1):.0%} of suppliers"},
        {"label": "Low Risk", "value": k["low"], "color": config.RISK_COLORS["LOW"], "sub": f"{k['low'] / max(k['total'], 1):.0%} of suppliers"},
        {"label": "Average Risk Score", "value": f"{k['avg_risk']:.1f}%", "sub": "0 = safest · 100 = riskiest"},
    ])

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.05, 1.2, 1.6])
    with c1:
        ui.risk_health_card(k["avg_risk"], k["high"], k["total"])
    with c2:
        with ui.card():
            ui.show(ui.donut(ctx.suppliers["Risk_Level"].value_counts().to_dict(), height=300), key="ov_donut")
    with c3:
        with ui.card():
            ui.show(ui.histogram(ctx.records["Risk_Score"], "Risk Score Distribution (all records)", height=300), key="ov_hist")

    ui.section("Risk Trend", "Average risk score over time · pick a supplier to see whether its risk is increasing, stable or decreasing")
    with ui.card():
        risk_trend_block(ctx, key="ov_trend", height=380)

    ui.section("High-Risk Suppliers", "Highest-risk suppliers first · sortable · use the search box or the sidebar filters to narrow down")
    table = ctx.suppliers.copy()
    metric_cols = [c for c in OVERVIEW_COLUMNS.get(ctx.profile or "", []) if c in table.columns]
    if not metric_cols:
        metric_cols = [c for c in table.select_dtypes("number").columns if c not in ("Risk_Score", "Avg_Risk_Score", "Max_Risk_Score", "Records", "Cluster")][:4]
    table["Status"] = table["Risk_Level"].map(STATUS)
    cols = ["Supplier_ID", "Supplier_Name", "Risk_Score", "Risk_Level"] + metric_cols + ["Status"]
    if "Cluster_Label" in table.columns:
        cols.append("Cluster_Label")
    for c in ("Country", "Industry", "Records", "Last_Seen"):
        if c in table.columns:
            cols.append(c)
    f1, f2 = st.columns([3, 1.4])
    search = f1.text_input("Search supplier", placeholder="Search by supplier ID or name…", key="ov_search", label_visibility="collapsed")
    scope = f2.selectbox("Show", ["All suppliers", "High risk only", "High + Medium"], key="ov_scope", label_visibility="collapsed")
    if search:
        m = table["Supplier_ID"].str.contains(search, case=False, na=False) | table["Supplier_Name"].str.contains(search, case=False, na=False)
        table = table[m]
    if scope == "High risk only":
        table = table[table["Risk_Level"] == "HIGH"]
    elif scope == "High + Medium":
        table = table[table["Risk_Level"].isin(["HIGH", "MEDIUM"])]
    table = table.sort_values("Risk_Score", ascending=False)
    ui.dataframe(table[cols], height=420, key="ov_table")
    st.download_button("Download supplier risk table (CSV)", table[cols].to_csv(index=False).encode("utf-8"), "supplier_risk_overview.csv", "text/csv")

    groups = [c for c in ("Country", "Industry") if c in ctx.suppliers.columns]
    if groups:
        ui.section("Risk by " + " and ".join(groups))
        cols_ = st.columns(len(groups))
        for col, by in zip(cols_, groups):
            g = group_risk(ctx.suppliers, by)
            if g.empty:
                continue
            with col:
                with ui.card():
                    ui.show(ui.bar(g, by, "Avg_Risk_Score", f"Average risk by {by.lower()}", color="Avg_Risk_Score", text=True, ylabel="Avg risk score", xlabel=by, height=320), key=f"ov_group_{by}")
