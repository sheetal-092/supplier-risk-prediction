"""🏠 Overview page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, ui
from src.analytics import group_risk, kpis
from src.pages.common import empty_state, model_banner, risk_trend_block
from src.state import DataContext

OVERVIEW_COLUMNS = {
    "operational": ["On_Time_Delivery", "Delivery_Delay_Days", "Quality_Score", "Lead_Time_Days", "Defect_Rate"],
    "financial": ["Financial_Stress_Index", "Leverage", "Current_Ratio", "Profit_Margin", "Cash_Flow_Million_USD"],
}


def dataset_summary_card(ctx: DataContext) -> None:
    s = ctx.summary
    if s is None:
        return
    ui.section("📁 Dataset Mode", f"Source: <b>{ctx.source_label}</b> · detected {s.schema.profile_label.lower()} · processing engine: {ctx.report.get('engine', 'pandas')}")
    ui.kpi_row([
        {"label": "Records", "value": f"{s.records:,}"},
        {"label": "Columns", "value": s.columns},
        {"label": "Missing values", "value": f"{s.missing_cells:,}", "sub": f"{len(s.missing_by_column)} columns affected", "color": config.RISK_COLORS["MEDIUM"] if s.missing_cells else config.PRIMARY},
        {"label": "Duplicate records", "value": f"{s.duplicate_rows:,}", "color": config.RISK_COLORS["MEDIUM"] if s.duplicate_rows else config.PRIMARY},
        {"label": "Numerical columns", "value": len(s.numeric_columns)},
        {"label": "Categorical columns", "value": len(s.categorical_columns), "sub": f"{len(s.datetime_columns)} datetime"},
    ])
    with st.expander("Dataset preview, validation and preprocessing details"):
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


def render(ctx: DataContext) -> None:
    ui.page_title("Overview", "Supplier risk portfolio", config.APP_SUBTITLE)
    if ctx.is_live:
        ui.simulated_notice()
    else:
        dataset_summary_card(ctx)
    if not ctx.ready:
        empty_state(ctx)
        return
    model_banner(ctx)

    k = kpis(ctx.suppliers, ctx.records)
    ui.section("Key indicators", "Supplier-level risk based on the current aggregation setting" + (" · filters active" if ctx.filters_active else ""))
    ui.kpi_row([
        {"label": "Total Suppliers", "value": f"{k['total']:,}", "sub": f"{k['records']:,} records"},
        {"label": "High Risk", "value": k["high"], "color": config.RISK_COLORS["HIGH"], "sub": f"{k['high'] / max(k['total'], 1):.0%} of suppliers"},
        {"label": "Medium Risk", "value": k["medium"], "color": config.RISK_COLORS["MEDIUM"], "sub": f"{k['medium'] / max(k['total'], 1):.0%} of suppliers"},
        {"label": "Low Risk", "value": k["low"], "color": config.RISK_COLORS["LOW"], "sub": f"{k['low'] / max(k['total'], 1):.0%} of suppliers"},
        {"label": "Average Risk Score", "value": f"{k['avg_risk']:.1f}%", "sub": "0 = safest, 100 = riskiest"},
    ])

    c1, c2 = st.columns([2, 3])
    with c1:
        ui.show(ui.donut(ctx.suppliers["Risk_Level"].value_counts().to_dict()), key="ov_donut")
    with c2:
        ui.show(ui.histogram(ctx.records["Risk_Score"], "Risk Score Distribution (all records)"), key="ov_hist")

    groups = [c for c in ("Country", "Industry") if c in ctx.suppliers.columns]
    if groups:
        cols = st.columns(len(groups))
        for col, by in zip(cols, groups):
            g = group_risk(ctx.suppliers, by)
            if g.empty:
                continue
            fig = ui.bar(g, by, "Avg_Risk_Score", f"Average Risk by {by}", color="Avg_Risk_Score", text=True, ylabel="Avg risk score", xlabel=by)
            with col:
                ui.show(fig, key=f"ov_group_{by}")

    ui.section("Supplier Risk Overview", "Sortable table - click a column header to sort, use the search box to filter")
    table = ctx.suppliers.copy()
    metric_cols = [c for c in OVERVIEW_COLUMNS.get(ctx.profile or "", []) if c in table.columns]
    if not metric_cols:
        metric_cols = [c for c in table.select_dtypes("number").columns if c not in ("Risk_Score", "Avg_Risk_Score", "Max_Risk_Score", "Records", "Cluster")][:4]
    cols = ["Supplier_ID", "Supplier_Name", "Risk_Score", "Risk_Level"] + metric_cols + ["Risk_Status"]
    if "Cluster_Label" in table.columns:
        cols.append("Cluster_Label")
    for c in ("Country", "Industry", "Records", "Last_Seen"):
        if c in table.columns:
            cols.append(c)
    search = st.text_input("Search supplier", placeholder="Type an ID or name…", key="ov_search", label_visibility="collapsed")
    if search:
        m = table["Supplier_ID"].str.contains(search, case=False, na=False) | table["Supplier_Name"].str.contains(search, case=False, na=False)
        table = table[m]
    ui.dataframe(table[cols], height=420, key="ov_table")
    st.download_button("Download supplier risk table (CSV)", table[cols].to_csv(index=False).encode("utf-8"), "supplier_risk_overview.csv", "text/csv")

    ui.section("Risk Trend", "Average risk score over time - pick a supplier to see whether its risk is increasing, stable or decreasing")
    risk_trend_block(ctx, key="ov_trend")
