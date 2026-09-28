"""🤖 Risk Prediction page."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import config, ui
from src.pages.common import empty_state, model_banner, risk_trend_block
from src.prediction import explain_rows
from src.schema import feature_label
from src.state import DataContext


def _feature_values(ctx: DataContext, sup_rows: pd.DataFrame) -> dict:
    feats = [f for f in ctx.profile_cfg("key_features") if f in sup_rows.columns]
    if not feats:
        feats = [c for c in sup_rows.select_dtypes("number").columns if not c.startswith(("Risk_", "Model_", "Record_"))][:6]
    latest = sup_rows.tail(1).iloc[0]
    return {f: latest[f] for f in feats}


def render(ctx: DataContext) -> None:
    ui.page_title("🤖 Supplier Risk Prediction", "XGBoost risk scoring with SHAP explanations")
    if ctx.is_live:
        ui.simulated_notice()
    if not ctx.ready:
        empty_state(ctx)
        return
    model_banner(ctx)

    sups = ctx.suppliers.sort_values("Risk_Score", ascending=False)
    if sups.empty:
        st.info("No suppliers match the current filters.")
        return
    labels = {row["Supplier_ID"]: f"{row['Supplier_ID']} - {row['Supplier_Name']}  ({row['Risk_Level']}, {row['Risk_Score']:.0f}%)" for _, row in sups.iterrows()}
    ids = list(labels)
    default = st.session_state.get("pred_supplier")
    idx = ids.index(default) if default in ids else 0
    sid = st.selectbox("Select supplier", ids, index=idx, format_func=lambda s: labels[s], key="pred_supplier")
    sup = sups[sups["Supplier_ID"] == sid].iloc[0]
    rows = ctx.records[ctx.records["Supplier_ID"] == sid]
    rows = rows.sort_values("Date") if "Date" in rows.columns else rows
    latest = rows.tail(1).iloc[0]

    level = sup["Risk_Level"]
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        meta = "".join(f"<div style='font-size:13px;color:{config.MUTED};margin-top:4px'>{k}: <b style='color:{config.TEXT}'>{sup[k]}</b></div>"
                       for k in ("Industry", "Country", "Cluster_Label") if k in sup.index and pd.notna(sup[k]))
        st.markdown(
            f'<div class="srp-card"><div class="kpi-label">Supplier ID</div><div class="kpi-value" style="font-size:22px">{sup["Supplier_ID"]}</div>'
            f'<div class="kpi-label" style="margin-top:8px">Supplier Name</div><div style="font-size:16px;font-weight:600">{sup["Supplier_Name"]}</div>{meta}'
            f'<div style="margin-top:10px;font-size:12px;color:{config.MUTED}">{int(sup["Records"])} records analysed</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="srp-card" style="text-align:center"><div class="big-score-label">Risk Score</div>'
            f'<div class="big-score" style="color:{config.RISK_COLORS[level]}">{sup["Risk_Score"]:.0f}%</div>'
            f'<div style="margin:10px 0 6px 0">{ui.badge("RISK LEVEL: " + level, level)}</div>'
            f'<div class="big-score-label">Prediction probability</div><div style="font-size:20px;font-weight:600">{latest["Risk_Probability"]:.3f}</div>'
            f'<div style="font-size:11px;color:{config.MUTED}">latest record · avg {sup["Avg_Risk_Score"]:.1f}% · max {sup["Max_Risk_Score"]:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        ui.show(ui.gauge(float(sup["Risk_Score"]), "Current risk (supplier aggregate)"), key="pred_gauge")

    ui.section("Key features for this supplier", "Latest record values used by the model")
    ui.feature_cards(_feature_values(ctx, rows))
    if {"Financial_Risk_Probability", "News_Risk_Probability"} <= set(rows.columns):
        f, n = latest["Financial_Risk_Probability"], latest["News_Risk_Probability"]
        st.caption(f"Final risk = 0.6 × financial risk ({f:.3f}) + 0.4 × news risk ({n:.3f}) = {latest['Risk_Probability']:.3f}")

    ui.section("Why is this supplier risky?", "Feature contributions for the supplier's most recent records")
    bundle = ctx.scoring.bundle if ctx.scoring else None
    if bundle is None:
        st.warning("No trained model is attached to these scores (unsupervised index), so a model explanation is not available. "
                   "The chart below shows how far this supplier deviates from the portfolio median instead.")
        feats = [c for c in ctx.profile_cfg("features") if c in rows.columns]
        med, mad = ctx.records[feats].median(), (ctx.records[feats] - ctx.records[feats].median()).abs().median().replace(0, np.nan)
        dev = ((rows[feats].tail(3).mean() - med) / mad).fillna(0)
        contrib = pd.DataFrame({"Feature": feats, "Contribution": dev.values, "Value": rows[feats].tail(3).mean().values}).sort_values("Contribution", key=abs, ascending=False)
        ui.show(ui.contribution_chart(contrib, "Deviation from median (robust z-score)", "shap"), key="pred_dev")
    else:
        contrib, base, method = explain_rows(bundle, rows.tail(3))
        if contrib is None:
            st.info("Explanation not available for this model.")
        else:
            c1, c2 = st.columns([3, 2])
            with c1:
                title = "SHAP contributions (mean of last 3 records)" if method == "shap" else "Model feature importance (SHAP unavailable)"
                ui.show(ui.contribution_chart(contrib, title, method), key="pred_shap")
            with c2:
                if method == "shap":
                    pos = contrib[contrib["Contribution"] > 0].head(3)
                    neg = contrib[contrib["Contribution"] < 0].head(3)
                    st.markdown('<div class="srp-card"><h4>Interpretation</h4>' +
                                ("<b style='color:#991B1B'>Pushing risk up:</b><ul>" + "".join(f"<li>{feature_label(r.Feature)} = {ui.fmt_num(r.Value, 3)} (+{r.Contribution:.2f})</li>" for r in pos.itertuples()) + "</ul>" if not pos.empty else "") +
                                ("<b style='color:#166534'>Pulling risk down:</b><ul>" + "".join(f"<li>{feature_label(r.Feature)} = {ui.fmt_num(r.Value, 3)} ({r.Contribution:.2f})</li>" for r in neg.itertuples()) + "</ul>" if not neg.empty else "") +
                                f"<div style='font-size:12px;color:{config.MUTED}'>Values are SHAP contributions in log-odds relative to the model's base value "
                                f"({base:.3f} probability). Positive = increases predicted risk.</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="srp-card"><h4>Interpretation</h4>Global importance of each feature for the model. SHAP could not be computed for this model type.</div>', unsafe_allow_html=True)
            with st.expander("Model details"):
                st.write(f"**Algorithm:** {bundle.algorithm}")
                st.write(f"**Features ({len(bundle.features)}):** " + ", ".join(feature_label(f) for f in bundle.features))
                if bundle.metrics:
                    st.write("**Hold-out metrics:** " + " · ".join(f"{k}: {v}" for k, v in bundle.metrics.items()))
                if bundle.source:
                    st.caption(f"Trained on: {bundle.source}")

    ui.section("Risk trend for this supplier")
    risk_trend_block(ctx, key="pred_trend", default_supplier=sid, show_supplier_picker=False)

    ui.section("Recent records")
    show_cols = ["Date"] if "Date" in rows.columns else []
    show_cols += ["Risk_Score", "Risk_Level"] + [f for f in ctx.profile_cfg("features") if f in rows.columns]
    ui.dataframe(rows[show_cols].tail(15).iloc[::-1], height=320, key="pred_recent")
