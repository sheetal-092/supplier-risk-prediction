"""Central configuration: paths, colours, thresholds and feature profiles.

The dashboard supports two *feature profiles*:

* ``financial``   - the project's original dataset (revenue, debt, leverage,
                    current ratio, Z-score, cash-flow ...) with the
                    ``Disruption_Within_90_Days`` label.  This mirrors the
                    CAPSTONE_PROJECT notebook exactly.
* ``operational`` - delivery / quality / cost style supplier data (delivery
                    delay, defect rate, lead time ...).  This is what the live
                    simulator produces and what many public supplier datasets
                    look like.

Everything else in the app (analytics, clustering, anomaly detection, SHAP)
is driven by the profile detected from the columns, so both modes share one
pipeline.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

SAMPLE_FINANCIAL_XLSX = DATA_DIR / "supplier_risk_10000rows_19columns.xlsx"
SAMPLE_FINANCIAL_CSV = DATA_DIR / "sample_supplier_data.csv"
SAMPLE_OPERATIONAL_CSV = DATA_DIR / "sample_operational_supplier_data.csv"
SAMPLE_PROCESSED_XLSX = DATA_DIR / "supplier_risk_processed_with_xgboost_finbert_lstm.xlsx"

MODEL_REGISTRY_FILE = MODELS_DIR / "model_registry.json"

APP_TITLE = "SUPPLIER RISK PREDICTION"
APP_SUBTITLE = "Intelligent Supplier Risk Prediction and Analytics"

# --------------------------------------------------------------------------- #
# Colours (dark enterprise theme)
# --------------------------------------------------------------------------- #
PRIMARY = "#3B82F6"
PRIMARY_DARK = "#2563EB"
PRIMARY_LIGHT = "#60A5FA"
PRIMARY_SOFT = "rgba(59,130,246,0.16)"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
BORDER = "#1E3A5F"
CARD_BG = "#111C30"
CARD_BG_2 = "#0F172A"
PAGE_BG = "#0B1220"
SIDEBAR_BG = "#0F172A"
GRID = "rgba(30,58,95,0.55)"

RISK_COLORS = {
    "LOW": "#22C55E",
    "MEDIUM": "#F59E0B",
    "HIGH": "#EF4444",
}
RISK_SOFT = {
    "LOW": "rgba(34,197,94,0.16)",
    "MEDIUM": "rgba(245,158,11,0.16)",
    "HIGH": "rgba(239,68,68,0.16)",
}
RISK_ORDER = ["LOW", "MEDIUM", "HIGH"]

SEVERITY_ICONS = {
    "Critical": "🔴",
    "Warning": "🟠",
    "Attention": "🟡",
    "Normal": "🟢",
}
SEVERITY_COLORS = {
    "Critical": "#EF4444",
    "Warning": "#F97316",
    "Attention": "#EAB308",
    "Normal": "#22C55E",
}

# A sequential blue palette for neutral charts
BLUES = ["#1E40AF", "#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE"]
CATEGORICAL = ["#3B82F6", "#22D3EE", "#A78BFA", "#2DD4BF", "#FBBF24", "#F472B6", "#94A3B8", "#A3E635"]

# --------------------------------------------------------------------------- #
# Risk thresholds - identical to the notebook (Final_Risk_Score based)
# --------------------------------------------------------------------------- #
RISK_LOW_MAX = 0.33      # score < 0.33  -> LOW
RISK_MEDIUM_MAX = 0.66   # score < 0.66  -> MEDIUM, otherwise HIGH

# Weight used in the notebook to fuse financial + news risk
FINANCIAL_WEIGHT = 0.60
NEWS_WEIGHT = 0.40

# XGBoost hyper-parameters (copied from the notebook so on-the-fly training
# behaves the same as the reference model)
XGB_PARAMS = dict(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
)

# --------------------------------------------------------------------------- #
# Feature profiles
# --------------------------------------------------------------------------- #
FINANCIAL_FEATURES = [
    "Revenue_Million_USD",
    "Debt_Million_USD",
    "Assets_Million_USD",
    "Current_Assets_Million_USD",
    "Current_Liabilities_Million_USD",
    "Leverage",
    "Current_Ratio",
    "Financial_Stability_ZScore",
    "Profit_Margin",
    "Cash_Flow_Million_USD",
]
FINANCIAL_ENGINEERED = ["Financial_Stress_Index"]
FINANCIAL_MODEL_FEATURES = FINANCIAL_FEATURES + FINANCIAL_ENGINEERED
FINANCIAL_TARGET = "Disruption_Within_90_Days"

OPERATIONAL_FEATURES = [
    "Delivery_Delay_Days",
    "Defect_Rate",
    "Lead_Time_Days",
    "Order_Quantity",
    "Order_Value",
    "Quality_Score",
    "Cost_Variation",
    "Order_Fulfillment_Rate",
    "Performance_Score",
]
OPERATIONAL_TARGET = "Risk_Event"

PROFILES = {
    "financial": {
        "label": "Financial risk profile",
        "features": FINANCIAL_MODEL_FEATURES,
        "raw_features": FINANCIAL_FEATURES,
        "target": FINANCIAL_TARGET,
        "model_key": "xgboost_financial",
        # features shown as "important features" on the prediction page
        "key_features": [
            "Leverage",
            "Current_Ratio",
            "Financial_Stability_ZScore",
            "Profit_Margin",
            "Cash_Flow_Million_USD",
            "Financial_Stress_Index",
        ],
        # features used for K-Means segmentation (supplier level)
        "cluster_features": [
            "Leverage",
            "Current_Ratio",
            "Profit_Margin",
            "Cash_Flow_Million_USD",
            "Financial_Stress_Index",
        ],
        # features for anomaly detection (record level)
        "anomaly_features": [
            "Leverage",
            "Current_Ratio",
            "Financial_Stability_ZScore",
            "Profit_Margin",
            "Cash_Flow_Million_USD",
            "Debt_Million_USD",
            "Revenue_Million_USD",
        ],
    },
    "operational": {
        "label": "Operational risk profile",
        "features": OPERATIONAL_FEATURES,
        "raw_features": OPERATIONAL_FEATURES,
        "target": OPERATIONAL_TARGET,
        "model_key": "xgboost_operational",
        "key_features": [
            "Delivery_Delay_Days",
            "Defect_Rate",
            "Lead_Time_Days",
            "Cost_Variation",
            "Quality_Score",
            "Order_Fulfillment_Rate",
            "Performance_Score",
        ],
        "cluster_features": [
            "Delivery_Delay_Days",
            "Defect_Rate",
            "Lead_Time_Days",
            "Cost_Variation",
            "Quality_Score",
            "Order_Value",
        ],
        "anomaly_features": [
            "Delivery_Delay_Days",
            "Defect_Rate",
            "Lead_Time_Days",
            "Cost_Variation",
            "Order_Quantity",
            "Order_Value",
            "Quality_Score",
            "Order_Fulfillment_Rate",
        ],
    },
}

# Human friendly labels + units for canonical columns
FEATURE_LABELS = {
    "Delivery_Delay_Days": ("Delivery Delay", "days"),
    "Defect_Rate": ("Defect Rate", "%"),
    "Lead_Time_Days": ("Lead Time", "days"),
    "Order_Quantity": ("Order Quantity", "units"),
    "Order_Value": ("Order Value", "USD"),
    "Quality_Score": ("Quality Score", "/100"),
    "Cost_Variation": ("Cost Variation", "%"),
    "Order_Fulfillment_Rate": ("Order Fulfillment", "%"),
    "Performance_Score": ("Performance Score", "/100"),
    "On_Time_Delivery": ("On-time Delivery", "%"),
    "Rejection_Rate": ("Rejection Rate", "%"),
    "Unit_Price": ("Unit Price", "USD"),
    "Revenue_Million_USD": ("Revenue", "M USD"),
    "Debt_Million_USD": ("Debt", "M USD"),
    "Assets_Million_USD": ("Assets", "M USD"),
    "Current_Assets_Million_USD": ("Current Assets", "M USD"),
    "Current_Liabilities_Million_USD": ("Current Liabilities", "M USD"),
    "Leverage": ("Leverage", "ratio"),
    "Current_Ratio": ("Current Ratio", "ratio"),
    "Financial_Stability_ZScore": ("Stability Z-Score", "score"),
    "Profit_Margin": ("Profit Margin", "ratio"),
    "Cash_Flow_Million_USD": ("Cash Flow", "M USD"),
    "Financial_Stress_Index": ("Financial Stress Index", "index"),
    "News_Risk_Probability": ("News Risk", "prob"),
    "FinBERT_Sentiment_Score": ("News Sentiment", "score"),
    "Risk_Score": ("Risk Score", "%"),
    "Risk_Probability": ("Risk Probability", "prob"),
}

# Direction in which a feature is "bad" (used for anomaly alert wording and
# cluster labelling).  +1 = higher is worse, -1 = lower is worse, 0 = either.
FEATURE_DIRECTION = {
    "Delivery_Delay_Days": 1,
    "Defect_Rate": 1,
    "Lead_Time_Days": 1,
    "Cost_Variation": 0,
    "Order_Quantity": 0,
    "Order_Value": 0,
    "Quality_Score": -1,
    "Order_Fulfillment_Rate": -1,
    "Performance_Score": -1,
    "Rejection_Rate": 1,
    "Unit_Price": 0,
    "Leverage": 1,
    "Current_Ratio": -1,
    "Financial_Stability_ZScore": -1,
    "Profit_Margin": -1,
    "Cash_Flow_Million_USD": -1,
    "Debt_Million_USD": 1,
    "Revenue_Million_USD": -1,
    "Financial_Stress_Index": 1,
    "Assets_Million_USD": 0,
    "Current_Assets_Million_USD": 0,
    "Current_Liabilities_Million_USD": 1,
}

# Analytics page sections per profile: (tab title, [metrics])
ANALYTICS_SECTIONS = {
    "operational": [
        ("Delivery Performance", ["Delivery_Delay_Days", "On_Time_Delivery", "Lead_Time_Days"]),
        ("Quality Performance", ["Defect_Rate", "Quality_Score", "Rejection_Rate"]),
        ("Operational Performance", ["Lead_Time_Days", "Order_Fulfillment_Rate", "Order_Quantity", "Order_Value"]),
        ("Cost Analysis", ["Cost_Variation", "Unit_Price", "Order_Value"]),
    ],
    "financial": [
        ("Financial Health", ["Leverage", "Financial_Stability_ZScore", "Financial_Stress_Index"]),
        ("Liquidity", ["Current_Ratio", "Current_Assets_Million_USD", "Current_Liabilities_Million_USD", "Cash_Flow_Million_USD"]),
        ("Profitability", ["Revenue_Million_USD", "Profit_Margin", "Debt_Million_USD"]),
        ("News Sentiment", ["FinBERT_Sentiment_Score", "News_Risk_Probability"]),
    ],
}

# Live simulation limits (bounded so Streamlit never freezes)
LIVE_MAX_RECORDS = 6000        # rolling record buffer
LIVE_MAX_HISTORY = 360         # trend snapshots kept
LIVE_MAX_ALERTS = 300          # alert feed length
LIVE_MAX_BATCH = 100           # max records generated per refresh
LIVE_RATE_OPTIONS = [1, 2, 5, 10, 20]
LIVE_REFRESH_SECONDS = 2.0
