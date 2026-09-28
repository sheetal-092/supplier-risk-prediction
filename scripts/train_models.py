"""Train and persist the models used by the dashboard.

Run from the project root::

    python scripts/train_models.py

Produces in ``models/``:

* ``xgboost_financial_model.pkl``      - notebook XGBoost (financial features -> Disruption_Within_90_Days)
* ``random_forest_financial_model.pkl``
* ``xgboost_operational_model.pkl``    - XGBoost trained on simulator data (operational features -> Risk_Event)
* ``random_forest_operational_model.pkl``
* ``model_registry.json``              - feature lists, training medians, metrics, thresholds

and in ``data/``:

* ``sample_supplier_data.csv``                (CSV export of the original Excel dataset)
* ``sample_operational_supplier_data.csv``    (small simulated operational dataset for testing uploads)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.live_data import generate_training_data  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402


def evaluate(model, X_test, y_test) -> dict:
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
    }
    if len(np.unique(y_test)) == 2:
        out["roc_auc"] = round(float(roc_auc_score(y_test, prob)), 4)
    return out


def time_split(df: pd.DataFrame, features, target, frac=0.8):
    df_sorted = df.sort_values("Date").reset_index(drop=True) if "Date" in df.columns else df.reset_index(drop=True)
    split = int(len(df_sorted) * frac)
    train, test = df_sorted.iloc[:split], df_sorted.iloc[split:]
    return train[features], train[target], test[features], test[target]


def fit_xgb(X_train, y_train) -> XGBClassifier:
    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    model = XGBClassifier(**config.XGB_PARAMS, scale_pos_weight=neg / pos)
    model.fit(X_train, y_train)
    return model


def fit_rf(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def main() -> None:
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.DATA_DIR.mkdir(exist_ok=True)
    registry = {"created": datetime.now().isoformat(timespec="seconds"), "models": {}, "thresholds": {
        "low_max": config.RISK_LOW_MAX, "medium_max": config.RISK_MEDIUM_MAX,
        "financial_weight": config.FINANCIAL_WEIGHT, "news_weight": config.NEWS_WEIGHT}}

    # ------------------------------------------------------------------ financial
    print("Loading financial dataset ...")
    raw = pd.read_excel(config.SAMPLE_FINANCIAL_XLSX, sheet_name="Supplier_Risk_Data")
    raw.to_csv(config.SAMPLE_FINANCIAL_CSV, index=False)
    print(f"  exported {config.SAMPLE_FINANCIAL_CSV.name}: {raw.shape}")

    df, schema, report = preprocess(raw)
    print("  preprocessing:", report["steps"])
    feats = config.FINANCIAL_MODEL_FEATURES
    target = config.FINANCIAL_TARGET
    X_train, y_train, X_test, y_test = time_split(df, feats, target)
    print(f"  train rows {len(X_train)}, test rows {len(X_test)}")

    xgb_fin = fit_xgb(X_train, y_train)
    fin_metrics = evaluate(xgb_fin, X_test, y_test)
    print("  XGBoost financial:", fin_metrics)
    joblib.dump(xgb_fin, config.MODELS_DIR / "xgboost_financial_model.pkl")
    xgb_fin.save_model(str(config.MODELS_DIR / "xgboost_financial_model.json"))

    rf_fin = fit_rf(X_train, y_train)
    rf_fin_metrics = evaluate(rf_fin, X_test, y_test)
    print("  RandomForest financial:", rf_fin_metrics)
    joblib.dump(rf_fin, config.MODELS_DIR / "random_forest_financial_model.pkl")

    registry["models"]["xgboost_financial"] = {
        "file": "xgboost_financial_model.pkl",
        "algorithm": "XGBoost (XGBClassifier)",
        "profile": "financial",
        "features": feats,
        "target": target,
        "medians": {f: float(df[f].median()) for f in feats},
        "feature_ranges": {f: [float(df[f].quantile(0.01)), float(df[f].quantile(0.99))] for f in feats},
        "metrics": fin_metrics,
        "training_rows": int(len(X_train)),
        "positive_rate": round(float(y_train.mean()), 4),
        "source": "CAPSTONE_PROJECT.ipynb pipeline on supplier_risk_10000rows_19columns.xlsx",
    }
    registry["models"]["random_forest_financial"] = {
        "file": "random_forest_financial_model.pkl", "algorithm": "Random Forest", "profile": "financial",
        "features": feats, "target": target, "metrics": rf_fin_metrics,
    }

    # sentiment proxy: empirical FinBERT negative probability per reference label
    if config.SAMPLE_PROCESSED_XLSX.exists():
        proc = pd.read_excel(config.SAMPLE_PROCESSED_XLSX, sheet_name="Processed_Data")
        proxy = proc.groupby("Synthetic_News_Sentiment")["FinBERT_Negative_Probability"].mean().round(4).to_dict()
        registry["sentiment_proxy"] = {str(k): float(v) for k, v in proxy.items()}
        print("  sentiment proxy (FinBERT negative prob by label):", registry["sentiment_proxy"])

    # --------------------------------------------------------------- operational
    print("Generating simulated operational training data ...")
    sim_df = generate_training_data(n_records=36_000, n_suppliers=120, seed=7)
    sim_df.head(3000).to_csv(config.SAMPLE_OPERATIONAL_CSV, index=False)
    print(f"  simulated {sim_df.shape}, event rate {sim_df['Risk_Event'].mean():.3f}; exported sample CSV")
    op_feats = config.OPERATIONAL_FEATURES
    op_target = config.OPERATIONAL_TARGET
    X_train, y_train, X_test, y_test = time_split(sim_df, op_feats, op_target)
    xgb_op = fit_xgb(X_train, y_train)
    op_metrics = evaluate(xgb_op, X_test, y_test)
    print("  XGBoost operational:", op_metrics)
    joblib.dump(xgb_op, config.MODELS_DIR / "xgboost_operational_model.pkl")
    xgb_op.save_model(str(config.MODELS_DIR / "xgboost_operational_model.json"))
    rf_op = fit_rf(X_train, y_train)
    rf_op_metrics = evaluate(rf_op, X_test, y_test)
    print("  RandomForest operational:", rf_op_metrics)
    joblib.dump(rf_op, config.MODELS_DIR / "random_forest_operational_model.pkl")

    registry["models"]["xgboost_operational"] = {
        "file": "xgboost_operational_model.pkl",
        "algorithm": "XGBoost (XGBClassifier)",
        "profile": "operational",
        "features": op_feats,
        "target": op_target,
        "medians": {f: float(sim_df[f].median()) for f in op_feats},
        "feature_ranges": {f: [float(sim_df[f].quantile(0.01)), float(sim_df[f].quantile(0.99))] for f in op_feats},
        "metrics": op_metrics,
        "training_rows": int(len(X_train)),
        "positive_rate": round(float(y_train.mean()), 4),
        "source": "src/live_data.py simulator (SIMULATED / ARTIFICIAL DATA)",
    }
    registry["models"]["random_forest_operational"] = {
        "file": "random_forest_operational_model.pkl", "algorithm": "Random Forest", "profile": "operational",
        "features": op_feats, "target": op_target, "metrics": rf_op_metrics,
    }

    with open(config.MODEL_REGISTRY_FILE, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)
    print(f"Saved registry -> {config.MODEL_REGISTRY_FILE}")


if __name__ == "__main__":
    main()
