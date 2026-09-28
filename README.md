# SUPPLIER RISK PREDICTION

**Intelligent Supplier Risk Prediction and Analytics** – a dark-themed enterprise Streamlit dashboard that scores supplier risk with XGBoost, explains every prediction with SHAP, segments suppliers with K-Means, flags anomalies with Isolation Forest, and streams a simulated live supplier feed for real-time monitoring.

| Mode | What it does |
|------|--------------|
| 📁 **Dataset Mode** | Upload a CSV / Excel supplier dataset. Columns are matched automatically to the *financial* profile (the project dataset: revenue, debt, leverage, current ratio, Z-score, cash flow, news sentiment) or the *operational* profile (delivery delay, defect rate, lead time, cost variation, fulfilment …). |
| 🔴 **Live Data Mode** | No upload needed. A built-in simulator generates internally consistent supplier transactions in real time (clearly labelled **SIMULATED / ARTIFICIAL DATA**), scores them with the same XGBoost pipeline, and produces live alerts. |

Pages (horizontal navigation): 🏠 Overview · 📊 Supplier Analytics · 🤖 Risk Prediction · 🚨 Anomaly & Alerts · 📈 Live Monitoring

Design: dark navy enterprise theme (`#0B1220` background, `#111C30` cards, `#3B82F6` primary, green/orange/red risk colours) defined once in `src/config.py` and `.streamlit/config.toml`; custom header with data-mode and system-status chips, data-source switch, KPI cards, Risk Health card, early-warning cards and a unified alert feed.

---

## 1. Quick start (local)

```powershell
# from the project folder
uv venv --python 3.12 .venv                      # or:  py -3.12 -m venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt   # or: .venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python.exe scripts\train_models.py # only needed if models/ is empty
.venv\Scripts\python.exe -m streamlit run app.py
```

Or simply double-click **`run_dashboard.bat`** on Windows.

The app opens at <http://localhost:8501>. For access from **another laptop through a public URL** see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 2. Project structure

```
SUPPLIER RISK PREDICTION/
├── app.py                      # Streamlit entry point (sidebar, routing, auto-refresh)
├── requirements.txt            # pinned dependencies (Python 3.12)
├── Dockerfile                  # container deployment (Render / Railway / HF Spaces)
├── run_dashboard.bat           # one-click local launcher (Windows)
├── README.md · DEPLOYMENT.md
├── .streamlit/config.toml      # dark enterprise theme, server settings
├── data/
│   ├── supplier_risk_10000rows_19columns.xlsx        # original project dataset (financial profile)
│   ├── sample_supplier_data.csv                      # CSV export of the above
│   ├── sample_operational_supplier_data.csv          # simulated operational example (for upload tests)
│   └── supplier_risk_processed_with_xgboost_finbert_lstm.xlsx  # notebook output incl. FinBERT/LSTM columns
├── models/
│   ├── xgboost_financial_model.pkl  (+ .json)        # notebook XGBoost: financial features -> disruption
│   ├── random_forest_financial_model.pkl
│   ├── xgboost_operational_model.pkl (+ .json)       # XGBoost trained on simulator data (operational features)
│   ├── random_forest_operational_model.pkl
│   └── model_registry.json                           # features, medians, metrics, thresholds, sentiment proxy
├── scripts/
│   ├── train_models.py         # reproduces the notebook training + saves models/ and data/ samples
│   ├── smoke_test.py           # backend pipeline test (no UI)
│   └── app_test.py             # headless UI test (streamlit.testing AppTest)
└── src/
    ├── config.py               # colours, thresholds, feature profiles
    ├── schema.py               # column detection / canonical names
    ├── preprocessing.py        # loading, validation, cleaning, Financial Stress Index, optional PySpark
    ├── prediction.py           # model registry, scoring, news-risk fusion, SHAP
    ├── analytics.py            # KPIs, supplier table, trends, K-Means segmentation
    ├── anomaly_detection.py    # Isolation Forest alerts + live alert feed
    ├── live_data.py            # bounded supplier simulator
    ├── state.py                # session state, data context, filters
    ├── ui.py                   # CSS, KPI cards, Plotly chart factories
    └── pages/                  # overview, analytics, prediction, anomaly, live, common
```

K-Means and Isolation Forest are unsupervised and data-dependent, so they are fitted on the currently loaded data at run time (with the cluster count and anomaly share adjustable in the UI) rather than loaded from disk.

---

## 3. Machine-learning pipeline

The pipeline mirrors `CAPSTONE_PROJECT.ipynb`:

1. **Preprocessing** – parse dates, drop duplicates, median-impute numeric columns, `fillna("Unknown")` for text, engineer `Financial_Stress_Index = Leverage + 1 / max(Current_Ratio, 0.1)` (pandas by default, PySpark when enabled and available).
2. **XGBoost** (`n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, scale_pos_weight`) with a chronological 80/20 split.
   * Financial model hold-out: accuracy 0.90, ROC-AUC 0.84 (notebook: 0.90 / 0.83).
   * Operational model hold-out: accuracy 0.90, ROC-AUC 0.93 (on simulated data).
3. **Risk fusion** – when the dataset carries news risk, `Final = 0.6 × XGBoost financial risk + 0.4 × news risk` (FinBERT/LSTM column if present, otherwise an empirical proxy derived from the FinBERT results per sentiment label).
4. **Risk category** – `< 0.33 LOW`, `< 0.66 MEDIUM`, else `HIGH` (score shown as 0–100 %).
5. **SHAP** `TreeExplainer` explains each supplier's most recent records; feature importances are the fallback.
6. **K-Means** segments suppliers on standardised supplier-level features; cluster labels (Reliable / Delayed / Quality Risk / High Cost / Financially Stressed …) are derived from each cluster centre's strongest deviation, not assigned by hand.
7. **Early warning** compares each supplier's last 3 records with the 3 before them; a rise of 10+ risk points is reported as ⚠️ RISK INCREASING with the metrics that moved most.
8. **Isolation Forest** flags anomalous records; the metric with the largest robust z-score becomes the alert type, its magnitude the severity (🟡 Attention ≥ 2σ, 🟠 Warning ≥ 3σ, 🔴 Critical ≥ 4.5σ) and the 5–95 % band the normal range.

If an uploaded file does not match a pre-trained model the app says so and either trains an XGBoost model on the fly (when a binary label column exists) or shows a clearly labelled **unsupervised risk index** – it never fabricates model predictions.

---

## 4. Live Data Mode

* Every simulated supplier carries a hidden stress state that drifts through stable → deteriorating → recovering episodes. Delivery delay, defect rate, lead time, cost variation, fulfilment, quality and performance are all functions of that state, so deteriorating suppliers really do become high risk.
* Controls: ▶ Start · ⏸ Pause · ↻ Reset · data rate 1 / 2 / 5 / 10 / 20 records per second · number of suppliers.
* Updates use `st.fragment(run_every=…)` with a bounded record buffer (6 000 records), bounded trend history and a cap on records per refresh – no infinite loops.
* Alerts are generated from the model output and Isolation Forest: `HIGH RISK`, `RISK INCREASING`, `ANOMALY DETECTED`, each with supplier, risk level, timestamp and reason.

---

## 5. Testing

```powershell
.venv\Scripts\python.exe scripts\smoke_test.py   # pipeline on 7 dataset variants + simulator
.venv\Scripts\python.exe scripts\app_test.py     # headless UI: all pages, filters, mode switching, start/pause/reset
```

---

## 6. Data notice

The bundled datasets are **synthetic educational data** (see the README sheet in the Excel file). Live Data Mode generates **simulated / artificial data**. Neither represents real supplier records.
