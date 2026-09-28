"""Session state, data pipeline and filters shared by all pages.

``build_context`` is the single place where raw data (uploaded dataset **or**
the live simulator buffer) is turned into the object the pages render:
scored records, supplier table, segmentation and the active filters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from src import config
from src.analytics import SegmentationResult, segment_suppliers, supplier_table
from src.anomaly_detection import live_alerts
from src.live_data import LiveSimulator
from src.prediction import ScoringResult, get_registry, score_dataframe
from src.preprocessing import DataValidationError, DatasetSummary, load_sample_dataset, load_uploaded_file, preprocess, summarise_dataset
from src.schema import SchemaInfo, detect_schema

PAGES = ["🏠 Overview", "📊 Supplier Analytics", "🤖 Risk Prediction", "🚨 Anomaly & Alerts", "📈 Live Monitoring"]


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("mode", "Dataset")
    ss.setdefault("page", PAGES[0])
    ss.setdefault("agg", "recent")
    ss.setdefault("cluster_k", 4)
    ss.setdefault("use_spark", False)
    ss.setdefault("auto_refresh", True)
    ss.setdefault("refresh_seconds", 2)
    ss.setdefault("dataset", None)          # dict with raw, name, summary, processed, schema, report, scoring
    ss.setdefault("dataset_error", None)
    if "sim" not in ss:
        ss["sim"] = LiveSimulator(n_suppliers=40, rate=5)


# --------------------------------------------------------------------------- #
# Dataset mode
# --------------------------------------------------------------------------- #
def ingest_dataset(raw: pd.DataFrame, name: str) -> None:
    """Validate, preprocess and score an uploaded / sample dataset (stored in session)."""
    ss = st.session_state
    try:
        summary = summarise_dataset(raw)
        processed, schema, report = preprocess(raw, summary.schema, use_spark=ss.get("use_spark", False))
        scoring = score_dataframe(processed, schema)
        ss["dataset"] = {
            "raw": raw, "name": name, "summary": summary, "processed": processed,
            "schema": schema, "report": report, "scoring": scoring,
        }
        ss["dataset_error"] = None
    except DataValidationError as exc:
        ss["dataset"] = None
        ss["dataset_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        ss["dataset"] = None
        ss["dataset_error"] = f"Unexpected error while processing the dataset: {exc}"


def handle_upload(uploaded) -> None:
    ss = st.session_state
    if uploaded is None:
        return
    file_id = f"{uploaded.name}:{uploaded.size}"
    if ss.get("dataset_file_id") == file_id and ss.get("dataset") is not None:
        return
    try:
        raw = load_uploaded_file(uploaded)
    except DataValidationError as exc:
        ss["dataset"], ss["dataset_error"], ss["dataset_file_id"] = None, str(exc), file_id
        return
    ingest_dataset(raw, uploaded.name)
    ss["dataset_file_id"] = file_id


def load_sample(kind: str) -> None:
    ss = st.session_state
    try:
        raw = load_sample_dataset(kind)
    except DataValidationError as exc:
        ss["dataset"], ss["dataset_error"] = None, str(exc)
        return
    label = {"financial": "sample_supplier_data.csv", "operational": "sample_operational_supplier_data.csv",
             "processed": "supplier_risk_processed_with_xgboost_finbert_lstm.xlsx"}[kind]
    ingest_dataset(raw, label)
    ss["dataset_file_id"] = f"sample:{kind}"


# --------------------------------------------------------------------------- #
# Live mode
# --------------------------------------------------------------------------- #
def live_scorer(batch: pd.DataFrame) -> pd.DataFrame:
    res = score_dataframe(batch, allow_on_the_fly=False, allow_unsupervised=False)
    if not res.ok:
        raise RuntimeError("; ".join(res.errors + res.warnings) or "operational model unavailable")
    return res.df


def advance_simulation() -> int:
    sim: LiveSimulator = st.session_state["sim"]
    return sim.advance(live_scorer, live_alerts)


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
@dataclass
class DataContext:
    mode: str                                   # 'dataset' | 'live'
    source_label: str
    all_records: pd.DataFrame
    records: pd.DataFrame
    all_suppliers: pd.DataFrame
    suppliers: pd.DataFrame
    schema: Optional[SchemaInfo]
    profile: Optional[str]
    scoring: Optional[ScoringResult]
    segmentation: Optional[SegmentationResult] = None
    summary: Optional[DatasetSummary] = None
    report: Dict[str, object] = field(default_factory=dict)
    sim: Optional[LiveSimulator] = None
    filters: Dict[str, object] = field(default_factory=dict)
    empty_reason: str = ""

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def ready(self) -> bool:
        return not self.all_records.empty and "Risk_Score" in self.all_records.columns

    @property
    def has_date(self) -> bool:
        return "Date" in self.all_records.columns

    @property
    def filters_active(self) -> bool:
        return len(self.suppliers) != len(self.all_suppliers) or len(self.records) != len(self.all_records)

    def profile_cfg(self, key: str) -> List[str]:
        if self.profile:
            return list(config.PROFILES[self.profile][key])
        # generic fallback: numeric columns
        return [c for c in self.all_records.select_dtypes("number").columns
                if not c.startswith(("Risk_", "Model_", "Record_")) and c not in ("Anomaly_Score",)][:8]


def build_context() -> DataContext:
    ss = st.session_state
    if ss["mode"] == "Live Data":
        sim: LiveSimulator = ss["sim"]
        records = sim.dataframe()
        schema = detect_schema(records) if not records.empty else None
        scoring = None
        if not records.empty and "Risk_Score" in records.columns:
            bundle = get_registry().for_profile("operational")
            scoring = ScoringResult(df=records, method="pretrained", model_key="xgboost_operational", profile="operational",
                                    features=bundle.features if bundle else [], bundle=bundle,
                                    messages=["Live records scored with the pre-trained XGBoost operational model."])
        ctx = DataContext(mode="live", source_label="Live simulator (SIMULATED / ARTIFICIAL DATA)",
                          all_records=records, records=records, all_suppliers=pd.DataFrame(), suppliers=pd.DataFrame(),
                          schema=schema, profile="operational" if not records.empty else None, scoring=scoring, sim=sim)
        if records.empty:
            ctx.empty_reason = "The simulation has not generated any records yet. Press ▶ Start in the sidebar."
    else:
        ds = ss.get("dataset")
        if not ds:
            return DataContext(mode="dataset", source_label="No dataset loaded", all_records=pd.DataFrame(), records=pd.DataFrame(),
                               all_suppliers=pd.DataFrame(), suppliers=pd.DataFrame(), schema=None, profile=None, scoring=None,
                               empty_reason=ss.get("dataset_error") or "Upload a supplier dataset (CSV / Excel) in the sidebar or load a sample dataset.")
        scoring: ScoringResult = ds["scoring"]
        records = scoring.df if scoring.ok else ds["processed"]
        ctx = DataContext(mode="dataset", source_label=ds["name"], all_records=records, records=records,
                          all_suppliers=pd.DataFrame(), suppliers=pd.DataFrame(), schema=ds["schema"], profile=ds["schema"].profile,
                          scoring=scoring, summary=ds["summary"], report=ds["report"])
        if not scoring.ok:
            ctx.empty_reason = "Risk scores could not be computed for this dataset. " + " ".join(scoring.errors + scoring.warnings)

    if not ctx.ready:
        return ctx

    ctx.all_suppliers = supplier_table(ctx.all_records, ss.get("agg", "recent"))
    ctx.segmentation = segment_suppliers(ctx.all_suppliers, ctx.profile_cfg("cluster_features"), int(ss.get("cluster_k", 4)), ctx.profile)
    if ctx.segmentation is not None:
        ctx.all_suppliers = ctx.all_suppliers.merge(ctx.segmentation.assignments, on="Supplier_ID", how="left")
    apply_filters(ctx)
    return ctx


# --------------------------------------------------------------------------- #
# Filters (sidebar)
# --------------------------------------------------------------------------- #
def render_filters(ctx: DataContext) -> None:
    """Sidebar filter widgets. Values are stored in session_state and applied in ``apply_filters``."""
    ss = st.session_state
    if not ctx.ready:
        return
    recs, sups = ctx.all_records, ctx.all_suppliers
    st.multiselect("Risk level", config.RISK_ORDER, default=ss.get("f_levels", config.RISK_ORDER), key="f_levels")
    st.slider("Risk score range", 0, 100, ss.get("f_score", (0, 100)), key="f_score")
    supplier_options = sups["Supplier_ID"].tolist()
    st.multiselect("Supplier", supplier_options, default=[s for s in ss.get("f_suppliers", []) if s in supplier_options],
                   key="f_suppliers", placeholder="All suppliers")
    for col, label in (("Country", "Country / location"), ("Industry", "Industry")):
        if col in sups.columns:
            opts = sorted(sups[col].dropna().astype(str).unique().tolist())
            st.multiselect(label, opts, default=[o for o in ss.get(f"f_{col}", []) if o in opts], key=f"f_{col}", placeholder=f"All {label.lower()}")
    if ctx.segmentation is not None:
        opts = list(dict.fromkeys(ctx.segmentation.assignments["Cluster_Label"].tolist()))
        st.multiselect("Supplier cluster", opts, default=[o for o in ss.get("f_cluster", []) if o in opts], key="f_cluster", placeholder="All clusters")
    if ctx.has_date and not ctx.is_live and recs["Date"].notna().any():
        dmin, dmax = recs["Date"].min().date(), recs["Date"].max().date()
        if dmin < dmax:
            cur = ss.get("f_dates", (dmin, dmax))
            try:
                cur = (max(cur[0], dmin), min(cur[1], dmax))
            except Exception:  # noqa: BLE001
                cur = (dmin, dmax)
            st.date_input("Date range", value=cur, min_value=dmin, max_value=dmax, key="f_dates")
    if st.button("Reset filters", width="stretch"):
        for k in [k for k in ss.keys() if str(k).startswith("f_")]:
            del ss[k]
        st.rerun()


def apply_filters(ctx: DataContext) -> None:
    ss = st.session_state
    recs, sups = ctx.all_records, ctx.all_suppliers
    rmask = pd.Series(True, index=recs.index)
    smask = pd.Series(True, index=sups.index)

    levels = ss.get("f_levels", config.RISK_ORDER)
    if levels and set(levels) != set(config.RISK_ORDER):
        smask &= sups["Risk_Level"].isin(levels)
    lo, hi = ss.get("f_score", (0, 100))
    if (lo, hi) != (0, 100):
        smask &= sups["Risk_Score"].between(lo, hi)
    chosen = ss.get("f_suppliers", [])
    if chosen:
        smask &= sups["Supplier_ID"].isin(chosen)
    for col in ("Country", "Industry"):
        vals = ss.get(f"f_{col}", [])
        if vals and col in sups.columns:
            smask &= sups[col].astype(str).isin(vals)
    clusters = ss.get("f_cluster", [])
    if clusters and "Cluster_Label" in sups.columns:
        smask &= sups["Cluster_Label"].isin(clusters)
    dates = ss.get("f_dates")
    if dates and ctx.has_date and not ctx.is_live and isinstance(dates, (tuple, list)) and len(dates) == 2:
        start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1]) + pd.Timedelta(days=1)
        rmask &= recs["Date"].between(start, end, inclusive="left") | recs["Date"].isna()

    keep_ids = set(sups.loc[smask, "Supplier_ID"])
    rmask &= recs["Supplier_ID"].isin(keep_ids)
    ctx.records = recs[rmask]
    ctx.suppliers = sups[smask & sups["Supplier_ID"].isin(ctx.records["Supplier_ID"].unique())]
    ctx.filters = {"levels": levels, "score": (lo, hi), "suppliers": chosen, "clusters": clusters, "dates": dates}
