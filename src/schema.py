"""Column detection and canonical naming.

Uploaded files rarely use exactly the column names the models were trained
with.  This module maps arbitrary column headers (``"Delivery delay (days)"``,
``"defect_rate_pct"`` ...) onto canonical names and detects which feature
profile (financial / operational) the data supports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src import config

# canonical name -> list of accepted aliases (normalised: lowercase, alnum only)
ALIASES: Dict[str, List[str]] = {
    # identifiers / dimensions
    "Supplier_ID": ["supplierid", "supplier", "vendorid", "suppliercode", "vendorcode", "supplierno", "vendor", "sid"],
    "Supplier_Name": ["suppliername", "vendorname", "name", "company", "companyname"],
    "Date": ["date", "orderdate", "deliverydate", "timestamp", "observationdate", "recorddate", "datetime", "period", "month"],
    "Industry": ["industry", "sector", "category", "segment"],
    "Country": ["country", "location", "region", "state", "city", "geography"],
    # operational
    "Delivery_Delay_Days": ["deliverydelay", "deliverydelaydays", "delaydays", "delay", "latedays", "deliverydelayindays", "daysdelayed", "deliverylateness", "avgdelay", "averagedelay"],
    "Defect_Rate": ["defectrate", "defectratepct", "defectpercentage", "defectratepercent", "defects", "defectpct", "defectivepercentage"],
    "Lead_Time_Days": ["leadtime", "leadtimedays", "leadtimeindays", "avgleadtime", "averageleadtime"],
    "Order_Quantity": ["orderquantity", "quantity", "qty", "orderqty", "units", "unitsordered"],
    "Order_Value": ["ordervalue", "orderamount", "totalvalue", "orderamountusd", "value", "amount", "totalcost", "spend", "ordervalueusd"],
    "Quality_Score": ["qualityscore", "quality", "qualityrating", "qualityindex"],
    "Cost_Variation": ["costvariation", "costvariance", "pricevariation", "pricevariance", "costdeviation", "pricedeviation", "costchange", "pricechange", "costvariationpct"],
    "Order_Fulfillment_Rate": ["orderfulfillment", "fulfillmentrate", "fulfilmentrate", "orderfulfillmentrate", "orderfulfilment", "fillrate", "fulfillment", "fulfilment"],
    "Performance_Score": ["performancescore", "performance", "supplierperformance", "overallperformance", "score"],
    "On_Time_Delivery": ["ontimedelivery", "ontimedeliveryrate", "ontimepct", "otd", "ontime", "ontimerate"],
    "Rejection_Rate": ["rejectionrate", "rejectrate", "rejected", "rejectionpct"],
    "Unit_Price": ["unitprice", "price", "priceperunit", "unitcost"],
    "Risk_Event": ["riskevent", "risklabel", "isrisky", "highrisk", "target", "label", "riskflag", "disruption", "disrupted", "event"],
    # financial (exact notebook names + a few variants)
    "Revenue_Million_USD": ["revenuemillionusd", "revenue", "revenueusd", "sales"],
    "Debt_Million_USD": ["debtmillionusd", "debt", "totaldebt"],
    "Assets_Million_USD": ["assetsmillionusd", "assets", "totalassets"],
    "Current_Assets_Million_USD": ["currentassetsmillionusd", "currentassets"],
    "Current_Liabilities_Million_USD": ["currentliabilitiesmillionusd", "currentliabilities"],
    "Leverage": ["leverage", "debttoassets", "leverageratio", "debtratio"],
    "Current_Ratio": ["currentratio", "liquidityratio"],
    "Financial_Stability_ZScore": ["financialstabilityzscore", "zscore", "altmanzscore", "stabilityzscore", "stabilityscore"],
    "Profit_Margin": ["profitmargin", "margin", "netmargin"],
    "Cash_Flow_Million_USD": ["cashflowmillionusd", "cashflow", "operatingcashflow"],
    "Financial_Stress_Index": ["financialstressindex", "stressindex"],
    "Disruption_Within_90_Days": ["disruptionwithin90days", "disruption90", "disruptionwithin90", "disruptionflag"],
    # optional pre-computed news/sentiment columns from the processed dataset
    "News_Risk_Probability": ["newsriskprobability", "newsrisk"],
    "FinBERT_Negative_Probability": ["finbertnegativeprobability", "negativeprobability", "finbertnegative"],
    "FinBERT_Sentiment_Score": ["finbertsentimentscore", "sentimentscore"],
    "Synthetic_News_Sentiment": ["syntheticnewssentiment", "newssentiment", "sentiment", "sentimentlabel"],
    "News_Headline": ["newsheadline", "headline"],
}

# Aliases that are too generic to trust unless nothing better exists
_WEAK = {"name", "score", "value", "amount", "price", "label", "target", "event", "delay", "quality", "performance", "id", "sid", "sector", "category", "segment", "month", "period", "units", "state", "city", "region", "location", "geography", "sales", "margin", "vendor", "supplier"}


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


@dataclass
class SchemaInfo:
    """Result of :func:`detect_schema`."""

    rename_map: Dict[str, str] = field(default_factory=dict)   # original -> canonical
    profile: Optional[str] = None                              # 'financial' | 'operational' | None
    present_features: List[str] = field(default_factory=list)
    missing_features: List[str] = field(default_factory=list)
    target: Optional[str] = None
    has_date: bool = False
    has_supplier_id: bool = False
    has_supplier_name: bool = False
    notes: List[str] = field(default_factory=list)
    coverage: Dict[str, float] = field(default_factory=dict)

    @property
    def profile_label(self) -> str:
        return config.PROFILES[self.profile]["label"] if self.profile else "Unknown profile"


def build_rename_map(columns) -> Dict[str, str]:
    """Return ``{original_column: canonical_name}`` for recognised columns."""
    rename: Dict[str, str] = {}
    taken: set = set()
    norm_cols = {c: normalise(c) for c in columns}

    # pass 1: exact canonical names or strong aliases
    for canonical, aliases in ALIASES.items():
        canon_norm = normalise(canonical)
        for col, n in norm_cols.items():
            if col in rename or canonical in taken:
                continue
            if n == canon_norm or (n in aliases and n not in _WEAK):
                rename[col] = canonical
                taken.add(canonical)
    # pass 2: weak aliases only if the canonical name is still unassigned
    for canonical, aliases in ALIASES.items():
        if canonical in taken:
            continue
        for col, n in norm_cols.items():
            if col in rename:
                continue
            if n in aliases:
                rename[col] = canonical
                taken.add(canonical)
                break
    return rename


def detect_schema(df: pd.DataFrame) -> SchemaInfo:
    """Detect canonical columns and the best matching feature profile."""
    info = SchemaInfo()
    info.rename_map = build_rename_map(df.columns)
    canonical_cols = set(info.rename_map.values()) | set(df.columns)

    info.has_date = "Date" in canonical_cols
    info.has_supplier_id = "Supplier_ID" in canonical_cols
    info.has_supplier_name = "Supplier_Name" in canonical_cols

    # coverage per profile (engineered features are derivable so they count if
    # their inputs are present)
    for name, prof in config.PROFILES.items():
        feats = list(prof["raw_features"])
        present = [f for f in feats if f in canonical_cols]
        info.coverage[name] = len(present) / max(len(feats), 1)

    best = max(info.coverage, key=info.coverage.get)
    if info.coverage[best] >= 0.6:
        info.profile = best
        prof = config.PROFILES[best]
        info.present_features = [f for f in prof["features"] if f in canonical_cols]
        # Financial_Stress_Index is derived from Leverage & Current_Ratio
        if best == "financial" and {"Leverage", "Current_Ratio"} <= canonical_cols:
            if "Financial_Stress_Index" not in info.present_features:
                info.present_features.append("Financial_Stress_Index")
        info.missing_features = [f for f in prof["features"] if f not in info.present_features]
        if prof["target"] in canonical_cols:
            info.target = prof["target"]
    else:
        info.notes.append(
            "The columns do not match the financial or operational feature profile "
            "well enough for the pre-trained XGBoost models."
        )

    # generic label detection (used for on-the-fly training)
    if info.target is None:
        for cand in ("Disruption_Within_90_Days", "Risk_Event"):
            if cand in canonical_cols:
                info.target = cand
                break

    if not info.has_supplier_id:
        info.notes.append("No supplier identifier column found - a synthetic Supplier_ID will be created per row.")
    if not info.has_date:
        info.notes.append("No date column found - time-based trend charts will use record order instead of dates.")
    return info


def apply_schema(df: pd.DataFrame, info: SchemaInfo) -> pd.DataFrame:
    """Return a copy of *df* with canonical column names applied."""
    out = df.copy()
    # avoid clobbering an existing canonical column with a renamed duplicate
    safe_map = {k: v for k, v in info.rename_map.items() if k != v and v not in out.columns}
    out = out.rename(columns=safe_map)
    return out


def feature_label(col: str) -> str:
    return config.FEATURE_LABELS.get(col, (col.replace("_", " "), ""))[0]


def feature_unit(col: str) -> str:
    return config.FEATURE_LABELS.get(col, (col, ""))[1]
