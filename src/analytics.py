"""Analytics helpers shared by Dataset Mode and Live Data Mode.

All functions take the *scored* record-level DataFrame (output of
``prediction.score_dataframe``) and return plain pandas objects so the page
modules only deal with presentation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config
from src.schema import feature_label

AGG_CHOICES = {
    "Recent average (last 3 records)": "recent",
    "Latest record": "latest",
    "Overall average": "mean",
}


# --------------------------------------------------------------------------- #
# Supplier level view
# --------------------------------------------------------------------------- #
def supplier_table(df: pd.DataFrame, agg: str = "recent", extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """One row per supplier with the current risk and key metrics."""
    if df.empty or "Risk_Score" not in df.columns:
        return pd.DataFrame()
    order_col = "Date" if "Date" in df.columns else ("Record_Index" if "Record_Index" in df.columns else None)
    d = df.sort_values(order_col) if order_col else df
    numeric_cols = [c for c in d.select_dtypes(include=[np.number]).columns if c not in ("Record_Index",)]
    keep_numeric = [c for c in numeric_cols if c not in ("Risk_Score", "Risk_Probability")]

    g = d.groupby("Supplier_ID", sort=False)
    if agg == "latest":
        cur = g.tail(1).set_index("Supplier_ID")
        risk = cur["Risk_Score"]
        metrics = cur[keep_numeric]
    elif agg == "mean":
        risk = g["Risk_Score"].mean()
        metrics = g[keep_numeric].mean()
    else:  # recent
        recent = g.tail(3)
        rg = recent.groupby("Supplier_ID", sort=False)
        risk = rg["Risk_Score"].mean()
        metrics = rg[keep_numeric].mean()

    base = pd.DataFrame({
        "Supplier_Name": g["Supplier_Name"].last(),
        "Risk_Score": risk.round(1),
        "Avg_Risk_Score": g["Risk_Score"].mean().round(1),
        "Max_Risk_Score": g["Risk_Score"].max().round(1),
        "Records": g.size(),
    })
    if order_col == "Date":
        base["Last_Seen"] = g["Date"].max()
    for col in ("Industry", "Country"):
        if col in d.columns:
            base[col] = g[col].last()
    base = base.join(metrics, how="left")
    base["Risk_Level"] = pd.cut(base["Risk_Score"] / 100.0,
                                bins=[-np.inf, config.RISK_LOW_MAX, config.RISK_MEDIUM_MAX, np.inf],
                                labels=["LOW", "MEDIUM", "HIGH"]).astype(str)
    base["Risk_Status"] = base["Risk_Level"].map({"LOW": "🟢 Stable", "MEDIUM": "🟠 Watch", "HIGH": "🔴 Critical"})
    base = base.reset_index().sort_values("Risk_Score", ascending=False).reset_index(drop=True)
    return base


def kpis(supplier_df: pd.DataFrame, record_df: pd.DataFrame) -> Dict[str, float]:
    if supplier_df.empty:
        return {"total": 0, "high": 0, "medium": 0, "low": 0, "avg_risk": 0.0, "records": len(record_df)}
    counts = supplier_df["Risk_Level"].value_counts()
    return {
        "total": int(len(supplier_df)),
        "high": int(counts.get("HIGH", 0)),
        "medium": int(counts.get("MEDIUM", 0)),
        "low": int(counts.get("LOW", 0)),
        "avg_risk": float(supplier_df["Risk_Score"].mean()),
        "records": int(len(record_df)),
    }


# --------------------------------------------------------------------------- #
# Trends
# --------------------------------------------------------------------------- #
def _period_rule(df: pd.DataFrame) -> str:
    span = df["Date"].max() - df["Date"].min()
    if span <= pd.Timedelta(hours=6):
        return "10s"
    if span <= pd.Timedelta(days=2):
        return "1h"
    if span <= pd.Timedelta(days=60):
        return "D"
    if span <= pd.Timedelta(days=400):
        return "W"
    return "MS"


def risk_over_time(df: pd.DataFrame, supplier_id: Optional[str] = None) -> pd.DataFrame:
    """Average risk score per period (optionally for one supplier)."""
    if df.empty or "Risk_Score" not in df.columns:
        return pd.DataFrame()
    d = df if supplier_id is None else df[df["Supplier_ID"] == supplier_id]
    if d.empty:
        return pd.DataFrame()
    if "Date" in d.columns and d["Date"].notna().any():
        d = d.dropna(subset=["Date"])
        rule = _period_rule(d)
        s = d.set_index("Date").sort_index()["Risk_Score"].resample(rule).mean().dropna()
        return s.rename("Risk_Score").reset_index().rename(columns={"Date": "Period"})
    idx = "Record_Index" if "Record_Index" in d.columns else None
    d = d.sort_values(idx) if idx else d
    out = d[["Risk_Score"]].reset_index(drop=True)
    out["Period"] = np.arange(len(out))
    return out[["Period", "Risk_Score"]]


def trend_status(series: pd.Series, window: int = 10) -> Tuple[str, float]:
    """Classify a risk series as Increasing / Stable / Decreasing from its slope."""
    s = pd.to_numeric(series, errors="coerce").dropna().tail(window)
    if len(s) < 3:
        return "Stable", 0.0
    x = np.arange(len(s))
    slope = float(np.polyfit(x, s.values, 1)[0])
    change = slope * (len(s) - 1)          # total change over the window in score points
    if change > 5:
        return "Increasing", change
    if change < -5:
        return "Decreasing", change
    return "Stable", change


# --------------------------------------------------------------------------- #
# Metric aggregations for analytics page
# --------------------------------------------------------------------------- #
def metric_over_time(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    if "Date" in df.columns and df["Date"].notna().any():
        d = df.dropna(subset=["Date"])
        rule = _period_rule(d)
        s = d.set_index("Date").sort_index()[metric].resample(rule).mean().dropna()
        return s.rename(metric).reset_index().rename(columns={"Date": "Period"})
    out = df[[metric]].reset_index(drop=True)
    out["Period"] = np.arange(len(out))
    return out[["Period", metric]]


def supplier_metric_ranking(supplier_df: pd.DataFrame, metric: str, n: int = 15, worst: bool = True) -> pd.DataFrame:
    if supplier_df.empty or metric not in supplier_df.columns:
        return pd.DataFrame()
    direction = config.FEATURE_DIRECTION.get(metric, 1)
    ascending = (direction == -1) if worst else (direction != -1)
    d = supplier_df[["Supplier_ID", "Supplier_Name", metric, "Risk_Level", "Risk_Score"]].dropna(subset=[metric])
    if direction == 0 and worst:
        d = d.assign(_abs=d[metric].abs()).sort_values("_abs", ascending=False).drop(columns="_abs")
    else:
        d = d.sort_values(metric, ascending=ascending)
    return d.head(n)


def correlation_matrix(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    cols = [c for c in features if c in df.columns]
    if "Risk_Score" in df.columns:
        cols = cols + ["Risk_Score"]
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].corr().round(2)


def group_risk(supplier_df: pd.DataFrame, by: str) -> pd.DataFrame:
    if supplier_df.empty or by not in supplier_df.columns:
        return pd.DataFrame()
    g = supplier_df.groupby(by)
    out = pd.DataFrame({
        "Suppliers": g.size(),
        "Avg_Risk_Score": g["Risk_Score"].mean().round(1),
        "High_Risk": g.apply(lambda x: int((x["Risk_Level"] == "HIGH").sum()), include_groups=False),
    }).reset_index().sort_values("Avg_Risk_Score", ascending=False)
    return out


# --------------------------------------------------------------------------- #
# Segmentation (K-Means)
# --------------------------------------------------------------------------- #
@dataclass
class SegmentationResult:
    assignments: pd.DataFrame            # Supplier_ID, Cluster, Cluster_Label
    summary: pd.DataFrame                # per cluster means + counts
    features: List[str]
    k: int
    labels: Dict[int, str] = field(default_factory=dict)
    message: str = ""
    pca: Optional[pd.DataFrame] = None   # 2-D projection for scatter


_OPERATIONAL_RULES = [
    # (feature, direction that indicates the label, label)
    ("Delivery_Delay_Days", 1, "Delayed Suppliers"),
    ("Defect_Rate", 1, "Quality Risk Suppliers"),
    ("Quality_Score", -1, "Quality Risk Suppliers"),
    ("Cost_Variation", 1, "High Cost Suppliers"),
    ("Order_Value", 1, "High Cost Suppliers"),
    ("Lead_Time_Days", 1, "Long Lead-time Suppliers"),
]
_FINANCIAL_RULES = [
    ("Financial_Stress_Index", 1, "Financially Stressed Suppliers"),
    ("Leverage", 1, "Highly Leveraged Suppliers"),
    ("Current_Ratio", -1, "Low Liquidity Suppliers"),
    ("Profit_Margin", -1, "Low Profitability Suppliers"),
    ("Cash_Flow_Million_USD", -1, "Weak Cash-flow Suppliers"),
]


def segment_suppliers(supplier_df: pd.DataFrame, features: List[str], k: int = 4, profile: Optional[str] = None) -> Optional[SegmentationResult]:
    """K-Means on standardised supplier-level features with data-driven labels."""
    feats = [f for f in features if f in supplier_df.columns and supplier_df[f].notna().any()]
    if len(feats) < 2 or len(supplier_df) < max(k, 4):
        return None
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception:  # noqa: BLE001
        return None

    X = supplier_df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0.0)
    k = int(min(k, max(2, len(X) // 3)))
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    clusters = km.labels_

    centers_z = pd.DataFrame(km.cluster_centers_, columns=feats)          # standardised centres
    rules = _OPERATIONAL_RULES if profile == "operational" else _FINANCIAL_RULES if profile == "financial" else []
    labels: Dict[int, str] = {}
    risk_by_cluster = pd.Series(supplier_df["Risk_Score"].values).groupby(clusters).mean() if "Risk_Score" in supplier_df else None

    # score each (cluster, rule) by how strongly the centre deviates in the rule's direction
    scored: List[Tuple[float, int, str]] = []
    for c in range(k):
        for feat, direction, label in rules:
            if feat in centers_z.columns:
                dev = centers_z.loc[c, feat] * direction
                scored.append((float(dev), c, label))
    scored.sort(reverse=True)
    used_labels: set = set()
    for dev, c, label in scored:
        if c in labels or label in used_labels:
            continue
        if dev > 0.35:                      # clear deviation in the "bad" direction
            labels[c] = label
            used_labels.add(label)
    # remaining clusters: reliable / moderate based on relative risk
    for c in range(k):
        if c in labels:
            continue
        if risk_by_cluster is not None and risk_by_cluster.loc[c] <= risk_by_cluster.median():
            name = "Reliable Suppliers"
        else:
            name = "Moderate Risk Suppliers"
        if name in used_labels:
            name = f"{name} ({c})"
        labels[c] = name
        used_labels.add(name)

    assignments = pd.DataFrame({
        "Supplier_ID": supplier_df["Supplier_ID"].values,
        "Cluster": clusters,
    })
    assignments["Cluster_Label"] = assignments["Cluster"].map(labels)

    summary = X.copy()
    summary["Cluster"] = clusters
    summary = summary.groupby("Cluster").mean().round(3)
    summary["Suppliers"] = pd.Series(clusters).value_counts().sort_index()
    if risk_by_cluster is not None:
        summary["Avg_Risk_Score"] = risk_by_cluster.round(1)
    summary["Cluster_Label"] = summary.index.map(labels)
    summary = summary.reset_index()

    pca_df = None
    try:
        comps = PCA(n_components=2, random_state=42).fit_transform(Z)
        pca_df = pd.DataFrame({"PC1": comps[:, 0], "PC2": comps[:, 1]})
    except Exception:  # noqa: BLE001
        pass

    return SegmentationResult(assignments=assignments, summary=summary, features=feats, k=k, labels=labels,
                              message=f"K-Means (k={k}) on {len(feats)} standardised supplier-level features: " + ", ".join(feature_label(f) for f in feats),
                              pca=pca_df)


# --------------------------------------------------------------------------- #
# Early warning system
# --------------------------------------------------------------------------- #
_REASON_WORDS = {1: "increased", -1: "dropped", 0: "changed"}


def early_warnings(records: pd.DataFrame, features: List[str], window: int = 3, min_change: float = 10.0) -> pd.DataFrame:
    """Suppliers whose risk rose by >= ``min_change`` points between two consecutive windows.

    Previous risk = mean of the ``window`` records before the last ``window``
    records; current risk = mean of the last ``window`` records.  Reasons are
    the features whose recent mean moved most in the risk direction
    (relative to the portfolio's standard deviation).
    """
    if records.empty or "Risk_Score" not in records.columns:
        return pd.DataFrame()
    order_col = "Date" if "Date" in records.columns else ("Record_Index" if "Record_Index" in records.columns else None)
    d = records.sort_values(order_col) if order_col else records
    feats = [f for f in features if f in d.columns and pd.api.types.is_numeric_dtype(d[f])]
    std = d[feats].std().replace(0, np.nan) if feats else pd.Series(dtype=float)
    rows = []
    for sid, g in d.groupby("Supplier_ID", sort=False):
        if len(g) < 2 * window:
            continue
        cur, prev = g.tail(window), g.iloc[-2 * window:-window]
        cur_risk, prev_risk = float(cur["Risk_Score"].mean()), float(prev["Risk_Score"].mean())
        change = cur_risk - prev_risk
        if change < min_change:
            continue
        reasons: List[Tuple[float, str]] = []
        for f in feats:
            delta = float(cur[f].mean() - prev[f].mean())
            if std.get(f) is None or np.isnan(std[f]):
                continue
            direction = config.FEATURE_DIRECTION.get(f, 0)
            z = delta / std[f]
            bad = z * direction if direction else abs(z)
            if bad > 0.25:
                reasons.append((bad, f"{feature_label(f)} {_REASON_WORDS[int(np.sign(delta))] if delta else 'changed'}"))
        reasons.sort(reverse=True)
        rows.append({
            "Supplier_ID": sid,
            "Supplier_Name": g["Supplier_Name"].iloc[-1] if "Supplier_Name" in g.columns else sid,
            "Previous_Risk": round(prev_risk, 1),
            "Current_Risk": round(cur_risk, 1),
            "Change": round(change, 1),
            "Reasons": ", ".join(r for _, r in reasons[:3]) or "Risk score rising",
            "Last_Seen": g[order_col].iloc[-1] if order_col else None,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("Change", ascending=False).reset_index(drop=True) if not out.empty else out
