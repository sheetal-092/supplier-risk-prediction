"""Anomaly detection with Isolation Forest + rule-based alert descriptions.

* :func:`detect_anomalies` fits an Isolation Forest on the profile's anomaly
  features (record level).  For every anomalous record the metric with the
  largest robust z-score decides the *alert type*, the magnitude decides the
  *severity*, and the interquantile band gives the *normal range*.
* :func:`live_alerts` produces the alert feed for Live Data Mode from a freshly
  scored batch: HIGH-RISK predictions, risk increases and anomalies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config
from src.schema import feature_label, feature_unit

ALERT_TYPES = {
    "Delivery_Delay_Days": "Unusual delivery delay",
    "Defect_Rate": "Abnormal defect rate",
    "Lead_Time_Days": "Unusual lead time",
    "Cost_Variation": "Unusual cost change",
    "Unit_Price": "Unusual price change",
    "Order_Quantity": "Abnormal order pattern",
    "Order_Value": "Abnormal order value",
    "Quality_Score": "Quality score drop",
    "Order_Fulfillment_Rate": "Fulfillment shortfall",
    "Performance_Score": "Performance drop",
    "Leverage": "Leverage spike",
    "Current_Ratio": "Liquidity drop",
    "Financial_Stability_ZScore": "Stability score drop",
    "Profit_Margin": "Margin deterioration",
    "Cash_Flow_Million_USD": "Cash-flow anomaly",
    "Debt_Million_USD": "Debt spike",
    "Revenue_Million_USD": "Revenue anomaly",
    "Financial_Stress_Index": "Financial stress spike",
}


@dataclass
class AnomalyResult:
    records: pd.DataFrame                  # scored df + Anomaly flags
    alerts: pd.DataFrame                   # one row per anomalous record
    features: List[str]
    contamination: float
    normal_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    message: str = ""

    @property
    def total(self) -> int:
        return int(len(self.alerts))

    @property
    def critical(self) -> int:
        return int((self.alerts["Severity"] == "Critical").sum()) if not self.alerts.empty else 0

    @property
    def suppliers_under_observation(self) -> int:
        if self.alerts.empty:
            return 0
        crit = set(self.alerts.loc[self.alerts["Severity"] == "Critical", "Supplier_ID"])
        return int(self.alerts["Supplier_ID"].nunique() - len(crit))


def _robust_z(df: pd.DataFrame, feats: List[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]], pd.Series, pd.Series]:
    med = df[feats].median()
    mad = (df[feats] - med).abs().median() * 1.4826
    mad = mad.replace(0, np.nan).fillna(df[feats].std().replace(0, np.nan)).fillna(1.0)
    z = (df[feats] - med) / mad
    ranges = {f: (float(df[f].quantile(0.05)), float(df[f].quantile(0.95))) for f in feats}
    return z, ranges, med, mad


def severity_from_z(z: float) -> str:
    z = abs(z)
    if z >= 4.5:
        return "Critical"
    if z >= 3.0:
        return "Warning"
    return "Attention"


def _fmt(v: float, unit: str) -> str:
    if abs(v) >= 1000:
        s = f"{v:,.0f}"
    elif abs(v) >= 10:
        s = f"{v:.1f}"
    else:
        s = f"{v:.2f}"
    return f"{s} {unit}".strip()


def detect_anomalies(df: pd.DataFrame, features: List[str], contamination: float = 0.03) -> Optional[AnomalyResult]:
    feats = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f]) and df[f].notna().any()]
    if len(feats) < 2 or len(df) < 20:
        return None
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except Exception:  # noqa: BLE001
        return None

    d = df.copy()
    X = d[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0.0)
    Z = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=200, contamination=float(contamination), random_state=42).fit(Z)
    d["Anomaly_Score"] = -iso.score_samples(Z)
    d["Is_Anomaly"] = iso.predict(Z) == -1

    z, ranges, med, mad = _robust_z(X, feats)
    # only deviations in the "bad" direction count for directional metrics
    directional = z.copy()
    for f in feats:
        direction = config.FEATURE_DIRECTION.get(f, 0)
        if direction == 1:
            directional[f] = directional[f].clip(lower=0)
        elif direction == -1:
            directional[f] = (-directional[f]).clip(lower=0)
        else:
            directional[f] = directional[f].abs()
    top_feat = directional.idxmax(axis=1)
    top_z = directional.max(axis=1)

    rows: List[Dict[str, object]] = []
    anomalies = d[d["Is_Anomaly"]]
    for idx, row in anomalies.iterrows():
        f = top_feat.loc[idx]
        zval = float(top_z.loc[idx])
        # sanity: if nothing deviates in a bad direction fall back to absolute z
        if zval < 1.0:
            f = z.loc[idx].abs().idxmax()
            zval = float(z.loc[idx].abs().max())
        lo, hi = ranges[f]
        unit = feature_unit(f)
        rows.append({
            "Supplier_ID": row["Supplier_ID"],
            "Supplier_Name": row.get("Supplier_Name", row["Supplier_ID"]),
            "Alert_Type": ALERT_TYPES.get(f, f"Unusual {feature_label(f).lower()}"),
            "Metric": f,
            "Severity": severity_from_z(zval),
            "Detected_Value": float(X.loc[idx, f]),
            "Detected": _fmt(float(X.loc[idx, f]), unit),
            "Normal_Range": f"{_fmt(lo, '')} - {_fmt(hi, unit)}",
            "Deviation_Z": round(zval, 2),
            "Risk_Level": row.get("Risk_Level", ""),
            "Risk_Score": row.get("Risk_Score", np.nan),
            "Date": row.get("Date", pd.NaT),
            "Anomaly_Score": round(float(row["Anomaly_Score"]), 4),
        })
    alerts = pd.DataFrame(rows)
    if not alerts.empty:
        sev_rank = {"Critical": 0, "Warning": 1, "Attention": 2}
        alerts["Status"] = alerts["Severity"].map(lambda s: f"{config.SEVERITY_ICONS[s]} {s}")
        alerts = alerts.sort_values(["Severity", "Deviation_Z"], key=lambda s: s.map(sev_rank) if s.name == "Severity" else -s).reset_index(drop=True)
    return AnomalyResult(records=d, alerts=alerts, features=feats, contamination=contamination, normal_ranges=ranges,
                         message=f"Isolation Forest on {len(feats)} features, contamination={contamination:.0%}")


def anomaly_counts_by_metric(alerts: pd.DataFrame) -> pd.DataFrame:
    if alerts.empty:
        return pd.DataFrame()
    out = alerts.groupby(["Alert_Type", "Severity"]).size().rename("Count").reset_index()
    return out


# --------------------------------------------------------------------------- #
# Live alert feed
# --------------------------------------------------------------------------- #
def live_alerts(sim, batch: pd.DataFrame, features: Optional[List[str]] = None) -> List[Dict[str, object]]:
    """Generate alerts from a freshly scored live batch.

    Alert types
    -----------
    * ``HIGH RISK``        - model puts the supplier in the HIGH band
    * ``RISK INCREASING``  - supplier's risk rose >= 15 points vs its previous value
    * ``ANOMALY DETECTED`` - Isolation Forest flags the record against the
                             recent rolling window
    """
    feats = features or config.PROFILES["operational"]["anomaly_features"]
    alerts: List[Dict[str, object]] = []
    now = datetime.now()
    recent = sim.dataframe()

    # anomaly model on the rolling window (cheap: window is bounded)
    anomaly_ids: set = set()
    reason_by_idx: Dict[int, str] = {}
    if len(recent) >= 60:
        try:
            window = recent.tail(1500)
            res = detect_anomalies(window, feats, contamination=0.03)
            if res is not None and not res.alerts.empty:
                tail_n = len(batch)
                flagged = res.records.tail(tail_n)
                flagged = flagged[flagged["Is_Anomaly"]]
                for _, r in flagged.iterrows():
                    anomaly_ids.add(r["Supplier_ID"])
                    match = res.alerts[(res.alerts["Supplier_ID"] == r["Supplier_ID"])]
                    if not match.empty:
                        m = match.iloc[-1]
                        reason_by_idx[r["Supplier_ID"]] = f"{m['Alert_Type']} ({m['Detected']}, normal {m['Normal_Range']})"
        except Exception:  # noqa: BLE001
            pass

    for _, row in batch.iterrows():
        sid = row["Supplier_ID"]
        score = float(row.get("Risk_Score", np.nan))
        if np.isnan(score):
            continue
        prev = sim.previous_risk(sid)
        sim.remember_risk(sid, score)
        level = row.get("Risk_Level", "")
        top_reason = _top_metric_reason(row)

        if level == "HIGH" and sim.cooldown_ok(f"high:{sid}", 40):
            alerts.append(_alert(now, sid, row, "HIGH RISK", "HIGH", "🔴", top_reason, score))
        elif prev is not None and score - prev >= 15 and score >= 33 and sim.cooldown_ok(f"inc:{sid}", 40):
            alerts.append(_alert(now, sid, row, "RISK INCREASING", level, "🟠",
                                 f"Risk rose {prev:.0f} -> {score:.0f}; {top_reason}", score))
        if sid in anomaly_ids and sim.cooldown_ok(f"anom:{sid}", 40):
            alerts.append(_alert(now, sid, row, "ANOMALY DETECTED", level, "🟡",
                                 reason_by_idx.get(sid, top_reason), score))
    return alerts


def _alert(ts: datetime, sid: str, row: pd.Series, kind: str, level: str, icon: str, reason: str, score: float) -> Dict[str, object]:
    return {
        "Time": ts,
        "Timestamp": ts.strftime("%H:%M:%S"),
        "Icon": icon,
        "Supplier_ID": sid,
        "Supplier_Name": row.get("Supplier_Name", sid),
        "Alert_Type": kind,
        "Risk_Level": level,
        "Risk_Score": round(score, 1),
        "Reason": reason,
    }


def _top_metric_reason(row: pd.Series) -> str:
    """Human readable reason based on the most extreme operational metric."""
    checks = [
        ("Delivery_Delay_Days", row.get("Delivery_Delay_Days"), 5, "Delivery delay {v:.1f} days"),
        ("Defect_Rate", row.get("Defect_Rate"), 5, "Defect rate {v:.1f}%"),
        ("Cost_Variation", row.get("Cost_Variation"), 10, "Cost variation {v:+.1f}%"),
        ("Order_Fulfillment_Rate", row.get("Order_Fulfillment_Rate"), None, "Fulfillment {v:.0f}%"),
        ("Quality_Score", row.get("Quality_Score"), None, "Quality score {v:.0f}"),
        ("Lead_Time_Days", row.get("Lead_Time_Days"), 25, "Lead time {v:.0f} days"),
    ]
    best: Optional[Tuple[float, str]] = None
    for name, val, thresh, template in checks:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        v = float(val)
        if name == "Order_Fulfillment_Rate":
            sev = max(0.0, (90 - v) / 10)
        elif name == "Quality_Score":
            sev = max(0.0, (80 - v) / 10)
        elif name == "Cost_Variation":
            sev = abs(v) / thresh
        else:
            sev = v / thresh
        if best is None or sev > best[0]:
            best = (sev, template.format(v=v))
    return best[1] if best else "Model risk score elevated"


# --------------------------------------------------------------------------- #
# Unified alert feed for Dataset Mode
# --------------------------------------------------------------------------- #
def build_alert_feed(supplier_df: pd.DataFrame, anomaly_alerts: Optional[pd.DataFrame], warnings: Optional[pd.DataFrame],
                     records: Optional[pd.DataFrame] = None, features: Optional[List[str]] = None, limit: int = 40) -> pd.DataFrame:
    """Combine HIGH RISK suppliers, RISK INCREASING warnings and Isolation-Forest anomalies into one feed.

    Everything is derived from the current data: no alert is invented.
    """
    rows: List[Dict[str, object]] = []

    def ts(value) -> str:
        try:
            if value is None or pd.isna(value):
                return ""
            return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
        except Exception:  # noqa: BLE001
            return str(value)

    # reasons for high-risk suppliers: metric with the largest bad deviation vs portfolio
    reason_by_supplier: Dict[str, str] = {}
    if records is not None and features:
        feats = [f for f in features if f in records.columns and pd.api.types.is_numeric_dtype(records[f])]
        if feats:
            z, ranges, med, mad = _robust_z(records[feats].fillna(records[feats].median()), feats)
            order_col = "Date" if "Date" in records.columns else None
            latest = (records.sort_values(order_col) if order_col else records).groupby("Supplier_ID").tail(1)
            for idx, row in latest.iterrows():
                best, best_v = None, 0.0
                for f in feats:
                    direction = config.FEATURE_DIRECTION.get(f, 0)
                    v = z.loc[idx, f] * direction if direction else abs(z.loc[idx, f])
                    if v > best_v:
                        best, best_v = f, float(v)
                if best is not None and best_v >= 1.0:
                    reason_by_supplier[row["Supplier_ID"]] = f"{ALERT_TYPES.get(best, feature_label(best))} ({_fmt(float(row[best]), feature_unit(best))})"

    if supplier_df is not None and not supplier_df.empty:
        high = supplier_df[supplier_df["Risk_Level"] == "HIGH"].sort_values("Risk_Score", ascending=False)
        for _, s in high.iterrows():
            rows.append({"Time": s.get("Last_Seen"), "Timestamp": ts(s.get("Last_Seen")), "Icon": "🔴", "Supplier_ID": s["Supplier_ID"],
                         "Supplier_Name": s.get("Supplier_Name", ""), "Alert_Type": "HIGH RISK", "Severity": "Critical", "Risk_Level": "HIGH",
                         "Risk_Score": s["Risk_Score"], "Reason": reason_by_supplier.get(s["Supplier_ID"], f"Model risk score {s['Risk_Score']:.0f}%"),
                         "Detected": f"{s['Risk_Score']:.0f}%"})
    if warnings is not None and not warnings.empty:
        for _, w in warnings.iterrows():
            rows.append({"Time": w.get("Last_Seen"), "Timestamp": ts(w.get("Last_Seen")), "Icon": "🟠", "Supplier_ID": w["Supplier_ID"],
                         "Supplier_Name": w.get("Supplier_Name", ""), "Alert_Type": "RISK INCREASING", "Severity": "Warning",
                         "Risk_Level": "HIGH" if w["Current_Risk"] >= config.RISK_MEDIUM_MAX * 100 else "MEDIUM" if w["Current_Risk"] >= config.RISK_LOW_MAX * 100 else "LOW",
                         "Risk_Score": w["Current_Risk"], "Reason": f"{w['Previous_Risk']:.0f}% → {w['Current_Risk']:.0f}%: {w['Reasons']}", "Detected": f"{w['Change']:+.0f}%"})
    if anomaly_alerts is not None and not anomaly_alerts.empty:
        recent = anomaly_alerts.sort_values("Date", ascending=False) if "Date" in anomaly_alerts.columns else anomaly_alerts
        for _, a in recent.head(limit).iterrows():
            rows.append({"Time": a.get("Date"), "Timestamp": ts(a.get("Date")), "Icon": "🟡", "Supplier_ID": a["Supplier_ID"],
                         "Supplier_Name": a.get("Supplier_Name", ""), "Alert_Type": "ANOMALY DETECTED", "Severity": a["Severity"],
                         "Risk_Level": a.get("Risk_Level", ""), "Risk_Score": a.get("Risk_Score", np.nan),
                         "Reason": f"{a['Alert_Type']}: {a['Detected']} (normal {a['Normal_Range']})", "Detected": a["Detected"]})
    feed = pd.DataFrame(rows)
    if feed.empty:
        return feed
    sev_rank = {"Critical": 0, "Warning": 1, "Attention": 2}
    feed["_r"] = feed["Severity"].map(sev_rank).fillna(3)
    feed["_t"] = pd.to_datetime(feed["Time"], errors="coerce")
    feed = feed.sort_values(["_r", "_t"], ascending=[True, False]).drop(columns=["_r", "_t"]).reset_index(drop=True)
    return feed
