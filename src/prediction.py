"""Risk prediction: model registry, compatibility checks, scoring and SHAP.

Scoring strategy (in order of preference):

1. **Pre-trained XGBoost** for the detected profile (financial / operational).
   Missing features (up to a limit) are filled with the training medians.
2. **On-the-fly XGBoost** trained on the uploaded data when a binary label
   column exists but the columns do not match a pre-trained model.
3. **Unsupervised risk index** (Isolation-Forest based, clearly labelled as a
   heuristic) when no model can be applied.  This is never presented as an
   XGBoost prediction.

If the dataset already carries a news-risk column (FinBERT output from the
notebook) the notebook's fusion ``0.6 * financial + 0.4 * news`` is applied.
If only the reference sentiment label exists, an empirical proxy (mean FinBERT
negative probability per label, computed from the processed dataset during
training) is used and reported as such.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src import config
from src.schema import SchemaInfo, detect_schema

MAX_MISSING_FRACTION = 0.3


@dataclass
class ModelBundle:
    key: str
    model: object
    features: List[str]
    profile: str
    algorithm: str
    medians: Dict[str, float]
    metrics: Dict[str, float] = field(default_factory=dict)
    source: str = ""
    training_rows: int = 0


@dataclass
class ScoringResult:
    df: pd.DataFrame
    method: str                    # 'pretrained' | 'on_the_fly' | 'unsupervised' | 'none'
    model_key: Optional[str]
    profile: Optional[str]
    features: List[str]
    messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    news_source: Optional[str] = None
    bundle: Optional[ModelBundle] = None

    @property
    def ok(self) -> bool:
        return "Risk_Score" in self.df.columns


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class ModelRegistry:
    def __init__(self) -> None:
        self.meta: Dict[str, object] = {}
        self.bundles: Dict[str, ModelBundle] = {}
        self.load_errors: List[str] = []
        self.sentiment_proxy: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not config.MODEL_REGISTRY_FILE.exists():
            self.load_errors.append(
                "models/model_registry.json not found. Run `python scripts/train_models.py` to create the models."
            )
            return
        try:
            with open(config.MODEL_REGISTRY_FILE, "r", encoding="utf-8") as fh:
                self.meta = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            self.load_errors.append(f"Model registry could not be read: {exc}")
            return
        self.sentiment_proxy = {k: float(v) for k, v in self.meta.get("sentiment_proxy", {}).items()}
        for key, entry in self.meta.get("models", {}).items():
            path = config.MODELS_DIR / entry.get("file", "")
            if not path.exists():
                self.load_errors.append(f"Model file missing: {path.name}")
                continue
            try:
                model = joblib.load(path)
            except Exception as exc:  # noqa: BLE001
                self.load_errors.append(f"Could not load {path.name}: {exc}")
                continue
            self.bundles[key] = ModelBundle(
                key=key,
                model=model,
                features=list(entry.get("features", [])),
                profile=entry.get("profile", ""),
                algorithm=entry.get("algorithm", key),
                medians={k: float(v) for k, v in entry.get("medians", {}).items()},
                metrics=entry.get("metrics", {}),
                source=entry.get("source", ""),
                training_rows=int(entry.get("training_rows", 0)),
            )

    @property
    def available(self) -> bool:
        return bool(self.bundles)

    def get(self, key: str) -> Optional[ModelBundle]:
        return self.bundles.get(key)

    def for_profile(self, profile: str, algorithm: str = "xgboost") -> Optional[ModelBundle]:
        return self.bundles.get(f"{algorithm}_{profile}")


_REGISTRY: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ModelRegistry()
    return _REGISTRY


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def risk_level(score01: float) -> str:
    if score01 < config.RISK_LOW_MAX:
        return "LOW"
    if score01 < config.RISK_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def risk_levels(series: pd.Series) -> pd.Series:
    return pd.cut(
        series.astype(float).clip(0, 1),
        bins=[-np.inf, config.RISK_LOW_MAX, config.RISK_MEDIUM_MAX, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)


def _prepare_matrix(df: pd.DataFrame, bundle: ModelBundle) -> Tuple[pd.DataFrame, List[str]]:
    X = pd.DataFrame(index=df.index)
    missing: List[str] = []
    for f in bundle.features:
        if f in df.columns:
            X[f] = pd.to_numeric(df[f], errors="coerce")
            if X[f].isna().any():
                X[f] = X[f].fillna(bundle.medians.get(f, float(X[f].median()) if X[f].notna().any() else 0.0))
        else:
            missing.append(f)
            X[f] = bundle.medians.get(f, 0.0)
    return X, missing


def _news_risk(df: pd.DataFrame, registry: ModelRegistry) -> Tuple[Optional[pd.Series], Optional[str]]:
    if "News_Risk_Probability" in df.columns:
        return pd.to_numeric(df["News_Risk_Probability"], errors="coerce").fillna(0.0).clip(0, 1), "FinBERT/LSTM news risk column"
    if "FinBERT_Negative_Probability" in df.columns:
        return pd.to_numeric(df["FinBERT_Negative_Probability"], errors="coerce").fillna(0.0).clip(0, 1), "FinBERT negative probability column"
    if "Synthetic_News_Sentiment" in df.columns and registry.sentiment_proxy:
        labels = df["Synthetic_News_Sentiment"].astype(str).str.strip().str.title()
        mapped = labels.map(registry.sentiment_proxy)
        if mapped.notna().mean() > 0.5:
            return mapped.fillna(float(np.mean(list(registry.sentiment_proxy.values())))).clip(0, 1), \
                "sentiment-label proxy (mean FinBERT negative probability per label)"
    return None, None


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def score_dataframe(df: pd.DataFrame, schema: Optional[SchemaInfo] = None, registry: Optional[ModelRegistry] = None,
                    allow_on_the_fly: bool = True, allow_unsupervised: bool = True) -> ScoringResult:
    """Append ``Risk_Probability``, ``Risk_Score`` (0-100) and ``Risk_Level`` to a copy of *df*."""
    registry = registry or get_registry()
    schema = schema or detect_schema(df)
    out = df.copy()
    result = ScoringResult(df=out, method="none", model_key=None, profile=schema.profile, features=[])
    result.errors.extend(registry.load_errors)

    if out.empty:
        result.errors.append("The dataset is empty - nothing to score.")
        return result

    # 1. pre-trained model ------------------------------------------------
    bundle = registry.for_profile(schema.profile) if schema.profile else None
    if bundle is not None:
        X, missing = _prepare_matrix(out, bundle)
        if len(missing) / max(len(bundle.features), 1) <= MAX_MISSING_FRACTION:
            try:
                prob = bundle.model.predict_proba(X)[:, 1]
                out["Model_Risk_Probability"] = prob
                result.method, result.model_key, result.features, result.bundle = "pretrained", bundle.key, bundle.features, bundle
                result.messages.append(f"Scored with pre-trained {bundle.algorithm} ({config.PROFILES[bundle.profile]['label']}).")
                if missing:
                    result.warnings.append("Missing features filled with training medians: " + ", ".join(missing))
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"The pre-trained model could not score this dataset: {exc}")
        else:
            result.warnings.append(
                f"Pre-trained model needs {len(bundle.features)} features but {len(missing)} are missing "
                f"({', '.join(missing[:6])}{'...' if len(missing) > 6 else ''})."
            )
    elif schema.profile and not registry.available:
        result.errors.append("No trained models are available (models folder empty).")

    # 2. on-the-fly training ---------------------------------------------
    if "Model_Risk_Probability" not in out.columns and allow_on_the_fly and schema.target and schema.target in out.columns:
        feats = [c for c in out.select_dtypes(include=[np.number]).columns
                 if c not in (schema.target, "Record_Index") and not c.startswith(("Risk_", "Model_", "XGBoost_", "Final_", "LSTM_", "FinBERT_", "News_Risk"))]
        y = pd.to_numeric(out[schema.target], errors="coerce").fillna(0).astype(int)
        if len(feats) >= 2 and y.nunique() == 2 and len(out) >= 50:
            try:
                from xgboost import XGBClassifier

                X = out[feats].apply(pd.to_numeric, errors="coerce")
                X = X.fillna(X.median())
                neg, pos = max(int((y == 0).sum()), 1), max(int((y == 1).sum()), 1)
                model = XGBClassifier(**config.XGB_PARAMS, scale_pos_weight=neg / pos)
                model.fit(X, y)
                out["Model_Risk_Probability"] = model.predict_proba(X)[:, 1]
                bundle = ModelBundle(key="xgboost_on_the_fly", model=model, features=feats, profile=schema.profile or "custom",
                                     algorithm="XGBoost (trained on uploaded data)", medians={f: float(X[f].median()) for f in feats},
                                     training_rows=len(X), source="uploaded dataset")
                result.method, result.model_key, result.features, result.bundle = "on_the_fly", bundle.key, feats, bundle
                result.messages.append(
                    f"No compatible pre-trained model - trained an XGBoost model on the uploaded data "
                    f"using label '{schema.target}' and {len(feats)} numeric features (in-sample scores)."
                )
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"On-the-fly training failed: {exc}")

    # 3. unsupervised fallback --------------------------------------------
    if "Model_Risk_Probability" not in out.columns and allow_unsupervised:
        feats = [c for c in out.select_dtypes(include=[np.number]).columns if c not in ("Record_Index",)]
        if len(feats) >= 2 and len(out) >= 10:
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler

                X = out[feats].apply(pd.to_numeric, errors="coerce")
                X = X.fillna(X.median()).fillna(0.0)
                Z = StandardScaler().fit_transform(X)
                iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42).fit(Z)
                raw = -iso.score_samples(Z)  # higher = more anomalous
                rank = pd.Series(raw, index=out.index).rank(pct=True)
                out["Model_Risk_Probability"] = rank.values
                result.method, result.model_key, result.features = "unsupervised", "isolation_forest_index", feats
                result.warnings.append(
                    "No trained model matches this dataset and no label column was found. Showing an UNSUPERVISED "
                    "risk index (Isolation-Forest anomaly rank). This is a heuristic, not an XGBoost prediction."
                )
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"Unsupervised fallback failed: {exc}")
        else:
            result.errors.append("Not enough numeric columns to compute any risk score.")

    if "Model_Risk_Probability" not in out.columns:
        return result

    # fusion with news risk (notebook: 0.6 financial + 0.4 news) --------------
    news, news_src = _news_risk(out, registry)
    if news is not None and result.method == "pretrained" and result.profile == "financial":
        out["Financial_Risk_Probability"] = out["Model_Risk_Probability"]
        out["News_Risk_Probability"] = news.values
        out["Risk_Probability"] = config.FINANCIAL_WEIGHT * out["Financial_Risk_Probability"] + config.NEWS_WEIGHT * out["News_Risk_Probability"]
        result.news_source = news_src
        result.messages.append(f"Final risk = 0.6 x financial (XGBoost) + 0.4 x news risk ({news_src}).")
    else:
        out["Risk_Probability"] = out["Model_Risk_Probability"]

    out["Risk_Probability"] = out["Risk_Probability"].astype(float).clip(0, 1)
    out["Risk_Score"] = (out["Risk_Probability"] * 100).round(1)
    out["Risk_Level"] = risk_levels(out["Risk_Probability"])
    result.df = out
    return result


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #
def explain_rows(bundle: ModelBundle, rows: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[float], str]:
    """SHAP contributions for *rows* (averaged) -> (contrib_df, base_value, method).

    Falls back to model feature importances when SHAP is unavailable.
    """
    X, _ = _prepare_matrix(rows, bundle)
    try:
        import shap

        explainer = shap.TreeExplainer(bundle.model)
        values = explainer.shap_values(X)
        if isinstance(values, list):          # RandomForest returns [class0, class1]
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:                  # (n, features, classes)
            values = values[:, :, 1]
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.ravel(base)[-1])
        contrib = pd.DataFrame({
            "Feature": bundle.features,
            "Contribution": values.mean(axis=0),
            "Value": X.mean(axis=0).values,
        })
        contrib["Abs"] = contrib["Contribution"].abs()
        contrib = contrib.sort_values("Abs", ascending=False).drop(columns="Abs").reset_index(drop=True)
        return contrib, float(base), "shap"
    except Exception:  # noqa: BLE001
        pass
    try:
        imp = getattr(bundle.model, "feature_importances_", None)
        if imp is None:
            return None, None, "none"
        contrib = pd.DataFrame({"Feature": bundle.features, "Contribution": imp, "Value": X.mean(axis=0).values})
        return contrib.sort_values("Contribution", ascending=False).reset_index(drop=True), None, "importance"
    except Exception:  # noqa: BLE001
        return None, None, "none"


def global_importance(bundle: ModelBundle) -> Optional[pd.DataFrame]:
    imp = getattr(bundle.model, "feature_importances_", None)
    if imp is None:
        return None
    return pd.DataFrame({"Feature": bundle.features, "Importance": imp}).sort_values("Importance", ascending=False)
