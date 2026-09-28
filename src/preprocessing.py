"""Data loading, validation and preprocessing.

The cleaning steps reproduce the CAPSTONE_PROJECT notebook:

1. parse ``Date``            (``pd.to_datetime(errors="coerce")``)
2. drop exact duplicates
3. numeric coercion + median imputation for feature columns
4. categorical / text ``fillna("Unknown")``
5. engineered ``Financial_Stress_Index = Leverage + 1 / max(Current_Ratio, 0.1)``

Step 5 (and the de-duplication) was done with PySpark in the notebook.  The
same logic is available here through :func:`spark_preprocess` and is used
automatically for large datasets when PySpark + Java are available; otherwise
the identical pandas implementation runs.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config
from src.schema import SchemaInfo, apply_schema, detect_schema

SPARK_ROW_THRESHOLD = 150_000   # rows above which PySpark is preferred


class DataValidationError(Exception):
    """Raised for user-facing dataset problems (empty file, unreadable ...)."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_uploaded_file(uploaded) -> pd.DataFrame:
    """Read a Streamlit ``UploadedFile`` (CSV / XLSX / XLS) into a DataFrame."""
    name = getattr(uploaded, "name", "uploaded")
    raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
    if raw is None or len(raw) == 0:
        raise DataValidationError(f"'{name}' is empty (0 bytes).")
    lower = name.lower()
    try:
        if lower.endswith((".xlsx", ".xlsm", ".xls")):
            xls = pd.ExcelFile(io.BytesIO(raw))
            # pick the first sheet that has more than a handful of rows
            best_sheet = xls.sheet_names[0]
            best_rows = -1
            for sh in xls.sheet_names:
                try:
                    n = len(xls.parse(sh, nrows=50))
                except Exception:  # pragma: no cover - defensive
                    continue
                if n > best_rows:
                    best_rows, best_sheet = n, sh
            df = xls.parse(best_sheet)
        elif lower.endswith((".csv", ".txt")):
            df = _read_csv_bytes(raw)
        else:
            # unknown extension - try CSV then Excel
            try:
                df = _read_csv_bytes(raw)
            except Exception:
                df = pd.read_excel(io.BytesIO(raw))
    except DataValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataValidationError(
            f"Could not read '{name}'. Make sure it is a valid CSV or Excel file. Details: {exc}"
        ) from exc
    return _basic_frame_checks(df, name)


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise DataValidationError(f"CSV could not be parsed: {last_exc}")


def load_sample_dataset(kind: str = "financial") -> pd.DataFrame:
    """Load one of the bundled sample datasets."""
    if kind == "operational":
        path = config.SAMPLE_OPERATIONAL_CSV
        if not path.exists():
            raise DataValidationError("Sample operational dataset not found in the data folder.")
        return _basic_frame_checks(pd.read_csv(path), path.name)
    if kind == "processed":
        path = config.SAMPLE_PROCESSED_XLSX
        if not path.exists():
            raise DataValidationError("Processed sample dataset not found in the data folder.")
        return _basic_frame_checks(pd.read_excel(path, sheet_name="Processed_Data"), path.name)
    if config.SAMPLE_FINANCIAL_CSV.exists():
        return _basic_frame_checks(pd.read_csv(config.SAMPLE_FINANCIAL_CSV), config.SAMPLE_FINANCIAL_CSV.name)
    if config.SAMPLE_FINANCIAL_XLSX.exists():
        return _basic_frame_checks(
            pd.read_excel(config.SAMPLE_FINANCIAL_XLSX, sheet_name="Supplier_Risk_Data"),
            config.SAMPLE_FINANCIAL_XLSX.name,
        )
    raise DataValidationError("No sample dataset found in the data folder.")


def _basic_frame_checks(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df is None or df.shape[0] == 0:
        raise DataValidationError(f"'{name}' contains no data rows.")
    if df.shape[1] == 0:
        raise DataValidationError(f"'{name}' contains no columns.")
    # drop fully empty columns / unnamed index columns
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all")
    if df.shape[1] == 0:
        raise DataValidationError(f"'{name}' only contains empty columns.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# Profiling / validation summary
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSummary:
    records: int
    columns: int
    missing_cells: int
    missing_by_column: Dict[str, int]
    duplicate_rows: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    memory_mb: float
    schema: SchemaInfo
    warnings: List[str] = field(default_factory=list)


def summarise_dataset(df: pd.DataFrame) -> DatasetSummary:
    schema = detect_schema(df)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and c not in datetime_cols]
    missing = df.isnull().sum()
    warnings: List[str] = []
    if len(numeric) == 0:
        warnings.append("No numeric columns detected - risk prediction and analytics need numeric features.")
    if schema.profile is None:
        warnings.append(
            "Columns do not match the pre-trained model profiles. Prediction will fall back to "
            "on-the-fly training (if a label column exists) or an unsupervised risk index."
        )
    elif schema.missing_features:
        warnings.append(
            "Missing model features will be filled with training medians: " + ", ".join(schema.missing_features)
        )
    return DatasetSummary(
        records=int(df.shape[0]),
        columns=int(df.shape[1]),
        missing_cells=int(missing.sum()),
        missing_by_column={k: int(v) for k, v in missing[missing > 0].items()},
        duplicate_rows=int(df.duplicated().sum()),
        numeric_columns=numeric,
        categorical_columns=categorical,
        datetime_columns=datetime_cols,
        memory_mb=float(df.memory_usage(deep=True).sum() / 1e6),
        schema=schema,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def financial_stress_index(leverage: pd.Series, current_ratio: pd.Series) -> pd.Series:
    """Notebook feature: ``coalesce(Leverage,0) + 1/greatest(coalesce(Current_Ratio,1),0.1)``."""
    lev = pd.to_numeric(leverage, errors="coerce").fillna(0.0)
    cr = pd.to_numeric(current_ratio, errors="coerce").fillna(1.0).clip(lower=0.1)
    return lev + 1.0 / cr


def preprocess(df_raw: pd.DataFrame, schema: Optional[SchemaInfo] = None, use_spark: bool = False) -> Tuple[pd.DataFrame, SchemaInfo, Dict[str, object]]:
    """Clean *df_raw* and return ``(processed_df, schema, report)``.

    The original DataFrame is never modified.
    """
    schema = schema or detect_schema(df_raw)
    report: Dict[str, object] = {"engine": "pandas", "steps": []}
    df = apply_schema(df_raw, schema)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before - len(df)
    report["steps"].append(f"Removed {before - len(df)} duplicate rows")

    # --- identifiers -------------------------------------------------------
    if "Supplier_ID" not in df.columns:
        df["Supplier_ID"] = [f"ROW{idx + 1:05d}" for idx in range(len(df))]
        report["steps"].append("Created synthetic Supplier_ID")
    df["Supplier_ID"] = df["Supplier_ID"].astype(str).str.strip()
    if "Supplier_Name" not in df.columns:
        df["Supplier_Name"] = df["Supplier_ID"]
    df["Supplier_Name"] = df["Supplier_Name"].fillna(df["Supplier_ID"]).astype(str)

    # --- dates -------------------------------------------------------------
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        n_bad = int(df["Date"].isna().sum())
        if n_bad == len(df):
            df = df.drop(columns=["Date"])
            schema.has_date = False
            report["steps"].append("Date column could not be parsed and was ignored")
        elif n_bad:
            report["steps"].append(f"{n_bad} unparseable dates set to missing")
    if "Date" in df.columns:
        df = df.sort_values(["Date", "Supplier_ID"], na_position="last").reset_index(drop=True)
    else:
        df["Record_Index"] = np.arange(len(df))

    # --- numeric features --------------------------------------------------
    numeric_candidates = [c for c in df.columns if c not in ("Supplier_ID", "Supplier_Name", "Date")]
    imputed: Dict[str, float] = {}
    for col in numeric_candidates:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col]
        else:
            coerced = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce")
            # accept as numeric only if most values converted
            if coerced.notna().mean() >= 0.8 and df[col].notna().any():
                series = coerced
            else:
                continue
        n_missing = int(series.isna().sum())
        if n_missing:
            median = float(series.median()) if series.notna().any() else 0.0
            series = series.fillna(median)
            imputed[col] = median
        df[col] = series.astype(float) if not pd.api.types.is_bool_dtype(series) else series.astype(int)
    if imputed:
        report["steps"].append(f"Median-imputed {len(imputed)} numeric columns")
    report["imputed"] = imputed

    # --- categorical / text ------------------------------------------------
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].fillna("Unknown").astype(str)

    # --- percentage style columns given as fractions -----------------------
    for col in ("Defect_Rate", "Order_Fulfillment_Rate", "On_Time_Delivery", "Rejection_Rate"):
        if col in df.columns and df[col].max() <= 1.0 and df[col].min() >= 0:
            df[col] = df[col] * 100.0
            report["steps"].append(f"{col} converted from fraction to percent")

    # --- engineered features ----------------------------------------------
    if {"Leverage", "Current_Ratio"} <= set(df.columns):
        if use_spark:
            spark_df, ok = spark_preprocess(df)
            if ok:
                df = spark_df
                report["engine"] = "pyspark"
            else:
                df["Financial_Stress_Index"] = financial_stress_index(df["Leverage"], df["Current_Ratio"])
                report["steps"].append("PySpark unavailable - used pandas for Financial_Stress_Index")
        else:
            df["Financial_Stress_Index"] = financial_stress_index(df["Leverage"], df["Current_Ratio"])
        report["steps"].append("Engineered Financial_Stress_Index")
    if "Delivery_Delay_Days" in df.columns and "On_Time_Delivery" not in df.columns:
        df["On_Time_Delivery"] = (df["Delivery_Delay_Days"] <= 0).astype(float) * 100.0
        report["steps"].append("Derived On_Time_Delivery from Delivery_Delay_Days")
    if "Performance_Score" not in df.columns and {"Quality_Score", "Order_Fulfillment_Rate", "Delivery_Delay_Days"} <= set(df.columns):
        delay_pen = (1 - df["Delivery_Delay_Days"].clip(0, 30) / 30) * 100
        df["Performance_Score"] = (0.4 * df["Quality_Score"] + 0.4 * df["Order_Fulfillment_Rate"] + 0.2 * delay_pen).clip(0, 100)
        report["steps"].append("Derived Performance_Score")
    if "News_Combined" not in df.columns and {"News_Headline", "News_Text"} <= set(df.columns):
        df["News_Combined"] = df["News_Headline"].str.strip() + ". " + df["News_Text"].str.strip()

    # --- target ------------------------------------------------------------
    if schema.target and schema.target in df.columns:
        df[schema.target] = pd.to_numeric(df[schema.target], errors="coerce").fillna(0).astype(int).clip(0, 1)

    # re-detect after canonicalisation so downstream sees the final picture
    schema = detect_schema(df)
    return df, schema, report


# --------------------------------------------------------------------------- #
# Optional PySpark path (Big-Data engine)
# --------------------------------------------------------------------------- #
def spark_available() -> bool:
    try:
        import pyspark  # noqa: F401
        import shutil

        return shutil.which("java") is not None
    except Exception:  # noqa: BLE001
        return False


def spark_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Run the notebook's Spark steps (dropDuplicates + Financial_Stress_Index).

    Returns ``(dataframe, success)``.  Any failure falls back gracefully.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F

        spark = (
            SparkSession.builder.appName("SupplierRiskPreprocessing")
            .master("local[*]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )
        pdf = df.copy()
        if "Date" in pdf.columns:
            pdf["Date"] = pd.to_datetime(pdf["Date"], errors="coerce")
        sdf = spark.createDataFrame(pdf)
        sdf = sdf.dropDuplicates()
        sdf = sdf.withColumn(
            "Financial_Stress_Index",
            F.coalesce(F.col("Leverage"), F.lit(0.0))
            + (F.lit(1.0) / F.greatest(F.coalesce(F.col("Current_Ratio"), F.lit(1.0)), F.lit(0.1))),
        )
        if "Date" in pdf.columns:
            sdf = sdf.orderBy("Date", "Supplier_ID")
        out = sdf.toPandas()
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"])
        return out.reset_index(drop=True), True
    except Exception:  # noqa: BLE001
        return df, False
