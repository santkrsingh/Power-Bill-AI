"""
PowerGuard AI – Data Ingestion & Validation Module
Handles CSV/Excel upload, structural validation, and data cleaning.
"""

import io
import pandas as pd
import numpy as np
from typing import Tuple


# ─── Required schema ──────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "consumer_id",
    "meter_id",
    "billing_month",
    "previous_reading",
    "current_reading",
    "billed_units",
    "bill_amount",
]

NUMERIC_COLUMNS = [
    "previous_reading",
    "current_reading",
    "billed_units",
    "bill_amount",
]

OPTIONAL_COLUMNS = [
    "billing_date",
    "consumer_type",
    "tariff_category",
    "city_zone",
    "actual_units_from_reading",
]


# ─── Public API ───────────────────────────────────────────────────────────────

def ingest_file(file_obj, filename: str) -> Tuple[pd.DataFrame, list[str]]:
    """
    Read a CSV or Excel file and return (raw_df, warnings).
    Raises ValueError on unreadable / unsupported file types.
    """
    warnings = []
    ext = filename.rsplit(".", 1)[-1].lower()

    try:
        if ext == "csv":
            df = pd.read_csv(file_obj)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(file_obj)
        else:
            raise ValueError(f"Unsupported file type: .{ext}. Use CSV or Excel.")
    except Exception as exc:
        raise ValueError(f"Could not read file: {exc}") from exc

    warnings.extend(_check_columns(df))
    return df, warnings


def ingest_json(records: list[dict]) -> Tuple[pd.DataFrame, list[str]]:
    """Accept a list of dicts (from API JSON body) and return (df, warnings)."""
    df = pd.DataFrame(records)
    warnings = _check_columns(df)
    return df, warnings


def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Perform full validation and cleaning.
    Returns (cleaned_df, quality_report).
    """
    report: dict = {
        "original_rows": len(df),
        "issues": [],
    }

    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── 1. Coerce numeric columns ────────────────────────────────────────────
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            new_na = df[col].isna().sum() - before_na
            if new_na > 0:
                report["issues"].append(
                    f"{new_na} non-numeric value(s) coerced to NaN in '{col}'."
                )

    # ── 2. Remove exact duplicates ───────────────────────────────────────────
    dups = df.duplicated().sum()
    if dups:
        df = df.drop_duplicates()
        report["issues"].append(f"Removed {dups} exact duplicate row(s).")

    # ── 3. Flag / drop rows where consumer_id or meter_id is null ───────────
    missing_key = df["consumer_id"].isna() | df["meter_id"].isna()
    if missing_key.sum():
        df = df[~missing_key]
        report["issues"].append(
            f"Dropped {missing_key.sum()} row(s) with missing consumer_id / meter_id."
        )

    # ── 4. Negative readings check ───────────────────────────────────────────
    for col in ["previous_reading", "current_reading", "billed_units", "bill_amount"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg:
                df.loc[df[col] < 0, col] = np.nan
                report["issues"].append(
                    f"Set {neg} negative value(s) to NaN in '{col}'."
                )

    # ── 5. Logical consistency: current >= previous ──────────────────────────
    if {"previous_reading", "current_reading"}.issubset(df.columns):
        inversion = df["current_reading"] < df["previous_reading"]
        inv_count = inversion.sum()
        if inv_count:
            df.loc[inversion, ["previous_reading", "current_reading"]] = np.nan
            report["issues"].append(
                f"Nullified {inv_count} row(s) where current_reading < previous_reading."
            )

    # ── 6. Compute actual_units if missing ───────────────────────────────────
    if "actual_units_from_reading" not in df.columns:
        df["actual_units_from_reading"] = (
            df["current_reading"] - df["previous_reading"]
        ).round(2)

    # ── 7. Fill missing billed_units from bill_amount (rate ~6.50) ───────────
    rate_estimate = 6.50
    mask_bu = df["billed_units"].isna() & df["bill_amount"].notna()
    df.loc[mask_bu, "billed_units"] = (df.loc[mask_bu, "bill_amount"] / rate_estimate).round(2)
    if mask_bu.sum():
        report["issues"].append(
            f"Imputed {mask_bu.sum()} missing billed_units from bill_amount."
        )

    mask_ba = df["bill_amount"].isna() & df["billed_units"].notna()
    df.loc[mask_ba, "bill_amount"] = (df.loc[mask_ba, "billed_units"] * rate_estimate).round(2)
    if mask_ba.sum():
        report["issues"].append(
            f"Imputed {mask_ba.sum()} missing bill_amounts from billed_units."
        )

    # ── 8. Parse billing_month / billing_date ────────────────────────────────
    if "billing_month" in df.columns:
        df["billing_month"] = df["billing_month"].astype(str).str.strip()

    report["cleaned_rows"] = len(df)
    report["rows_removed"] = report["original_rows"] - report["cleaned_rows"]
    report["clean"] = len(report["issues"]) == 0

    return df, report


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _check_columns(df: pd.DataFrame) -> list[str]:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    warnings = []
    if missing:
        warnings.append(
            f"Missing required column(s): {', '.join(missing)}. "
            "Results may be incomplete."
        )
    return warnings
