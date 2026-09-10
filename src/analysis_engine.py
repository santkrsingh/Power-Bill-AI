"""
PowerGuard AI – Consumption Analysis & Fraud Detection Engine
Deterministic, rule-based + statistical anomaly detection (no LLM dependency).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class AnomalyFlag:
    code: str
    description: str
    severity: str          # "low" | "medium" | "high"
    deviation_pct: float = 0.0


@dataclass
class ConsumerAnalysis:
    consumer_id: str
    meter_id: str
    consumer_type: str = "unknown"
    city_zone: str = "unknown"
    num_bills: int = 0

    # Consumption stats
    avg_consumption: float = 0.0
    std_consumption: float = 0.0
    latest_consumption: float = 0.0
    min_consumption: float = 0.0
    max_consumption: float = 0.0
    consumption_trend: float = 0.0    # slope (units/month)

    # Anomaly detection
    anomaly_flags: list[AnomalyFlag] = field(default_factory=list)
    anomaly_score: float = 0.0       # 0-1 from IsolationForest
    risk_score: int = 0              # 0-100
    risk_level: str = "Normal"

    # Mismatch
    reading_mismatch_count: int = 0
    reading_mismatch_pct: float = 0.0

    # Per-month details
    monthly_detail: list[dict] = field(default_factory=list)


# ─── Feature Engineering ──────────────────────────────────────────────────────

def _safe_pct_deviation(value: float, reference: float) -> float:
    """Percentage deviation of value from reference. Returns 0 if reference≈0."""
    if abs(reference) < 1e-6:
        return 0.0
    return round((value - reference) / reference * 100, 2)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived feature columns to the cleaned billing dataframe.
    """
    df = df.copy()
    df = df.sort_values(["consumer_id", "billing_month"]).reset_index(drop=True)

    # Per-consumer historical stats
    grp = df.groupby("consumer_id")["billed_units"]
    df["hist_mean"]   = grp.transform("mean")
    df["hist_std"]    = grp.transform("std").fillna(0)
    df["hist_median"] = grp.transform("median")

    # Rolling 3-month average (shift to avoid data leakage)
    df["rolling_3m_avg"] = (
        df.groupby("consumer_id")["billed_units"]
          .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    # Deviation from historical mean
    df["dev_from_mean_pct"] = (
        (df["billed_units"] - df["hist_mean"]) / df["hist_mean"].replace(0, np.nan) * 100
    ).round(2)

    # Month-over-month change
    df["mom_change"] = df.groupby("consumer_id")["billed_units"].diff()
    df["mom_change_pct"] = (
        df["mom_change"] / df.groupby("consumer_id")["billed_units"]
              .shift(1).replace(0, np.nan) * 100
    ).round(2)

    # Reading vs billed mismatch (allow 2% tolerance)
    if "actual_units_from_reading" in df.columns:
        df["reading_billed_diff"] = (
            df["actual_units_from_reading"] - df["billed_units"]
        ).abs().round(2)
        df["reading_mismatch"] = (
            df["reading_billed_diff"] >
            (df["actual_units_from_reading"].abs() * 0.02 + 1)
        )
    else:
        df["reading_billed_diff"] = 0.0
        df["reading_mismatch"] = False

    # Z-score of billed_units per consumer
    df["zscore"] = df.groupby("consumer_id")["billed_units"].transform(
        lambda s: stats.zscore(s, nan_policy="omit") if len(s) > 2 else pd.Series(0, index=s.index)
    )

    # Bill amount per unit (should be roughly constant)
    df["rate_per_unit"] = (
        df["bill_amount"] / df["billed_units"].replace(0, np.nan)
    ).round(4)

    return df


# ─── Rule-based checks ────────────────────────────────────────────────────────

# Thresholds
SUDDEN_DROP_THRESHOLD   = -60.0   # % drop from hist mean  → high
SUDDEN_SPIKE_THRESHOLD  = +200.0  # % spike from hist mean → medium
ZERO_CONSUMPTION_MAX    = 5.0     # units; near-zero       → high
LARGE_DEV_THRESHOLD     = 80.0    # |dev| > 80%            → medium
MISMATCH_MIN_DIFF       = 5.0     # units difference counts as mismatch
CONSECUTIVE_LOW_MONTHS  = 3       # run of N below 30% of mean
CONSECUTIVE_HIGH_MONTHS = 3       # run of N above 200% of mean
ZSCORE_HIGH             = 2.5
ZSCORE_EXTREME          = 3.5


def _run_rule_checks(consumer_id: str, cdf: pd.DataFrame) -> list[AnomalyFlag]:
    """Apply deterministic rule-based fraud checks for a single consumer."""
    flags: list[AnomalyFlag] = []
    hist_mean = cdf["hist_mean"].iloc[0]

    # ── Rule 1: Sudden Drop ───────────────────────────────────────────────────
    drop_rows = cdf[cdf["dev_from_mean_pct"] < SUDDEN_DROP_THRESHOLD]
    if len(drop_rows):
        worst = drop_rows["dev_from_mean_pct"].min()
        flags.append(AnomalyFlag(
            code="SUDDEN_DROP",
            description=(
                f"Consumption dropped {abs(worst):.1f}% below historical average "
                f"in {len(drop_rows)} billing cycle(s). Possible meter bypass or tampering."
            ),
            severity="high",
            deviation_pct=worst,
        ))

    # ── Rule 2: Unusual Spike ─────────────────────────────────────────────────
    spike_rows = cdf[cdf["dev_from_mean_pct"] > SUDDEN_SPIKE_THRESHOLD]
    if len(spike_rows):
        worst = spike_rows["dev_from_mean_pct"].max()
        flags.append(AnomalyFlag(
            code="UNUSUAL_SPIKE",
            description=(
                f"Consumption spiked {worst:.1f}% above historical average "
                f"in {len(spike_rows)} cycle(s). Possible illegal connection."
            ),
            severity="medium",
            deviation_pct=worst,
        ))

    # ── Rule 3: Zero / Near-Zero Consumption ─────────────────────────────────
    zero_rows = cdf[cdf["billed_units"] <= ZERO_CONSUMPTION_MAX]
    if len(zero_rows) >= 2 and hist_mean > 30:
        flags.append(AnomalyFlag(
            code="ZERO_CONSUMPTION",
            description=(
                f"Near-zero consumption (≤{ZERO_CONSUMPTION_MAX} units) recorded "
                f"in {len(zero_rows)} cycle(s) despite active connection history."
            ),
            severity="high",
            deviation_pct=-100.0,
        ))

    # ── Rule 4: Large Deviation ───────────────────────────────────────────────
    large_dev = cdf[cdf["dev_from_mean_pct"].abs() > LARGE_DEV_THRESHOLD]
    if len(large_dev) >= 3:
        flags.append(AnomalyFlag(
            code="LARGE_DEVIATION",
            description=(
                f"Consumption deviated >{LARGE_DEV_THRESHOLD}% from historical mean "
                f"in {len(large_dev)} cycles."
            ),
            severity="medium",
            deviation_pct=large_dev["dev_from_mean_pct"].mean(),
        ))

    # ── Rule 5: Consecutive Abnormal Low ─────────────────────────────────────
    low_mask = cdf["billed_units"] < (hist_mean * 0.30)
    consecutive_low = _max_consecutive(low_mask)
    if consecutive_low >= CONSECUTIVE_LOW_MONTHS:
        flags.append(AnomalyFlag(
            code="REPEATED_LOW",
            description=(
                f"Consumption remained below 30% of historical average for "
                f"{consecutive_low} consecutive months."
            ),
            severity="high",
            deviation_pct=-70.0,
        ))

    # ── Rule 6: Consecutive Abnormal High ────────────────────────────────────
    high_mask = cdf["billed_units"] > (hist_mean * 3.0)
    consecutive_high = _max_consecutive(high_mask)
    if consecutive_high >= CONSECUTIVE_HIGH_MONTHS:
        flags.append(AnomalyFlag(
            code="REPEATED_HIGH",
            description=(
                f"Consumption stayed above 300% of historical average for "
                f"{consecutive_high} consecutive months. Possible unauthorized usage."
            ),
            severity="medium",
            deviation_pct=200.0,
        ))

    # ── Rule 7: Reading vs Billed Mismatch ───────────────────────────────────
    mismatch_rows = cdf[cdf["reading_mismatch"] == True]
    if len(mismatch_rows) >= 2:
        avg_diff = mismatch_rows["reading_billed_diff"].mean()
        flags.append(AnomalyFlag(
            code="READING_MISMATCH",
            description=(
                f"Meter reading and billed units differ significantly "
                f"(avg. {avg_diff:.1f} units off) in {len(mismatch_rows)} cycle(s)."
            ),
            severity="high",
            deviation_pct=0.0,
        ))

    # ── Rule 8: Z-Score Extreme ───────────────────────────────────────────────
    extreme_rows = cdf[cdf["zscore"].abs() > ZSCORE_EXTREME]
    if len(extreme_rows):
        flags.append(AnomalyFlag(
            code="STATISTICAL_OUTLIER",
            description=(
                f"{len(extreme_rows)} billing cycle(s) are statistical outliers "
                f"(|z-score| > {ZSCORE_EXTREME})."
            ),
            severity="medium",
            deviation_pct=0.0,
        ))

    return flags


def _max_consecutive(mask: pd.Series) -> int:
    """Return the length of the longest consecutive True run."""
    max_run = 0
    run = 0
    for v in mask:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


# ─── ML Anomaly Detection (Isolation Forest) ─────────────────────────────────

FEATURE_COLS = [
    "billed_units",
    "dev_from_mean_pct",
    "mom_change_pct",
    "zscore",
    "reading_billed_diff",
]


def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit IsolationForest on per-consumer aggregated features.
    Adds 'if_anomaly_score' column (0-1, higher = more anomalous).
    """
    df = df.copy()

    feat_df = df.groupby("consumer_id").agg(
        mean_units        = ("billed_units", "mean"),
        std_units         = ("billed_units", "std"),
        min_units         = ("billed_units", "min"),
        max_units         = ("billed_units", "max"),
        mean_dev_pct      = ("dev_from_mean_pct", "mean"),
        max_abs_dev_pct   = ("dev_from_mean_pct", lambda x: x.abs().max()),
        mean_mom_change   = ("mom_change_pct", lambda x: x.abs().mean()),
        mismatch_count    = ("reading_mismatch", "sum"),
        mean_zscore_abs   = ("zscore", lambda x: x.abs().mean()),
        max_zscore_abs    = ("zscore", lambda x: x.abs().max()),
        zero_count        = ("billed_units", lambda x: (x <= 5).sum()),
    ).reset_index()

    feat_df = feat_df.fillna(0)

    X = feat_df.drop(columns=["consumer_id"]).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(
        n_estimators=200,
        contamination=0.20,
        random_state=42,
    )
    clf.fit(X_scaled)

    # decision_function: lower = more anomalous. Normalise to [0,1].
    raw_scores = clf.decision_function(X_scaled)
    normalised = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    feat_df["if_anomaly_score"] = normalised.round(4)

    return df.merge(feat_df[["consumer_id", "if_anomaly_score"]], on="consumer_id", how="left")


# ─── Main Orchestrator ────────────────────────────────────────────────────────

def analyze_all_consumers(df: pd.DataFrame) -> list[ConsumerAnalysis]:
    """
    Full pipeline:
      feature engineering → IsolationForest → rule checks → per-consumer analysis.
    """
    df_feat = engineer_features(df)
    df_feat = run_isolation_forest(df_feat)

    results: list[ConsumerAnalysis] = []

    for consumer_id, cdf in df_feat.groupby("consumer_id"):
        cdf = cdf.sort_values("billing_month").reset_index(drop=True)
        meta_row = cdf.iloc[0]

        flags = _run_rule_checks(consumer_id, cdf)

        # Consumption trend (linear regression slope)
        if len(cdf) >= 3:
            x = np.arange(len(cdf))
            valid = cdf["billed_units"].dropna()
            if len(valid) >= 3:
                slope, *_ = np.polyfit(x[:len(valid)], valid.values, 1)
            else:
                slope = 0.0
        else:
            slope = 0.0

        # Mismatch stats
        mismatch_count = int(cdf["reading_mismatch"].sum())
        mismatch_pct = round(mismatch_count / max(len(cdf), 1) * 100, 1)

        # IF anomaly score (same for all rows of a consumer)
        if_score = float(cdf["if_anomaly_score"].iloc[0])

        # Build monthly detail
        monthly = []
        for _, row in cdf.iterrows():
            monthly.append({
                "billing_month":    row["billing_month"],
                "billed_units":     row["billed_units"],
                "bill_amount":      row.get("bill_amount", 0),
                "dev_from_mean_pct": row["dev_from_mean_pct"],
                "mom_change_pct":    row["mom_change_pct"],
                "zscore":            row["zscore"],
                "reading_mismatch":  bool(row["reading_mismatch"]),
                "reading_billed_diff": row["reading_billed_diff"],
            })

        analysis = ConsumerAnalysis(
            consumer_id       = str(consumer_id),
            meter_id          = str(meta_row.get("meter_id", "")),
            consumer_type     = str(meta_row.get("consumer_type", "unknown")),
            city_zone         = str(meta_row.get("city_zone", "unknown")),
            num_bills         = len(cdf),
            avg_consumption   = round(float(cdf["billed_units"].mean()), 2),
            std_consumption   = round(float(cdf["billed_units"].std()), 2),
            latest_consumption= round(float(cdf["billed_units"].iloc[-1]), 2),
            min_consumption   = round(float(cdf["billed_units"].min()), 2),
            max_consumption   = round(float(cdf["billed_units"].max()), 2),
            consumption_trend = round(float(slope), 3),
            anomaly_flags     = flags,
            anomaly_score     = if_score,
            reading_mismatch_count = mismatch_count,
            reading_mismatch_pct   = mismatch_pct,
            monthly_detail    = monthly,
        )
        results.append(analysis)

    return results
