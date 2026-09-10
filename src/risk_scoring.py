"""
PowerGuard AI – Risk Scoring Module
Converts anomaly flags + ML score into a deterministic 0-100 risk score.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis_engine import ConsumerAnalysis, AnomalyFlag


# ─── Scoring weights ──────────────────────────────────────────────────────────

# Each flag code has a base contribution (0-100 scale)
FLAG_WEIGHTS: dict[str, int] = {
    "SUDDEN_DROP":         35,
    "ZERO_CONSUMPTION":    35,
    "REPEATED_LOW":        30,
    "READING_MISMATCH":    30,
    "UNUSUAL_SPIKE":       20,
    "LARGE_DEVIATION":     18,
    "REPEATED_HIGH":       18,
    "STATISTICAL_OUTLIER": 12,
}

# Severity multipliers
SEVERITY_MULT: dict[str, float] = {
    "high":   1.0,
    "medium": 0.65,
    "low":    0.35,
}

# ML model score weight (max contribution)
IF_SCORE_MAX_CONTRIBUTION = 25

# Risk bands
RISK_BANDS = [
    (81, 100, "High Risk"),
    (61, 80,  "Suspicious"),
    (31, 60,  "Low Risk"),
    (0,  30,  "Normal"),
]


def classify_risk(score: int) -> str:
    for lo, hi, label in RISK_BANDS:
        if lo <= score <= hi:
            return label
    return "Normal"


def compute_risk_score(analysis: "ConsumerAnalysis") -> tuple[int, str]:
    """
    Returns (risk_score 0-100, risk_level string).
    Deterministic — pure arithmetic, no LLM.
    """
    # ── Rule-based component ─────────────────────────────────────────────────
    rule_total = 0.0
    seen_codes: set[str] = set()

    for flag in analysis.anomaly_flags:
        if flag.code in seen_codes:
            continue
        seen_codes.add(flag.code)
        base   = FLAG_WEIGHTS.get(flag.code, 10)
        mult   = SEVERITY_MULT.get(flag.severity, 0.5)
        rule_total += base * mult

    # Mismatch bonus (in addition to flag, proportional to frequency)
    if analysis.reading_mismatch_pct > 0:
        mismatch_bonus = min(15, analysis.reading_mismatch_pct * 0.5)
        rule_total += mismatch_bonus

    rule_capped = min(75, rule_total)   # rule-based max contribution = 75

    # ── ML component ─────────────────────────────────────────────────────────
    ml_contribution = analysis.anomaly_score * IF_SCORE_MAX_CONTRIBUTION

    # ── Combined score ────────────────────────────────────────────────────────
    raw = rule_capped + ml_contribution
    final = max(0, min(100, round(raw)))

    return final, classify_risk(final)


def score_all(analyses: list["ConsumerAnalysis"]) -> list["ConsumerAnalysis"]:
    """Apply risk scoring to every ConsumerAnalysis in-place and return sorted list."""
    for a in analyses:
        a.risk_score, a.risk_level = compute_risk_score(a)

    # Sort: highest risk first
    analyses.sort(key=lambda x: x.risk_score, reverse=True)
    return analyses


def build_summary_stats(analyses: list["ConsumerAnalysis"]) -> dict:
    """Aggregate statistics for the dashboard."""
    total = len(analyses)
    if total == 0:
        return {}

    level_counts = {"Normal": 0, "Low Risk": 0, "Suspicious": 0, "High Risk": 0}
    for a in analyses:
        level_counts[a.risk_level] = level_counts.get(a.risk_level, 0) + 1

    total_bills = sum(a.num_bills for a in analyses)
    mismatch_consumers = sum(1 for a in analyses if a.reading_mismatch_count > 0)

    high_risk = [a for a in analyses if a.risk_level == "High Risk"]
    suspicious = [a for a in analyses if a.risk_level == "Suspicious"]
    flagged = high_risk + suspicious

    flag_code_counts: dict[str, int] = {}
    for a in analyses:
        for f in a.anomaly_flags:
            flag_code_counts[f.code] = flag_code_counts.get(f.code, 0) + 1

    return {
        "total_consumers": total,
        "total_bills": total_bills,
        "normal_consumers": level_counts["Normal"],
        "low_risk_consumers": level_counts["Low Risk"],
        "suspicious_consumers": level_counts["Suspicious"],
        "high_risk_consumers": level_counts["High Risk"],
        "flagged_consumers": len(flagged),
        "mismatch_consumers": mismatch_consumers,
        "top_10_high_risk": [
            {
                "consumer_id":  a.consumer_id,
                "meter_id":     a.meter_id,
                "risk_score":   a.risk_score,
                "risk_level":   a.risk_level,
                "consumer_type": a.consumer_type,
                "city_zone":    a.city_zone,
                "flags":        [f.code for f in a.anomaly_flags],
            }
            for a in analyses[:10]
        ],
        "risk_distribution": level_counts,
        "flag_frequency": flag_code_counts,
        "avg_risk_score": round(
            sum(a.risk_score for a in analyses) / total, 1
        ),
    }
