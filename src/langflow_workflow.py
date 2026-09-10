"""
PowerGuard AI – Langflow Workflow Definition
Exports a Langflow-compatible JSON workflow for visual orchestration.
Also provides a Python-level orchestration pipeline that mirrors the flow.
"""

from __future__ import annotations
import json
from pathlib import Path


# ─── Langflow JSON export ─────────────────────────────────────────────────────

LANGFLOW_WORKFLOW = {
    "name": "PowerGuard AI - Electricity Fraud Detection",
    "description": (
        "End-to-end electricity billing fraud detection workflow using IBM Granite "
        "via watsonx.ai. Covers data ingestion, validation, feature engineering, "
        "anomaly detection, risk scoring, and explainable AI."
    ),
    "version": "1.0.0",
    "nodes": [
        {
            "id": "node_01",
            "type": "DataInputNode",
            "label": "1. Data Input",
            "description": "Accept CSV/Excel billing data or JSON records via API.",
            "inputs": ["file_upload", "json_payload"],
            "outputs": ["raw_dataframe"],
            "ibm_component": False,
            "position": {"x": 100, "y": 200},
        },
        {
            "id": "node_02",
            "type": "DataValidationNode",
            "label": "2. Data Validation",
            "description": (
                "Check required columns, remove duplicates, fix missing values, "
                "validate current_reading >= previous_reading."
            ),
            "inputs": ["raw_dataframe"],
            "outputs": ["validated_dataframe", "quality_report"],
            "ibm_component": False,
            "position": {"x": 300, "y": 200},
        },
        {
            "id": "node_03",
            "type": "PreprocessingNode",
            "label": "3. Data Preprocessing",
            "description": "Normalise columns, impute missing values, parse dates.",
            "inputs": ["validated_dataframe"],
            "outputs": ["clean_dataframe"],
            "ibm_component": False,
            "position": {"x": 500, "y": 200},
        },
        {
            "id": "node_04",
            "type": "FeatureEngineeringNode",
            "label": "4. Feature Engineering",
            "description": (
                "Compute historical averages, deviation %, month-over-month change, "
                "z-scores, rolling averages, rate-per-unit."
            ),
            "inputs": ["clean_dataframe"],
            "outputs": ["feature_dataframe"],
            "ibm_component": False,
            "position": {"x": 700, "y": 200},
        },
        {
            "id": "node_05",
            "type": "ConsumptionAnalysisNode",
            "label": "5. Consumption Analysis",
            "description": (
                "Calculate actual vs billed consumption, trend slopes, "
                "reading vs billed-unit mismatches."
            ),
            "inputs": ["feature_dataframe"],
            "outputs": ["consumption_stats"],
            "ibm_component": False,
            "position": {"x": 900, "y": 100},
        },
        {
            "id": "node_06",
            "type": "RuleBasedFraudNode",
            "label": "6. Rule-Based Fraud Checks",
            "description": (
                "Apply deterministic rules: sudden drop, spike, zero consumption, "
                "repeated low/high, reading mismatch, z-score outliers."
            ),
            "inputs": ["feature_dataframe"],
            "outputs": ["anomaly_flags"],
            "ibm_component": False,
            "position": {"x": 900, "y": 300},
        },
        {
            "id": "node_07",
            "type": "MLAnomalyNode",
            "label": "7. ML Anomaly Detection",
            "description": (
                "Isolation Forest on aggregated consumer features. "
                "Produces normalised anomaly score 0-1 per consumer."
            ),
            "inputs": ["feature_dataframe"],
            "outputs": ["ml_anomaly_scores"],
            "ibm_component": False,
            "position": {"x": 1100, "y": 200},
        },
        {
            "id": "node_08",
            "type": "RiskScoringNode",
            "label": "8. Risk Scoring",
            "description": (
                "Combine rule flags + ML score into 0-100 risk score. "
                "Classify: Normal / Low Risk / Suspicious / High Risk."
            ),
            "inputs": ["anomaly_flags", "ml_anomaly_scores"],
            "outputs": ["risk_scored_analyses"],
            "ibm_component": False,
            "position": {"x": 1300, "y": 200},
        },
        {
            "id": "node_09",
            "type": "IBMGraniteNode",
            "label": "9. IBM Granite (watsonx.ai)",
            "description": (
                "IBM Granite 3 8B Instruct model via watsonx.ai. "
                "Generates natural-language explanations, investigation summaries, "
                "and handles agent queries."
            ),
            "inputs": ["risk_scored_analyses", "user_query"],
            "outputs": ["nl_explanation", "agent_response"],
            "ibm_component": True,
            "ibm_model": "ibm/granite-3-8b-instruct",
            "ibm_service": "IBM watsonx.ai",
            "position": {"x": 1500, "y": 200},
        },
        {
            "id": "node_10",
            "type": "ExplainableAINode",
            "label": "10. Explainable AI Output",
            "description": (
                "Combine structured anomaly data with IBM Granite NL explanation "
                "into a complete, human-readable investigation note."
            ),
            "inputs": ["risk_scored_analyses", "nl_explanation"],
            "outputs": ["xai_report"],
            "ibm_component": True,
            "position": {"x": 1700, "y": 100},
        },
        {
            "id": "node_11",
            "type": "ReportDashboardNode",
            "label": "11. Final Report / Dashboard",
            "description": (
                "Present dashboard with risk distribution, top-10 flagged consumers, "
                "trends, and downloadable PDF investigation report."
            ),
            "inputs": ["xai_report"],
            "outputs": ["dashboard_data", "pdf_report"],
            "ibm_component": False,
            "position": {"x": 1900, "y": 200},
        },
    ],
    "edges": [
        {"from": "node_01", "to": "node_02"},
        {"from": "node_02", "to": "node_03"},
        {"from": "node_03", "to": "node_04"},
        {"from": "node_04", "to": "node_05"},
        {"from": "node_04", "to": "node_06"},
        {"from": "node_04", "to": "node_07"},
        {"from": "node_05", "to": "node_08"},
        {"from": "node_06", "to": "node_08"},
        {"from": "node_07", "to": "node_08"},
        {"from": "node_08", "to": "node_09"},
        {"from": "node_09", "to": "node_10"},
        {"from": "node_08", "to": "node_10"},
        {"from": "node_10", "to": "node_11"},
    ],
    "ibm_components": {
        "watsonx_ai": {
            "model_id": "ibm/granite-3-8b-instruct",
            "url": "https://us-south.ml.cloud.ibm.com",
            "usage": [
                "Generate natural-language fraud investigation explanations",
                "Handle AI Agent NL queries",
                "Produce investigation summaries",
                "Explain anomaly patterns in consumer-friendly language",
            ],
        },
        "ibm_orchestrate": {
            "usage": [
                "Orchestrate multi-step fraud detection workflow",
                "Route agent queries to appropriate tools",
                "Schedule periodic batch analysis",
                "Trigger alerts for High Risk consumers",
            ],
        },
    },
}


def export_langflow_json(output_path: str | None = None) -> str:
    """Return the Langflow workflow JSON string and optionally save to file."""
    payload = json.dumps(LANGFLOW_WORKFLOW, indent=2)
    if output_path:
        Path(output_path).write_text(payload, encoding="utf-8")
    return payload


# ─── Python orchestration pipeline (mirrors Langflow flow) ───────────────────

def run_full_pipeline(df_raw, quality_report_out: dict | None = None) -> tuple:
    """
    Execute the complete PowerGuard AI pipeline programmatically.
    Returns (analyses, summary_stats).
    """
    try:
        from .data_ingestion   import validate_and_clean
        from .analysis_engine  import analyze_all_consumers
        from .risk_scoring     import score_all, build_summary_stats
    except ImportError:
        from data_ingestion   import validate_and_clean
        from analysis_engine  import analyze_all_consumers
        from risk_scoring     import score_all, build_summary_stats

    # Nodes 2-3: Validate + clean
    df_clean, quality_report = validate_and_clean(df_raw)
    if quality_report_out is not None:
        quality_report_out.update(quality_report)

    # Nodes 4-7: Feature engineering + analysis + anomaly detection (inside analyze_all_consumers)
    analyses = analyze_all_consumers(df_clean)

    # Node 8: Risk scoring
    analyses = score_all(analyses)

    # Node 11: Summary stats
    summary_stats = build_summary_stats(analyses)

    return analyses, summary_stats
