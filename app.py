"""
PowerGuard AI – Flask Backend API
Serves dashboard data, runs analysis pipeline, handles agent queries.
"""

from __future__ import annotations
import os
import io
import json
import dataclasses
from pathlib import Path
from functools import lru_cache

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

# ── Local imports ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_ingestion    import ingest_file, ingest_json, validate_and_clean
from analysis_engine   import analyze_all_consumers, ConsumerAnalysis, AnomalyFlag
from risk_scoring      import score_all, build_summary_stats
from ibm_integration   import generate_explanation, agent_query, ibm_available
from langflow_workflow import run_full_pipeline, export_langflow_json
from report_generator  import generate_pdf_report, build_json_report

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
app.secret_key = os.getenv("FLASK_SECRET_KEY", "powerguard-dev-key")

# ── In-memory state (single session for demo) ─────────────────────────────────
_state: dict = {
    "analyses":      [],
    "summary_stats": {},
    "quality_report": {},
    "explanations":  {},
    "loaded": False,
}


# ─── Helper: serialise ConsumerAnalysis → dict ───────────────────────────────

def _flag_to_dict(f: AnomalyFlag) -> dict:
    return dataclasses.asdict(f)


def _analysis_to_dict(a: ConsumerAnalysis) -> dict:
    d = dataclasses.asdict(a)
    return d


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the React/HTML dashboard."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "ok",
        "ibm_connected": ibm_available(),
        "data_loaded": _state["loaded"],
        "consumers_loaded": len(_state["analyses"]),
    })


@app.route("/api/load-demo", methods=["POST"])
def load_demo_data():
    """Generate and load demo electricity billing data."""
    try:
        # Generate demo dataset
        from data_generator import build_dataset
        df_raw, _ = build_dataset()

        quality_report: dict = {}
        analyses, summary = run_full_pipeline(df_raw, quality_report)

        _state["analyses"]       = analyses
        _state["summary_stats"]  = summary
        _state["quality_report"] = quality_report
        _state["explanations"]   = {}
        _state["loaded"]         = True

        return jsonify({
            "success": True,
            "message": f"Loaded {len(analyses)} consumers from demo dataset.",
            "summary": summary,
            "quality_report": quality_report,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Upload CSV/Excel billing data."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"success": False, "error": "Empty filename."}), 400

    try:
        warnings: list[str] = []
        df_raw, warnings = ingest_file(f.stream, f.filename)

        quality_report: dict = {}
        analyses, summary = run_full_pipeline(df_raw, quality_report)

        _state["analyses"]       = analyses
        _state["summary_stats"]  = summary
        _state["quality_report"] = quality_report
        _state["explanations"]   = {}
        _state["loaded"]         = True

        return jsonify({
            "success": True,
            "message": f"Processed {len(analyses)} consumers.",
            "warnings": warnings,
            "summary": summary,
            "quality_report": quality_report,
        })
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 422
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/summary")
def get_summary():
    if not _state["loaded"]:
        return jsonify({"error": "No data loaded. Use /api/load-demo or /api/upload first."}), 404
    return jsonify(_state["summary_stats"])


@app.route("/api/consumers")
def get_consumers():
    """Return paginated consumer list sorted by risk score."""
    if not _state["loaded"]:
        return jsonify({"error": "No data loaded."}), 404

    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 20))
    risk  = request.args.get("risk", "")   # filter by risk level

    analyses = _state["analyses"]
    if risk:
        analyses = [a for a in analyses if a.risk_level == risk]

    total = len(analyses)
    start = (page - 1) * limit
    end   = start + limit
    page_items = analyses[start:end]

    return jsonify({
        "total": total,
        "page":  page,
        "limit": limit,
        "consumers": [_analysis_to_dict(a) for a in page_items],
    })


@app.route("/api/consumer/<consumer_id>")
def get_consumer(consumer_id: str):
    """Return full analysis for a specific consumer."""
    if not _state["loaded"]:
        return jsonify({"error": "No data loaded."}), 404

    found = next(
        (a for a in _state["analyses"] if a.consumer_id == consumer_id), None
    )
    if not found:
        return jsonify({"error": f"Consumer {consumer_id} not found."}), 404

    result = _analysis_to_dict(found)

    # Add explanation (cache it)
    if consumer_id not in _state["explanations"]:
        _state["explanations"][consumer_id] = generate_explanation(found)
    result["explanation"] = _state["explanations"][consumer_id]

    return jsonify(result)


@app.route("/api/agent", methods=["POST"])
def agent_endpoint():
    """AI Agent natural-language query endpoint."""
    if not _state["loaded"]:
        return jsonify({"error": "No data loaded. Load demo data first."}), 404

    body = request.get_json(silent=True) or {}
    query = body.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query."}), 400

    try:
        response = agent_query(
            query,
            _state["analyses"],
            _state["summary_stats"],
        )
        return jsonify({
            "query":    query,
            "response": response,
            "ibm_used": ibm_available(),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/report/json")
def download_json_report():
    """Download full investigation report as JSON."""
    if not _state["loaded"]:
        return jsonify({"error": "No data loaded."}), 404

    # Ensure all explanations are generated
    for a in _state["analyses"]:
        if a.consumer_id not in _state["explanations"]:
            _state["explanations"][a.consumer_id] = generate_explanation(a)

    report = build_json_report(
        _state["analyses"],
        _state["explanations"],
        _state["summary_stats"],
    )
    buf = io.BytesIO(json.dumps(report, indent=2).encode())
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/json",
        as_attachment=True,
        download_name="powerguard_investigation_report.json",
    )


@app.route("/api/report/pdf")
def download_pdf_report():
    """Download investigation report as PDF."""
    if not _state["loaded"]:
        return jsonify({"error": "No data loaded."}), 404

    # Ensure explanations exist for flagged consumers
    for a in _state["analyses"]:
        if a.risk_score > 30 and a.consumer_id not in _state["explanations"]:
            _state["explanations"][a.consumer_id] = generate_explanation(a)

    pdf_bytes = generate_pdf_report(
        _state["analyses"],
        _state["explanations"],
        _state["summary_stats"],
    )
    buf = io.BytesIO(pdf_bytes)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="powerguard_investigation_report.pdf",
    )


@app.route("/api/langflow-workflow")
def get_langflow_workflow():
    """Return Langflow workflow JSON for visual inspection."""
    return jsonify(json.loads(export_langflow_json()))


@app.route("/api/top-risk")
def top_risk():
    """Return top N consumers by risk score."""
    if not _state["loaded"]:
        return jsonify({"error": "No data loaded."}), 404
    n = int(request.args.get("n", 10))
    top = _state["analyses"][:n]
    return jsonify([_analysis_to_dict(a) for a in top])


@app.route("/api/quality-report")
def quality_report():
    return jsonify(_state.get("quality_report", {}))


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"[PowerGuard AI] Starting on http://localhost:{port}")
    print(f"[PowerGuard AI] IBM watsonx connected: {ibm_available()}")
    app.run(host="0.0.0.0", port=port, debug=debug)
