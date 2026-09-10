"""
PowerGuard AI – Investigation Report Generator
Produces structured PDF and JSON investigation reports.
"""

from __future__ import annotations
import io
import json
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis_engine import ConsumerAnalysis

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ─── Colour palette ───────────────────────────────────────────────────────────
COLORS = {
    "High Risk":   "#dc2626",
    "Suspicious":  "#d97706",
    "Low Risk":    "#2563eb",
    "Normal":      "#16a34a",
    "header_bg":   "#1e3a5f",
    "row_alt":     "#f0f4ff",
    "border":      "#cbd5e1",
}


# ─── JSON report ──────────────────────────────────────────────────────────────

def build_json_report(
    analyses: list["ConsumerAnalysis"],
    explanations: dict[str, str],
    summary_stats: dict,
) -> dict:
    """Build a full structured JSON report."""
    report_rows = []
    for a in analyses:
        if a.risk_score == 0 and not a.anomaly_flags:
            continue  # Skip perfectly normal consumers from detailed report

        latest = a.monthly_detail[-1] if a.monthly_detail else {}
        row = {
            "consumer_id":          a.consumer_id,
            "meter_id":             a.meter_id,
            "consumer_type":        a.consumer_type,
            "city_zone":            a.city_zone,
            "billing_period":       (
                f"{a.monthly_detail[0]['billing_month']} to "
                f"{a.monthly_detail[-1]['billing_month']}"
                if a.monthly_detail else "N/A"
            ),
            "latest_consumption":   a.latest_consumption,
            "historical_avg":       a.avg_consumption,
            "deviation_pct":        latest.get("dev_from_mean_pct", 0),
            "anomaly_score":        round(a.anomaly_score, 4),
            "risk_score":           a.risk_score,
            "risk_level":           a.risk_level,
            "fraud_indicators":     [f.code for f in a.anomaly_flags],
            "flag_details":         [
                {
                    "code": f.code,
                    "severity": f.severity,
                    "description": f.description,
                }
                for f in a.anomaly_flags
            ],
            "reading_mismatch_count": a.reading_mismatch_count,
            "reading_mismatch_pct":   a.reading_mismatch_pct,
            "ai_explanation":         explanations.get(a.consumer_id, ""),
            "investigation_priority": (
                "IMMEDIATE" if a.risk_level == "High Risk" else
                "HIGH"      if a.risk_level == "Suspicious" else
                "ROUTINE"
            ),
        }
        report_rows.append(row)

    return {
        "report_title":    "PowerGuard AI – Electricity Billing Fraud Investigation Report",
        "generated_at":    datetime.now().isoformat(),
        "summary":         summary_stats,
        "flagged_consumers": report_rows,
    }


# ─── PDF report ───────────────────────────────────────────────────────────────

def _rl_color(hex_str: str):
    """Convert hex colour to ReportLab HexColor."""
    return HexColor(hex_str)


def generate_pdf_report(
    analyses: list["ConsumerAnalysis"],
    explanations: dict[str, str],
    summary_stats: dict,
) -> bytes:
    """
    Generate a professional PDF investigation report.
    Returns raw PDF bytes. Falls back to a plain-text PDF if reportlab unavailable.
    """
    if not REPORTLAB_AVAILABLE:
        return _plain_text_pdf(analyses, summary_stats)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()

    H1 = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        textColor=_rl_color(COLORS["header_bg"]),
        fontSize=18, spaceAfter=4,
    )
    H2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        textColor=_rl_color(COLORS["header_bg"]),
        fontSize=13, spaceBefore=12, spaceAfter=4,
    )
    NORMAL = ParagraphStyle(
        "NORMAL", parent=styles["Normal"],
        fontSize=9, leading=13,
    )
    SMALL = ParagraphStyle(
        "SMALL", parent=styles["Normal"],
        fontSize=8, leading=11, textColor=_rl_color("#475569"),
    )

    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("⚡ PowerGuard AI", H1))
    story.append(Paragraph("Electricity Billing Fraud Investigation Report", H2))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Powered by IBM watsonx.ai (Granite Model)",
        SMALL,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=_rl_color(COLORS["header_bg"])))
    story.append(Spacer(1, 0.3 * cm))

    # ── Executive Summary ─────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", H2))
    s = summary_stats
    summary_data = [
        ["Metric", "Value"],
        ["Total Consumers Analysed", str(s.get("total_consumers", 0))],
        ["Total Bills Analysed",     str(s.get("total_bills", 0))],
        ["Normal Consumers",         str(s.get("normal_consumers", 0))],
        ["Low Risk",                 str(s.get("low_risk_consumers", 0))],
        ["Suspicious",               str(s.get("suspicious_consumers", 0))],
        ["High Risk",                str(s.get("high_risk_consumers", 0))],
        ["Meter/Billing Mismatches", str(s.get("mismatch_consumers", 0))],
        ["Average Risk Score",       f"{s.get('avg_risk_score', 0)}/100"],
    ]
    summary_table = Table(summary_data, colWidths=[9 * cm, 6 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _rl_color(COLORS["header_bg"])),
        ("TEXTCOLOR",  (0, 0), (-1, 0), white),
        ("FONTSIZE",   (0, 0), (-1, 0), 10),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, _rl_color(COLORS["row_alt"])]),
        ("GRID",       (0, 0), (-1, -1), 0.5, _rl_color(COLORS["border"])),
        ("FONTSIZE",   (0, 1), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.5 * cm))

    # ── Flagged Consumers Table ───────────────────────────────────────────────
    flagged = [a for a in analyses if a.risk_level in ("High Risk", "Suspicious")]
    if flagged:
        story.append(Paragraph(f"Flagged Consumers ({len(flagged)})", H2))
        header = ["Consumer ID", "Risk Score", "Risk Level", "Flags", "Priority"]
        rows = [header]
        for a in flagged:
            flags = "\n".join(f.code for f in a.anomaly_flags) or "—"
            priority = "IMMEDIATE" if a.risk_level == "High Risk" else "HIGH"
            rows.append([
                a.consumer_id,
                f"{a.risk_score}/100",
                a.risk_level,
                flags,
                priority,
            ])
        flag_table = Table(rows, colWidths=[3.5 * cm, 2.5 * cm, 3 * cm, 4.5 * cm, 3 * cm])
        flag_style = [
            ("BACKGROUND", (0, 0), (-1, 0), _rl_color(COLORS["header_bg"])),
            ("TEXTCOLOR",  (0, 0), (-1, 0), white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("GRID",       (0, 0), (-1, -1), 0.4, _rl_color(COLORS["border"])),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ]
        for i, a in enumerate(flagged, start=1):
            color_hex = COLORS.get(a.risk_level, "#ffffff")
            flag_style.append(
                ("BACKGROUND", (1, i), (2, i), _rl_color(color_hex + "33"))
            )
        flag_table.setStyle(TableStyle(flag_style))
        story.append(flag_table)
        story.append(Spacer(1, 0.5 * cm))

    # ── Individual Consumer Details ───────────────────────────────────────────
    story.append(Paragraph("Individual Investigation Notes", H2))
    for a in analyses:
        if a.risk_score < 31 and not a.anomaly_flags:
            continue
        color_hex = COLORS.get(a.risk_level, "#16a34a")
        story.append(Paragraph(
            f'<font color="{color_hex}">■</font> '
            f'<b>{a.consumer_id}</b> (Meter: {a.meter_id}) — '
            f'Risk Score: <b>{a.risk_score}/100</b> [{a.risk_level}]',
            NORMAL,
        ))
        explanation = explanations.get(a.consumer_id, "No explanation available.")
        story.append(Paragraph(explanation, SMALL))
        story.append(Spacer(1, 0.25 * cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=_rl_color("#94a3b8")))
    story.append(Paragraph(
        "PowerGuard AI — Powered by IBM watsonx.ai (Granite) | "
        "For investigation purposes only. Anomalies do not confirm fraud.",
        SMALL,
    ))

    doc.build(story)
    return buf.getvalue()


def _plain_text_pdf(analyses: list, summary_stats: dict) -> bytes:
    """Minimal PDF without reportlab (plain bytes placeholder)."""
    lines = [
        "PowerGuard AI – Investigation Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Summary:",
    ]
    for k, v in summary_stats.items():
        if not isinstance(v, (dict, list)):
            lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Flagged Consumers:")
    for a in analyses:
        if a.risk_score > 30:
            lines.append(
                f"  {a.consumer_id} | Score: {a.risk_score} | Level: {a.risk_level}"
            )
    text = "\n".join(lines)
    return text.encode("utf-8")
