"""
PowerGuard AI – LLM Integration (Groq API)
Uses Groq-hosted models (llama-3.3-70b-versatile) for NL explanations and agent responses.
Falls back to template-based explanations when the Groq key is unavailable.
"""

from __future__ import annotations
import os
import json
import re
import urllib.request
import urllib.error
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis_engine import ConsumerAnalysis

# ─── Groq client ──────────────────────────────────────────────────────────────

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_groq_key: str | None = None
_groq_ok:  bool | None = None   # None = not tested yet


def _get_groq_key() -> str | None:
    global _groq_key
    if _groq_key is None:
        _groq_key = os.getenv("GROQ_API_KEY", "").strip() or None
    return _groq_key


def ibm_available() -> bool:
    """
    Named 'ibm_available' for API compatibility.
    Returns True when the Groq key is set and the endpoint is reachable.
    """
    global _groq_ok
    if _groq_ok is not None:
        return _groq_ok
    key = _get_groq_key()
    if not key:
        _groq_ok = False
        return False
    # Quick connectivity check with a minimal payload
    try:
        _call_groq("Say OK", max_tokens=5)
        _groq_ok = True
    except Exception:
        _groq_ok = False
    return _groq_ok


def _call_groq(prompt: str, max_tokens: int = 600) -> str:
    """
    Call Groq chat completions endpoint.
    Returns generated text or raises RuntimeError.
    """
    key = _get_groq_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY not set.")

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        GROQ_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="ignore")
        raise RuntimeError(f"Groq HTTP {exc.code}: {body[:300]}") from exc
    except Exception as exc:
        raise RuntimeError(f"Groq call failed: {exc}") from exc


# ─── Prompt builders ──────────────────────────────────────────────────────────

def _build_explanation_prompt(analysis: "ConsumerAnalysis") -> str:
    flags_text = "\n".join(
        f"  - [{f.severity.upper()}] {f.code}: {f.description}"
        for f in analysis.anomaly_flags
    ) or "  - No specific rule-based flags."

    monthly_summary = ""
    if analysis.monthly_detail:
        last3 = analysis.monthly_detail[-3:]
        lines = []
        for m in last3:
            lines.append(
                f"  {m['billing_month']}: {m['billed_units']:.1f} units "
                f"(dev: {m['dev_from_mean_pct']:+.1f}%, mismatch: {m['reading_mismatch']})"
            )
        monthly_summary = "Last 3 months:\n" + "\n".join(lines)

    trend = (
        "Decreasing" if analysis.consumption_trend < -1 else
        "Increasing" if analysis.consumption_trend > 1 else
        "Stable"
    )

    return f"""You are PowerGuard AI, an electricity fraud analyst assistant.
Your task is to write a concise, professional investigation explanation for a consumer.
IMPORTANT: Never state that fraud is confirmed. Use terms: potential fraud, suspicious activity, anomaly, requires investigation.

Consumer Details:
- Consumer ID: {analysis.consumer_id}
- Meter ID: {analysis.meter_id}
- Risk Score: {analysis.risk_score}/100
- Risk Level: {analysis.risk_level}
- Consumer Type: {analysis.consumer_type}
- City Zone: {analysis.city_zone}
- Average Historical Consumption: {analysis.avg_consumption:.1f} units/month
- Latest Consumption: {analysis.latest_consumption:.1f} units/month
- Consumption Trend: {trend}
- Meter/Billing Mismatches: {analysis.reading_mismatch_count} out of {analysis.num_bills} cycles ({analysis.reading_mismatch_pct:.1f}%)
- ML Anomaly Score: {analysis.anomaly_score:.2f}/1.00

Detected Anomalies:
{flags_text}

{monthly_summary}

Write a 4-6 sentence professional investigation note covering:
1. Risk classification and score
2. Key anomalies detected
3. Historical comparison
4. Meter/billing observations
5. Recommended investigation priority

Begin directly with: "Consumer {analysis.consumer_id} is classified as..."
"""


def _build_agent_prompt(user_query: str, context_json: str) -> str:
    return f"""You are PowerGuard AI, an intelligent electricity fraud detection assistant.
You help electricity distribution company investigators analyze billing data and identify suspicious consumers.
Always be professional, factual, and never accuse — use terms like "suspicious", "anomaly", "requires investigation".

Current Dataset Context (JSON summary):
{context_json}

User Query: {user_query}

Provide a clear, helpful, concise response (3-8 sentences).
If listing consumers, use bullet points.
If no relevant data is available for the query, say so politely.
"""


# ─── Template-based fallback (no LLM required) ────────────────────────────────

def _template_explanation(analysis: "ConsumerAnalysis") -> str:
    flag_names = [f.code.replace("_", " ").title() for f in analysis.anomaly_flags]
    flags_str = ", ".join(flag_names) if flag_names else "no specific rule violations"

    latest_dev = ""
    if analysis.monthly_detail:
        last = analysis.monthly_detail[-1]
        dev = last.get("dev_from_mean_pct", 0)
        latest_dev = (
            f" In the most recent billing cycle, consumption was "
            f"{abs(dev):.1f}% {'below' if dev < 0 else 'above'} the historical average."
        )

    trend_str = (
        "a declining trend" if analysis.consumption_trend < -2 else
        "an increasing trend" if analysis.consumption_trend > 2 else
        "a stable trend"
    )

    mismatch_str = ""
    if analysis.reading_mismatch_count > 0:
        mismatch_str = (
            f" Additionally, meter readings and billed units are inconsistent in "
            f"{analysis.reading_mismatch_count} out of {analysis.num_bills} billing cycles "
            f"({analysis.reading_mismatch_pct:.1f}%), which requires verification."
        )

    priority = (
        "IMMEDIATE" if analysis.risk_level == "High Risk" else
        "HIGH"      if analysis.risk_level == "Suspicious" else
        "ROUTINE"
    )

    return (
        f"Consumer {analysis.consumer_id} (Meter: {analysis.meter_id}) is classified as "
        f"{analysis.risk_level} with a risk score of {analysis.risk_score}/100. "
        f"The analysis detected the following anomalies: {flags_str}. "
        f"Historical average consumption is {analysis.avg_consumption:.1f} units/month, "
        f"and consumption shows {trend_str}.{latest_dev}{mismatch_str} "
        f"The ML anomaly model assigned a score of {analysis.anomaly_score:.2f}/1.00. "
        f"Recommended investigation priority: {priority}."
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_explanation(analysis: "ConsumerAnalysis") -> str:
    """
    Generate an XAI explanation for a consumer.
    Uses Groq LLM if available, otherwise template fallback.
    """
    if ibm_available():
        try:
            prompt = _build_explanation_prompt(analysis)
            return _call_groq(prompt, max_tokens=500)
        except RuntimeError as exc:
            print(f"[Groq] explanation error: {exc}")
    return _template_explanation(analysis)


def agent_query(
    user_query: str,
    analyses: list["ConsumerAnalysis"],
    summary_stats: dict,
) -> str:
    """
    Process a natural-language agent query.
    Routes deterministically first, then optionally enhances with Groq LLM.
    """
    deterministic_result = _deterministic_agent_router(user_query, analyses, summary_stats)

    if deterministic_result:
        if ibm_available():
            try:
                ctx = json.dumps({
                    "summary": summary_stats,
                    "deterministic_answer": deterministic_result[:2000],
                }, indent=2)
                prompt = _build_agent_prompt(user_query, ctx)
                return _call_groq(prompt, max_tokens=600)
            except RuntimeError as exc:
                print(f"[Groq] agent error: {exc}")
        return deterministic_result

    # Open-ended query — send straight to LLM
    if ibm_available():
        try:
            ctx = json.dumps(summary_stats, indent=2)
            prompt = _build_agent_prompt(user_query, ctx)
            return _call_groq(prompt, max_tokens=600)
        except RuntimeError as exc:
            print(f"[Groq] agent error: {exc}")

    return (
        "I couldn't process that query with the current dataset. "
        "Try asking about specific consumers, risk levels, or consumption patterns."
    )


def _deterministic_agent_router(
    query: str,
    analyses: list["ConsumerAnalysis"],
    summary_stats: dict,
) -> str | None:
    """Route well-known query patterns to deterministic answers."""
    q = query.lower()

    # ── Top N suspicious / high risk ─────────────────────────────────────────
    if any(kw in q for kw in ["top", "highest risk", "most suspicious", "fraud report", "summarize"]):
        flagged = [a for a in analyses if a.risk_level in ("High Risk", "Suspicious")][:10]
        if not flagged:
            return "No suspicious consumers found in the current dataset."
        lines = [f"**Top {len(flagged)} flagged consumers:**"]
        for a in flagged:
            flags = ", ".join(f.code for f in a.anomaly_flags) or "None"
            lines.append(
                f"• {a.consumer_id} (Score: {a.risk_score}/100, Level: {a.risk_level}) "
                f"— Flags: {flags}"
            )
        return "\n".join(lines)

    # ── Why is C### flagged ───────────────────────────────────────────────────
    cid_match = re.search(r"c\d+", q)
    if cid_match:
        cid = cid_match.group(0).upper()
        found = next((a for a in analyses if a.consumer_id.upper() == cid), None)
        if found:
            return generate_explanation(found)
        return f"Consumer {cid} was not found in the current dataset."

    # ── Sudden drop / low consumption ────────────────────────────────────────
    if "drop" in q or "decrease" in q or "low consumption" in q:
        dropped = [
            a for a in analyses
            if any(f.code in ("SUDDEN_DROP", "ZERO_CONSUMPTION", "REPEATED_LOW")
                   for f in a.anomaly_flags)
        ]
        if not dropped:
            return "No consumers with sudden consumption drops detected."
        lines = [f"**{len(dropped)} consumers with consumption drop anomalies:**"]
        for a in dropped[:15]:
            lines.append(
                f"• {a.consumer_id} — Avg: {a.avg_consumption:.1f} units, "
                f"Latest: {a.latest_consumption:.1f} units (Score: {a.risk_score}/100)"
            )
        return "\n".join(lines)

    # ── Meter / billed unit mismatch ─────────────────────────────────────────
    if "mismatch" in q or "meter reading" in q or "billed unit" in q:
        mismatched = [a for a in analyses if a.reading_mismatch_count > 0]
        if not mismatched:
            return "No meter reading / billed unit mismatches found."
        lines = [f"**{len(mismatched)} consumers with meter/billing mismatches:**"]
        for a in mismatched[:15]:
            lines.append(
                f"• {a.consumer_id} — {a.reading_mismatch_count}/{a.num_bills} cycles "
                f"({a.reading_mismatch_pct:.1f}%) — Risk: {a.risk_level}"
            )
        return "\n".join(lines)

    # ── Overall summary ───────────────────────────────────────────────────────
    if any(kw in q for kw in ["overall", "summary", "dataset", "how many", "total"]):
        s = summary_stats
        return (
            f"**Dataset Summary:**\n"
            f"• Total consumers analysed: {s.get('total_consumers', 0)}\n"
            f"• Total bills analysed: {s.get('total_bills', 0)}\n"
            f"• Normal: {s.get('normal_consumers', 0)}\n"
            f"• Low Risk: {s.get('low_risk_consumers', 0)}\n"
            f"• Suspicious: {s.get('suspicious_consumers', 0)}\n"
            f"• High Risk: {s.get('high_risk_consumers', 0)}\n"
            f"• Meter/billing mismatches: {s.get('mismatch_consumers', 0)} consumers\n"
            f"• Average risk score: {s.get('avg_risk_score', 0)}/100"
        )

    return None  # No deterministic match → LLM handles it
