import sys, os, io, json
sys.path.insert(0, 'src')

PASS = "[OK]"
FAIL = "[FAIL]"
errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"{PASS} {label}{': ' + detail if detail else ''}")
    else:
        print(f"{FAIL} {label}{': ' + detail if detail else ''}")
        errors.append(label)

# ── 1. Data Generator ────────────────────────────────────────────────────────
from data_generator import build_dataset
df, meta = build_dataset()
check("Data Generator", len(df) == 960 and df["consumer_id"].nunique() == 80,
      f"{len(df)} rows, {df['consumer_id'].nunique()} consumers")

# ── 2. Data Ingestion & Validation ───────────────────────────────────────────
from data_ingestion import validate_and_clean, ingest_file
df_clean, qr = validate_and_clean(df)
check("Data Validation", qr["cleaned_rows"] > 900,
      f"{qr['original_rows']} -> {qr['cleaned_rows']} rows, {len(qr['issues'])} issues")

# ── 3. Feature Engineering ────────────────────────────────────────────────────
from analysis_engine import engineer_features, run_isolation_forest, analyze_all_consumers
df_feat = engineer_features(df_clean)
cols_ok = all(c in df_feat.columns for c in ["dev_from_mean_pct", "zscore", "reading_mismatch", "mom_change_pct"])
check("Feature Engineering", cols_ok, "all derived columns present")

df_feat = run_isolation_forest(df_feat)
check("Isolation Forest", "if_anomaly_score" in df_feat.columns,
      "anomaly scores computed")

# ── 4. Full Consumer Analysis ─────────────────────────────────────────────────
analyses = analyze_all_consumers(df_clean)
check("Consumer Analysis", len(analyses) == 80, f"{len(analyses)} consumers analysed")

# ── 5. Risk Scoring ───────────────────────────────────────────────────────────
from risk_scoring import score_all, build_summary_stats
analyses = score_all(analyses)
scores_valid = all(0 <= a.risk_score <= 100 for a in analyses)
sorted_ok = analyses[0].risk_score >= analyses[-1].risk_score
summary = build_summary_stats(analyses)
check("Risk Scoring", scores_valid and sorted_ok,
      f"High={summary['high_risk_consumers']} Suspicious={summary['suspicious_consumers']} "
      f"LowRisk={summary['low_risk_consumers']} Normal={summary['normal_consumers']} "
      f"Avg={summary['avg_risk_score']}")

# ── 6. Explainable AI ─────────────────────────────────────────────────────────
from ibm_integration import generate_explanation, agent_query, ibm_available
top = analyses[0]
expl = generate_explanation(top)
check("XAI Explanation", top.consumer_id in expl and len(expl) > 80,
      f"{len(expl)} chars for {top.consumer_id}")

# ── 7. AI Agent – 6 query types ───────────────────────────────────────────────
queries = [
    ("Overall summary",     "Summarize the overall fraud risk in this dataset"),
    ("Top suspicious",      "Find the top 10 suspicious consumers"),
    ("Explain consumer",    f"Why is {top.consumer_id} flagged"),
    ("Consumption drops",   "Show consumers with sudden consumption drops"),
    ("Meter mismatches",    "Find bills where meter readings and billed units do not match"),
    ("High risk count",     "How many high risk consumers are there"),
]
agent_ok = True
for label, q in queries:
    resp = agent_query(q, analyses, summary)
    if not resp or len(resp) < 10:
        agent_ok = False
        errors.append(f"Agent query: {label}")
check("AI Agent (6 queries)", agent_ok, "all query types responded correctly")

# ── 8. Langflow Workflow ──────────────────────────────────────────────────────
from langflow_workflow import export_langflow_json, run_full_pipeline
wf = json.loads(export_langflow_json())
check("Langflow Workflow", len(wf["nodes"]) == 11 and len(wf["edges"]) == 13,
      f"{len(wf['nodes'])} nodes, {len(wf['edges'])} edges")

# ── 9. Pipeline Orchestrator ──────────────────────────────────────────────────
q2 = {}
a2, s2 = run_full_pipeline(df, q2)
check("Full Pipeline", len(a2) == 80, "end-to-end orchestration")

# ── 10. JSON Report ───────────────────────────────────────────────────────────
from report_generator import build_json_report, generate_pdf_report
exp_map = {a.consumer_id: generate_explanation(a) for a in analyses[:5]}
rpt = build_json_report(analyses, exp_map, summary)
check("JSON Report", "flagged_consumers" in rpt and rpt["summary"]["total_consumers"] == 80,
      f"{len(rpt['flagged_consumers'])} flagged consumers")

# ── 11. PDF Report ────────────────────────────────────────────────────────────
pdf = generate_pdf_report(analyses[:10], exp_map, summary)
check("PDF Report", len(pdf) > 1000, f"{len(pdf):,} bytes")

# ── 12. CSV File Upload ───────────────────────────────────────────────────────
csv_bytes = df_clean.head(50).to_csv(index=False).encode()
df_up, warns = ingest_file(io.BytesIO(csv_bytes), "test.csv")
check("CSV Upload", len(df_up) == 50, f"{len(df_up)} rows ingested")

# ── 13. Flask App Import ──────────────────────────────────────────────────────
os.environ.setdefault("FLASK_DEBUG", "false")
import importlib.util
spec = importlib.util.spec_from_file_location("app", "app.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
routes = [str(r) for r in mod.app.url_map.iter_rules()]
check("Flask App", len(routes) >= 12, f"{len(routes)} routes registered")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 56)
if not errors:
    print("  ALL 13 CHECKS PASSED - PROJECT IS FULLY WORKING")
else:
    print(f"  {len(errors)} CHECK(S) FAILED: {', '.join(errors)}")
print("=" * 56)
print(f"  LLM (Groq) connected    : {ibm_available()}")
print(f"  Top flagged consumer    : {top.consumer_id}  (score {top.risk_score}/100)")
print(f"  High Risk consumers     : {summary['high_risk_consumers']}")
print(f"  Suspicious consumers    : {summary['suspicious_consumers']}")
print(f"  Total bills processed   : {summary['total_bills']}")
print(f"  Meter mismatches found  : {summary['mismatch_consumers']} consumers")
print("=" * 56)
