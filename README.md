# PowerGuard AI — Electricity Billing Fraud Detection

> **AI-powered electricity billing fraud detection system built on IBM watsonx.ai (Granite) + IBM Orchestrate + Langflow**

---

## System Architecture

```
Data Input → Validation → Preprocessing → Feature Engineering
    → Consumption Analysis → Rule-Based Fraud Checks
    → ML Anomaly Detection (Isolation Forest)
    → Risk Scoring → IBM Granite (watsonx.ai)
    → Explainable AI → Dashboard / Report
```

### IBM Technology Stack
| Component | Usage |
|-----------|-------|
| **IBM watsonx.ai** | Hosts the Granite 3 8B Instruct model |
| **IBM Granite Model** | Natural-language fraud explanations, AI Agent queries, investigation summaries |
| **IBM Orchestrate** | Workflow orchestration, agent tool routing, batch scheduling |
| **IBM Langflow** | Visual pipeline editor for the fraud detection workflow |

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- pip

### 2. Clone / unzip the project
```bash
cd powerguard-ai
```

### 3. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure IBM credentials (optional — system works without them)
```bash
cp .env.example .env
```
Edit `.env` and fill in:
- `IBM_API_KEY` — your IBM Cloud API key
- `IBM_PROJECT_ID` — your watsonx.ai project ID
- `IBM_WATSONX_URL` — region endpoint (default: `https://us-south.ml.cloud.ibm.com`)
- `IBM_MODEL_ID` — model to use (default: `ibm/granite-3-8b-instruct`)

> **Without IBM credentials** the system runs in *template mode* — all analysis, fraud detection, risk scoring, and structured explanations still work. IBM Granite adds richer natural-language explanations and agent responses.

### 6. Start the application
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 7. Generate sample data
Click **"Load Demo Data"** in the sidebar. The system will:
1. Generate 80 consumers × 12 months of billing records (960 rows)
2. Embed realistic fraud patterns (sudden drops, meter mismatches, spikes, zero consumption)
3. Run the full analysis pipeline
4. Display the dashboard

---

## Project Structure

```
powerguard-ai/
├── app.py                      # Flask API server (main entry point)
├── requirements.txt
├── .env.example                # IBM credentials template
├── frontend/
│   └── index.html              # Single-page dashboard (vanilla JS + Chart.js)
├── src/
│   ├── data_generator.py       # Demo dataset generator with embedded fraud
│   ├── data_ingestion.py       # CSV/Excel ingest + validation + cleaning
│   ├── analysis_engine.py      # Feature engineering + rule checks + Isolation Forest
│   ├── risk_scoring.py         # 0-100 risk score + risk level classification
│   ├── ibm_integration.py      # IBM watsonx.ai / Granite client + agent logic
│   ├── langflow_workflow.py    # Langflow JSON export + Python pipeline
│   └── report_generator.py     # PDF + JSON investigation report generation
└── data/                       # Generated data files (auto-created)
```

---

## Core Features

### 1. Data Input & Validation
- Accepts **CSV** and **Excel (.xlsx/.xls)** files via browser upload
- Accepts **JSON** via API
- Required fields: `consumer_id`, `meter_id`, `billing_month`, `previous_reading`, `current_reading`, `billed_units`, `bill_amount`
- Auto-fixes: duplicates, negative values, missing readings, logical inconsistencies (`current < previous`)

### 2. Fraud Detection Rules
| Rule | Code | Severity |
|------|------|----------|
| Consumption drops >60% below historical average | `SUDDEN_DROP` | High |
| Near-zero consumption (≤5 units) × 2+ cycles | `ZERO_CONSUMPTION` | High |
| Consumption below 30% for 3+ consecutive months | `REPEATED_LOW` | High |
| Meter reading ≠ billed units (>2% diff, 2+ cycles) | `READING_MISMATCH` | High |
| Consumption spikes >200% above average | `UNUSUAL_SPIKE` | Medium |
| Deviation >80% from mean in 3+ cycles | `LARGE_DEVIATION` | Medium |
| Consumption above 300% for 3+ consecutive months | `REPEATED_HIGH` | Medium |
| Z-score outlier (|z| > 3.5) | `STATISTICAL_OUTLIER` | Medium |

### 3. ML Anomaly Detection
- **Isolation Forest** (200 trees, 20% contamination) on per-consumer aggregated features
- Features: mean units, std units, max absolute deviation %, mean MoM change, mismatch count, max z-score, zero-consumption count
- Produces a normalised **anomaly score 0–1** per consumer

### 4. Risk Scoring (0–100)
```
Rule-based component (max 75):
  Each flag × severity multiplier (High=1.0, Medium=0.65, Low=0.35)
  + mismatch frequency bonus (max 15)

ML component (max 25):
  Isolation Forest anomaly score × 25

Final = rule_component + ml_component (capped at 100)

0–30:   Normal
31–60:  Low Risk
61–80:  Suspicious
81–100: High Risk
```

### 5. IBM AI Agent Queries
The AI Agent supports natural-language queries such as:
- *"Analyze this month's electricity bills"*
- *"Find the top 10 suspicious consumers"*
- *"Why is Consumer C102 flagged?"*
- *"Show consumers with sudden consumption drops"*
- *"Find bills where meter readings and billed units do not match"*
- *"Generate a fraud investigation report"*
- *"Summarize the overall fraud risk in this dataset"*

The agent first routes well-known queries deterministically (fast, reliable), then optionally enhances the response with IBM Granite for richer natural language.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Dashboard UI |
| `GET /api/status` | GET | System status + IBM connection |
| `POST /api/load-demo` | POST | Generate + load demo dataset |
| `POST /api/upload` | POST | Upload CSV/Excel file |
| `GET /api/summary` | GET | Dashboard summary stats |
| `GET /api/consumers` | GET | Paginated consumer list (`?page=1&limit=20&risk=High+Risk`) |
| `GET /api/consumer/{id}` | GET | Full analysis for one consumer |
| `POST /api/agent` | POST | AI Agent query `{"query": "..."}` |
| `GET /api/report/json` | GET | Download JSON investigation report |
| `GET /api/report/pdf` | GET | Download PDF investigation report |
| `GET /api/langflow-workflow` | GET | Langflow workflow JSON |
| `GET /api/top-risk` | GET | Top N consumers (`?n=10`) |
| `GET /api/quality-report` | GET | Data quality report |

---

## IBM Langflow / Orchestrate Integration

### Langflow Workflow
Export the Langflow-compatible JSON via:
```bash
python -c "from src.langflow_workflow import export_langflow_json; export_langflow_json('langflow_workflow.json')"
```
Import `langflow_workflow.json` into your IBM Langflow instance to view and edit the visual pipeline.

### IBM Orchestrate
Configure IBM Orchestrate to:
1. Call `POST /api/load-demo` or `POST /api/upload` to trigger analysis
2. Call `GET /api/top-risk` to retrieve high-risk consumers for alerts
3. Call `POST /api/agent` with natural-language queries for investigation summaries
4. Schedule periodic batch runs and route alerts to investigators

---

## Demo Fraud Patterns Injected

| Pattern | Consumers | Description |
|---------|-----------|-------------|
| `sudden_drop` | ~4 | Months 4-8: consumption drops 80-95% |
| `reading_mismatch` | ~4 | 3 months: billed units = 25-50% of actual meter diff |
| `spike_anomaly` | ~4 | 2 random months with 3-5× consumption spike |
| `zero_consumption` | ~4 | 3-5 consecutive near-zero months |
| `repeated_low` | ~4 | Every other month: consumption 10-20% of normal |

---

## Important Disclaimers

- This system detects **potential fraud indicators and anomalies** — it does **not confirm fraud**.
- All flagged consumers should be reviewed by a qualified investigator.
- The system uses terms such as *suspicious activity, potential fraud, anomaly, requires investigation*.
- Risk scores are indicative and should be used as a triage tool, not as definitive proof.

---

*PowerGuard AI — Built with IBM watsonx.ai, IBM Granite, IBM Orchestrate, and IBM Langflow*
