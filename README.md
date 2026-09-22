# Autonomous Multi-Agent Industrial Incident Response System

An autonomous software-based incident response system for industrial condition monitoring.
## 🎥 Demo

[▶️ Watch the Autonomous Incident Response Demo](https://drive.google.com/file/d/1_VnxJCY_ez1uJpPnjSlsbNP1uUr9eACs/view?usp=sharing)

A short walkthrough demonstrating dynamic sensor-stream monitoring, ML-based anomaly detection, multi-agent investigation, simulated drone inspection, RAG-grounded evidence retrieval, autonomous escalation, and sandbox-safe execution.

The system processes a simulated industrial sensor stream, detects anomalous equipment behavior using a supervised machine-learning ensemble, investigates incidents through multiple specialized agents, retrieves relevant maintenance knowledge using RAG, simulates drone-based inspection evidence, and produces a safety-controlled response decision.

**Live Application:**  
https://autonomous-industrial-incident-response-production.up.railway.app

The deployed Streamlit application provides an interactive demonstration of:

- Dynamic industrial sensor-stream monitoring
- ML-based anomaly detection
- Multi-agent incident investigation
- Simulated drone inspection
- RAG-based maintenance evidence retrieval
- Evidence-based escalation
- Safety-controlled execution
- LLM-generated operator incident reports

---

## Overview

Industrial monitoring systems can detect abnormal sensor behavior, but identifying the cause of an incident and deciding what action should follow often requires multiple stages of investigation.

This project demonstrates an **agentic AI approach to industrial incident response**.

Instead of using a single model to directly produce an action, the system separates the workflow into specialized components:

1. Sensor data analysis
2. Machine-learning anomaly detection
3. Statistical analysis
4. Investigation decision
5. Simulated drone inspection
6. Maintenance knowledge retrieval
7. Evidence-based escalation
8. Safety-controlled action execution
9. Structured incident logging

The system is designed so that incoming sensor measurements drive the investigation and response rather than directly passing an incident label to the agents.

---

# System Architecture

```text
                Simulated Sensor Stream
                         |
                         v
                +-------------------+
                | Data Analyst Agent|
                +-------------------+
                         |
             +-----------+-----------+
             |                       |
             v                       v
     ML Anomaly Detection    Statistical Analysis
             |                       |
             +-----------+-----------+
                         |
                         v
                +---------------------+
                | Drone Commander     |
                | Agent               |
                +---------------------+
                         |
                         v
                +---------------------+
                | Simulated Drone     |
                | Inspection          |
                +---------------------+
                         |
                         v
                +---------------------+
                | RAG Knowledge       |
                | Retrieval           |
                +---------------------+
                         |
                         v
                +---------------------+
                | Escalation Agent    |
                +---------------------+
                         |
                         v
                +---------------------+
                | Safety Sandbox      |
                +---------------------+
                         |
                         v
                +---------------------+
                | Final Incident      |
                | Decision            |
                +---------------------+
                         |
                         v
                +---------------------+
                | LLM Operator Report |
                | (explanation only)  |
                +---------------------+
                        |
                        v
                Structured Incident Log
```

---

# Key Features

- Stateful streaming analysis of industrial sensor measurements
- Supervised machine-learning anomaly detection
- ExtraTrees + RandomForest soft-voting ensemble
- 88 engineered features
- Multi-sensor statistical analysis
- Autonomous incident investigation
- Drone-orchestration layer
- Simulated drone inspection evidence
- Retrieval-Augmented Generation (RAG)
- Maintenance knowledge base
- Evidence-based escalation decisions
- Safety-controlled action execution
- Structured agent traces
- Incident JSON logging
- Scenario-based validation
- Streamlit monitoring dashboard
- Dynamic sensor-stream replay

---

# Machine Learning

The final anomaly detector uses a supervised ensemble consisting of:

- ExtraTrees Classifier
- RandomForest Classifier
- Soft-voting ensemble

The model operates on **88 engineered features** derived from the industrial sensor measurements.

The data was divided at the experiment level into training, validation, and unseen test experiments.

## Model Performance

The final model was evaluated on an unseen held-out test set.

| Metric | Result |
|---|---:|
| Accuracy | 90.14% |
| Precision | 90.55% |
| Recall | 80.07% |
| F1 Score | 84.99% |
| ROC-AUC | 94.50% |
| PR-AUC | 93.13% |
| False Alarm Rate | 4.47% |
| Miss Alarm Rate | 19.93% |

These results are from the project's held-out test evaluation and are separate from the scenario decision-match validation.

---

# Agentic Investigation Pipeline

## 1. Data Analyst Agent

The Data Analyst Agent is responsible for analyzing the incoming sensor window.

It combines:

- Machine-learning anomaly probability
- Statistical anomaly detection
- Sensor-level analysis
- Affected-sensor identification
- Incident severity estimation

The agent produces structured evidence that is passed to the next stage.

---

## 2. Drone Commander Agent

The Drone Commander determines whether additional physical inspection should be requested based on the evidence produced by the Data Analyst Agent.

The system can map detected sensor categories to inspection areas such as:

```text
Accelerometer → Pump mechanical area
Pressure      → Pump pressure system
Temperature   → Pump thermal area
Current       → Pump electrical area
```

The purpose of this component is to demonstrate an autonomous orchestration layer that decides when additional inspection evidence is required.

---

## 3. Simulated Drone Inspection

The project includes a deterministic drone simulator.

The simulator produces inspection evidence such as:

- Inspection target
- Visual anomaly detection
- Inspection confidence
- Observations

For example, a mechanical pump investigation can produce simulated observations related to abnormal vibration or mechanical behavior.

### Important Scope

The drone is **software-simulated**.

This project does not claim to:

- Control a physical drone
- Access a physical camera
- Perform real-world computer vision
- Control industrial equipment

The drone layer demonstrates how a future physical inspection system could be orchestrated by the incident-response pipeline.

---

# RAG Knowledge Retrieval

The project contains a local maintenance knowledge base used by the incident-response pipeline.

The knowledge base includes documents covering:

- Pump faults
- Vibration procedures
- Pressure procedures
- Escalation policies

The RAG component retrieves relevant knowledge based on the detected incident evidence.

The retrieved information is then passed to the escalation stage as additional evidence.

This allows the response decision to incorporate both:

```text
Sensor Evidence
      +
ML Evidence
      +
Inspection Evidence
      +
Maintenance Knowledge
```

---

# Escalation Agent

The Escalation Agent combines evidence from multiple stages of the pipeline.

Evidence can include:

- ML anomaly detection
- ML anomaly probability
- Number of affected sensors
- Simulated visual inspection
- Drone inspection confidence
- Retrieved maintenance knowledge

The resulting evidence score and incident severity are used to determine an appropriate response.

Possible response categories include:

```text
continued_monitoring
enhanced_monitoring
request_maintenance_inspection
priority_maintenance_escalation
collect_additional_diagnostic_evidence
```

---

# Safety Sandbox

All operational actions pass through a safety-controlled sandbox.

The sandbox uses an allow-list of permitted actions and prevents arbitrary operational commands from being executed.

Example permitted actions include:

- Continued monitoring
- Enhanced monitoring
- Maintenance inspection request
- Priority maintenance escalation
- Additional diagnostic evidence collection

The current implementation is **simulation-only**.

```text
real_system_modified = False
```

No real industrial system or physical equipment is modified by this project.

This provides a safety boundary between autonomous reasoning and operational execution.

---

# LLM Operator Incident Report

After the agents reach a decision, an LLM (Google Gemini) writes a short plain-language incident report
for the human operator. The report is built only from evidence the agents already produced (affected sensors,
ML probability, drone finding, decision) and the maintenance procedure retrieved by the RAG module.

The LLM does **not** make or change any decision and does not trigger any action. All decisions come from the
ML model and the rule-based agents. If the API is unavailable (no key, no internet, rate limit), a deterministic
template report is used instead, and the dashboard labels it "template fallback".

The report is shown in the dashboard under **Operator Incident Report**, only for incidents that reach
the escalation stage. Normal operation produces no report.

Setup: set `GEMINI_API_KEY` in your local `.env` (a free key is available from Google AI Studio).
The system also runs without a key, using the template fallback.

---

# Dynamic Sensor Streaming

The project includes a stateful streaming pipeline that replays industrial condition-monitoring measurements as a **virtual sensor stream**.

The streaming system maintains state across incoming batches and performs rolling-window analysis.

The Streamlit dashboard supports:

- Selecting a source experiment
- Selecting a starting row
- Configuring batch size
- Configuring analysis-window size
- Starting dynamic monitoring
- Processing the next batch
- Resetting the stream
- Viewing current incident state
- Viewing investigation results
- Viewing final response decisions

This enables a real-time-style demonstration using replayed industrial measurements.

---

# Autonomous Decision Flow

A typical investigation follows this process:

```text
Incoming Sensor Measurements
            |
            v
      Data Analysis
            |
            v
     ML Prediction
            |
            v
   Severity Estimation
            |
            v
   Should Inspect?
        /       \
      No         Yes
      |           |
      |           v
      |     Drone Inspection
      |           |
      +-----+-----+
            |
            v
     RAG Retrieval
            |
            v
    Evidence Scoring
            |
            v
    Escalation Decision
            |
            v
      Safety Sandbox
            |
            v
     Incident Logging
```

The important design principle is that each stage contributes evidence to the next stage rather than having one monolithic component make the entire decision.

---

# Scenario Validation

Three predefined scenarios are included:

```text
scenarios/
├── scenario_normal.json
├── scenario_ambiguous.json
└── scenario_pump_fault.json
```

They represent:

### Normal Operation

Expected response:

```text
NORMAL
continued_monitoring
```

### Ambiguous Condition

Expected response:

```text
MEDIUM
enhanced_monitoring
```

### Pump Fault

Expected response:

```text
CRITICAL
priority_maintenance_escalation
```

The scenario runner validates whether the autonomous pipeline produces the expected response category for each predefined sensor window.

The expected scenario labels are used **only for validation** and are not passed to the agents during inference.

---

# Project Structure

```text
autonomous-industrial-incident-response/
│
├── data/
│   ├── raw/
│   │   └── SKAB sensor CSV files
│   └── processed/
│       └── skab_processed.parquet
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_model_training.ipynb
│
├── scenarios/
│   ├── scenario_normal.json
│   ├── scenario_ambiguous.json
│   └── scenario_pump_fault.json
│
├── src/
│   ├── agents/
│   │   ├── data_analyst_agent.py
│   │   ├── drone_commander_agent.py
│   │   ├── drone_simulator.py
│   │   ├── escalation_agent.py
│   │   ├── model_tools.py
│   │   ├── orchestrator.py
│   │   └── tools.py
│   │
│   ├── dashboard/
│   │   └── app.py
│   │
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── streaming_features.py
│   │
│   ├── logging/
│   │   └── logger.py
│   │
│   ├── models/
│   │   ├── artifacts/
│   │   │   ├── supervised_detector.joblib
│   │   │   └── model reports
│   │   ├── baseline.py
│   │   ├── compare_models.py
│   │   ├── evaluate.py
│   │   ├── pca_detector.py
│   │   ├── supervised_detector.py
│   │   └── train.py
│   │
│   ├── rag/
│   │   ├── knowledge_base/
│   │   ├── ingest.py
│   │   ├── knowledge_index.json
│   │   └── retriever.py
│   │
│   ├── sandbox/
│   │   └── executor.py
│   │
│   └── streaming/
│       ├── dynamic_pipeline.py
│       └── simulator.py
│
├── tests/
│   ├── __init__.py
│   └── run_scenarios.py
│
├── .env.example
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Git
- Git LFS
- Retrieval-Augmented Generation
- Stateful streaming pipeline

---

# Dataset

The project uses measurements from the **SKAB (Skoltech Anomaly Benchmark)** industrial condition-monitoring dataset.

The dataset contains sensor measurements from industrial equipment experiments and is used here to develop and evaluate the anomaly-detection and incident-response pipeline.

The measurements are replayed as a **software-simulated sensor stream** for the dashboard and autonomous investigation workflow.

---

# Local Installation

## 1. Clone the repository

```bash
git clone https://github.com/chapadayanita/autonomous-industrial-incident-response.git
cd autonomous-industrial-incident-response
```

The repository uses Git LFS for the trained supervised model.

Make sure Git LFS is installed before working with the model artifact.

---

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 4. Configure environment variables

Create your local environment file from the example:

```powershell
Copy-Item .env.example .env
```

Add any required local configuration to `.env`.

The `.env` file is intentionally excluded from Git using `.gitignore`.

**Never commit secrets or API keys to the repository.**

---

# Run the Dashboard

From the project root:

```powershell
streamlit run src/dashboard/app.py
```

The Streamlit dashboard provides both scenario-based validation and dynamic sensor-stream monitoring.

---

# Run Scenario Validation

From the project root:

```powershell
python tests/run_scenarios.py
```

This executes the predefined scenarios through the autonomous incident-response pipeline and verifies the resulting decisions.

---

# Model Artifacts

The final supervised model is stored at:

```text
src/models/artifacts/supervised_detector.joblib
```

Because the trained model is approximately 194 MB, it is stored using **Git LFS** rather than normal Git object storage.

The repository includes the corresponding `.gitattributes` configuration.

---

# Deployment

The Streamlit dashboard can be deployed as a Python web service using a platform such as Render.

### Build Command

```text
pip install -r requirements.txt
```

### Start Command

```text
streamlit run src/dashboard/app.py --server.address=0.0.0.0 --server.port=$PORT
```

The deployed application provides the same software-based monitoring and incident-response workflow through the Streamlit dashboard.

---

# Safety and Scope

This project is a software prototype demonstrating autonomous incident-response orchestration.

The following components are simulated:

- Physical sensor streaming
- Drone hardware
- Drone inspection
- Visual inspection evidence
- Industrial equipment interaction
- Operational response execution

The safety sandbox ensures that the current prototype does not modify or control a real industrial system.

The project demonstrates the **software architecture and agentic decision-making workflow** that could later be integrated with physical industrial systems.

---

# Engineering Highlights

This project demonstrates:

### Agentic orchestration

Multiple specialized components collaborate through a structured investigation pipeline rather than relying on a single model.

### Evidence-based decisions

Response decisions incorporate multiple sources of evidence including ML predictions, statistical analysis, simulated inspection results, and maintenance knowledge.

### Stateful streaming

Sensor measurements are processed incrementally using a stateful streaming pipeline and rolling analysis windows.

### Safety-aware execution

Operational actions are constrained by a sandbox allow-list and remain simulation-only.

### Auditability

The orchestrator produces structured traces and incident logs containing the stages and evidence used during an investigation.

### Reproducible validation

Predefined scenarios allow the complete autonomous pipeline to be repeatedly tested against known sensor windows.

---

# Limitations and Future Extensions

Potential future extensions include:

- Integration with live industrial sensor streams
- Real drone hardware integration
- Real computer-vision-based inspection
- Industrial IoT protocols
- More advanced vector-based RAG retrieval
- Human-in-the-loop approval for high-risk actions
- Production-grade observability
- Containerized deployment
- Integration with industrial maintenance-management systems

These are future extensions; the current implementation intentionally keeps physical inspection and operational execution simulated.

---

# Author

**Dayanita Chapa**

GitHub:

https://github.com/chapadayanita

Project:

https://github.com/chapadayanita/autonomous-industrial-incident-response

