## PROJECT OVERVIEW STATEMENT (POS)

### Project Name:  
**Multi-Agent ML Pipeline (LangGraph AutoML CLI)**  

### Project Manager:  
**TBD**

### Problem/Opportunity:
Teams and individual practitioners often spend significant time repeatedly performing the same end-to-end machine learning workflow steps—data ingestion, profiling, cleaning, feature engineering, model training, evaluation, and reporting—while still lacking consistent **auditability**, **data-safety checks (PII/leakage)**, and **repeatable artifacts**. This creates delays, quality variance, and elevated risk of invalid results (e.g., leakage-driven “too good” scores).

### Goal (SMART):
Deliver a production-ready **CLI-based, multi-agent automated ML pipeline** that, for any tabular CSV/Excel dataset, can run end-to-end to:
- infer/confirm target + problem type, optionally via human-in-the-loop questions  
- profile data (schema/missingness/cardinality) and flag potential **PII** and **leakage** risks  
- clean and preprocess data without leakage (fit transforms on train only)  
- train **3–6** candidate models (baseline-first), evaluate using an appropriate strategy (train/test or CV), and select a best model  
- generate a reproducible **run folder** containing `report.md`, metrics JSON, plots, and saved model artifacts  

**Time-related:** achieve this as a stable v1 within **4 weeks** of project start, validated on at least **3** representative datasets (classification + regression).

### Objectives:
- **Objective 1 — Usable CLI entrypoint**
  - **Outcome:** A command-line interface that accepts input file, optional target/problem type/metric, iteration limit, streaming, and interactive mode.
  - **Time frame:** By end of Week 1
  - **Measure:** `python -m app.main --file <path>` runs and creates a new run directory with logs.
  - **Action:** Implement/maintain argument parsing, config validation, run directory creation, and pipeline invocation.

- **Objective 2 — Deterministic orchestration with iterative quality control**
  - **Outcome:** A LangGraph workflow that executes the full ML lifecycle and iterates when blocking issues are found (or until iteration budget is exhausted).
  - **Time frame:** By end of Week 2
  - **Measure:** Graph completes with a `stop_reason`, iteration count, and recorded decisions; loops when blocking issues exist.
  - **Action:** Define the graph nodes/routing, maintain shared `PipelineState`, and persist decision logs.

- **Objective 3 — Data profiling + risk detection (PII and leakage)**
  - **Outcome:** A profiling step that summarizes dataset structure and flags PII candidates and leakage risks before modeling.
  - **Time frame:** By end of Week 2
  - **Measure:** `data_profile.json` exists; `pii_warnings` and `leakage_warnings` populate when applicable.
  - **Action:** Profile data, infer target/problem type when needed, and run PII/leakage detection rules with saved artifacts.

- **Objective 4 — Cleaning and preprocessing that preserves validity**
  - **Outcome:** Data cleaning plan execution and preprocessing pipeline that avoids target corruption and prevents leakage (train-only fitting).
  - **Time frame:** By end of Week 3
  - **Measure:** `cleaned_data.csv`, `cleaning_report.json`, `preprocessor.joblib`, and transformed train/test (or CV) feature files are produced; target is preserved.
  - **Action:** Generate cleaning steps (LLM-assisted with deterministic fallback), apply cleaning, split data appropriately, fit/transform preprocessor, and encode target safely.

- **Objective 5 — Model training, evaluation, and best-model selection**
  - **Outcome:** Train multiple model candidates under runtime constraints and select the best model using a user-specified or default metric.
  - **Time frame:** By end of Week 4
  - **Measure:** At least 3 successful trained models per run (where feasible), `evaluation_results.json`, model comparison plot(s), and a populated `best_model` record.
  - **Action:** Select candidates (baseline first), train with timeouts, compute metrics/plots (classification or regression), and choose the best model consistently.

- **Objective 6 — Reporting and reproducibility artifacts**
  - **Outcome:** A stakeholder-readable `report.md` and a run manifest capturing configuration, outputs, and key decisions.
  - **Time frame:** By end of Week 4
  - **Measure:** `report.md` includes dataset overview, preprocessing summary, model results table, issues/warnings, limitations, and artifact index; `run_manifest.json` created.
  - **Action:** Assemble report content from pipeline state, persist artifacts, and include reproducibility fields (run id, random state, iterations).

### Success Criteria:
- The pipeline completes end-to-end on at least **3** datasets (including at least **1 classification** and **1 regression**) and produces:
  - **`report.md`** + **`run_manifest.json`**
  - saved **models** (`.joblib`), **metrics** (`.json`), and **plots** (`.png`)
- The pipeline reliably trains and evaluates **multiple** models and selects a best model using the configured primary metric.
- The system surfaces and documents **PII/leakage warnings** and flags **suspiciously high scores** as blocking issues (triggering iteration where possible).
- Results are reproducible at the run level: each run has a unique run id, captured configuration, and a clear artifact trail.

### Assumptions, Risks, Obstacles:
- **Assumptions**
  - Input data is tabular and readable as CSV/Excel; target exists or can be inferred.
  - Runtime environment has required Python packages installed; optional libraries (XGBoost/LightGBM/SHAP) may be absent.
  - An `OPENAI_API_KEY` is available when LLM-assisted planning/cleaning/model selection/report sections are desired.
  - Users have authorization to process data and accept responsibility for any PII present in inputs.

- **Risks**
  - **LLM dependency risk:** missing API key, cost, rate limits, and non-deterministic outputs can impact plan quality and consistency.
  - **Data leakage risk:** despite safeguards, users may provide inherently leaky datasets/columns; leakage can inflate metrics.
  - **PII handling risk:** heuristic detection may miss PII (false negatives) or over-flag (false positives).
  - **Scalability risk:** very large datasets can exceed memory/time budgets; some algorithms are unsuitable at scale.
  - **Metric misuse risk:** incorrect metric choice for the business goal can lead to wrong “best model” decisions.

- **Obstacles**
  - Missing/invalid `OPENAI_API_KEY` blocks LLM-enhanced behavior (pipeline must fall back deterministically).
  - Ambiguous targets or messy schemas can prevent correct inference and halt early.
  - Heavy class imbalance can cause unstable training; resampling methods may fail or introduce artifacts.
  - Environment limitations (no optional dependencies, limited compute) can reduce model diversity/performance.

### Prepared By:  
**(Your Name)**  

### Date:  
**2026-02-07**

### Approved By:  
**TBD**  

### Date:  
**TBD**

