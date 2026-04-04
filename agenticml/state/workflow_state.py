"""
Shared workflow state for the orchestrator-centered ML pipeline.

This module defines the WorkflowState TypedDict that flows between all nodes
in the LangGraph.  The orchestrator reads and writes this state to decide
which specialised agent to invoke next.
"""

from typing import TypedDict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================================
# Issue Severity Levels
# ============================================================================

class IssueSeverity:
    INFO = "info"
    WARN = "warn"
    BLOCKING = "blocking"


# ============================================================================
# Data Classes (preserved from original codebase)
# ============================================================================

@dataclass
class DataSummary:
    """Summary statistics and metadata about the dataset."""
    n_rows: int = 0
    n_cols: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    missing_counts: dict[str, int] = field(default_factory=dict)
    missing_percentages: dict[str, float] = field(default_factory=dict)
    cardinality: dict[str, int] = field(default_factory=dict)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    high_cardinality_columns: list[str] = field(default_factory=list)
    sample_values: dict[str, list] = field(default_factory=dict)
    numeric_stats: dict[str, dict] = field(default_factory=dict)


@dataclass
class CleaningStep:
    """A single cleaning operation."""
    action: str
    column: Optional[str] = None
    params: dict = field(default_factory=dict)
    rationale: str = ""


@dataclass
class CleaningPlan:
    """Plan for data cleaning operations."""
    steps: list[CleaningStep] = field(default_factory=list)
    rationale: str = ""


@dataclass
class CleaningReport:
    """Report of cleaning operations performed."""
    steps_executed: list[dict] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0
    changes_summary: str = ""


@dataclass
class PreprocessingPlan:
    """Plan for feature preprocessing."""
    numeric_strategy: str = "standard"
    categorical_strategy: str = "onehot"
    handle_unknown: str = "ignore"
    datetime_features: list[str] = field(default_factory=list)
    text_strategy: str = "tfidf"
    columns_to_drop: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class SplitPlan:
    """Plan for train/test splitting."""
    strategy: str = "stratified"
    test_size: float = 0.2
    cv_folds: int = 5
    time_column: Optional[str] = None
    random_state: int = 42
    rationale: str = ""


@dataclass
class ModelCandidate:
    """A model candidate with its configuration."""
    name: str
    model_type: str
    params: dict = field(default_factory=dict)
    is_baseline: bool = False
    training_time: float = 0.0
    model_path: Optional[str] = None


@dataclass
class EvaluationResult:
    """Evaluation results for a single model."""
    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[list[list[int]]] = None
    classification_report: Optional[dict] = None
    feature_importances: Optional[dict[str, float]] = None
    plots: list[str] = field(default_factory=list)
    cv_scores: Optional[list[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None


@dataclass
class Artifact:
    """Reference to a saved artifact."""
    name: str
    path: str
    artifact_type: str
    created_at: str = ""


# ============================================================================
# Workflow State
# ============================================================================

class WorkflowState(TypedDict, total=False):
    """
    Shared state for the orchestrator-centered LangGraph pipeline.

    All fields are optional (total=False) so agents can update incrementally.
    The orchestrator inspects this state to decide which agent to invoke next.
    """

    # -- Run metadata --------------------------------------------------------
    run_id: str
    file_path: str
    run_dir: str
    started_at: str

    # -- User inputs (may be inferred by profiling agent) --------------------
    target: Optional[str]
    problem_type: Optional[str]          # "classification" | "regression"
    user_metric: Optional[str]           # e.g. "f1", "rmse"
    max_iterations: int
    user_constraints: dict[str, Any]
    verbose: bool

    # -- Current user intent (set by orchestrator per turn) ------------------
    user_query: str                      # raw natural-language query
    user_intent: Optional[dict]          # parsed intent JSON from LLM
    current_phase: Optional[str]         # which agent is executing right now
    next_agent: Optional[str]            # next agent the orchestrator chose

    # -- Execution history ---------------------------------------------------
    execution_history: list[dict]        # [{"agent": "...", "timestamp": "...", "status": "..."}]

    # -- Dataset & profiling -------------------------------------------------
    raw_data_path: Optional[str]
    data_summary: Optional[dict]         # serialised DataSummary
    missing_value_summary: Optional[dict]
    outlier_summary: Optional[dict]      # per-column outlier stats from EDA
    high_correlation_pairs: list[dict]   # feature pairs with |corr| > threshold
    pii_warnings: list[dict]
    leakage_warnings: list[dict]

    # -- Preprocessing / cleaning --------------------------------------------
    cleaned_data_path: Optional[str]
    cleaning_plan: Optional[dict]
    cleaning_report: Optional[dict]

    # -- Feature engineering -------------------------------------------------
    preprocessing_plan: Optional[dict]
    preprocessor_path: Optional[str]
    split_plan: Optional[dict]
    feature_engineering_applied: list[dict]
    smote_applied: bool
    new_features_created: list[str]
    feature_representations: Optional[dict]  # summary of final feature set

    # -- Train / test data ---------------------------------------------------
    train_data_path: Optional[str]
    test_data_path: Optional[str]
    cv_splitter: bool
    feature_names: list[str]
    n_train_samples: int
    n_test_samples: int
    n_features: int

    # -- Model selection & training ------------------------------------------
    selected_models: list[dict]          # LLM-chosen model configs
    model_candidates: list[dict]
    trained_models: list[dict]
    retrain_history: list[dict]          # [{"model": "...", "attempt": N, "params": {...}, "score": float, "metric": "..."}]

    # -- Evaluation ----------------------------------------------------------
    evaluation_results: list[dict]
    best_model: Optional[dict]
    best_model_path: Optional[str]
    evaluation_issues: list[dict]        # issues found during evaluation (replaces critic_issues)

    # -- Insights & visualisation --------------------------------------------
    generated_insights: Optional[dict]   # executive summary, findings, etc.
    generated_plots: list[str]           # paths to generated plot files

    # -- Iteration control ---------------------------------------------------
    iteration: int
    stop_reason: Optional[str]

    # -- Artifacts & audit trail ---------------------------------------------
    artifacts: list[dict]
    decision_log: list[dict]
    errors: list[dict]


# ============================================================================
# State Helpers
# ============================================================================

def create_initial_state(
    run_id: str,
    file_path: str,
    run_dir: str,
    target: Optional[str] = None,
    problem_type: Optional[str] = None,
    user_metric: Optional[str] = None,
    max_iterations: int = 5,
    user_constraints: Optional[dict] = None,
    verbose: bool = False,
    user_query: str = "",
) -> WorkflowState:
    """Create the initial state for a pipeline run."""
    return WorkflowState(
        run_id=run_id,
        file_path=file_path,
        run_dir=run_dir,
        started_at=datetime.now().isoformat(),
        target=target,
        problem_type=problem_type,
        user_metric=user_metric,
        max_iterations=max_iterations,
        user_constraints=user_constraints or {},
        verbose=verbose,
        user_query=user_query,
        user_intent=None,
        current_phase=None,
        next_agent=None,
        execution_history=[],
        pii_warnings=[],
        leakage_warnings=[],
        feature_engineering_applied=[],
        smote_applied=False,
        new_features_created=[],
        selected_models=[],
        model_candidates=[],
        trained_models=[],
        retrain_history=[],
        evaluation_results=[],
        evaluation_issues=[],
        generated_plots=[],
        iteration=0,
        artifacts=[],
        decision_log=[],
        errors=[],
    )


def log_decision(
    state: WorkflowState,
    agent: str,
    decision: str,
    rationale: str,
    details: Optional[dict] = None,
) -> None:
    """Log an agent decision to the state for auditability."""
    if "decision_log" not in state:
        state["decision_log"] = []
    state["decision_log"].append({
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "decision": decision,
        "rationale": rationale,
        "details": details or {},
        "iteration": state.get("iteration", 0),
    })


def add_artifact(
    state: WorkflowState,
    name: str,
    path: str,
    artifact_type: str,
) -> None:
    """Add an artifact reference to the state."""
    if "artifacts" not in state:
        state["artifacts"] = []
    state["artifacts"].append({
        "name": name,
        "path": path,
        "artifact_type": artifact_type,
        "created_at": datetime.now().isoformat(),
    })


def add_error(
    state: WorkflowState,
    agent: str,
    error: str,
    details: Optional[dict] = None,
) -> None:
    """Log an error to the state."""
    if "errors" not in state:
        state["errors"] = []
    state["errors"].append({
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "error": error,
        "details": details or {},
        "iteration": state.get("iteration", 0),
    })


def record_execution(
    state: WorkflowState,
    agent: str,
    status: str = "completed",
) -> None:
    """Append an entry to execution_history."""
    if "execution_history" not in state:
        state["execution_history"] = []
    state["execution_history"].append({
        "agent": agent,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "iteration": state.get("iteration", 0),
    })
