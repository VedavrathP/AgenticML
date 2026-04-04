"""
SwayamML — Streamlit UI for the orchestrator-centered ML pipeline.

Handles dataset upload, chat-based interaction with the agent pipeline,
streaming progress display, and rich artifact rendering (plots, reports,
decision logs).
"""

import json
import os
from collections import defaultdict

import pandas as pd
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.state.workflow_state import WorkflowState, create_initial_state
from agenticml.graph.builder import run_graph_streaming
from agenticml.services.llm_service import get_llm, invoke_llm_json
from agenticml.ml.config import get_config
from agenticml.ml.tools.utils import generate_run_id, create_run_directory

# ---------------------------------------------------------------------------
# Secrets — load from Streamlit secrets (cloud or local .streamlit/secrets.toml)
# and inject into os.environ so the backend LLM service picks them up.
# ---------------------------------------------------------------------------
for _secret_key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
    if _secret_key in st.secrets and not os.environ.get(_secret_key):
        os.environ[_secret_key] = st.secrets[_secret_key]


# ═══════════════════════════════════════════════════════════════════════════
# Answer synthesis — LLM reads state and answers the user's question
# ═══════════════════════════════════════════════════════════════════════════

ANSWER_SYSTEM_PROMPT = """\
You are a helpful ML assistant. The user asked a question and the ML pipeline \
has executed relevant agents to gather the data needed to answer.

You are given:
1. The user's original question
2. Relevant data extracted from the pipeline state

Your job is to **directly answer the user's question** in clear, concise \
markdown. Use tables when showing columnar data. Be specific with numbers. \
If the data doesn't contain what the user asked for, say so honestly.

You must respond with a JSON object:
{"answer": "<your markdown-formatted answer>"}
"""


def _extract_answer_context(state: WorkflowState) -> dict:
    """Pull the most relevant data from state for answer synthesis."""
    ctx: dict = {}

    ds = state.get("data_summary")
    if ds:
        ctx["dataset"] = {
            "n_rows": ds.get("n_rows"),
            "n_cols": ds.get("n_cols"),
            "columns": ds.get("columns"),
            "numeric_columns": ds.get("numeric_columns"),
            "categorical_columns": ds.get("categorical_columns"),
        }

    mvs = state.get("missing_value_summary")
    if mvs:
        ctx["missing_values"] = mvs

    outlier = state.get("outlier_summary")
    if outlier:
        ctx["outlier_summary"] = outlier

    corr = state.get("high_correlation_pairs")
    if corr:
        ctx["high_correlation_pairs"] = corr

    cr = state.get("cleaning_report")
    if cr:
        stats = cr.get("stats", {}) if isinstance(cr, dict) else {}
        ctx["cleaning_summary"] = {
            "rows_before": stats.get("rows_before"),
            "rows_after": stats.get("rows_after"),
            "rows_removed": stats.get("rows_removed"),
            "cols_before": stats.get("cols_before"),
            "cols_after": stats.get("cols_after"),
        }

    bm = state.get("best_model")
    if bm:
        ctx["best_model"] = {
            "name": bm.get("name"),
            "primary_score": bm.get("primary_score"),
            "metrics": bm.get("metrics"),
        }

    results = state.get("evaluation_results")
    if results:
        ctx["model_results"] = [
            {"name": r.get("name"), "score": r.get("primary_score"), "metrics": r.get("metrics")}
            for r in results if r.get("success")
        ]

    ctx["target"] = state.get("target")
    ctx["problem_type"] = state.get("problem_type")

    return ctx


def _synthesize_answer(user_query: str, state: WorkflowState) -> str:
    """Use the LLM to answer the user's question from pipeline state data."""
    try:
        config = get_config()
        llm = get_llm(config)
        context = _extract_answer_context(state)

        prompt = (
            f"User's question: \"{user_query}\"\n\n"
            f"Pipeline data:\n{json.dumps(context, indent=2, default=str)}\n\n"
            "Answer the user's question directly based on this data."
        )

        messages = [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        result = invoke_llm_json(
            llm, messages,
            agent_name="AnswerSynthesis",
            step_description="Answer user question",
            verbose=False,
        )
        return result.get("answer", "I couldn't generate an answer from the available data.")
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — artifact rendering
# ═══════════════════════════════════════════════════════════════════════════

AGENT_LABELS = {
    "orchestrator": "Orchestrator",
    "dataset_profiling": "Dataset Profiling",
    "data_preprocessing": "Data Preprocessing",
    "feature_engineering": "Feature Engineering",
    "model_selection": "Model Selection",
    "model_training": "Model Training",
    "evaluation": "Evaluation",
    "insight_visualization": "Insight & Visualisation",
}


def _render_summary(state: WorkflowState) -> None:
    """Render a top-level summary card."""
    cols = st.columns(3)

    ds = state.get("data_summary") or {}
    with cols[0]:
        st.metric("Rows", f"{ds.get('n_rows', '—'):,}" if isinstance(ds.get("n_rows"), int) else "—")
        st.metric("Columns", ds.get("n_cols", "—"))

    with cols[1]:
        st.metric("Target", state.get("target", "—"))
        st.metric("Problem", (state.get("problem_type") or "—").title())

    bm = state.get("best_model") or {}
    with cols[2]:
        st.metric("Best Model", bm.get("name", "—"))
        score = bm.get("primary_score")
        st.metric(
            state.get("user_metric", "Score") or "Score",
            f"{score:.4f}" if isinstance(score, (int, float)) else "—",
        )

    history = state.get("execution_history", [])
    agents_run = [e["agent"] for e in history if e.get("status") == "completed"]
    if agents_run:
        labels = [AGENT_LABELS.get(a, a) for a in agents_run]
        st.caption(f"Agents executed: {' → '.join(labels)}")


def _render_eda_plots(state: WorkflowState) -> None:
    """Display EDA plots in a 2-column grid."""
    eda_artifacts = [
        a for a in state.get("artifacts", [])
        if a.get("artifact_type") == "plot"
        and os.path.basename(a.get("path", "")).startswith("eda_")
    ]
    if not eda_artifacts:
        st.info("No EDA plots generated yet. Run the pipeline to see visualisations.")
        return

    col_a, col_b = st.columns(2)
    for idx, artifact in enumerate(eda_artifacts):
        path = artifact["path"]
        if not os.path.isfile(path):
            continue
        caption = (
            os.path.basename(path)
            .replace("eda_", "")
            .replace(".png", "")
            .replace("_", " ")
            .title()
        )
        target_col = col_a if idx % 2 == 0 else col_b
        with target_col:
            st.image(path, caption=caption, use_container_width=True)


def _render_model_plots(state: WorkflowState) -> None:
    """Display model evaluation plots grouped by model name."""
    model_artifacts = [
        a for a in state.get("artifacts", [])
        if a.get("artifact_type") == "plot"
        and not os.path.basename(a.get("path", "")).startswith("eda_")
        and os.path.isfile(a.get("path", ""))
    ]
    if not model_artifacts:
        st.info("No model plots generated yet.")
        return

    grouped: dict[str, list[dict]] = defaultdict(list)
    for a in model_artifacts:
        fname = os.path.basename(a["path"])
        parts = fname.replace(".png", "").split("_", 1)
        model_name = parts[0] if len(parts) > 1 else "General"
        grouped[model_name].append(a)

    for model_name, artifacts in sorted(grouped.items()):
        with st.expander(model_name, expanded=False):
            c1, c2 = st.columns(2)
            for idx, a in enumerate(artifacts):
                caption = (
                    os.path.basename(a["path"])
                    .replace(".png", "")
                    .replace("_", " ")
                    .title()
                )
                target = c1 if idx % 2 == 0 else c2
                with target:
                    st.image(a["path"], caption=caption, use_container_width=True)


def _render_metrics_table(state: WorkflowState) -> None:
    """Show evaluation results as a table."""
    results = state.get("evaluation_results", [])
    successful = [r for r in results if r.get("success")]
    if not successful:
        st.info("No evaluation results yet.")
        return

    rows = []
    for r in sorted(successful, key=lambda x: x.get("primary_score", 0), reverse=True):
        row = {"Model": r.get("name", "?"), "Score": r.get("primary_score", 0)}
        for k, v in r.get("metrics", {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                row[k.upper()] = round(v, 4)
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_report(state: WorkflowState) -> None:
    """Render the full markdown report if it exists."""
    run_dir = state.get("run_dir", "")
    report_path = os.path.join(run_dir, "report.md")
    if os.path.isfile(report_path):
        with open(report_path) as f:
            st.markdown(f.read())
    else:
        st.info("Report not generated yet. Run the full pipeline to produce a report.")


def _render_decision_log(state: WorkflowState) -> None:
    """Show the decision log as expandable entries."""
    log = state.get("decision_log", [])
    if not log:
        st.info("No decisions recorded yet.")
        return

    for entry in log:
        agent = AGENT_LABELS.get(entry.get("agent", ""), entry.get("agent", ""))
        decision = entry.get("decision", "")
        with st.expander(f"{agent}: {decision}"):
            st.write(f"**Rationale:** {entry.get('rationale', '—')}")
            details = entry.get("details")
            if details:
                st.json(details)


def _render_errors(state: WorkflowState) -> None:
    """Show errors if any."""
    errors = state.get("errors", [])
    if not errors:
        st.success("No errors.")
        return
    for err in errors:
        st.error(f"**{err.get('agent', '?')}:** {err.get('error', '?')}")


def _build_response_text(state: WorkflowState) -> str:
    """Build a concise text summary for the chat history."""
    parts: list[str] = []

    history = state.get("execution_history", [])
    agents_run = [e["agent"] for e in history if e.get("status") == "completed"]
    if agents_run:
        labels = [AGENT_LABELS.get(a, a) for a in agents_run]
        parts.append(f"**Agents executed:** {' → '.join(labels)}")

    ds = state.get("data_summary")
    if ds:
        parts.append(
            f"**Dataset:** {ds.get('n_rows', '?')} rows × {ds.get('n_cols', '?')} cols  \n"
            f"**Target:** {state.get('target', 'N/A')} ({state.get('problem_type', 'N/A')})"
        )

    bm = state.get("best_model")
    if bm:
        score = bm.get("primary_score", 0)
        parts.append(
            f"**Best model:** {bm.get('name', 'N/A')}  \n"
            f"**Score:** {score:.4f}"
        )

    insights = state.get("generated_insights")
    if insights and insights.get("executive_summary"):
        parts.append(f"**Summary:** {insights['executive_summary']}")

    errors = state.get("errors", [])
    if errors:
        parts.append(f"**Error:** {errors[-1].get('error', '?')}")

    return "\n\n".join(parts) if parts else "Done — no new results to display."


def _render_artifacts(state: WorkflowState) -> None:
    """Render the full tabbed artifact display."""
    tab_names = ["Summary", "EDA Plots", "Model Plots", "Metrics", "Report", "Decisions"]
    if state.get("errors"):
        tab_names.append("Errors")

    tabs = st.tabs(tab_names)

    with tabs[0]:
        _render_summary(state)
    with tabs[1]:
        _render_eda_plots(state)
    with tabs[2]:
        _render_model_plots(state)
    with tabs[3]:
        _render_metrics_table(state)
    with tabs[4]:
        _render_report(state)
    with tabs[5]:
        _render_decision_log(state)
    if state.get("errors"):
        with tabs[6]:
            _render_errors(state)


# ═══════════════════════════════════════════════════════════════════════════
# Page config & session state
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="SwayamML", page_icon="🧪", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — upload, preview, quick actions
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("SwayamML")
    st.caption("Orchestrator-Centered ML Pipeline")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        data_dir = os.path.join("runs", "_uploads")
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file_path = os.path.abspath(file_path)
        st.success(f"Uploaded: {uploaded_file.name}")

        try:
            preview_df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
            st.caption(f"{preview_df.shape[0]:,} rows × {preview_df.shape[1]} columns")
            st.dataframe(preview_df.head(), use_container_width=True, height=180)

            target_options = ["(auto-detect)"] + list(preview_df.columns)
            chosen_target = st.selectbox("Target column", target_options)
            if chosen_target != "(auto-detect)":
                st.session_state.chosen_target = chosen_target
            else:
                st.session_state.pop("chosen_target", None)
        except Exception:
            pass

    st.markdown("---")
    st.markdown("**Quick actions**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Full Pipeline", use_container_width=True):
            st.session_state.pending_query = "Run the full ML pipeline end to end"
    with c2:
        if st.button("Profile Data", use_container_width=True):
            st.session_state.pending_query = "Profile this dataset"

    c3, c4 = st.columns(2)
    with c3:
        if st.button("Show Nulls", use_container_width=True):
            st.session_state.pending_query = "Show null values in the dataset"
    with c4:
        if st.button("Compare Models", use_container_width=True):
            st.session_state.pending_query = "Compare all trained models"

# ═══════════════════════════════════════════════════════════════════════════
# Main area — chat
# ═══════════════════════════════════════════════════════════════════════════

st.header("SwayamML")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ═══════════════════════════════════════════════════════════════════════════
# Query input + pipeline execution
# ═══════════════════════════════════════════════════════════════════════════

pending = st.session_state.pop("pending_query", None)
user_query = st.chat_input("Ask the ML agent anything...") or pending

if user_query:
    file_path = st.session_state.get("uploaded_file_path")
    if not file_path:
        st.warning("Please upload a dataset first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    ws: WorkflowState | None = st.session_state.workflow_state
    target = st.session_state.get("chosen_target")

    if ws is None:
        run_id = generate_run_id()
        run_dir = create_run_directory("runs", run_id)
        ws = create_initial_state(
            run_id=run_id,
            file_path=file_path,
            run_dir=run_dir,
            user_query=user_query,
            target=target,
        )
    else:
        ws["user_query"] = user_query
        ws["next_agent"] = None

    with st.chat_message("assistant"):
        final_state = None

        with st.status("Running pipeline...", expanded=True) as status:
            seen_agents: set[str] = set()
            try:
                for snapshot in run_graph_streaming(ws):
                    final_state = snapshot
                    history = snapshot.get("execution_history", [])
                    for entry in history:
                        agent = entry.get("agent", "")
                        if agent and agent not in seen_agents:
                            seen_agents.add(agent)
                            label = AGENT_LABELS.get(agent, agent)
                            status.update(label=f"Running: {label}...")
                            st.write(f"✓ {label}")

                status.update(label="Pipeline complete", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Pipeline failed", state="error")
                st.error(f"**Error:** {exc}")
                final_state = ws

        if final_state:
            st.session_state.workflow_state = final_state

            answer = _synthesize_answer(user_query, final_state)
            if answer:
                st.markdown(answer)
                response_text = answer
            else:
                response_text = _build_response_text(final_state)
                st.markdown(response_text)

            _render_artifacts(final_state)
        else:
            response_text = "Pipeline returned no results."
            st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

# ═══════════════════════════════════════════════════════════════════════════
# Show artifacts from last run (when no new query)
# ═══════════════════════════════════════════════════════════════════════════

if not user_query and st.session_state.workflow_state is not None:
    with st.expander("Last run artifacts", expanded=False):
        _render_artifacts(st.session_state.workflow_state)
