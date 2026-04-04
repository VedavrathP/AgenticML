"""
AgenticML ML subpackage — automated machine learning pipeline.

This module provides backward-compatible ``run()`` entry point that now
uses the new orchestrator-centered graph internally.

Usage::

    from agenticml import ml
    result = ml.run("data.csv", target="price", verbose=True)
"""

import os

from agenticml.state.workflow_state import create_initial_state
from agenticml.graph.builder import run_graph, run_graph_streaming
from agenticml.ml.config import get_config, reset_config
from agenticml.ml.tools.utils import generate_run_id, create_run_directory, setup_logging


def run(
    file_path: str,
    target: str | None = None,
    problem_type: str | None = None,
    metric: str | None = None,
    max_iterations: int = 5,
    verbose: bool = False,
    runs_dir: str = "runs",
    stream: bool = False,
    model: str = "gpt-4o",
    api_key: str | None = None,
    user_query: str = "Run the full ML pipeline end to end",
):
    """
    One-call entry point to run the full ML pipeline.

    Args:
        file_path: Path to CSV or Excel data file.
        target: Target column name (auto-detected if None).
        problem_type: ``"classification"`` or ``"regression"`` (auto-detected if None).
        metric: Primary evaluation metric (e.g. ``"f1"``, ``"rmse"``). Auto-selected if None.
        max_iterations: Maximum pipeline iterations (default 5).
        verbose: Print LLM prompts and responses at each agent step.
        runs_dir: Directory to store run artifacts.
        stream: Enable streaming output to see per-node progress.
        model: LLM model name — provider is auto-detected from the name.
        api_key: LLM API key. Falls back to the provider's environment variable if None.
        user_query: Natural-language query for the orchestrator.

    Returns:
        The final workflow state dictionary.
    """
    config = get_config()
    config.llm_model = model
    if api_key:
        config.llm_api_key = api_key

    run_id = generate_run_id()
    run_dir = create_run_directory(runs_dir, run_id)
    setup_logging(run_id, run_dir)

    initial_state = create_initial_state(
        run_id=run_id,
        file_path=os.path.abspath(file_path),
        run_dir=run_dir,
        target=target,
        problem_type=problem_type,
        user_metric=metric,
        max_iterations=max_iterations,
        verbose=verbose,
        user_query=user_query,
    )

    if stream:
        final_state = None
        for output in run_graph_streaming(initial_state):
            for node_name, node_state in output.items():
                if node_name != "__end__":
                    final_state = node_state
        return final_state

    return run_graph(initial_state)


__all__ = [
    "run",
    "run_graph",
    "run_graph_streaming",
    "create_initial_state",
    "get_config",
    "reset_config",
]
