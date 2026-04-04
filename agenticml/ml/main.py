"""
CLI entrypoint for the Orchestrator-Centered ML Pipeline.

Usage:
    python -m agenticml --file data.csv --target target_column
    python -m agenticml --file data.csv --problem_type classification --metric f1
    python -m agenticml --file data.csv --max_iterations 3
    python -m agenticml --file data.csv --query "profile this dataset"
"""

import argparse
import os
import sys

from agenticml.state.workflow_state import create_initial_state
from agenticml.ml.config import get_config
from agenticml.graph.builder import run_graph, run_graph_streaming
from agenticml.ml.tools.utils import (
    generate_run_id,
    create_run_directory,
    setup_logging,
    validate_file_exists,
    validate_problem_type,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AgenticML — Orchestrator-Centered ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m agenticml --file data.csv
    python -m agenticml --file data.csv --target price
    python -m agenticml --file data.csv --query "profile this dataset"
    python -m agenticml --file data.csv --query "run only Random Forest"
    python -m agenticml --file data.csv --stream
        """,
    )

    parser.add_argument("--file", "-f", type=str, required=True,
                        help="Path to the input data file (CSV or Excel)")
    parser.add_argument("--target", "-t", type=str, default=None,
                        help="Name of the target column (auto-detected if omitted)")
    parser.add_argument("--problem_type", "-p", type=str,
                        choices=["classification", "regression"], default=None,
                        help="Problem type (auto-detected if omitted)")
    parser.add_argument("--metric", "-m", type=str, default=None,
                        help="Primary evaluation metric (e.g. f1, roc_auc, rmse, r2)")
    parser.add_argument("--max_iterations", type=int, default=5,
                        help="Maximum number of pipeline iterations (default: 5)")
    parser.add_argument("--runs_dir", type=str, default="runs",
                        help="Directory to store run artifacts (default: runs)")
    parser.add_argument("--stream", action="store_true",
                        help="Enable streaming output")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print LLM prompts and responses")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model name (provider auto-detected)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="LLM API key (falls back to env var)")
    parser.add_argument("--query", "-q", type=str,
                        default="Run the full ML pipeline end to end",
                        help="Natural-language query for the orchestrator")

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate configuration
    config = get_config()
    config.llm_model = args.model
    if args.api_key:
        config.llm_api_key = args.api_key
    config_issues = config.validate()

    if config_issues:
        print("Configuration issues:")
        for issue in config_issues:
            print(f"  - {issue}")

        from agenticml.services.llm_service import detect_provider, _ENV_KEY_MAP
        try:
            provider = detect_provider(args.model)
            env_var = _ENV_KEY_MAP.get(provider, "OPENAI_API_KEY")
        except ValueError:
            env_var = "OPENAI_API_KEY"
        if env_var in str(config_issues):
            print(f"\nPlease set the {env_var} environment variable:")
            print(f"  export {env_var}='your-api-key'")
        sys.exit(1)

    # Validate input file
    valid, error = validate_file_exists(args.file)
    if not valid:
        print(f"Error: {error}")
        sys.exit(1)

    # Validate problem type
    if args.problem_type:
        valid, error = validate_problem_type(args.problem_type)
        if not valid:
            print(f"Error: {error}")
            sys.exit(1)

    # Generate run ID and create directory
    run_id = generate_run_id()
    run_dir = create_run_directory(args.runs_dir, run_id)
    logger = setup_logging(run_id, run_dir)

    # Banner
    print("=" * 60)
    print("  AgenticML — Orchestrator-Centered Pipeline")
    print("=" * 60)
    print(f"  Run ID:      {run_id}")
    print(f"  Input file:  {args.file}")
    print(f"  Target:      {args.target or 'auto-detect'}")
    print(f"  Problem:     {args.problem_type or 'auto-detect'}")
    print(f"  Metric:      {args.metric or 'auto-select'}")
    print(f"  Max iters:   {args.max_iterations}")
    print(f"  Query:       {args.query}")
    print(f"  Output dir:  {run_dir}")
    print("=" * 60)
    print()

    logger.info(f"Starting pipeline run: {run_id}")

    # Create initial state
    initial_state = create_initial_state(
        run_id=run_id,
        file_path=os.path.abspath(args.file),
        run_dir=run_dir,
        target=args.target,
        problem_type=args.problem_type,
        user_metric=args.metric,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        user_query=args.query,
    )

    # Run the pipeline
    try:
        if args.stream:
            print("Running pipeline with streaming output...\n")
            final_state = None
            for output in run_graph_streaming(initial_state):
                for node_name, node_state in output.items():
                    if node_name != "__end__":
                        iteration = node_state.get("iteration", 0)
                        print(f"[Iteration {iteration + 1}] Completed: {node_name}")
                        final_state = node_state
            print()
        else:
            print("Running pipeline...\n")
            final_state = run_graph(initial_state)

        # Summary
        print("=" * 60)
        print("  Pipeline Complete")
        print("=" * 60)

        if final_state:
            best_model = final_state.get("best_model", {})
            history = final_state.get("execution_history", [])
            agents_run = [e["agent"] for e in history if e.get("status") == "completed"]

            print(f"  Agents run:  {', '.join(agents_run)}")

            if best_model:
                print(f"  Best model:  {best_model.get('name', 'N/A')}")
                print(f"  Score:       {best_model.get('primary_score', 0):.4f}")

            print()
            print(f"  Report:      {os.path.join(run_dir, 'report.md')}")
            print(f"  Artifacts:   {run_dir}")

        print("=" * 60)
        logger.info(f"Pipeline completed: {final_state.get('stop_reason', 'unknown')}")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.exception("Pipeline failed with error")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
