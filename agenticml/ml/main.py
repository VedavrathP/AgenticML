"""
CLI entrypoint for the Multi-Agent ML Pipeline.

Usage:
    python -m app.main --file data.csv --target target_column
    python -m app.main --file data.csv --problem_type classification --metric f1
    python -m app.main --file data.csv --max_iterations 3
    python -m app.main --file data.csv --interactive  # Enable human-in-the-loop
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

from agenticml.ml.state import create_initial_state, PipelineState
from agenticml.ml.config import get_config
from agenticml.ml.graph import run_pipeline, run_pipeline_with_streaming, set_human_interaction_callback
from agenticml.ml.tools.utils import (
    generate_run_id,
    create_run_directory,
    setup_logging,
    validate_file_exists,
    validate_problem_type
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent ML Pipeline - Automated machine learning with LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with auto-detection
    python -m app.main --file data.csv

    # Specify target column
    python -m app.main --file data.csv --target price

    # Specify problem type and metric
    python -m app.main --file data.csv --target label --problem_type classification --metric f1

    # Limit iterations
    python -m app.main --file data.csv --max_iterations 3

    # Enable streaming output
    python -m app.main --file data.csv --stream
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--file", "-f",
        type=str,
        required=True,
        help="Path to the input data file (CSV or Excel)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--target", "-t",
        type=str,
        default=None,
        help="Name of the target column (will be inferred if not provided)"
    )
    
    parser.add_argument(
        "--problem_type", "-p",
        type=str,
        choices=["classification", "regression"],
        default=None,
        help="Problem type: classification or regression (will be inferred if not provided)"
    )
    
    parser.add_argument(
        "--metric", "-m",
        type=str,
        default=None,
        help="Primary evaluation metric (e.g., f1, roc_auc, rmse, r2)"
    )
    
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum number of pipeline iterations (default: 5)"
    )
    
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs",
        help="Directory to store run artifacts (default: runs)"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output to see progress"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (prints LLM prompts and responses at each agent step)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model name (provider auto-detected, e.g. gpt-4o, claude-3-sonnet-20240229)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM API key (falls back to environment variable if not provided)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enable interactive mode for clarification questions (human-in-the-loop)"
    )
    
    parser.add_argument(
        "--auto-smote",
        action="store_true",
        help="Automatically apply SMOTE for imbalanced datasets"
    )
    
    return parser.parse_args()


def create_human_interaction_handler(verbose: bool = False):
    """
    Create a handler function for human interaction during the pipeline.
    
    This handler prompts the user for answers to clarification questions.
    """
    def handler(state: PipelineState) -> PipelineState:
        pending_questions = state.get("pending_questions", [])
        
        if not pending_questions:
            state["planning_complete"] = True
            return state
        
        print("\n" + "=" * 60)
        print("  PLANNING AGENT - Clarification Questions")
        print("=" * 60)
        print("\nThe planning agent has analyzed your data and has some questions:")
        print("(Press Enter to use the default option, or type your choice)\n")
        
        user_responses = {}
        
        for i, question in enumerate(pending_questions, 1):
            q_id = question.get("id")
            q_text = question.get("question", "")
            options = question.get("options", [])
            default = question.get("default", "")
            importance = question.get("importance", "optional")
            context = question.get("context", "")
            
            # Display question
            importance_marker = "🔴" if importance == "required" else "🟡" if importance == "recommended" else "⚪"
            print(f"{importance_marker} Question {i}: {q_text}")
            
            if context:
                print(f"   Context: {context}")
            
            if options:
                print("   Options:")
                for j, opt in enumerate(options, 1):
                    is_default = opt == default
                    marker = " [default]" if is_default else ""
                    print(f"     {j}. {opt}{marker}")
            
            # Get user input
            if options:
                prompt = f"   Your choice (1-{len(options)}, or type custom answer) [{default[:30]}...]: " if len(default) > 30 else f"   Your choice (1-{len(options)}, or type custom answer) [{default}]: "
            else:
                prompt = f"   Your answer [{default}]: "
            
            try:
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                user_input = ""
            
            # Process input
            if not user_input:
                # Use default
                answer = default
            elif options and user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(options):
                    answer = options[idx]
                else:
                    answer = user_input
            else:
                answer = user_input
            
            user_responses[q_id] = answer
            print(f"   → Selected: {answer}\n")
        
        state["user_responses"] = user_responses
        state["planning_complete"] = True
        
        print("=" * 60)
        print("  Continuing with pipeline...")
        print("=" * 60 + "\n")
        
        return state
    
    return handler


def main():
    """Main entry point for the ML pipeline."""
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
        
        from agenticml.ml.tools.llm_factory import detect_provider, _ENV_KEY_MAP
        try:
            provider = detect_provider(args.model)
            env_var = _ENV_KEY_MAP.get(provider, "OPENAI_API_KEY")
        except ValueError:
            env_var = "OPENAI_API_KEY"
        if env_var in str(config_issues):
            print(f"\nPlease set the {env_var} environment variable:")
            print(f"  export {env_var}='your-api-key'")
            print("\nOr create a .env file with:")
            print(f"  {env_var}=your-api-key")
        
        sys.exit(1)
    
    # Validate input file
    valid, error = validate_file_exists(args.file)
    if not valid:
        print(f"Error: {error}")
        sys.exit(1)
    
    # Validate problem type if provided
    if args.problem_type:
        valid, error = validate_problem_type(args.problem_type)
        if not valid:
            print(f"Error: {error}")
            sys.exit(1)
    
    # Generate run ID and create directory
    run_id = generate_run_id()
    run_dir = create_run_directory(args.runs_dir, run_id)
    
    # Setup logging
    logger = setup_logging(run_id, run_dir)
    
    # Print banner
    print("=" * 60)
    print("  Multi-Agent ML Pipeline")
    print("=" * 60)
    print(f"  Run ID:      {run_id}")
    print(f"  Input file:  {args.file}")
    print(f"  Target:      {args.target or 'auto-detect'}")
    print(f"  Problem:     {args.problem_type or 'auto-detect'}")
    print(f"  Metric:      {args.metric or 'auto-select'}")
    print(f"  Max iters:   {args.max_iterations}")
    print(f"  Interactive: {args.interactive}")
    print(f"  Output dir:  {run_dir}")
    print("=" * 60)
    print()
    
    logger.info(f"Starting pipeline run: {run_id}")
    logger.info(f"Input file: {args.file}")
    logger.info(f"Interactive mode: {args.interactive}")
    
    # Set up human interaction handler if interactive mode
    if args.interactive:
        handler = create_human_interaction_handler(verbose=args.verbose)
        set_human_interaction_callback(handler)
        print("📋 Interactive mode enabled - you will be asked clarification questions\n")
    
    # Build user constraints
    user_constraints = {}
    if args.auto_smote:
        user_constraints["auto_smote"] = True
    
    # Create initial state
    initial_state = create_initial_state(
        run_id=run_id,
        file_path=os.path.abspath(args.file),
        run_dir=run_dir,
        target=args.target,
        problem_type=args.problem_type,
        user_metric=args.metric,
        max_iterations=args.max_iterations,
        interactive_mode=args.interactive,
        user_constraints=user_constraints,
        verbose=args.verbose
    )
    
    # Run the pipeline
    try:
        if args.stream:
            print("Running pipeline with streaming output...\n")
            final_state = None
            
            for output in run_pipeline_with_streaming(initial_state):
                # Print node name and brief status
                for node_name, node_state in output.items():
                    if node_name != "__end__":
                        iteration = node_state.get("iteration", 0)
                        print(f"[Iteration {iteration + 1}] Completed: {node_name}")
                        
                        # Print any decisions made
                        decision_log = node_state.get("decision_log", [])
                        if decision_log:
                            latest = decision_log[-1]
                            if latest.get("agent") == node_name.replace("_node", ""):
                                print(f"  → {latest.get('decision', '')[:80]}")
                        
                        final_state = node_state
            
            print()
        else:
            print("Running pipeline...\n")
            final_state = run_pipeline(initial_state)
        
        # Print summary
        print("=" * 60)
        print("  Pipeline Complete")
        print("=" * 60)
        
        if final_state:
            stop_reason = final_state.get("stop_reason", "unknown")
            iterations = final_state.get("iteration", 0) + 1
            best_model = final_state.get("best_model", {})
            
            print(f"  Status:      {stop_reason}")
            print(f"  Iterations:  {iterations}")
            
            if best_model:
                print(f"  Best model:  {best_model.get('name', 'N/A')}")
                print(f"  Score:       {best_model.get('primary_score', 0):.4f}")
            
            # Print issues summary
            critic_issues = final_state.get("critic_issues", [])
            if critic_issues:
                blocking = sum(1 for i in critic_issues if i.get("severity") == "blocking")
                warnings = sum(1 for i in critic_issues if i.get("severity") == "warn")
                print(f"  Issues:      {blocking} blocking, {warnings} warnings")
            
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
