"""
Central LLM invocation helper with verbose output support.

Wraps all LLM calls so that when verbose mode is active, the prompt
sent to the model and the raw response are printed to the console.
"""

from typing import Any


def invoke_llm(
    llm: Any,
    messages: list,
    agent_name: str,
    step_description: str,
    verbose: bool = False
) -> Any:
    """
    Invoke an LLM and optionally print the prompt and response.

    Args:
        llm: A LangChain chat model instance (created by ``create_llm``).
        messages: List of LangChain message objects to send.
        agent_name: Human-readable agent name (e.g. "Cleaner").
        step_description: Short description of what this call does.
        verbose: When True, print prompt and response to console.

    Returns:
        The LLM response object (same as ``llm.invoke(messages)``).
    """
    if verbose:
        _print_prompt(agent_name, step_description, messages)

    response = llm.invoke(messages)

    if verbose:
        _print_response(response)

    return response


def _print_prompt(agent_name: str, step_description: str, messages: list) -> None:
    """Print a formatted header and the user prompt."""
    header = f"[{agent_name}] {step_description}"
    width = max(60, len(header) + 4)

    print()
    print("=" * width)
    print(f"  {header}")
    print("-" * width)
    print("  PROMPT:")
    print("-" * width)

    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        content = getattr(msg, "content", str(msg))
        # Truncate very long prompts to keep output readable
        if len(content) > 2000:
            content = content[:2000] + f"\n... ({len(content) - 2000} more characters)"
        print(f"  [{role}]")
        for line in content.splitlines():
            print(f"    {line}")
        print()


def _print_response(response: Any) -> None:
    """Print the raw LLM response."""
    content = getattr(response, "content", str(response))
    width = 60

    print("-" * width)
    print("  RESPONSE:")
    print("-" * width)
    for line in content.splitlines():
        print(f"    {line}")
    print("=" * width)
    print()
