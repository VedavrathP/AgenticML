"""
Centralised LLM service.

Every LLM call in the pipeline goes through ``invoke_llm_json`` which
enforces ``response_format={"type": "json_object"}`` so that all agent
decisions are guaranteed to be parseable JSON objects.
"""

import json
import os
from typing import Any, Optional

from langchain_core.messages import BaseMessage

# ---------------------------------------------------------------------------
# Provider detection (preserved from tools/llm_factory.py)
# ---------------------------------------------------------------------------

_MODEL_PREFIXES: list[tuple[str, str]] = [
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("chatgpt-", "openai"),
    ("claude-", "anthropic"),
    ("gemini-", "google"),
    ("models/gemini-", "google"),
]

_ENV_KEY_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

_INSTALL_HINT: dict[str, str] = {
    "openai": "pip install langchain-openai",
    "anthropic": "pip install langchain-anthropic",
    "google": "pip install langchain-google-genai",
}


def detect_provider(model: str) -> str:
    model_lower = model.lower()
    for prefix, provider in _MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot auto-detect provider for model '{model}'. "
        f"Supported prefixes: {', '.join(p for p, _ in _MODEL_PREFIXES)}"
    )


def resolve_api_key(provider: str, explicit_key: str | None = None) -> str:
    if explicit_key:
        return explicit_key
    env_var = _ENV_KEY_MAP.get(provider, "")
    return os.getenv(env_var, "")


# ---------------------------------------------------------------------------
# LLM creation
# ---------------------------------------------------------------------------

_llm_cache: dict[str, Any] = {}


def get_llm(config: Any) -> Any:
    """
    Create (or return cached) LangChain chat model from pipeline config.

    The model is configured with ``response_format={"type": "json_object"}``
    for providers that support it natively (OpenAI).  For other providers the
    system prompt must instruct the model to reply in JSON.
    """
    cache_key = f"{config.llm_model}:{config.llm_temperature}"
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    provider = detect_provider(config.llm_model)
    api_key = resolve_api_key(provider, config.llm_api_key)

    if not api_key:
        env_var = _ENV_KEY_MAP.get(provider, "???")
        raise ValueError(
            f"No API key for provider '{provider}'. "
            f"Set the {env_var} environment variable."
        )

    llm: Any

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            api_key=api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=config.llm_model,
            temperature=config.llm_temperature,
            api_key=api_key,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            google_api_key=api_key,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    _llm_cache[cache_key] = llm
    return llm


# ---------------------------------------------------------------------------
# Structured JSON invocation
# ---------------------------------------------------------------------------

def invoke_llm_json(
    llm: Any,
    messages: list[BaseMessage],
    agent_name: str,
    step_description: str,
    verbose: bool = False,
) -> dict:
    """
    Invoke the LLM and return a **parsed JSON dict**.

    The LLM is expected to have ``response_format=json_object`` set (OpenAI)
    or the system prompt must instruct it to reply in JSON (Anthropic/Google).
    This function parses the response and raises on failure -- there is no
    fallback to free-text parsing.

    Returns:
        Parsed JSON dict from the LLM response.

    Raises:
        ValueError: If the response cannot be parsed as JSON.
    """
    if verbose:
        _print_prompt(agent_name, step_description, messages)

    response = llm.invoke(messages)

    if verbose:
        _print_response(response)

    content: str = getattr(response, "content", str(response))

    # Strip markdown code-fence wrappers that some providers add
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"[{agent_name}] LLM response is not valid JSON.\n"
            f"Step: {step_description}\n"
            f"Raw response:\n{content[:500]}\n"
            f"Parse error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Verbose helpers (preserved from tools/llm.py)
# ---------------------------------------------------------------------------

def _print_prompt(agent_name: str, step_description: str, messages: list) -> None:
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
        if len(content) > 2000:
            content = content[:2000] + f"\n... ({len(content) - 2000} more characters)"
        print(f"  [{role}]")
        for line in content.splitlines():
            print(f"    {line}")
        print()


def _print_response(response: Any) -> None:
    content = getattr(response, "content", str(response))
    width = 60
    print("-" * width)
    print("  RESPONSE:")
    print("-" * width)
    for line in content.splitlines():
        print(f"    {line}")
    print("=" * width)
    print()
