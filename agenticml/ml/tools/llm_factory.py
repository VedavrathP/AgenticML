"""
LLM factory with automatic provider detection.

Detects the provider (OpenAI, Anthropic, Google) from the model name
and creates the corresponding LangChain chat model. Only the package
for the chosen provider needs to be installed.
"""

import os
from typing import Any

# Prefix -> provider mapping.  Checked in order; first match wins.
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
    "anthropic": "pip install langchain-anthropic   (or: pip install cortexml[anthropic])",
    "google": "pip install langchain-google-genai  (or: pip install cortexml[google])",
}


def detect_provider(model: str) -> str:
    """Return the provider name for a given model string.

    Raises ``ValueError`` if the model name cannot be mapped.
    """
    model_lower = model.lower()
    for prefix, provider in _MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot auto-detect provider for model '{model}'. "
        f"Supported prefixes: {', '.join(p for p, _ in _MODEL_PREFIXES)}"
    )


def resolve_api_key(provider: str, explicit_key: str | None) -> str:
    """Return the API key — explicit value first, then env var."""
    if explicit_key:
        return explicit_key
    env_var = _ENV_KEY_MAP.get(provider, "")
    return os.getenv(env_var, "")


def create_llm(config: Any) -> Any:
    """Create a LangChain chat model from the pipeline config.

    The provider is auto-detected from ``config.llm_model``.  Only the
    LangChain package for that provider is imported (lazy), so users
    only need to install the one they actually use.
    """
    provider = detect_provider(config.llm_model)
    api_key = resolve_api_key(provider, config.llm_api_key)

    if not api_key:
        env_var = _ENV_KEY_MAP.get(provider, "???")
        raise ValueError(
            f"No API key for provider '{provider}'. "
            f"Pass api_key= or set the {env_var} environment variable."
        )

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                f"langchain-openai is required for model '{config.llm_model}'. "
                f"Install it with: {_INSTALL_HINT['openai']}"
            )
        return ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            api_key=api_key,
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                f"langchain-anthropic is required for model '{config.llm_model}'. "
                f"Install it with: {_INSTALL_HINT['anthropic']}"
            )
        return ChatAnthropic(
            model=config.llm_model,
            temperature=config.llm_temperature,
            api_key=api_key,
        )

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                f"langchain-google-genai is required for model '{config.llm_model}'. "
                f"Install it with: {_INSTALL_HINT['google']}"
            )
        return ChatGoogleGenerativeAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            google_api_key=api_key,
        )

    raise ValueError(f"Unsupported provider: {provider}")
