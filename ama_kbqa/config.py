"""Configuration management for AMA KBQA system.

This module provides utilities to load configuration from config.toml
and initialize LLM clients for different providers (OpenRouter, LMStudio, KIT Ollama, etc.)
using the OpenAI client wrapper for compatibility.
"""

from __future__ import annotations
import os
import tomllib
from pathlib import Path
from typing import Literal, Optional
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Find the repository root (where config.toml should be located)
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.toml"

# Global config cache
_config_cache: Optional[dict] = None


def load_config() -> dict:
    """Load configuration from config.toml file.

    Returns:
        dict: The parsed configuration dictionary

    Raises:
        FileNotFoundError: If config.toml is not found
        ValueError: If config.toml cannot be parsed
    """
    global _config_cache

    # Return cached config if available
    if _config_cache is not None:
        return _config_cache

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {CONFIG_PATH}. "
            "Please create config.toml in the project root."
        )

    try:
        with open(CONFIG_PATH, "rb") as f:
            _config_cache = tomllib.load(f)
        logger.info(f"Configuration loaded from {CONFIG_PATH}")
        return _config_cache
    except Exception as e:
        raise ValueError(f"Failed to parse config.toml: {e}")


def get_chat_client() -> OpenAI:
    """Get an OpenAI client configured for chat/reasoning tasks.

    Returns:
        OpenAI: Configured OpenAI client instance

    Raises:
        ValueError: If the configured provider is not supported
        KeyError: If required environment variables are missing
    """
    config = load_config()
    provider = config["llm"]["chat_provider"]

    return _create_client(provider, model_type="chat")


def get_embedding_client() -> OpenAI:
    """Get an OpenAI client configured for embedding tasks.

    Returns:
        OpenAI: Configured OpenAI client instance

    Raises:
        ValueError: If the configured provider is not supported
        KeyError: If required environment variables are missing
    """
    config = load_config()
    provider = config["llm"]["embedding_provider"]

    return _create_client(provider, model_type="embedding")


def get_chat_model_name() -> str:
    """Get the configured chat model name.

    Returns:
        str: The chat model name
    """
    config = load_config()
    provider = config["llm"]["chat_provider"]
    return config[provider]["chat_model"]


def get_embedding_model_name() -> str:
    """Get the configured embedding model name.

    Returns:
        str: The embedding model name
    """
    config = load_config()
    provider = config["llm"]["embedding_provider"]

    # Some providers might not have a dedicated embedding model
    provider_config = config[provider]
    if "embedding_model" in provider_config:
        return provider_config["embedding_model"]
    else:
        # Fallback to chat model if no embedding model is specified
        logger.warning(
            f"No embedding model configured for {provider}, "
            f"falling back to chat model"
        )
        return provider_config["chat_model"]


def get_chat_temperature() -> float:
    """Get the configured chat temperature.

    Returns:
        float: The chat temperature value
    """
    config = load_config()
    return config["llm"].get("chat_temperature", 0.2)


def get_chat_max_tokens() -> int:
    """Get the configured max tokens for chat.

    Returns:
        int: The max tokens value
    """
    config = load_config()
    return config["llm"].get("chat_max_tokens", 4000)


def get_chat_model_provider() -> Optional[str]:
    """Get the configured chat model provider preference.

    This is used for OpenRouter provider routing to target specific
    provider endpoints (e.g., "minimax/fp8", "deepinfra/turbo").

    Returns:
        Optional[str]: The provider preference string, or None if not configured
    """
    config = load_config()
    provider = config["llm"]["chat_provider"]

    # Only relevant for OpenRouter
    if provider != "openrouter":
        return None

    return config.get("openrouter", {}).get("chat_model_provider")


def get_provider_preferences() -> Optional[dict]:
    """Build OpenRouter provider preferences object based on configuration.

    This constructs the 'provider' parameter for OpenRouter API requests
    according to the chat_model_provider setting in config.toml.

    Returns:
        Optional[dict]: Provider preferences dict for OpenRouter, or None if not applicable

    Example return value:
        {
            "order": ["minimax/fp8"],
            "allow_fallbacks": False
        }
    """
    config = load_config()
    provider = config["llm"]["chat_provider"]

    # Only relevant for OpenRouter
    if provider != "openrouter":
        return None

    provider_pref = get_chat_model_provider()
    if not provider_pref:
        # No specific provider preference, use default load balancing
        return None

    # Build provider preferences object
    # When a specific provider variant is specified, disable fallbacks
    # to ensure we only use that specific endpoint
    return {
        "order": [provider_pref],
        "allow_fallbacks": False
    }


def _create_client(
    provider: str,
    model_type: Literal["chat", "embedding"] = "chat"
) -> OpenAI:
    """Create an OpenAI client for the specified provider.

    Args:
        provider: The LLM provider name
        model_type: Type of model (chat or embedding)

    Returns:
        OpenAI: Configured OpenAI client instance

    Raises:
        ValueError: If the provider is not supported
        KeyError: If required environment variables are missing
    """
    config = load_config()

    if provider not in config:
        raise ValueError(
            f"Provider '{provider}' not found in config.toml. "
            f"Available providers: {list(config.keys())}"
        )

    provider_config = config[provider]
    base_url = provider_config["base_url"]

    # Get API key from environment or config
    api_key = _get_api_key(provider, provider_config)

    # Create OpenAI client with provider-specific settings
    client_kwargs = {
        "base_url": base_url,
        "api_key": api_key,
        "timeout": 60.0,
        "max_retries": 3
    }

    # Add OpenRouter-specific headers for rankings
    if provider == "openrouter":
        client_kwargs["default_headers"] = {
            "HTTP-Referer": "https://github.com/MaxKlat29/AMAKBQA",
            "X-Title": "ama-kbqa"
        }

    client = OpenAI(**client_kwargs)

    logger.info(
        f"Created {model_type} client for provider '{provider}' "
        f"at {base_url}"
    )

    return client


def _get_api_key(provider: str, provider_config: dict) -> str:
    """Get API key for the specified provider.

    Args:
        provider: The provider name
        provider_config: The provider configuration dictionary

    Returns:
        str: The API key

    Raises:
        KeyError: If the required environment variable is not set
    """
    # Check if API key is specified directly in config
    if "api_key" in provider_config:
        return provider_config["api_key"]

    # Map providers to their environment variable names
    env_var_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "kit_ollama": "KIT_OLLAMA_TOKEN",
        "deepseek": "DEEPSEEK_API_KEY",
        "zai": "ZAI_API_KEY",
        "lm_studio": "LM_STUDIO_API_KEY"  # Optional for local
    }

    env_var = env_var_map.get(provider)

    if not env_var:
        raise ValueError(
            f"No API key configuration found for provider '{provider}'"
        )

    # For local providers like lm_studio, use a default if not set
    if provider == "lm_studio":
        return os.getenv(env_var, "lm-studio")

    # For remote providers, require the environment variable
    api_key = os.getenv(env_var)
    if not api_key:
        raise KeyError(
            f"Environment variable '{env_var}' is required for provider '{provider}' "
            f"but is not set. Please add it to your .env file."
        )

    return api_key


def get_database_config() -> dict:
    """Get database configuration.

    Returns:
        dict: Database configuration including Qdrant and Virtuoso settings
    """
    config = load_config()
    return config.get("database", {})


def get_search_config() -> dict:
    """Get search configuration.

    Returns:
        dict: Search parameters like top_n and score_threshold
    """
    config = load_config()
    return config.get("search", {})


# Convenience functions for commonly used config values
def get_qdrant_host() -> str:
    """Get Qdrant host from config."""
    return get_database_config().get("qdrant_host", "localhost")


def get_qdrant_port() -> int:
    """Get Qdrant port from config."""
    return get_database_config().get("qdrant_port", 6333)


def get_virtuoso_endpoint() -> str:
    """Get Virtuoso SPARQL endpoint from config."""
    return get_database_config().get(
        "virtuoso_endpoint",
        "http://localhost:8890/sparql"
    )


def get_collection_entities() -> str:
    """Get entities collection name from config."""
    return get_database_config().get("collection_entities", "kqapro-entities")


def get_collection_relations() -> str:
    """Get relations collection name from config."""
    return get_database_config().get("collection_relations", "kqapro-relations")


def get_top_n() -> int:
    """Get top N search results from config."""
    return get_search_config().get("top_n", 5)


def get_score_threshold() -> float:
    """Get score threshold for vector search from config."""
    return get_search_config().get("score_threshold", 0.7)
