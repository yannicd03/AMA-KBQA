"""Configuration management for AMA KBQA system.

This module provides utilities to load configuration from config.toml
and initialize LLM clients for different providers (OpenRouter, KIT Ollama, KIT)
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


def get_chat_client(max_retries: Optional[int] = None) -> OpenAI:
    """Get an OpenAI client configured for chat/reasoning tasks.

    Args:
        max_retries: Override the SDK's own retry count. Pass 0 when the caller
            wraps this client in a :class:`ama_kbqa.llm.retry.TransientRetry`
            (e.g. ``BaseKBQAAgent``, ``Orchestrator``) — layering the SDK's
            retries under that stepped backoff would double the backoff and,
            per ``ama_kbqa/llm/kit.py``'s rationale, still miss KIT's non-5xx
            "Open WebUI: Server Connection Error" transient. Leave unset
            (default SDK retries) for callers that don't retry themselves.

    Returns:
        OpenAI: Configured OpenAI client instance

    Raises:
        ValueError: If the configured provider is not supported
        KeyError: If required environment variables are missing
    """
    config = load_config()
    provider = config["llm"]["chat_provider"]

    return _create_client(provider, model_type="chat", max_retries=max_retries)


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
    return config["llm"].get("chat_temperature", 1.0)


def get_chat_max_tokens() -> int:
    """Get the configured max tokens for chat.

    Returns:
        int: The max tokens value
    """
    config = load_config()
    return config["llm"].get("chat_max_tokens", 16000)


# Env var carrying the run seed across the process boundary into the agent and
# the MCP server subprocesses (which inherit os.environ). The benchmark sets it
# from --seed; LLM call sites pass it through as the OpenAI `seed` parameter so
# a run is reproducible (same seed) and independently re-rollable (new seed).
LLM_SEED_ENV = "AMA_LLM_SEED"


def get_chat_seed() -> Optional[int]:
    """Return the LLM sampling seed, or None when unset.

    Read from the ``AMA_LLM_SEED`` environment variable (set by the benchmark
    from ``--seed``) so it crosses into MCP subprocesses, falling back to the
    optional ``[llm] chat_seed`` config key. Returns None when neither is set,
    in which case call sites omit the seed and the provider samples freely.
    """
    raw = os.environ.get(LLM_SEED_ENV)
    if raw is None or str(raw).strip() == "":
        config = load_config()
        raw = config.get("llm", {}).get("chat_seed")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


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
    model_type: Literal["chat", "embedding"] = "chat",
    max_retries: Optional[int] = None,
) -> OpenAI:
    """Create an OpenAI client for the specified provider.

    Args:
        provider: The LLM provider name
        model_type: Type of model (chat or embedding)
        max_retries: Override the SDK's own retry count (default 3 when None).
            Pass 0 for a client that a caller wraps in its own retry layer.

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

    # Create OpenAI client with provider-specific settings.
    # We pass a structured httpx.Timeout so socket-level hangs trip the
    # connect/read/write/pool budgets individually rather than waiting on the
    # single bulk `timeout=60.0` value, which we observed not firing on the
    # KIT endpoint (2026-05-08 gemma seed=44 first attempt: 24h hang at
    # iteration 17 of question 4, sleeping at 0% CPU). Tight connect/pool
    # budgets ensure we surface a TimeoutError rather than block the event
    # loop indefinitely; read=60 keeps existing behaviour for healthy calls.
    import httpx  # local import to avoid front-loading the dep at module import
    client_kwargs = {
        "base_url": base_url,
        "api_key": api_key,
        "timeout": httpx.Timeout(connect=20.0, read=60.0, write=10.0, pool=5.0),
        "max_retries": 3 if max_retries is None else max_retries,
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

    # Local providers that don't require an API key (e.g., self-hosted llama.cpp)
    if provider == "llamacpp":
        return "sk-no-key-required"

    # Map providers to their environment variable names
    env_var_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "kit": "KIT_API_KEY",
    }

    env_var = env_var_map.get(provider)

    if not env_var:
        raise ValueError(
            f"No API key configuration found for provider '{provider}'"
        )

    api_key = os.getenv(env_var)
    if not api_key:
        raise KeyError(
            f"Environment variable '{env_var}' is required for provider '{provider}' "
            f"but is not set. Please add it to your .env file."
        )

    return api_key


def get_benchmark_concurrency() -> int:
    """Number of questions to process concurrently in the benchmark runner.

    Default 1 preserves the original strictly-serial behavior exactly (opt-in
    concurrency only). Values > 1 spin up a pool of that many isolated agents
    (each with its own MCP subprocess). A CLI flag (--concurrency) overrides
    this. Respect provider rate limits when raising it.

    Returns:
        int: concurrency level (>= 1)
    """
    config = load_config()
    try:
        value = int(config.get("benchmark", {}).get("concurrency", 1))
    except (TypeError, ValueError):
        return 1
    return max(1, value)


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


# Synthesis configuration functions
def get_synthesis_client(max_retries: Optional[int] = None) -> OpenAI:
    """Get an OpenAI client configured for final answer synthesis.

    Args:
        max_retries: Override the SDK's own retry count. Pass 0 when the caller
            wraps this client in a :class:`ama_kbqa.llm.retry.TransientRetry`
            (see ``get_chat_client`` for the full rationale).

    Returns:
        OpenAI: Configured OpenAI client instance

    Raises:
        ValueError: If the configured provider is not supported
        KeyError: If required environment variables are missing
    """
    config = load_config()

    # Check if synthesis section exists, fallback to chat provider
    if "synthesis" in config and "synthesis_provider" in config["synthesis"]:
        provider = config["synthesis"]["synthesis_provider"]
    else:
        # Fallback to chat provider if synthesis not configured
        logger.warning(
            "Synthesis provider not configured, falling back to chat provider"
        )
        provider = config["llm"]["chat_provider"]

    return _create_client(provider, model_type="chat", max_retries=max_retries)


def get_synthesis_model_name() -> str:
    """Get the configured synthesis model name.

    Returns:
        str: The synthesis model name
    """
    config = load_config()

    # Check if synthesis section exists, fallback to chat model
    if "synthesis" in config and "synthesis_model" in config["synthesis"]:
        return config["synthesis"]["synthesis_model"]
    else:
        # Fallback to chat model if synthesis not configured
        logger.warning(
            "Synthesis model not configured, falling back to chat model"
        )
        provider = config["llm"]["chat_provider"]
        return config[provider]["chat_model"]


def get_synthesis_temperature() -> float:
    """Get the configured synthesis temperature.

    Returns:
        float: The synthesis temperature value
    """
    config = load_config()

    # Check if synthesis section exists, fallback to chat temperature
    if "synthesis" in config:
        return config["synthesis"].get("synthesis_temperature", 0.2)
    else:
        # Fallback to chat temperature if synthesis not configured
        return config["llm"].get("chat_temperature", 1.0)


def get_synthesis_max_tokens() -> int:
    """Get the configured max tokens for synthesis.

    Returns:
        int: The max tokens value
    """
    config = load_config()

    # Check if synthesis section exists, fallback to default
    if "synthesis" in config:
        return config["synthesis"].get("synthesis_max_tokens", 2000)
    else:
        # Default to 2000 tokens for synthesis
        return 2000


def get_auto_inject_journal() -> bool:
    """Whether to auto-inject journal summaries into the tool-loop context.

    When True (default), the base agent:
      - periodically injects a GetJournalSummary refresh every N iterations
      - injects an "answer now" prompt right after the agent calls
        GetJournalSummary itself

    When False, the agent still has GetJournalSummary available as a tool but
    no journal text is auto-pushed into the conversation. The agent must read
    its own tool results to track state.

    Returns:
        bool: True to enable auto-injection (default), False to disable.
    """
    config = load_config()
    return bool(config.get("agent", {}).get("auto_inject_journal", True))


def get_zero_tool_call_retry() -> bool:
    """Whether to re-prompt the agent when it emits a final answer with zero tool calls.

    Some models (notably gemma-4-31b on the KIT endpoint) occasionally bypass
    RULE 0 (MANDATORY TOOL USE) and answer from prior knowledge. When this
    setting is True (default), the tool loop detects a zero-tool-call answer,
    injects a corrective user message, and continues so the agent can recover.

    Returns:
        bool: True to enable retry (default), False to disable.
    """
    config = load_config()
    return bool(config.get("agent", {}).get("zero_tool_call_retry", True))


def get_zero_tool_call_retry_max() -> int:
    """Maximum number of zero-tool-call retries per question.

    Returns:
        int: max retries (default 1). 1 is enough to recover the gemma cases
        observed in the 2026-05-03 fixbundle audit; raising it costs LLM calls
        on the cases that genuinely cannot be answered.
    """
    config = load_config()
    return int(config.get("agent", {}).get("zero_tool_call_retry_max", 1))


def get_synthesis_enabled() -> bool:
    """Whether to run the dedicated synthesis LLM step after the tool loop.

    When False, the agent's own final assistant message is returned directly
    as the answer, skipping the second (synthesis) LLM call entirely.

    Returns:
        bool: True if synthesis is enabled (default), False to bypass.
    """
    config = load_config()
    return bool(config.get("synthesis", {}).get("synthesis_enabled", True))


def get_synthesis_mode() -> str:
    """Get the synthesis answer style.

    Returns:
        "benchmark" (short exact-match answers) or
        "conversational" (verbose, user-friendly answers). Default: "benchmark".
    """
    config = load_config()
    mode = config.get("synthesis", {}).get("synthesis_mode", "benchmark")
    if mode not in ("benchmark", "conversational"):
        logger.warning(
            f"Unknown synthesis_mode '{mode}', falling back to 'benchmark'"
        )
        return "benchmark"
    return mode


def get_synthesis_provider_preferences() -> Optional[dict]:
    """Build provider preferences object for synthesis based on configuration.

    This is similar to get_provider_preferences() but for synthesis.

    Returns:
        Optional[dict]: Provider preferences dict, or None if not applicable
    """
    config = load_config()

    # Check if synthesis section exists, fallback to chat provider
    if "synthesis" in config and "synthesis_provider" in config["synthesis"]:
        provider = config["synthesis"]["synthesis_provider"]
    else:
        provider = config["llm"]["chat_provider"]

    # Only relevant for OpenRouter
    if provider != "openrouter":
        return None

    # Check if there's a synthesis-specific provider preference
    if "synthesis" in config and "synthesis_model_provider" in config["synthesis"]:
        provider_pref = config["synthesis"]["synthesis_model_provider"]
    else:
        # Fallback to chat model provider preference
        provider_pref = get_chat_model_provider()

    if not provider_pref:
        return None

    return {
        "order": [provider_pref],
        "allow_fallbacks": False
    }


# ==============================================================================
# Retrieval (hybrid search + reranker) Configuration Functions
# ==============================================================================

def _parse_env_bool(raw: str) -> bool:
    """Parse a boolean from an environment-variable string."""
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Environment-variable overrides for the [retrieval] section. The frontend
# sets these (AMA_RETRIEVAL_<KEY>) so the settings cross the process boundary
# into the MCP server subprocesses, which inherit the parent environment at
# launch. Env always wins over config.toml; unset vars leave the toml value.
_RETRIEVAL_ENV_PARSERS = {
    "hybrid_enabled": _parse_env_bool,
    "fusion": str,
    "prefetch_limit": int,
    "reranker_enabled": _parse_env_bool,
    "reranker_model": str,
    "rerank_candidates": int,
    "rerank_threshold": float,
}

RETRIEVAL_ENV_PREFIX = "AMA_RETRIEVAL_"


def get_retrieval_config() -> dict:
    """Get the [retrieval] configuration section.

    Values from config.toml are overlaid with ``AMA_RETRIEVAL_<KEY>``
    environment variables (e.g. ``AMA_RETRIEVAL_HYBRID_ENABLED=true``).
    The env channel exists so per-session frontend toggles reach the MCP
    server subprocesses, which inherit the environment when spawned.

    Returns:
        dict: Hybrid-search and reranker settings (empty dict if absent).
    """
    config = load_config()
    section = dict(config.get("retrieval", {}))
    for key, parse in _RETRIEVAL_ENV_PARSERS.items():
        raw = os.environ.get(RETRIEVAL_ENV_PREFIX + key.upper())
        if raw is None:
            continue
        try:
            section[key] = parse(raw)
        except ValueError:
            logger.warning(
                f"Invalid value {raw!r} for {RETRIEVAL_ENV_PREFIX + key.upper()}; "
                f"keeping config.toml value"
            )
    return section


def get_hybrid_enabled() -> bool:
    """Whether hybrid (dense + BM25) retrieval is enabled.

    When False (default), retrieval runs pure dense vector search,
    matching the legacy behavior exactly.

    Returns:
        bool: True to query both the dense and BM25 prefetch branches.
    """
    return bool(get_retrieval_config().get("hybrid_enabled", False))


def get_fusion() -> str:
    """Get the score-fusion method for hybrid retrieval.

    Returns:
        "rrf" (Reciprocal Rank Fusion, default) or "dbsf"
        (Distribution-Based Score Fusion).
    """
    fusion = get_retrieval_config().get("fusion", "rrf")
    if fusion not in ("rrf", "dbsf"):
        logger.warning(f"Unknown fusion method '{fusion}', falling back to 'rrf'")
        return "rrf"
    return fusion


def get_prefetch_limit() -> int:
    """Per-branch candidate count fetched before fusion in hybrid mode.

    Returns:
        int: prefetch limit (default 20).
    """
    return int(get_retrieval_config().get("prefetch_limit", 20))


def get_reranker_enabled() -> bool:
    """Whether the cross-encoder reranker stage is enabled.

    Independent of the hybrid toggle: when True the reranker also
    re-scores pure-dense results.

    Returns:
        bool: True to rerank retrieval candidates (default False).
    """
    return bool(get_retrieval_config().get("reranker_enabled", False))


def get_reranker_model() -> str:
    """Get the cross-encoder reranker model name.

    Returns:
        str: HuggingFace model id (default Alibaba-NLP/gte-reranker-modernbert-base).
    """
    return get_retrieval_config().get(
        "reranker_model", "Alibaba-NLP/gte-reranker-modernbert-base"
    )


def get_rerank_candidates() -> int:
    """How many fused/dense hits are fed to the cross-encoder.

    Returns:
        int: rerank candidate pool size (default 20).
    """
    return int(get_retrieval_config().get("rerank_candidates", 20))


def get_rerank_threshold() -> Optional[float]:
    """Optional gate on cross-encoder scores.

    Returns:
        Optional[float]: minimum rerank score to keep a hit, or None for no gate.
    """
    value = get_retrieval_config().get("rerank_threshold")
    return float(value) if value is not None else None


# ==============================================================================
# SciQA / ORKG Configuration Functions
# ==============================================================================

def get_sciqa_collection_entities() -> str:
    """Get the SciQA entities collection name from config.

    Returns:
        str: The SciQA entities collection name
    """
    config = load_config()
    return config.get("search", {}).get("sciqa_collection_entities", "sciqa-entities")


def get_sciqa_collection_relations() -> str:
    """Get the SciQA relations collection name from config.

    Returns:
        str: The SciQA relations collection name
    """
    config = load_config()
    return config.get("search", {}).get("sciqa_collection_relations", "sciqa-relations")


def get_sciqa_virtuoso_graph() -> str:
    """Get the SciQA Virtuoso graph URI from config.

    Returns:
        str: The SciQA Virtuoso graph URI
    """
    config = load_config()
    return config.get("search", {}).get("sciqa_virtuoso_graph", "http://sciqa.org/kg")


def get_sciqa_config() -> dict:
    """Get the full SciQA configuration section.

    Returns:
        dict: The SciQA configuration dictionary with thresholds and paths
    """
    config = load_config()
    return config.get("sciqa", {})


def get_sciqa_dataset_path(dataset_type: str = "handcrafted") -> str:
    """Get the path to a SciQA dataset.

    Args:
        dataset_type: Either "handcrafted" or "autogenerated"

    Returns:
        str: The path to the dataset directory

    Raises:
        ValueError: If dataset_type is not valid
    """
    if dataset_type.lower() not in ("handcrafted", "autogenerated", "auto"):
        raise ValueError(
            f"Invalid dataset_type '{dataset_type}'. "
            "Must be 'handcrafted', 'autogenerated', or 'auto'."
        )

    config = load_config()
    sciqa_config = config.get("sciqa", {})

    if dataset_type.lower() in ("autogenerated", "auto"):
        return sciqa_config.get(
            "autogenerated_dataset",
            "db/datasets/SciQA/Autogenerated"
        )
    else:
        return sciqa_config.get(
            "handcrafted_dataset",
            "db/datasets/SciQA/Handcrafted"
        )


def get_sciqa_entity_threshold() -> float:
    """Get the SciQA entity search score threshold.

    Returns:
        float: The entity search threshold (default: 0.70)
    """
    config = load_config()
    return config.get("sciqa", {}).get("entity_threshold", 0.70)


def get_sciqa_relation_threshold() -> float:
    """Get the SciQA relation search score threshold.

    Returns:
        float: The relation search threshold (default: 0.65)
    """
    config = load_config()
    return config.get("sciqa", {}).get("relation_threshold", 0.65)
