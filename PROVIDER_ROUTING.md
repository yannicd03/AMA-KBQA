# OpenRouter Provider Routing Integration

This document explains how to use OpenRouter's provider routing feature in the AMA KBQA system.

## Overview

OpenRouter allows you to control which provider serves your model requests through provider routing. This is useful when you want to:
- Target specific provider endpoints (e.g., FP8 quantized versions)
- Optimize for price, throughput, or latency
- Ensure specific providers handle your requests

## Configuration

### Basic Setup

In `config.toml`, set the `chat_model_provider` field under the `[openrouter]` section:

```toml
[openrouter]
base_url = "https://openrouter.ai/api/v1"
chat_model = "minimax/minimax-m2"
embedding_model = "qwen/qwen3-embedding-8b"
chat_model_provider = "minimax/fp8"  # Target the FP8 quantized endpoint
```

### Provider Preference Formats

The `chat_model_provider` field can specify:

1. **Specific Provider Endpoint**: `"minimax/fp8"` - Targets the FP8 variant
2. **Provider Name**: `"anthropic"` - Targets any Anthropic endpoint
3. **Multiple Providers**: Currently set one at a time in config

### Finding Provider Slugs

To find the correct provider slug:
1. Visit the model page on OpenRouter (e.g., https://openrouter.ai/models/minimax/minimax-m2)
2. Look at the "Providers" section
3. Click the copy button next to the provider name to get the exact slug
4. The slug includes any variants like "/fp8", "/turbo", etc.

## Code Integration

### In Config Module (`ama_kbqa/config.py`)

Two new functions are available:

```python
from ama_kbqa.config import get_chat_model_provider, get_provider_preferences

# Get the raw provider preference string
provider = get_chat_model_provider()  # Returns "minimax/fp8" or None

# Get the formatted provider preferences object
prefs = get_provider_preferences()
# Returns: {"order": ["minimax/fp8"], "allow_fallbacks": False}
# or None if not configured
```

### In API Calls

When making OpenRouter API calls, include the provider preferences:

```python
from ama_kbqa.config import get_chat_client, get_chat_model_name, get_provider_preferences

client = get_chat_client()
model = get_chat_model_name()

# Build API call parameters
call_params = {
    "model": model,
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.2
}

# Add provider preferences if configured
provider_prefs = get_provider_preferences()
if provider_prefs:
    call_params["provider"] = provider_prefs

# Make the API call
completion = client.chat.completions.create(**call_params)
```

### Example: Updated Server Code

The `orchestrator_server.py` has been updated to use provider routing:

```python
from ama_kbqa.config import get_provider_preferences

# In your tool function
call_params = {
    "model": CHAT_MODEL_ID,
    "messages": messages,
    "response_format": {"type": "json_object"}
}

# Add OpenRouter provider preferences if configured
provider_prefs = get_provider_preferences()
if provider_prefs:
    call_params["provider"] = provider_prefs
    logger.debug(f"Using provider preferences: {provider_prefs}")

completion = client.chat.completions.create(**call_params)
```

## Advanced Options

While the current implementation uses `order` and `allow_fallbacks`, you can extend `get_provider_preferences()` to support additional OpenRouter features:

### Sorting by Throughput or Price

```python
def get_provider_preferences() -> Optional[dict]:
    config = load_config()
    # ... existing code ...

    # Example: Sort by throughput instead of using fixed order
    return {
        "sort": "throughput"  # or "price" or "latency"
    }
```

### Enabling Fallbacks

```python
return {
    "order": [provider_pref],
    "allow_fallbacks": True  # Allow other providers if primary fails
}
```

### Filtering by Quantization

```python
return {
    "quantizations": ["fp8"],  # Only use FP8 quantized providers
}
```

### Setting Maximum Price

```python
return {
    "max_price": {
        "prompt": 1.0,      # Max $1 per million prompt tokens
        "completion": 2.0   # Max $2 per million completion tokens
    }
}
```

### Data Collection Policies

```python
return {
    "data_collection": "deny",  # Only use providers that don't store data
}
```

### Zero Data Retention (ZDR)

```python
return {
    "zdr": True,  # Only route to ZDR endpoints
}
```

## Behavior

### When Provider Preference is Set

- The system will ONLY use the specified provider endpoint
- `allow_fallbacks` is set to `False` by default
- If the specified provider is unavailable, the request will fail
- This ensures you always get the exact provider/quantization you requested

### When No Provider Preference is Set

- OpenRouter uses its default load balancing strategy
- Requests are distributed across providers based on price and uptime
- Automatic failover to backup providers

## Testing

To verify provider routing is working:

1. Set `chat_model_provider` in `config.toml`
2. Make an API call through any agent or server
3. Check the logs for: `Using provider preferences: {'order': ['...'], 'allow_fallbacks': False}`
4. Verify the response headers (if logging) to see which provider was used

## Troubleshooting

### Request Fails with "No providers available"

**Cause**: The specified provider doesn't support your model or is unavailable.

**Solution**:
- Verify the provider slug is correct
- Check that the provider actually offers your model
- Set `allow_fallbacks: True` to allow backup providers

### Provider Preference Not Applied

**Cause**: You might not be using OpenRouter as the chat provider.

**Solution**:
- Verify `chat_provider = "openrouter"` in `[llm]` section
- The provider routing only works with OpenRouter

### Different Provider Being Used

**Cause**: The `chat_model_provider` field might not be set or is incorrect.

**Solution**:
- Check `config.toml` has the `chat_model_provider` field under `[openrouter]`
- Verify the spelling of the provider slug

## Reference

For more details on OpenRouter provider routing, see:
- [OpenRouter Provider Routing Documentation](https://openrouter.ai/docs/guides/routing/provider-selection)
