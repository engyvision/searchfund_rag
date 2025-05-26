# LLM Providers

This directory contains provider implementations for different LLM services. Each provider implements the `BaseLLMProvider` interface, which standardizes the API for common LLM operations:

- Embedding generation
- Query clarification
- Answer generation
- Document reranking

## Available Providers

- **OpenAI Provider**: Implements the full interface using OpenAI's API
- **Perplexity Provider**: Implements reasoning capabilities (clarification, generation, reranking) with Perplexity API

## Provider Architecture

Each provider:
1. Inherits from `BaseLLMProvider` abstract base class
2. Accepts a full configuration dictionary in its constructor
3. Extracts provider-specific settings from the configuration
4. Implements all required methods for the abstraction layer

## Dynamic Provider Discovery

The system automatically discovers providers through two mechanisms:

1. **Built-in providers**: Providers included in the core system
2. **Entry point providers**: Third-party providers registered via entry points

This allows adding new providers without modifying any existing code.

## Adding a Built-in Provider

To add a provider to the core system:

1. Create a new file `your_provider.py` that implements the `BaseLLMProvider` interface:
   ```python
   class YourProvider(BaseLLMProvider):
       # Default environment variable for API key
       DEFAULT_API_KEY_ENV: ClassVar[str] = "YOUR_PROVIDER_API_KEY"
       
       # Static provider name
       PROVIDER_NAME: ClassVar[str] = "your_provider"
       
       # Default model names
       DEFAULT_EMBEDDING_MODEL = "your-embedding-model"
       DEFAULT_COMPLETION_MODEL = "your-completion-model"
       DEFAULT_CLARIFICATION_MODEL = "your-clarification-model"
       
       def __init__(self, config: Dict[str, Any]):
           # Extract settings from config
           api_key = self._get_api_key_from_config(config)
           # Initialize your client
           # ...
   ```

2. Add your module to the list in `__init__.py`:
   ```python
   # Import built-in provider modules
   from src.llm.providers import openai_provider, perplexity_provider, your_provider
   
   # Scan modules for provider classes
   modules_to_scan = [
       openai_provider,
       perplexity_provider,
       your_provider
   ]
   ```

## Creating a Third-Party Provider

To create a provider that can be distributed separately:

1. Create a new Python package with your provider implementation
2. Register it using entry points in setup.py:
   ```python
   setup(
       name="your_provider_package",
       # ...
       entry_points={
           "webscraper.llm_providers": [
               "your_provider = your_provider_package:YourProviderClass",
           ],
       },
   )
   ```

See the `examples/anthropic_provider` directory for a complete example.

## Manual Provider Registration

During development, you can register providers directly:

```python
from src.llm.providers import PROVIDER_REGISTRY
from your_module import YourProvider

# Register the provider
PROVIDER_REGISTRY["your_provider"] = YourProvider
```

## Configuration

Create a configuration file in `config/` that references your provider:
```yaml
embeddings:
  provider: your_provider
  model: your-embedding-model
  api_key: ${YOUR_PROVIDER_API_KEY}

query_clarification:
  provider: your_provider
  model: your-clarification-model
  api_key: ${YOUR_PROVIDER_API_KEY}
```

The system is designed to be fully configurable from YAML:
- No hardcoded model names or settings in the code
- All provider-specific settings are extracted from the configuration
- New settings can be added without changing the provider interface
- Environment variables in the config are automatically resolved