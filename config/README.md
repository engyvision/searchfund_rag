# Configuration Files

This directory contains configuration files for the web scraper project. These files control various aspects of the application, including which LLM providers and models to use for different functions.

## LLM Configuration Files

- `llm_config_default.yaml`: Default configuration using OpenAI for all LLM functions
- `llm_config_perplexity_hybrid.yaml`: Hybrid configuration using OpenAI for embeddings and Perplexity for reasoning functions
- `llm_config_perplexity_test.yaml`: Test configuration using Perplexity only for query clarification
- `llm_config_custom_models.yaml`: Example configuration with custom model selections

## Format

Each LLM configuration file specifies which provider to use for each function and any provider-specific settings:

```yaml
embeddings:
  provider: openai
  model: text-embedding-3-small
  api_key: ${OPENAI_API_KEY}  # Will be filled from environment variable
  # Add any provider-specific settings here

query_clarification:
  provider: perplexity
  model: pplx-7b-online
  api_key: ${PERPLEXITY_API_KEY}
  # Add any provider-specific settings here

answer_generation:
  provider: perplexity
  model: pplx-7b-online
  api_key: ${PERPLEXITY_API_KEY}
  # Add any provider-specific settings here

reranking:
  provider: perplexity
  model: pplx-7b-online
  api_key: ${PERPLEXITY_API_KEY}
  # Add any provider-specific settings here
```

## Extending the Configuration

You can add any provider-specific settings to each function configuration block. The provider implementations will extract the settings they need from the configuration. This makes it easy to add new settings without changing code:

```yaml
embeddings:
  provider: openai
  model: text-embedding-3-small
  api_key: ${OPENAI_API_KEY}
  dimensions: 1536  # Example of a provider-specific setting
  batch_size: 100   # Example of a provider-specific setting
```

## Usage

Set the `LLM_CONFIG_NAME` environment variable to choose which configuration to use:

```bash
# In .env file or environment
LLM_CONFIG_NAME=llm_config_default.yaml
```

## Adding New Providers

To add support for a new LLM provider:

1. Create a new provider implementation in `src/llm/providers/` that inherits from `BaseLLMProvider`
2. Register the provider in `src/llm/providers/factory.py`
3. Create a configuration file here that references your provider

The factory will automatically pick up the new provider and pass the configuration to it.