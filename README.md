# Web Scraper Project

A web scraper and RAG (Retrieval Augmented Generation) system for search fund information.

## Project Structure

The project is organized into several modules:

```
src/
├── core/           # Core functionality
│   ├── config.py   # Configuration management
│   └── logging.py  # Centralized logging setup
├── data/           # Data processing functionality
│   ├── embeddings.py  # Embedding generation
│   ├── indexing.py    # Vector index management (FAISS)
│   └── preprocessing.py # Text preprocessing
├── scrapers/       # Web/PDF scraping functionality 
│   ├── pdf_extractor.py
│   └── web_scraper.py
├── retrieval/      # Retrieval functionality
│   └── retrieval.py
├── llm/            # LLM integration
│   ├── query_clarification.py
│   └── answer_generation.py
├── app/            # Web application
│   └── streamlit_app.py
└── scripts/        # Command-line scripts
    ├── generate_embeddings.py
    ├── create_index.py
    └── test_rag.py
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:

```
# Required for embedding generation
OPENAI_API_KEY=your-openai-api-key

# Optional: for using Perplexity as a provider
PERPLEXITY_API_KEY=your-perplexity-api-key

# Optional: configure which LLM config to use
LLM_CONFIG_NAME=llm_config_default.yaml
```

## Usage

### Scraping Documents

Scrape PDF documents from the IESE search funds website:

```bash
python src/scripts/scrape_documents.py --output-dir data/IESE
```

You can also automatically extract text from the downloaded PDFs:

```bash
python src/scripts/scrape_documents.py --output-dir data/IESE --extract-text --text-output-dir data/processed_txt
```

### Generating Embeddings

Generate embeddings for preprocessed text files:

```bash
python src/scripts/generate_embeddings.py --input-dir data/preprocessed_data --output-file data/embeddings.json
```

### Creating FAISS Index

Create a FAISS index from embeddings:

```bash
python src/scripts/create_index.py --embeddings-file data/embeddings.json --index-file data/faiss_index.idx
```

### Testing the RAG Pipeline

Test the RAG pipeline with a query:

```bash
python src/scripts/test_rag.py "How do search funds work?"
```

### Testing LLM Providers

Test the LLM provider abstraction:

```bash
# Test with default configuration
python src/scripts/test_provider.py

# Test with a specific query
python src/scripts/test_provider.py --query "What is a search fund?"

# Test with Perplexity configuration
LLM_CONFIG_NAME=llm_config_perplexity_test.yaml python src/scripts/test_provider.py

# Test with custom models configuration
LLM_CONFIG_NAME=llm_config_custom_models.yaml python src/scripts/test_provider.py --show-config

# Test with demo Anthropic provider (dynamic registration)
LLM_CONFIG_NAME=llm_config_anthropic.yaml python src/scripts/test_provider.py --register-demo-provider
```

### Running the Web Application

Run the Streamlit web application:

```bash
streamlit run src/app/streamlit_app.py
```

## Architecture

The system is built around a RAG (Retrieval Augmented Generation) architecture:

1. **Query Clarification**: The system clarifies the user's query to make it more specific and targeted.
2. **Document Retrieval**: The system retrieves relevant documents using vector similarity search with FAISS.
3. **Answer Generation**: The system generates an answer based on the retrieved documents using an LLM.

## Configuration

The system is configured through a centralized configuration system that supports:

- Environment variables (through `.env` file)
- Streamlit secrets (for deployment)
- YAML configuration files

### LLM Configuration

The project supports multiple LLM providers through a provider abstraction layer. You can switch between providers by setting the `LLM_CONFIG_NAME` environment variable:

```bash
# Use OpenAI for all LLM functions (default)
export LLM_CONFIG_NAME=llm_config_default.yaml

# Use OpenAI for embeddings and Perplexity for other functions
export LLM_CONFIG_NAME=llm_config_perplexity_hybrid.yaml

# Use custom model selection
export LLM_CONFIG_NAME=llm_config_custom_models.yaml
```

The configuration files in the `config/` directory define which provider and model to use for each function:

```yaml
embeddings:
  provider: openai
  model: text-embedding-3-large  # Choose any supported model
  api_key: ${OPENAI_API_KEY}  # Environment variable reference

query_clarification:
  provider: perplexity
  model: pplx-7b-online
  api_key: ${PERPLEXITY_API_KEY}
```

These configurations control:
- Embeddings generation
- Query clarification
- Answer generation
- Document reranking

The system supports adding new providers in two ways:

### Option 1: Built-in Providers

For providers that should be part of the core system:

1. Create a new provider implementation in `src/llm/providers/` that inherits from `BaseLLMProvider`
2. Add it to the built-in modules list in `src/llm/providers/__init__.py`
3. Create a configuration file in `config/` that references your provider

### Option 2: Third-Party Providers (Plugin System)

For providers that should be distributed separately:

1. Create a new Python package with your provider implementation
2. Register it using Python entry points in your setup.py:
   ```python
   entry_points={
       "webscraper.llm_providers": [
           "your_provider = your_package:YourProviderClass",
       ],
   }
   ```
3. Install the package in the same environment as the web scraper

The provider system automatically discovers and loads providers from both sources
at runtime, with no need to modify existing code. See the `examples/anthropic_provider`
directory for a complete example of a third-party provider.

All model names and settings are read from the configuration files without hardcoded values,
making it easy to experiment with different models and providers.

## Contributing

See the [CLAUDE.md](CLAUDE.md) file for code style guidelines and development practices.