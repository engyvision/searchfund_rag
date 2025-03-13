# Web Scraper Project Guidelines

## Commands
- **Run application**: `streamlit run src/app.py`
- **Run scraper**: `python src/scrape.py`
- **Test RAG pipeline**: `python src/rag_test.py`
- **Generate embeddings**: `python src/gen_embeddings.py`
- **Create FAISS index**: `python src/FAISS_index.py`

## Improved Project Structure
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

## Code Style
- **Imports**: 
  - Standard library first, third-party packages second, local modules last
  - Group related imports together
  - Use absolute imports rather than relative

- **Formatting**: 
  - 4 spaces for indentation, 2 blank lines between top-level functions
  - Maximum line length of 88 characters (Black formatter standard)
  - Use trailing commas in multi-line collections

- **Types**: 
  - Use type hints for ALL function parameters and return values
  - Use generic types (List, Dict, Optional) from typing module
  - Add TypedDict for complex dictionary structures

- **Naming**: 
  - `snake_case` for functions/variables, `UPPER_CASE` for constants
  - Descriptive and specific names (e.g., `retrieve_similar_documents` not just `retrieve`)
  - Class names in `PascalCase`
  - Boolean variables with `is_`, `has_`, or `should_` prefix

- **Error handling**: 
  - Use specific exception types with descriptive messages
  - Create custom exception classes for domain-specific errors
  - Always log the exception with stack trace for debugging

- **Logging**: 
  - Include both console and file handlers with appropriate levels
  - Use structured logging with context data
  - Different log levels for development vs. production

- **Documentation**: 
  - Module-level docstrings explaining purpose and usage
  - Function docstrings using Google style format 
  - Examples for complex functions
  - Architecture documentation explaining component interactions
  - Comments explaining "why" not "what"

## Testing Strategy
- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test interactions between components
- **End-to-end tests**: Test complete workflows
- **Fixtures**: Reuse test data and configurations

## Configuration Management
- Store configuration in environment variables and YAML files
- Use centralized configuration with validation
- Don't hardcode sensitive values like API keys