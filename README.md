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

2. Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your-api-key
```

## Usage

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

## Contributing

See the [CLAUDE.md](CLAUDE.md) file for code style guidelines and development practices.