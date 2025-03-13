# Web Scraper Project Guidelines

## Commands
- **Run application**: `streamlit run src/app.py`
- **Run scraper**: `python src/scrape.py`
- **Test RAG pipeline**: `python src/rag_test.py`
- **Generate embeddings**: `python src/gen_embeddings.py`
- **Create FAISS index**: `python src/FAISS_index.py`

## Code Style
- **Imports**: Standard library first, third-party packages second, local modules last
- **Formatting**: 4 spaces for indentation, 2 blank lines between top-level functions
- **Types**: Use type hints for function parameters and return values
- **Naming**: `snake_case` for functions/variables, `UPPER_CASE` for constants
- **Error handling**: Use specific exception types with descriptive messages
- **Logging**: Include both console and file handlers with appropriate levels
- **Documentation**: Include docstrings for functions and descriptive comments

## Project Structure
- **src/**: Source code
- **data/**: Contains raw files, embeddings, and FAISS index
- **docs/**: Documentation and reference materials