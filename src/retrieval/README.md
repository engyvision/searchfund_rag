# Enhanced Retrieval System

This directory contains an advanced retrieval system implementing a "Reranked Contextual Embedding + Contextual BM25" approach based on Anthropic's recommendations. This hybrid approach combines multiple retrieval methods and adds reranking for significantly improved retrieval quality.

## Components

### 1. Contextual Embeddings (`contextual_embeddings.py`)

The contextual embedding retriever enhances traditional vector search by considering the query context when creating document embeddings. This helps address the "vocabulary mismatch problem" where documents may use different terms than the query but still be semantically relevant.

- Uses the query as context when creating document embeddings
- Generates more query-focused document representations
- Improves retrieval of semantically similar content

### 2. BM25 Retrieval (`bm25.py`)

The BM25 retriever uses a proven keyword-based retrieval algorithm that excels at finding documents with exact term matches. It complements embedding-based retrieval for queries where exact term matching is important.

- Implements the Okapi BM25 algorithm for keyword-based search  
- Focuses on exact term matches and term frequency
- Excels at finding documents containing specific terms from the query

### 3. Hybrid Retrieval (`hybrid.py`)

The hybrid retriever combines results from both contextual embeddings and BM25 for more robust retrieval. By merging the strengths of both approaches, it produces better results for various query types.

- Combines semantic (contextual embeddings) and lexical (BM25) results
- Uses weighted scoring to balance both retrieval methods
- Provides more robust retrieval across different query types

### 4. LLM Reranking (`reranker.py`)

The LLM reranker uses a language model to evaluate the relevance of retrieved documents to the query. This final step significantly improves retrieval precision by bringing the most relevant documents to the top.

- Uses an LLM to evaluate document relevance to the query
- Assigns relevance scores and explanations for each document
- Reorders documents based on relevance judgment
- Helps filter out false positives from other retrieval methods

## Main Interface

The main retrieval interface (`retrieval.py`) integrates all these components into a unified system that provides significant improvements over traditional vector search alone.

## Usage

You can use the enhanced retrieval system with default settings:

```python
from src.retrieval import DocumentRetriever

# Initialize with all enhancements enabled
retriever = DocumentRetriever(
    use_hybrid_retrieval=True,
    use_reranking=True
)

# Retrieve documents
results = retriever.retrieve_documents("your query here", top_k=5)
```

Or with specific configurations:

```python
# Use only contextual embeddings + BM25 without reranking
retriever = DocumentRetriever(
    use_hybrid_retrieval=True,
    use_reranking=False
)

# Use traditional vector search with reranking
retriever = DocumentRetriever(
    use_hybrid_retrieval=False,
    use_reranking=True
)

# Use traditional vector search only
retriever = DocumentRetriever(
    use_hybrid_retrieval=False,
    use_reranking=False
)
```

## Testing

You can evaluate different retrieval configurations using the test script:

```bash
python src/scripts/test_retrieval.py
```

This script compares retrieval performance across different configurations and provides metrics on retrieval time and relevance.