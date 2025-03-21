from typing import List, Dict, Any, Optional
import numpy as np
from openai import OpenAI
from src.core import OPENAI_CONFIG
from src.core.logging import get_logger

# Initialize logger
logger = get_logger("retrieval.contextual_embeddings")

class ContextualEmbeddingRetriever:
    """Retriever that uses query context to guide document embedding and retrieval."""
    
    def __init__(self, index, embedding_function, top_k: int = 10):
        """
        Initialize the contextual embedding retriever.
        
        Args:
            index: FAISS index containing document embeddings
            embedding_function: Function to create embeddings
            top_k: Number of documents to retrieve
        """
        self.index = index
        self.embedding_function = embedding_function
        self.top_k = top_k
        self.client = OpenAI(api_key=OPENAI_CONFIG.api_key)
    
    def _get_contextual_embedding_prompt(self, query: str) -> str:
        """Generate a prompt that includes the query context for embedding."""
        return f"""
        You are embedding a document to determine if it's relevant to the following query:
        
        Query: {query}
        
        Document: {{document}}
        """
    
    def _embed_documents_with_context(self, documents: List[str], query: str) -> np.ndarray:
        """Embed documents with query context."""
        prompt_template = self._get_contextual_embedding_prompt(query)
        
        # Calculate approximately how many tokens the prompt template uses
        # (query + instructions, but without the document)
        base_prompt_tokens = len(prompt_template.format(document="").split())
        
        # Set a max token limit for documents to keep total under the model's limit (8192)
        # Leave some margin for the tokenization differences
        max_doc_tokens = 2000  # Setting a much smaller limit for safety
        
        contextual_docs = []
        for doc in documents:
            # Truncate document if it's too long - being much more aggressive with truncation
            doc_preview = doc[:5000]  # Take only the first part of the document as a preview
            if len(doc_preview.split()) > max_doc_tokens:
                # Simple truncation by words (approximate)
                truncated_doc = " ".join(doc_preview.split()[:max_doc_tokens]) + "..."
                contextual_docs.append(prompt_template.format(document=truncated_doc))
            else:
                contextual_docs.append(prompt_template.format(document=doc_preview))
        
        # Process in batches to avoid memory issues or API limits
        batch_size = 5  # Smaller batch size for safety
        all_embeddings = []
        
        for i in range(0, len(contextual_docs), batch_size):
            batch = contextual_docs[i:i+batch_size]
            try:
                batch_embeddings = self.embedding_function(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                # If batch fails, try one by one
                for doc in batch:
                    try:
                        # Try with an even more aggressively truncated document
                        truncated_doc = doc[:1000] + "..."
                        single_embedding = self.embedding_function([truncated_doc])
                        all_embeddings.extend(single_embedding)
                    except Exception as inner_e:
                        # If still failing, use query embedding as fallback
                        query_embedding = self.embedding_function([query])
                        all_embeddings.extend(query_embedding)
        
        return np.array(all_embeddings)
    
    def retrieve(self, query: str, documents: List[str], document_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Retrieve documents using contextual embeddings.
        
        Args:
            query: User query
            documents: List of document texts
            document_metadata: Metadata for each document
            
        Returns:
            List of retrieved documents with their metadata and scores
        """
        # Create query embedding
        query_embedding = self.embedding_function([query])[0]
        
        # Create contextual document embeddings (this would be done in batches in practice)
        document_embeddings = self._embed_documents_with_context(documents, query)
        
        # Calculate similarities
        similarities = np.dot(document_embeddings, query_embedding)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        # Return results
        results = [
            {
                "document": documents[i],
                "metadata": document_metadata[i],
                "score": float(similarities[i])
            }
            for i in top_indices
        ]
        
        # Log embedding retrieval results
        for i, result in enumerate(results):
            doc_preview = result["document"][:50].replace('\n', ' ').strip()
            score = result.get("score", 0)
            
            # Get metadata for logging
            meta = result.get("metadata", {})
            file_info = meta.get("file", "unknown")
            if isinstance(file_info, str) and len(file_info) > 20:
                file_info = file_info[:20] + "..."
                
            logger.debug(f"Contextual embedding result {i+1}: file={file_info}, score={score:.3f} - '{doc_preview}...'")
        
        return results