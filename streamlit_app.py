import os
import sys
import streamlit as st

# Make this the main entry point for Streamlit Cloud
# This file should be at the root of the repository

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import necessary components
from src.core.logging import get_logger
from src.llm.query_clarification import QueryClarifier
from src.llm.answer_generation import AnswerGenerator
from src.retrieval.retrieval import DocumentRetriever

# Initialize logger
logger = get_logger("app.streamlit")

# Initialize components
query_clarifier = None
document_retriever = None
answer_generator = None

def initialize_components(use_hybrid_retrieval: bool = True, use_reranking: bool = True, use_contextual_index: bool = False):
    """Initialize application components.
    
    Args:
        use_hybrid_retrieval: Whether to use hybrid retrieval
        use_reranking: Whether to use LLM reranking
        use_contextual_index: Whether to use the contextual index file
    """
    global query_clarifier, document_retriever, answer_generator
    
    if query_clarifier is None:
        with st.spinner("Loading query clarifier..."):
            query_clarifier = QueryClarifier()
    
    # Always recreate document retriever to apply new settings
    with st.spinner("Loading document retriever..."):
        from src.core.config import FAISS_CONFIG
        
        # Override index file if using contextual index
        if use_contextual_index:
            # Check if contextual index exists
            import os
            # Use relative path that works across environments
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            contextual_index_file = os.path.join(data_dir, "faiss_contextual_index.idx")
            contextual_metadata_file = os.path.join(data_dir, "faiss_contextual_metadata.json")
            
            if os.path.exists(contextual_index_file) and os.path.exists(contextual_metadata_file):
                # Import and initialize FAISSIndexer directly to override settings
                from src.data.indexing import FAISSIndexer
                faiss_indexer = FAISSIndexer(
                    index_file=contextual_index_file,
                    metadata_file=contextual_metadata_file
                )
                
                document_retriever = DocumentRetriever(
                    faiss_indexer=faiss_indexer,
                    use_hybrid_retrieval=use_hybrid_retrieval,
                    use_reranking=use_reranking
                )
                st.success("Using contextual index file for better retrieval!")
            else:
                st.warning("Contextual index not found. Using standard index.")
                document_retriever = DocumentRetriever(
                    use_hybrid_retrieval=use_hybrid_retrieval,
                    use_reranking=use_reranking
                )
        else:
            document_retriever = DocumentRetriever(
                use_hybrid_retrieval=use_hybrid_retrieval,
                use_reranking=use_reranking
            )
    
    if answer_generator is None:
        with st.spinner("Loading answer generator..."):
            answer_generator = AnswerGenerator()

def process_query(
    query: str, 
    top_k: int = 3, 
    include_source_explanations: bool = True,
    reasoning_effort: str = "medium"
):
    """Process a user query.
    
    Args:
        query: The user's query
        top_k: Number of documents to retrieve
        include_source_explanations: Whether to include source explanations in the answer
        reasoning_effort: Reasoning effort for the model (low, medium, high)
        
    Returns:
        tuple: Clarified query, retrieved documents, and final answer
    """
    # Step 1: Clarify the query
    clarified_query = query_clarifier.clarify_query(query)
    
    # Step 2: Retrieve documents
    context = document_retriever.retrieve_and_format(
        clarified_query, 
        top_k=top_k,
        include_metadata=True
    )
    
    # Step 3: Generate answer
    answer = answer_generator.generate_answer(
        clarified_query,
        context,
        include_source_explanations=include_source_explanations,
        reasoning_effort=reasoning_effort
    )
    
    return clarified_query, context, answer

def main():
    """Main function for the streamlit app."""
    st.title("Search Fund Assistant")
    st.write("Ask any question about search funds and get expert answers!")
    
    # Retrieval method settings
    st.sidebar.header("Retrieval Settings")
    
    # Add option to use contextual index
    import os
    # Use relative path that works across environments
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    contextual_index_file = os.path.join(data_dir, "faiss_contextual_index.idx")
    contextual_metadata_file = os.path.join(data_dir, "faiss_contextual_metadata.json")
    contextual_index_exists = os.path.exists(contextual_index_file) and os.path.exists(contextual_metadata_file)
    
    if contextual_index_exists:
        use_contextual_index = st.sidebar.checkbox("Use Contextual Index (Recommended)", value=True, 
                                                help="Use the pre-chunked contextual index for better retrieval")
    else:
        use_contextual_index = False
        st.sidebar.info("Contextual index not available. Run `python src/scripts/create_contextual_index.py --resume` to create it.")
    
    # Allow users to toggle hybrid retrieval and reranking
    use_hybrid_retrieval = st.sidebar.checkbox("Use Hybrid Retrieval (BM25 + Contextual Embeddings)", value=True)
    use_reranking = st.sidebar.checkbox("Use LLM Reranking", value=True)
    
    # Allow users to set the number of documents to retrieve
    top_k = st.sidebar.slider("Number of documents to retrieve", min_value=1, max_value=10, value=3)
    
    # Allow users to adjust LLM settings
    st.sidebar.header("LLM Settings")
    reasoning_effort = st.sidebar.selectbox(
        "Reasoning effort", 
        options=["low", "medium", "high"],
        index=1
    )
    include_source_explanations = st.sidebar.checkbox("Include source explanations", value=True)
    
    # Initialize components with user settings
    try:
        initialize_components(use_hybrid_retrieval, use_reranking, use_contextual_index)
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.info("Using fallback initialization...")
        # Fallback to basic retrieval without advanced features
        use_hybrid_retrieval = False
        use_reranking = False
        use_contextual_index = False
        initialize_components(False, False, False)
    
    # Create a text input for the user query
    query = st.text_input("Enter your query:")
    
    # Show retrieval method being used
    retrieval_method = "📚 Using: "
    
    # Start with index type
    if use_contextual_index:
        retrieval_method += "Contextual Index with "
    else:
        retrieval_method += "Standard Index with "
        
    # Add retrieval method
    if use_hybrid_retrieval and use_reranking:
        retrieval_method += "Hybrid Retrieval (BM25 + Contextual Embeddings) with LLM Reranking"
    elif use_hybrid_retrieval:
        retrieval_method += "Hybrid Retrieval (BM25 + Contextual Embeddings)"
    elif use_reranking:
        retrieval_method += "Vector Search with LLM Reranking"
    else:
        retrieval_method += "Basic Vector Search"
    
    st.write(retrieval_method)
    
    # Create a button to submit the query
    if st.button("Submit"):
        if query.strip():
            with st.spinner("Processing..."):
                # Process the query
                clarified_query, context, answer = process_query(
                    query,
                    top_k=top_k,
                    include_source_explanations=include_source_explanations,
                    reasoning_effort=reasoning_effort
                )
                
                # Display results
                st.subheader("Clarified Query")
                st.write(clarified_query)
                
                st.subheader("Retrieved Documents")
                st.expander("View Document Context", expanded=False).text(context)
                
                st.subheader("Final Answer")
                # Use markdown for proper formatting
                st.markdown(answer)
            
            st.success("Done!")
        else:
            st.error("Please enter a valid query.")

if __name__ == "__main__":
    main()