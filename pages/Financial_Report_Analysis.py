"""
Financial Report Analysis - RAG-Based PDF Analysis
Integrated with Stockify AI Stock Predictor

This page provides:
- PDF document upload and processing
- Intelligent Q&A using RAG (Retrieval-Augmented Generation)
- Vector embeddings and semantic search
- Financial report analysis with LM Studio
"""

import streamlit as st
import os
import sys
import tempfile

# LangChain imports
from langchain_community.vectorstores import FAISS

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modular components
from services.embeddings_service import get_embeddings_model
from services.rag_pipeline import RAGPipeline
from utils.pdf_utils import load_pdf_with_langchain, split_documents
from utils.stock_utils import init_local_llm, display_llm_sidebar_status

def initialize_session_state():
    """
    Initialize Streamlit session state variables for RAG pipeline
    
    Creates session state entries for:
    - Vector store (FAISS)
    - RAG pipeline instance
    - Embeddings model
    - Chat history
    - Uploaded file tracking
    - Document chunk count
    - LLM service instance
    """
    if 'rag_vector_store' not in st.session_state:
        st.session_state.rag_vector_store = None
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'rag_embeddings' not in st.session_state:
        st.session_state.rag_embeddings = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'total_chunks' not in st.session_state:
        st.session_state.total_chunks = 0


def process_pdf_document(uploaded_file, tmp_path: str):
    """
    Process uploaded PDF document and create vector store
    
    Args:
        uploaded_file: Streamlit uploaded file object
        tmp_path: Path to temporary PDF file
        
    Returns:
        tuple: (vector_store, chunks_count) or (None, 0) on error
    """
    try:
        # Load PDF using LangChain
        st.info("📄 Loading PDF with LangChain...")
        documents = load_pdf_with_langchain(tmp_path)
        st.info(f"✅ Loaded {len(documents)} pages")
        
        # Split documents into chunks
        st.info("✂️ Splitting into chunks...")
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
        st.info(f"📊 Created {len(chunks)} text chunks")
        
        # Create embeddings model
        st.info("🔮 Initializing embeddings model...")
        embeddings = get_embeddings_model()
        
        # Create FAISS vector store
        st.info("💾 Building FAISS vector store...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        return vector_store, len(chunks), embeddings
        
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, 0, None



# Main Streamlit Application
def main():
    """
    Main Streamlit application for Financial Report Analysis
    
    Workflow:
    1. Upload PDF financial document
    2. Process and chunk the document
    3. Generate embeddings and create vector store
    4. Initialize RAG pipeline
    5. Ask questions and get AI-powered answers with citations
    """
    
    # Page configuration
    st.set_page_config(
        page_title="Financial Report Analysis - Stockify", 
        page_icon="📄",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize LLM service
    llm_service = init_local_llm()
    
    # Title and description
    st.title('📄 Financial Report Analysis')
    st.markdown("*AI-powered document analysis using RAG (Retrieval-Augmented Generation)*")
    
    # Sidebar
    st.sidebar.markdown("### ℹ️ **About:**")
    st.sidebar.info("""
This page provides AI-enhanced financial report analysis using:
- **RAG Pipeline** for intelligent Q&A
- **Vector Embeddings** for semantic search
- **Local LLM** for context-aware answers
- **PDF Processing** for document analysis
""")
    
    # Display LLM status
    display_llm_sidebar_status(llm_service)
    
    st.sidebar.markdown("### 🔍 **Features:**")
    st.sidebar.info("""
- **Upload PDF Reports** (10-K, 10-Q, Annual Reports)
- **Ask Questions** about financial data
- **Get AI Insights** with source citations
- **Vector Search** for relevant information
""")
    
    # Check LLM connection first
    if llm_service.check_connection():
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📤 Upload Financial Document")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload a PDF financial report (10-K, 10-Q, Annual Report, etc.)",
                type=['pdf'],
                help="Upload a financial document to analyze with AI"
            )
            
            # Reset session state when file is removed
            if uploaded_file is None:
                if st.session_state.uploaded_file_name is not None:
                    st.session_state.uploaded_file_name = None
                    st.session_state.rag_vector_store = None
                    st.session_state.rag_pipeline = None
                    st.session_state.chat_history = []
                    st.session_state.total_chunks = 0
            
            if uploaded_file is not None:
                # Check if this is a new file
                if st.session_state.uploaded_file_name != uploaded_file.name:
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.rag_vector_store = None
                    st.session_state.rag_pipeline = None
                    st.session_state.chat_history = []
                
                # Display file info
                st.success(f"✅ File uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.2f} KB)")
                
                # Process PDF button
                if st.session_state.rag_vector_store is None:
                    if st.button("🔄 Process Document", type="primary"):
                        # Container for processing steps
                        processing_container = st.container()
                        
                        with st.spinner("Processing PDF with LangChain and FAISS..."):
                            with processing_container:
                                with st.expander("📋 Processing Steps", expanded=True):
                                    # Save uploaded file temporarily
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                        tmp_file.write(uploaded_file.getvalue())
                                        tmp_path = tmp_file.name
                                    
                                    # Process the PDF
                                    vector_store, chunks_count, embeddings = process_pdf_document(uploaded_file, tmp_path)
                                    
                                    if vector_store is not None:
                                        st.session_state.rag_vector_store = vector_store
                                        st.session_state.rag_embeddings = embeddings
                                        st.session_state.total_chunks = chunks_count
                                        
                                        # Initialize RAG pipeline
                                        st.info("⚙️ Initializing RAG pipeline...")
                                        st.session_state.rag_pipeline = RAGPipeline(
                                            llm_service=llm_service,
                                            vector_store=vector_store,
                                            top_k=4
                                        )
                                        
                                        # Clean up temp file
                                        os.unlink(tmp_path)
                                        
                                        st.success("✅ Document processed successfully with FAISS! You can now ask questions.")
                                    else:
                                        # Clean up temp file on error
                                        os.unlink(tmp_path)
                else:
                    st.success("✅ Document processed and ready for questions")
                    with st.expander("📋 View Processing Details", expanded=False):
                        st.info(f"📄 Document: {st.session_state.uploaded_file_name}")
                        st.info(f"📊 Total chunks created: {st.session_state.total_chunks}")
                        st.info(f"🔍 Vector store: FAISS with LM Studio embeddings")
                        st.info(f"💬 Questions asked: {len(st.session_state.chat_history)}")
                    
                    if st.button("🔄 Process Different Document"):
                        st.session_state.rag_vector_store = None
                        st.session_state.rag_pipeline = None
                        st.session_state.chat_history = []
                        st.rerun()
        
        with col2:
            st.header("📊 Docs Stats")
            if st.session_state.rag_vector_store:
                col3, col4 = st.columns([1, 1])
                with col3:
                    st.metric("Total Chunks", st.session_state.get('total_chunks', 0))
                with col4:
                    st.metric("Questions Asked", len(st.session_state.chat_history))
            else:
                st.info("Upload and process a document to see stats")
        
        # Chat Interface
        if st.session_state.rag_pipeline is not None:
            st.markdown("---")
            st.header("💬 Ask Questions About the Document")
            st.success("✅ AI Question Answering Available")
            
            # Display chat history
            for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
                with st.expander(f"Q{i+1}: {question[:80]}...", expanded=(i == len(st.session_state.chat_history) - 1)):
                    st.markdown(f"**Question:** {question}")
                    st.markdown(f"**Answer:** {answer}")
                    
                    if sources:
                        with st.expander(f"📚 View Sources ({len(sources)})", expanded=False):
                            for j, source in enumerate(sources, 1):
                                st.caption(f"{j}. Relevance: {source['relevance_score']:.2%}")
                                st.text(source['content'][:200] + "...")
                                if 'page' in source['metadata']:
                                    st.caption(f"Page: {source['metadata']['page']}")
                                st.divider()
            
            # Question input
            question = st.text_input(
                "Ask a question about the financial report:",
                placeholder="e.g., What was the total revenue in 2024?",
                key="question_input"
            )
            
            col_ask, col_clear = st.columns([1, 4])
            with col_ask:
                ask_button = st.button("🔍 Ask", type="primary")
            with col_clear:
                if st.button("🗑️ Clear History"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            if ask_button and question:
                with st.spinner("Searching and generating answer..."):
                    try:
                        result = st.session_state.rag_pipeline.query(question, return_sources=True)
                        st.session_state.chat_history.append((question, result['answer'], result['sources']))
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Upload and process a PDF document to start asking questions")
    else:
        st.markdown("---")
        st.warning("🤖 AI Document Analysis Unavailable")
        st.info("💡 To enable AI-powered document analysis:")
        st.markdown("""
        1. Install and start **LM Studio** from [lmstudio.ai](https://lmstudio.ai)
        2. Download a financial analysis model (recommended: Gemma or Llama)
        3. Start the local server on `localhost:1234`
        4. Refresh this page to enable AI features
        """)


if __name__ == "__main__":
    main()
