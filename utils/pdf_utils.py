"""
PDF Utilities - Document Loading and Text Processing
Handles PDF loading, text splitting, and chunk creation for RAG pipeline
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LangChainDocument

# PDF support check
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def is_pdf_available() -> bool:
    """
    Check if PDF processing support is available
    
    Returns:
        bool: True if PyPDF2 is installed, False otherwise
    """
    return PDF_AVAILABLE


def get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """
    Get LangChain text splitter for optimal chunking
    Uses RecursiveCharacterTextSplitter for smart splitting at natural boundaries
    
    Args:
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        RecursiveCharacterTextSplitter configured for document processing
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def load_pdf_with_langchain(filepath: str) -> List[LangChainDocument]:
    """
    Load PDF using LangChain's PyPDFLoader
    
    Creates one Document object per page with metadata including:
    - Page number
    - Source file path
    
    Args:
        filepath: Absolute or relative path to the PDF file
        
    Returns:
        List[LangChainDocument]: List of Document objects (one per page)
        
    Raises:
        ImportError: If PyPDF2 is not installed
        Exception: If PDF loading or parsing fails
    """
    if not PDF_AVAILABLE:
        raise ImportError("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")
    
    try:
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        return documents
    except Exception as e:
        raise Exception(f"Error loading PDF with LangChain: {e}")


def split_documents(documents: List[LangChainDocument], 
                    chunk_size: int = 1000, 
                    chunk_overlap: int = 200) -> List[LangChainDocument]:
    """
    Split documents into smaller chunks for embedding and retrieval
    
    Uses intelligent splitting at natural text boundaries (paragraphs,
    sentences, etc.) while maintaining context through overlap.
    
    Args:
        documents: List of LangChain Document objects to split
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List[LangChainDocument]: List of chunked Document objects with preserved metadata
    """
    text_splitter = get_text_splitter(chunk_size, chunk_overlap)
    return text_splitter.split_documents(documents)
