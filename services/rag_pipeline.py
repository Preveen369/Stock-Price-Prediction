"""
RAG Pipeline - Retrieval-Augmented Generation
Handles document retrieval and question answering using vector stores and LLM
"""

import streamlit as st
from typing import Dict
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from services.local_llm_service import LocalLLMService


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline using FAISS and LLM
    
    Implements a complete RAG workflow:
    1. Query embedding generation
    2. Similarity search in vector store
    3. Context retrieval and formatting
    4. LLM-based answer generation
    5. Source citation
    """
    
    def __init__(self, 
                 llm_service: LocalLLMService,
                 vector_store: FAISS,
                 top_k: int = 3):
        """
        Initialize RAG pipeline with LLM and vector store
        
        Args:
            llm_service: Configured LocalLLMService instance
            vector_store: FAISS vector store containing document embeddings
            top_k: Number of top relevant documents to retrieve
        """
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.top_k = top_k
        
        # Create custom prompt template for financial analysis
        self.prompt_template = PromptTemplate(
            template="""You are a financial analyst AI assistant. Use the following context from the financial document to answer the question accurately and concisely.

Context: {context}

Question: {question}

Provide a detailed answer based on the context. If the context doesn't contain enough information, clearly state that.

Answer:""",
            input_variables=["context", "question"]
        )
        
    def query(self, question: str, return_sources: bool = True) -> Dict:
        """
        Query the RAG pipeline and return answer with source documents
        
        Args:
            question: User's question about the document
            return_sources: Whether to include source documents in response
            
        Returns:
            Dict: Dictionary containing 'answer' and 'sources' keys
                  - answer: AI-generated response text
                  - sources: List of source document snippets with metadata
        """
        try:
            # Retrieve relevant documents using FAISS similarity search
            # Note: similarity_search_with_score handles embedding internally
            docs_with_scores = self.vector_store.similarity_search_with_score(question, k=self.top_k)
            
            if not docs_with_scores:
                return {
                    'answer': "No relevant context found to answer the question.",
                    'sources': []
                }
            
            # Build context from retrieved documents
            context = "\n\n".join([
                f"[Source {i+1} - Relevance: {1-score:.2%}]\n{doc.page_content}"
                for i, (doc, score) in enumerate(docs_with_scores)
            ])
            
            # Create prompt
            prompt = self.prompt_template.format(context=context, question=question)
            
            # Generate answer using LLM
            if hasattr(self.llm_service, 'client') and self.llm_service.client:
                response = self.llm_service.client.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
            else:
                answer = "LLM service not available. Please ensure LM Studio is running."
            
            result = {
                'answer': answer,
                'sources': []
            }
            
            if return_sources:
                result['sources'] = [
                    {
                        'content': doc.page_content[:200] + "...",
                        'metadata': doc.metadata,
                        'relevance_score': 1 - score  # Convert distance to similarity
                    }
                    for doc, score in docs_with_scores
                ]
            
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.error(f"Query error details: {error_details}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': []
            }
