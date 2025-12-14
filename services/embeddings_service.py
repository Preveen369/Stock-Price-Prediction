"""
Embeddings Service - LM Studio Embeddings Integration
Handles text embedding generation for vector storage and retrieval
"""

import streamlit as st
import requests
from typing import List
from langchain_core.embeddings import Embeddings


class LMStudioEmbeddings(Embeddings):
    """
    Custom embeddings class for LM Studio that properly handles the API format
    Inherits from LangChain's Embeddings base class for full compatibility
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:1234"):
        """
        Initialize LM Studio embeddings client
        
        Args:
            base_url: Base URL for LM Studio server
        """
        self.base_url = base_url
        self.embeddings_endpoint = f"{base_url}/v1/embeddings"
        self.model_name = "text-embedding-nomic-embed-text-v1.5"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using LM Studio's embedding API
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List[List[float]]: List of embedding vectors (384-dimensional)
        """
        embeddings = []
        total = len(texts)
        
        # Create progress bar
        progress_bar = st.progress(0, text=f"Generating embeddings: 0/{total}")
        
        # Process each text individually (LM Studio expects single strings)
        for idx, text in enumerate(texts):
            payload = {
                "model": self.model_name,
                "input": text  # Single string, not array
            }
            
            try:
                response = requests.post(self.embeddings_endpoint, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                embedding = data['data'][0]['embedding']
                embeddings.append(embedding)
            except Exception as e:
                st.warning(f"Error generating embedding {idx+1}/{total}: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * 384)
            
            # Update progress
            progress = (idx + 1) / total
            progress_bar.progress(progress, text=f"Generating embeddings: {idx+1}/{total}")
        
        # Clear progress bar
        progress_bar.empty()
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text for similarity search
        
        Args:
            text: Query text to embed
            
        Returns:
            List[float]: Embedding vector (384-dimensional)
        """
        payload = {
            "model": self.model_name,
            "input": text
        }
        
        try:
            response = requests.post(self.embeddings_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data['data'][0]['embedding']
        except Exception as e:
            st.error(f"Error generating query embedding: {e}")
            return [0.0] * 384


def get_embeddings_model(base_url: str = "http://127.0.0.1:1234") -> LMStudioEmbeddings:
    """
    Factory function to create LM Studio embeddings model instance
    
    Args:
        base_url: Base URL for LM Studio server
        
    Returns:
        LMStudioEmbeddings: Configured embeddings model for LangChain integration
    """
    return LMStudioEmbeddings(base_url=base_url)
