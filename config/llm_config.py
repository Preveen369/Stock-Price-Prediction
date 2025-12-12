import os
from dotenv import load_dotenv

load_dotenv()

# LM Studio Configuration
LM_STUDIO_CONFIG = {
    "base_url": os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1"),  # Allow env override
    "api_key": "lm-studio",  # LM Studio doesn't require real API key
    "model_name": "google/gemma-3-4b",  # Will be overridden by actual model
    "max_tokens": int(os.getenv("LM_STUDIO_MAX_TOKENS", "2048")),
    "temperature": float(os.getenv("LM_STUDIO_TEMPERATURE", "0.3")),
    "timeout": int(os.getenv("LM_STUDIO_TIMEOUT", "30"))
}

# Recommended model for financial analysis
RECOMMENDED_MODELS = [
    "google/gemma-3-4b"
]