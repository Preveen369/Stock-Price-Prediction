"""LM Studio configuration — loads environment variables and exposes
LM_STUDIO_CONFIG and RECOMMENDED_MODELS.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# Read env vars with sensible defaults and type coercion
LM_STUDIO_BASE_V1 = os.getenv("LM_STUDIO_V1_URL", "http://localhost:1234/v1")
LM_STUDIO_BASE_ROOT = os.getenv("LM_STUDIO_ROOT_URL", "http://localhost:1234")

LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-3-4b")
LM_STUDIO_EMBEDDING_MODEL = os.getenv("LM_STUDIO_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")

LM_STUDIO_MAX_TOKENS = int(os.getenv("LM_STUDIO_MAX_TOKENS", "2048"))
LM_STUDIO_TEMPERATURE = float(os.getenv("LM_STUDIO_TEMPERATURE", "0.3"))
LM_STUDIO_TIMEOUT = int(os.getenv("LM_STUDIO_TIMEOUT", "30"))


# LM Studio Configuration exported for use across services
LM_STUDIO_CONFIG = {
    # Preferred base URL for API calls that expect the v1 prefix
    "base_url": LM_STUDIO_BASE_V1,
    # Root URL without the /v1 suffix (useful for services that append /v1
    # themselves or call other endpoints)
    "base_root": LM_STUDIO_BASE_ROOT,
    "api_key": LM_STUDIO_API_KEY,
    "model_name": LM_STUDIO_MODEL,
    "embedding_model": LM_STUDIO_EMBEDDING_MODEL,
    "max_tokens": LM_STUDIO_MAX_TOKENS,
    "temperature": LM_STUDIO_TEMPERATURE,
    "timeout": LM_STUDIO_TIMEOUT,
}


# Recommended models list (kept for backwards compatibility / docs)
RECOMMENDED_MODELS = [LM_STUDIO_MODEL]