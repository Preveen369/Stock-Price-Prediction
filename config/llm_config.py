"""
LLM Configuration - LM Studio Settings
Centralized configuration for local LLM service integration.

This module reads LM Studio related environment variables from the project's
`.env` (via python-dotenv) and provides a normalized `LM_STUDIO_CONFIG` dict
that is safe to use across services in the codebase.

Notes:
- `LM_STUDIO_URL` can be set with or without the trailing `/v1` suffix. The
  configuration exposes both a normalized v1 endpoint (`base_url`) and a
  root endpoint without `/v1` (`base_root`) so services that expect either
  format can use the correct value.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# Helper: normalize URL and produce both forms (root and /v1)
def _normalize_lm_studio_url(raw_url: str) -> tuple[str, str]:
    url = (raw_url or "").strip().rstrip("/")
    # If user provided a url that already includes /v1, keep it but also
    # expose the root form without the suffix.
    if url.endswith("/v1"):
        base_root = url[:-3].rstrip("/")
        base_v1 = base_root + "/v1"
    else:
        base_root = url
        base_v1 = base_root + "/v1"
    return base_root, base_v1


# Read env vars with sensible defaults and type coercion
RAW_LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-3-4b")
LM_STUDIO_EMBEDDING_MODEL = os.getenv("LM_STUDIO_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")

LM_STUDIO_MAX_TOKENS = int(os.getenv("LM_STUDIO_MAX_TOKENS", "2048"))
LM_STUDIO_TEMPERATURE = float(os.getenv("LM_STUDIO_TEMPERATURE", "0.3"))
LM_STUDIO_TIMEOUT = int(os.getenv("LM_STUDIO_TIMEOUT", "30"))


# Normalized base URLs
LM_STUDIO_BASE_ROOT, LM_STUDIO_BASE_V1 = _normalize_lm_studio_url(RAW_LM_STUDIO_URL)


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