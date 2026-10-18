"""
MediTrace - Live Gemini Embedding API Smoke Test
Verifies live connectivity, model configuration, and vector dimensionality
using the Google Gemini API.

CLINICAL SAFETY NOTICE:
-----------------------
This script runs a live smoke test against the Gemini embedding API using a
synthetic reference string. It does not process real patient records.
"""

import math
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Attempt to load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except ImportError:
    env_file = os.path.join(REPO_ROOT, ".env")
    if os.path.isfile(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v

from app.services.embedding_service import (
    EmbeddingAPIError,
    EmbeddingConfigurationError,
    EmbeddingService,
    EmbeddingValidationError,
)

SYNTHETIC_QUERY = "Synthetic clinical guideline retrieval test."
EXPECTED_DIMENSION = 768


def run_live_smoke_test() -> int:
    """
    Execute live API smoke test against the configured Gemini embedding model.
    Prints only model, dimension, and sanitized success/failure status.
    """
    # 1. Initialize service
    try:
        service = EmbeddingService()
    except EmbeddingConfigurationError:
        print("Status: FAILED")
        print("Category: Configuration Error (Missing or Invalid GEMINI_API_KEY)")
        return 1
    except Exception as exc:
        print("Status: FAILED")
        print(f"Category: Initialization Error ({type(exc).__name__})")
        return 1

    configured_model = service.model
    print(f"Configured Model: {configured_model}")

    # 2. Execute live embedding request
    try:
        embedding = service.embed_query(SYNTHETIC_QUERY)
    except EmbeddingAPIError as exc:
        print("Status: FAILED")

        message = str(exc).lower()

        if "401" in message or "authentication" in message:
            category = "Authentication Error"
        elif "404" in message or "model not found" in message:
            category = "Model Availability Error"
        elif "429" in message or "rate limit" in message:
            category = "Quota or Rate Limit Error"
        elif "403" in message or "permission" in message:
            category = "Permission Error"
        else:
            category = "API Communication Error"

        print(f"Category: {category}")
        return 1
    except EmbeddingValidationError:
        print("Status: FAILED")
        print("Category: Response Validation Error")
        return 1
    except Exception as exc:
        print("Status: FAILED")
        print(f"Category: Unexpected Error ({type(exc).__name__})")
        return 1

    # 3. Verify returned vector characteristics
    if not isinstance(embedding, list) or len(embedding) == 0:
        print("Status: FAILED")
        print("Category: Empty Embedding Returned")
        return 1

    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in embedding):
        print("Status: FAILED")
        print("Category: Non-finite Vector Values Detected")
        return 1

    actual_dim = len(embedding)
    print(f"Returned Vector Dimension: {actual_dim}")

    if actual_dim != EXPECTED_DIMENSION:
        print("Status: FAILED")
        print(f"Category: Dimensionality Mismatch (Expected {EXPECTED_DIMENSION}, Got {actual_dim})")
        return 1

    print("Status: SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(run_live_smoke_test())
