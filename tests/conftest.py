"""
Pytest configuration for MediTrace test suite.
Ensures existing unit tests have deterministic mock configuration available by default,
while allowing live workflow integration tests to test both live and mock engine modes explicitly.
"""

import os
import pytest
from backend.config import settings

# By default for legacy unit tests, enable mock mode unless explicitly set
if "MEDITRACE_MOCK_MODE" not in os.environ:
    os.environ["MEDITRACE_MOCK_MODE"] = "true"
    settings.mock_mode = True
