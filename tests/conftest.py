"""Test settings: the required settings have no defaults in code, so tests set
them before session_memory is imported. Nothing here is a real host."""

import os

os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:1")
os.environ.setdefault("SESSION_MEMORY_MODEL", "test-model")
