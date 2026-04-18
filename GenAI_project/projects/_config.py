"""Shared config loader used by every script in this pack."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from this folder, if present
load_dotenv(Path(__file__).resolve().parent / ".env")

PROJECT_ID = os.environ.get("PROJECT_ID", "genai-project-493620")
LOCATION = os.environ.get("LOCATION", "us-east4")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_PRO_MODEL = os.environ.get("GEMINI_PRO_MODEL", "gemini-2.0-pro-exp")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-004")
STAGING_BUCKET = os.environ.get("STAGING_BUCKET", "")
BQ_DATASET = os.environ.get("BQ_DATASET", "ai_engineer_labs")
VECTOR_INDEX_ID = os.environ.get("VECTOR_INDEX_ID", "")
VECTOR_INDEX_ENDPOINT_ID = os.environ.get("VECTOR_INDEX_ENDPOINT_ID", "")


def init_vertex():
    """Initialize the Vertex AI SDK once. Called by every lab script."""
    import vertexai
    vertexai.init(project=PROJECT_ID, location=LOCATION)
