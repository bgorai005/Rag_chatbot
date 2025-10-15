# File: config.py
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Persistent docs directory
DOCS_DIR = os.getenv("DOCS_DIR", "./documents")
os.makedirs(DOCS_DIR, exist_ok=True)