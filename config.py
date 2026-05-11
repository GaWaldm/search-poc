import os
try:
    import streamlit as st
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
    LIVINGDOCS_API_TOKEN = st.secrets["LIVINGDOCS_API_TOKEN"]
except:
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    LIVINGDOCS_API_TOKEN = os.environ.get("LIVINGDOCS_API_TOKEN", "")

CLAUDE_MODEL = "claude-sonnet-4-5"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CHROMA_PATH = "data/chroma"
PDF_PATH = "data/pdfs"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5
