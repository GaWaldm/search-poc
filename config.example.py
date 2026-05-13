import os

# HCMS API
HCMS_API_URL = "https://your-hcms-instance.example.com/api/latest/publications"
HCMS_BASE_URL = "https://your-hcms-instance.example.com"

# Öffentliche Webseite
SITE_BASE_URL = "https://your-public-site.example.com"

try:
    import streamlit as st
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
    HCMS_API_TOKEN = st.secrets["HCMS_API_TOKEN"]
except:
    ANTHROPIC_API_KEY = "sk-ant-..."  # Anthropic API Key
    HCMS_API_TOKEN = "eyJ..."         # HCMS API Token

CLAUDE_MODEL = "claude-sonnet-4-5"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CHROMA_PATH = "data/chroma"
PDF_PATH = "data/pdfs"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5