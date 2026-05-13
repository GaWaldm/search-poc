from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import CHROMA_PATH, EMBEDDING_MODEL

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

# Alle PDF-Chunks holen
data = db.get(where={"content_type": "pdf"})

# Lokale PDFs haben keine URL als document_id
local_ids = [
    id for id, meta in zip(data["ids"], data["metadatas"]) 
    if not meta.get("document_id", "").startswith("http")
]

print(f"Lokale PDF-Chunks gefunden: {len(local_ids)}")

if local_ids:
    db.delete(ids=local_ids)
    print(f"✓ {len(local_ids)} lokale PDF-Chunks gelöscht")
    print(f"✓ Gesamt im Index: {db._collection.count()} Chunks")