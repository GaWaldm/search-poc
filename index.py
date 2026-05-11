from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ingest import fetch_articles, extract_article_data
from config import CHROMA_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def build_index(limit=20):
    """Artikel abrufen, chunken und in Chroma indexieren"""
    
    # 1. Embedding-Modell laden
    print(f"Lade Embedding-Modell: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("Modell geladen ✓")
    
    # 2. Artikel abrufen
    articles = fetch_articles(limit=limit)
    
    # 3. Text splitten
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    all_chunks = []
    all_metadata = []
    
    for article in articles:
        data = extract_article_data(article)
        
        if not data["text"].strip():
            continue
        
        # Text in Chunks aufteilen
        chunks = splitter.split_text(data["text"])
        
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "document_id": str(data["document_id"]),
                "title": data["title"],
                "language": data["language"],
                "slug": data["slug"],
                "content_type": data["content_type"],
                "last_updated": data["last_updated"]
            })
    
    print(f"\n{len(all_chunks)} Chunks aus {len(articles)} Artikeln erstellt")
    
    # 4. In Chroma speichern
    print(f"Speichere in Chroma unter: {CHROMA_PATH}")
    
    # Bestehenden Index laden oder neu erstellen
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Bestehende Livingdocs-Chunks entfernen (verhindert Duplikate)
    existing = db.get(where={"content_type": {"$nin": ["pdf"]}})
    if existing and existing["ids"]:
        print(f"Entferne {len(existing['ids'])} bestehende Livingdocs-Chunks...")
    db.delete(ids=existing["ids"])

    # Neue Chunks hinzufügen
    db.add_texts(
        texts=all_chunks,
        metadatas=all_metadata
    )
    
    print(f"Index erfolgreich erstellt ✓")
    print(f"Gesamt: {db._collection.count()} Chunks indexiert")

if __name__ == "__main__":
    build_index(limit=20)