import os
import html
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHROMA_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, PDF_PATH

def detect_language_from_filename(filename):
    """Sprache aus Dateiname erraten"""
    filename_lower = filename.lower()
    if "_de" in filename_lower or "de_" in filename_lower or filename_lower.startswith("de "):
        return "de"
    elif "_fr" in filename_lower or "fr_" in filename_lower or filename_lower.startswith("fr "):
        return "fr"
    elif "_it" in filename_lower or "it_" in filename_lower or filename_lower.startswith("it "):
        return "it"
    return "de"  # Fallback

def extract_text_from_pdf(filepath):
    """Text aus PDF extrahieren"""
    try:
        reader = PdfReader(filepath)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 20:
                texts.append(text.strip())
        return "\n".join(texts)
    except Exception as e:
        print(f"  Fehler beim Lesen: {e}")
        return ""

def index_pdfs():
    """Alle PDFs im Ordner indexieren"""

    print(f"Lade Embedding-Modell: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Bestehenden Index laden
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    pdf_files = [f for f in os.listdir(PDF_PATH) if f.endswith(".pdf")]
    print(f"\n{len(pdf_files)} PDFs gefunden\n")

    total_chunks = 0

    for filename in pdf_files:
        filepath = os.path.join(PDF_PATH, filename)
        print(f"Verarbeite: {filename}")

        # Text extrahieren
        text = extract_text_from_pdf(filepath)

        if not text:
            print(f"  → Kein Text gefunden, übersprungen")
            continue

        print(f"  → {len(text)} Zeichen extrahiert")

        # Sprache erkennen
        language = detect_language_from_filename(filename)
        print(f"  → Sprache: {language}")

        # Chunken
        chunks = splitter.split_text(text)
        print(f"  → {len(chunks)} Chunks erstellt")

        # Metadaten
        metadatas = [{
            "document_id": filename,
            "title": filename.replace(".pdf", "").replace("_", " "),
            "language": language,
            "slug": filename,
            "content_type": "pdf",
            "last_updated": "",
            "source": filepath
        } for _ in chunks]

        # In bestehenden Index hinzufügen
        # Bestehende PDF-Chunks für diese Datei entfernen
        existing = db.get(where={"document_id": filename})
        if existing and existing["ids"]:
            print(f"  → Entferne {len(existing['ids'])} bestehende Chunks für {filename}")
            db.delete(ids=existing["ids"])

        # Neue Chunks hinzufügen
        db.add_texts(texts=chunks, metadatas=metadatas)
        total_chunks += len(chunks)

    print(f"\n✓ {total_chunks} Chunks aus {len(pdf_files)} PDFs indexiert")
    print(f"✓ Gesamt im Index: {db._collection.count()} Chunks")

if __name__ == "__main__":
    index_pdfs()