import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pypdf import PdfReader
import io
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from config import CHROMA_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def detect_language(text):
    """Sprache aus Text erkennen"""
    try:
        return detect(text[:500])
    except:
        return "de"

def detect_language_from_url(url):
    """Sprache aus URL-Pfad erkennen"""
    if "/fr/" in url:
        return "fr"
    elif "/it/" in url:
        return "it"
    elif "/de/" in url:
        return "de"
    return "de"

def fetch_pdf_from_url(pdf_url):
    """PDF direkt von URL laden und Text extrahieren"""
    try:
        print(f"  → PDF herunterladen: {pdf_url[:80]}...")
        response = requests.get(pdf_url, timeout=30)
        if response.status_code != 200:
            print(f"  → Fehler: {response.status_code}")
            return ""
        
        # PDF im Speicher lesen — nicht auf Disk speichern
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 20:
                texts.append(text.strip())
        
        return "\n".join(texts)
    except Exception as e:
        print(f"  → Fehler beim PDF-Lesen: {e}")
        return ""

def crawl_page(url):
    """Eine Webseite crawlen und PDF-Links finden"""
    try:
        print(f"\nCrawle: {url}")
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"  → Fehler: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Alle PDF-Links finden
        pdf_links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # Relativen Link zu absolutem machen
            full_url = urljoin(url, href)
            if full_url.lower().endswith(".pdf"):
                title = a_tag.get_text(strip=True) or a_tag.get("aria-label", "") or full_url
                pdf_links.append({
                    "url": full_url,
                    "title": title,
                    "source_page": url
                })
        
        print(f"  → {len(pdf_links)} PDF-Links gefunden")
        return pdf_links
    
    except Exception as e:
        print(f"  → Fehler beim Crawlen: {e}")
        return []

def index_pdfs_from_web(urls):
    """PDFs von Webseiten crawlen und indexieren"""
    
    print(f"Lade Embedding-Modell: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    total_chunks = 0
    total_pdfs = 0
    
    for url in urls:
        # PDF-Links auf der Seite finden
        pdf_links = crawl_page(url)
        
        for pdf_info in pdf_links:
            pdf_url = pdf_info["url"]
            title = pdf_info["title"]
            
            # Text extrahieren
            text = fetch_pdf_from_url(pdf_url)
            if not text:
                continue
            
            print(f"  → {len(text)} Zeichen extrahiert")
            
            # Sprache erkennen
            language = detect_language_from_url(pdf_url)
            print(f"  → Sprache: {language}")
            
            # Chunken
            chunks = splitter.split_text(text)
            print(f"  → {len(chunks)} Chunks erstellt")
            
            # Bestehende Chunks für diese PDF entfernen
            existing = db.get(where={"document_id": pdf_url})
            if existing and existing["ids"]:
                print(f"  → Entferne {len(existing['ids'])} bestehende Chunks")
                db.delete(ids=existing["ids"])
            
            # Metadaten
            metadatas = [{
                "document_id": pdf_url,
                "title": title,
                "language": language,
                "slug": pdf_url,
                "content_type": "pdf",
                "last_updated": "",
                "source": pdf_info["source_page"]
            } for _ in chunks]
            
            # Indexieren
            db.add_texts(texts=chunks, metadatas=metadatas)
            total_chunks += len(chunks)
            total_pdfs += 1
    
    print(f"\n✓ {total_chunks} Chunks aus {total_pdfs} PDFs indexiert")
    print(f"✓ Gesamt im Index: {db._collection.count()} Chunks")

# Test
if __name__ == "__main__":
    # Seiten die gecrawlt werden sollen
    URLS = [
        "https://prod-sandbox01-ssg.scs.scs-sdweb.ch/de/der-bund-kurz-erklaert",
        "https://prod-sandbox01-ssg.scs.scs-sdweb.ch/fr/french-translation-of-der-bund-kurz-erklart",
        "https://prod-sandbox01-ssg.scs.scs-sdweb.ch/it/italian-translation-of-la-confederation-en-bref",
    ]
    
    index_pdfs_from_web(URLS)