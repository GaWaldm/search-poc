from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langdetect import detect
from config import CHROMA_PATH, EMBEDDING_MODEL

def detect_language(text):
    """Sprache der Anfrage erkennen"""
    try:
        lang = detect(text)
        print(f"Erkannte Sprache: {lang}")
        return lang
    except:
        return None

def search(query, k=5):
    """Semantische Suche mit Sprachpriorisierung"""
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Sprache der Anfrage erkennen
    language = detect_language(query)
    
    print(f"\nSuchanfrage: '{query}'")
    
    # Erst in erkannter Sprache suchen
    results = []
    if language:
        results = db.similarity_search(
            query,
            k=k,
            filter={"language": language}
        )
        print(f"{len(results)} Treffer in Sprache '{language}' gefunden")
    
    # Falls weniger als 2 Treffer — ohne Sprachfilter ergänzen
    if len(results) < 2:
        print("Zu wenig Treffer in dieser Sprache — ergänze andere Sprachen...")
        extra_results = db.similarity_search(query, k=k)
        
        # Duplikate vermeiden
        existing_texts = {r.page_content for r in results}
        for r in extra_results:
            if r.page_content not in existing_texts:
                results.append(r)
                if len(results) >= k:
                    break
    
    print(f"\nTotal {len(results)} Treffer\n")
    
    for i, result in enumerate(results):
        lang = result.metadata.get('language')
        flag = "🇩🇪" if lang == "de" else "🇫🇷" if lang == "fr" else "🇮🇹" if lang == "it" else "🌐"
        print(f"--- Treffer {i+1} {flag} ---")
        print(f"Titel:    {result.metadata.get('title')}")
        print(f"Sprache:  {lang}")
        print(f"Typ:      {result.metadata.get('content_type')}")
        print(f"Text:     {result.page_content[:200]}")
        print()

if __name__ == "__main__":
    # Teste drei Sprachen
    search("saisonale Depression")
    print("=" * 60)
    search("dépression saisonnière")
    print("=" * 60)
    search("depressione stagionale")