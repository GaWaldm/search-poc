import requests
import json
import html
from config import HCMS_API_URL, HCMS_API_TOKEN, SITE_BASE_URL

BASE_URL = SITE_BASE_URL
HEADERS = {"Authorization": f"Bearer {HCMS_API_TOKEN}"}

# Komponenten die Referenzen enthalten
REFERENCE_COMPONENTS = {
    "faq-teaser",
    "howto-teaser",
    "reusable-content-teaser",
    "dynamic-faq-teaser"
}

# Komponenten ohne nutzbaren Text
SKIP_COMPONENTS = {"image"}

def fetch_articles(limit=10):
    """Artikel von HCMS API abrufen"""
    print(f"Abrufen von {limit} Artikeln...")
    response = requests.get(HCMS_API_URL, headers=HEADERS, params={"limit": limit})
    
    if response.status_code != 200:
        print(f"Fehler: {response.status_code}")
        return []
    
    data = response.json()
    print(f"{len(data)} Artikel erhalten")
    return data

def fetch_referenced_document(doc_id):
    """Ein referenziertes Dokument per ID abrufen"""
    url = f"{BASE_URL}/api/v1/documents/{doc_id}/latestPublication"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        return None
    return response.json()

def extract_text_from_components(components, resolve_references=True):
    """Text rekursiv aus Komponenten und Containern extrahieren"""
    texts = []
    
    for component in components:
        component_name = component.get("component", "")
        
        # Bilder überspringen
        if component_name in SKIP_COMPONENTS:
            continue
        
        # Referenzen auflösen
        if component_name in REFERENCE_COMPONENTS and resolve_references:
            content = component.get("content", {})
            for key, value in content.items():
                if isinstance(value, dict) and "params" in value:
                    params = value["params"]
                    
                    # Einzelne Referenz
                    if "teaser" in params and "$ref" in params["teaser"]:
                        ref_id = params["teaser"]["reference"]["id"]
                        print(f"  → Referenz auflösen: Dokument {ref_id}")
                        ref_doc = fetch_referenced_document(ref_id)
                        if ref_doc:
                            ref_components = ref_doc.get("content", [])
                            texts.extend(extract_text_from_components(ref_components, resolve_references=False))
                    
                    # Mehrere Referenzen
                    if "teasers" in params and "$ref" in params.get("teasers", {}):
                        references = params["teasers"].get("references", [])
                        for ref in references:
                            ref_id = ref["id"]
                            print(f"  → Referenz auflösen: Dokument {ref_id}")
                            ref_doc = fetch_referenced_document(ref_id)
                            if ref_doc:
                                ref_components = ref_doc.get("content", [])
                                texts.extend(extract_text_from_components(ref_components, resolve_references=False))
            continue
        
        # Direkte Textfelder
        content = component.get("content", {})
        for key, value in content.items():
            if isinstance(value, str) and len(value) > 10:
                texts.append(value)
        
        # Verschachtelte Container rekursiv
        containers = component.get("containers", {})
        for container_name, container_components in containers.items():
            if isinstance(container_components, list):
                texts.extend(extract_text_from_components(container_components, resolve_references))
    
    return texts

def extract_article_data(article):
    """Alle relevanten Daten aus einem Artikel extrahieren"""
    systemdata = article.get("systemdata", {})
    metadata = article.get("metadata", {})
    
    doc_id = systemdata.get("documentId", "unbekannt")
    title = metadata.get("title", "")
    description = metadata.get("description", "")
    language = metadata.get("language", {}).get("locale", "unbekannt")
    slug = metadata.get("slug", "")
    last_updated = systemdata.get("lastPublicationDate", "")
    content_type = systemdata.get("contentType", "")
    
    print(f"\nVerarbeite Artikel {doc_id} [{language}]: {title}")
    
    content_components = article.get("content", [])
    content_texts = extract_text_from_components(content_components)
    
    # Duplikate entfernen
    seen = set()
    unique_texts = []
    for t in [title, description] + content_texts:
        t_clean = t.strip()
        if t_clean and t_clean not in seen:
            seen.add(t_clean)
            unique_texts.append(t_clean)
    
    full_text = html.unescape(" ".join(unique_texts))
    
    return {
        "document_id": doc_id,
        "title": title,
        "language": language,
        "slug": slug,
        "last_updated": last_updated,
        "content_type": content_type,
        "text": full_text
    }

# Test
if __name__ == "__main__":
    articles = fetch_articles(limit=10)
    
    for article in articles:
        data = extract_article_data(article)
        print(f"Text (erste 400 Zeichen): {data['text'][:400]}")
        print("-" * 50)