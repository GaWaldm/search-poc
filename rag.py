import anthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langdetect import detect
from config import CHROMA_PATH, EMBEDDING_MODEL, ANTHROPIC_API_KEY, CLAUDE_MODEL

SUPPORTED_LANGUAGES = ["de", "fr", "it", "en", "rm"]

FLAGS = {"de": "🇩🇪", "fr": "🇫🇷", "it": "🇮🇹", "en": "🇬🇧", "rm": "🏔️"}

# Prompt-Templates pro Sprache
PROMPTS = {
    "de": """Du bist ein hilfreicher Assistent der Schweizer Bundesverwaltung.
Beantworte die Frage ausschliesslich auf Basis der folgenden Textabschnitte.
Wenn die Antwort nicht in den Textabschnitten steht, sage das ehrlich.
Antworte immer auf Deutsch, klar und bürgerfreundlich.

Textabschnitte:
{context}

Frage: {question}

Antwort:""",

    "fr": """Tu es un assistant utile de l'administration fédérale suisse.
Réponds à la question uniquement sur la base des extraits de texte suivants.
Si la réponse ne figure pas dans les extraits, dis-le honnêtement.
Réponds toujours en français, de manière claire et accessible aux citoyens.

Extraits de texte:
{context}

Question: {question}

Réponse:""",

    "it": """Sei un assistente utile dell'amministrazione federale svizzera.
Rispondi alla domanda esclusivamente sulla base dei seguenti estratti di testo.
Se la risposta non è contenuta negli estratti, dillo onestamente.
Rispondi sempre in italiano, in modo chiaro e accessibile ai cittadini.

Estratti di testo:
{context}

Domanda: {question}

Risposta:"""
}

def load_db():
    """Index laden"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def search_chunks(db, query, language, k=5):
    """Relevante Chunks suchen mit Sprachpriorisierung"""
    results = []

    # Erst in erkannter Sprache suchen
    results = db.similarity_search(query, k=k, filter={"language": language})

    # Fallback wenn zu wenig Treffer
    if len(results) < 2:
        extra = db.similarity_search(query, k=k)
        existing = {r.page_content for r in results}
        for r in extra:
            if r.page_content not in existing:
                results.append(r)
            if len(results) >= k:
                break

    return results

def ask(question, ui_lang="de"):
    """Frage stellen und Antwort von Claude erhalten"""

    db = load_db()

    # Sprache erkennen
    try:
        detected = detect(question)
        language = detected if detected in SUPPORTED_LANGUAGES else ui_lang
    except:
        language = ui_lang

    print(f"Erkannte Sprache: {language}")

    # Relevante Chunks suchen
    chunks = search_chunks(db, question, language)
    print(f"{len(chunks)} Chunks gefunden")

    # Kontext zusammenbauen
    context_parts = []
    sources = []

    for i, chunk in enumerate(chunks):
        title = chunk.metadata.get("title", "")
        content_type = chunk.metadata.get("content_type", "")
        lang = chunk.metadata.get("language", "")
        context_parts.append(f"[{i+1}] {chunk.page_content}")
        sources.append({
            "index": i+1,
            "title": title,
            "content_type": content_type,
            "language": lang,
            "flag": FLAGS.get(lang, "🌐")
        })

    context = "\n\n".join(context_parts)

    # Prompt zusammenbauen
    prompt_template = PROMPTS.get(language, PROMPTS["de"])
    prompt = prompt_template.format(context=context, question=question)

    # Claude aufrufen
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = message.content[0].text

    return {
        "question": question,
        "answer": answer,
        "language": language,
        "sources": sources
    }

# Test
if __name__ == "__main__":
    result = ask("Was ist die direkte Demokratie?")
    print(f"\nFrage: {result['question']}")
    print(f"\nAntwort:\n{result['answer']}")
    print(f"\nQuellen:")
    for s in result['sources']:
        print(f"  {s['flag']} [{s['index']}] {s['title']} ({s['content_type']})")