import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langdetect import detect
from config import CHROMA_PATH, EMBEDDING_MODEL
from rag import ask
from ingest_pdf import index_pdfs
import os

if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
    with st.spinner("Index wird aufgebaut... (einmalig, ca. 2 Minuten)"):
        index_pdfs()

SUPPORTED_LANGUAGES = ["de", "fr", "it", "en", "rm"]

# Übersetzungen
TRANSLATIONS = {
    "de": {
        "title": "Intelligente Suche - grins — Prototyp",
        "subtitle": "Semantische Suche über Inhalte eines Webauftrittes",
        "indexed": "Inhalte indexiert · Mehrsprachig (DE / FR / IT)",
        "placeholder": "Ihre Frage in natürlicher Sprache — z.B. «Wie beantrage ich einen Schweizer Pass?»",
        "only_my_lang": "Nur meine Sprache",
        "use_rag": "KI-Antwort (Beta)",
        "detected_lang": "Erkannte Sprache",
        "results_found": "Treffer gefunden",
        "fallback": "Wenige Treffer in dieser Sprache — weitere Sprachen werden angezeigt.",
        "updated": "Aktualisiert",
        "no_results": "Keine Treffer gefunden — bitte anders formulieren.",
        "loading": "Suche wird geladen...",
        "footer": "Prototyp · Semantische Suche auf Basis HCMS · Powered by sentence-transformers & Chroma"
    },
    "fr": {
        "title": "Recherche intelligente — Prototype",
        "subtitle": "Recherche sémantique sur les contenus d'une site web'",
        "indexed": "contenus indexés · Multilingue (DE / FR / IT)",
        "placeholder": "Votre question en langage naturel — p.ex. «Comment demander un passeport suisse?»",
        "only_my_lang": "Uniquement ma langue",
        "use_rag": "Réponse IA (Bêta)",
        "detected_lang": "Langue détectée",
        "results_found": "résultats trouvés",
        "fallback": "Peu de résultats dans cette langue — d'autres langues sont affichées.",
        "updated": "Mis à jour",
        "no_results": "Aucun résultat — veuillez reformuler votre question.",
        "loading": "Chargement de la recherche...",
        "footer": "Prototype · Recherche sémantique basée sur un HCMS · Powered by sentence-transformers & Chroma"
    },
    "it": {
        "title": "Ricerca intelligente — Prototipo",
        "subtitle": "Ricerca semantica sui contenuti del un sito web",
        "indexed": "contenuti indicizzati · Multilingue (DE / FR / IT)",
        "placeholder": "La sua domanda in linguaggio naturale — es. «Come richiedere un passaporto svizzero?»",
        "only_my_lang": "Solo la mia lingua",
        "use_rag": "Risposta IA (Beta)",
        "detected_lang": "Lingua rilevata",
        "results_found": "risultati trovati",
        "fallback": "Pochi risultati in questa lingua — vengono mostrate altre lingue.",
        "updated": "Aggiornato",
        "no_results": "Nessun risultato — si prega di riformulare la domanda.",
        "loading": "Caricamento della ricerca...",
        "footer": "Prototipo · Ricerca semantica basata su un HCMS · Powered by sentence-transformers & Chroma"
    }
}

# Sprach-Flags und Labels
FLAGS = {"de": "🇩🇪", "fr": "🇫🇷", "it": "🇮🇹", "en": "🇬🇧", "rm": "🏔️"}
LANG_LABELS = {"de": "Deutsch", "fr": "Français", "it": "Italiano", "en": "English", "rm": "Rumantsch"}

# Seite konfigurieren
st.set_page_config(
    page_title="Bundessuche",
    page_icon="🇨🇭",
    layout="centered"
)

# CSS
st.markdown("""
<style>
    .header-bar {
        background-color: #D4141C;
        padding: 12px 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 8px;
        border-radius: 4px;
    }
    .header-bar img {
        height: 40px;
    }
    .header-title {
        color: white;
        font-size: 18px;
        font-weight: 600;
        font-family: sans-serif;
    }
    .header-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 13px;
        font-family: sans-serif;
    }
    .result-card {
        border-left: 4px solid #D4141C;
        padding: 12px 16px;
        margin-bottom: 12px;
        background: #f9f9f9;
        border-radius: 0 4px 4px 0;
    }
    .result-title {
        font-weight: 600;
        font-size: 16px;
        color: #1a1a1a;
    }
    .result-meta {
        font-size: 12px;
        color: #666;
        margin-top: 4px;
    }
    .result-text {
        font-size: 14px;
        color: #333;
        margin-top: 8px;
        line-height: 1.5;
    }
    .stTextInput input {
        border: 2px solid #D4141C !important;
        border-radius: 4px !important;
        font-size: 16px !important;
    }
    .footer {
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #eee;
        font-size: 12px;
        color: #999;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sprache aus URL-Parameter lesen
params = st.query_params
default_lang = params.get("lang", "de")
if default_lang not in ["de", "fr", "it"]:
    default_lang = "de"

ui_lang = st.selectbox(
    "Sprache / Langue / Lingua",
    options=["de", "fr", "it"],
    index=["de", "fr", "it"].index(default_lang),
    format_func=lambda x: {"de": "🇩🇪 Deutsch", "fr": "🇫🇷 Français", "it": "🇮🇹 Italiano"}[x],
    label_visibility="collapsed"
)
t = TRANSLATIONS[ui_lang]

# Header
st.markdown(f"""
<div class="header-bar">
    <img src="https://www.admin.ch/images/swiss-logo-flag.svg"/>
    <div>
        <div class="header-title">🇨🇭 {t['title']}</div>
        <div class="header-subtitle">{t['subtitle']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Index laden
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return db

with st.spinner(t["loading"]):
    db = load_db()

total = len(db.get()["ids"])
st.caption(f"{total} {t['indexed']}")

# Suchfeld
query = st.text_input(
    "Suchanfrage",
    placeholder=t["placeholder"],
    label_visibility="collapsed"
)

# Optionen
col1, col2 = st.columns(2)
with col1:
    only_my_language = st.checkbox(t["only_my_lang"], value=False)
with col2:
    use_rag = st.checkbox(t["use_rag"], value=False)

# Suche
if query:

    # Sprache erkennen
    try:
        detected = detect(query)
        language = detected if detected in SUPPORTED_LANGUAGES else ui_lang
    except:
        language = ui_lang

    flag = FLAGS.get(language, "🌐")
    label = LANG_LABELS.get(language, language)
    st.caption(f"{t['detected_lang']}: {flag} {label}")

    # RAG-Modus
    if use_rag:
        with st.spinner("🤖 Claude sucht und antwortet..."):
            result = ask(query, ui_lang=ui_lang)

        # Antwort anzeigen
        st.markdown(f"""
        <div style="background:#f0f7f0; border-left: 4px solid #2e7d32; 
             padding: 16px; border-radius: 0 4px 4px 0; margin-bottom: 16px;">
            <div style="font-size:12px; color:#2e7d32; margin-bottom:8px;">
                🤖 KI-Antwort auf Basis eurer Inhalte (Beta)
            </div>
            <div style="font-size:15px; line-height:1.6;">
                {result['answer']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quellen
        st.markdown("**Verwendete Quellen:**")
        for s in result['sources']:
            st.markdown(f"{s['flag']} {s['title']} · `{s['content_type']}`")

    # Normale Suche
    else:
        results = []
        fallback_used = False

        if only_my_language and language:
            results = db.similarity_search(query, k=5, filter={"language": language})
            if len(results) < 2:
                fallback_used = True
                extra = db.similarity_search(query, k=5)
                existing = {r.page_content for r in results}
                for r in extra:
                    if r.page_content not in existing:
                        results.append(r)
                    if len(results) >= 5:
                        break
        else:
            results = db.similarity_search(query, k=5)

        if fallback_used:
            st.info(t["fallback"])

        if not results:
            st.warning(t["no_results"])
        else:
            st.markdown(f"**{len(results)} {t['results_found']}**")

            for result in results:
                lang = result.metadata.get("language", "")
                title = result.metadata.get("title", "Ohne Titel")
                content_type = result.metadata.get("content_type", "")
                text = result.page_content
                flag = FLAGS.get(lang, "🌐")
                updated = result.metadata.get("last_updated", "")[:10]

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">{flag} {title}</div>
                    <div class="result-meta">
                        {LANG_LABELS.get(lang, lang)} · {content_type} · 
                        {t['updated']}: {updated}
                    </div>
                    <div class="result-text">{text[:400]}...</div>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div class="footer">{t['footer']}</div>
""", unsafe_allow_html=True)
