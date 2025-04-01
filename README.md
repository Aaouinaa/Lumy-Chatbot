# Lumy Kundenservice RAG Chatbot

Ein Retrieval-Augmented Generation (RAG) Chatbot für Kundenservice mit lokaler LLM-Integration, Pinecone-Vektordatenbank und intelligenter Intent- und Sentiment-Analyse.

## Mitwirkende

- **Motseki Khoarai**  – https://github.com/maxmustermann](https://github.com/Lameshmo
- **Gina Weng**        – https://github.com/erikamusterfrau](https://github.com/ginaweng
- **Maxwell Cranston** – https://github.com/johndoe](https://github.com/RehlytC


## Überblick

Dieser Chatbot wurde entwickelt, um:
- Kundenanfragen intelligent zu beantworten durch Zugriff auf FAQ-Dokumente
- Intent und Stimmung in Kundenanfragen zu erkennen und entsprechend zu reagieren
- Benutzerfreundliche Schnittstelle für Kunden bereitzustellen
- Offline-fähig zu arbeiten mit lokaler Protokollierung, falls keine Datenbankverbindung besteht

  ## FAQ

Die FAQ basiert auf einem Datensatz, der auf [huggingface](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) zur Verfügung gestellt wurde:
- 26.8k Zeilen
- 5 Spalten
- 27 intents

## Hauptfunktionen

- **RAG-Architektur**: Kombiniert Retrieval (FAQ-Wissensbasis) mit generativen Antworten
- **Intent- und Sentiment-Analyse**: Erkennt Absicht und Stimmung in Benutzeranfragen
- **Gedankengang-Visualisierung**: Zeigt den Reasoning-Prozess des LLMs für Transparenz
- **Offline-Unterstützung**: Funktioniert auch ohne Internetverbindung durch lokale Speicherung
- **Fehlertoleranz**: Graceful degradation bei Netzwerkproblemen

## Installation

```bash
# Repository klonen
git clone https://github.com/IHR_USERNAME/Lumy-Chatbot.git
cd Lumy-Chatbot

# Python-Umgebung einrichten
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# oder .venv\Scripts\activate  # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt

# .env-Datei konfigurieren
cp .env.example .env
# Bearbeiten Sie die .env-Datei mit Ihren API-Keys
```

## Umgebungsvariablen

```
# Pinecone
PINECONE_API_KEY=ihr_pinecone_api_key
PINECONE_ENV=us-east1-gcp

# Supabase (optional)
SUPABASE_URL=https://ihre-supabase-url.supabase.co
SUPABASE_KEY=ihr_supabase_key
```

## Komponenten

- **streamlit_app.py**: Chat-Interface mit Streamlit
- **pinecone_rag.py**: RAG-Pipeline mit Pinecone-Integration
- **pinecone_rag_multi.py**: Erweiterte Version mit Multi-Namespace-Unterstützung
- **main.py**: Startpunkt der Anwendung
- **prepare_pinecone_index.py**: Skript zum Erstellen und Befüllen des Pinecone-Index

## Verwendung

```bash
# Lokales LLM starten (über Ollama oder LM Studio)
ollama run llama3:8b-instruct

# Pinecone-Index vorbereiten (falls noch nicht geschehen)
python prepare_pinecone_index.py

# Anwendung starten
python main.py

# Oder direkt Streamlit starten
streamlit run streamlit_app.py
```

## Technische Details

### LLM-Integration

Die Anwendung verwendet standardmäßig einen lokalen LLM, der über einen HTTP-Endpunkt angesprochen wird. Der Standardendpunkt ist `http://localhost:1234/v1/chat/completions`.

```python
# In pinecone_rag.py
class LocalLLM(LLM):
    model_name: str = "llama3:8b-instruct"
    temperature: float = 0.3
    max_tokens: int = 2048
    endpoint: str = "http://localhost:1234/v1/chat/completions"
```

### RAG-Architektur

Die Retrieval-Augmented Generation (RAG) Pipeline verwendet folgende Komponenten:

1. **Embedding-Modell**: SentenceTransformers (all-MiniLM-L6-v2)
2. **Vektordatenbank**: Pinecone
3. **Chat-Modell**: Lokales LLM (standardmäßig Llama 3)
4. **Prompt-Template**: Strukturiertes Format mit System-Anweisungen, Kontext und Benutzeranfrage

### Offline-Unterstützung

Der Chatbot speichert Konversationen lokal und versucht, sie mit Supabase zu synchronisieren, wenn eine Verbindung besteht. Bei Netzwerkfehlern bleiben die lokalen Daten erhalten.


## Weiterentwicklung

### Integration neuer FAQs

Neue FAQs können über das Skript `prepare_pinecone_index.py` in den Pinecone-Index aufgenommen werden:

```python
# Beispiel für das Hinzufügen neuer FAQs
new_faqs = [
    {"Q": "Wie kann ich mein Passwort zurücksetzen?", 
     "A": "Sie können Ihr Passwort über die 'Passwort vergessen'-Funktion auf der Login-Seite zurücksetzen..."}
]
```

### Anpassung des Prompt-Templates

Das Prompt-Template kann in `pinecone_rag.py` angepasst werden, um die Persönlichkeit und Antwortstruktur des Chatbots zu ändern.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe LICENSE-Datei für Details. 
