import streamlit as st
import json
import random
import os
import datetime
from pathlib import Path
import uuid
from typing import List, Tuple
from pinecone_rag import analyze_intent_and_sentiment, rag_pipeline_func, save_chat_log_pinecone
import string
import time

# Seitenkonfiguration
st.set_page_config(
    page_title="Lumy Kundenservice Chatbot",
    page_icon="💬",
    layout="wide"
)

# Titel und Einführung
st.title("Lumy Kundenservice Chatbot")
st.markdown("Stellen Sie Ihre Fragen zum Kundenservice.")

# Session ID für Streamlit
session_ticket_id = "TICKET-" + str(random.randint(10000, 99999))

# Session States initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "thinking" not in st.session_state:
    st.session_state.thinking = ""

if "chat_history_log" not in st.session_state:
    # Lade existierende Chat-Logs
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_log.json")
    if Path(log_file).exists():
        with open(log_file, "r", encoding="utf-8") as f:
            try:
                st.session_state.chat_history_log = json.load(f)
            except json.JSONDecodeError:
                st.session_state.chat_history_log = []
    else:
        st.session_state.chat_history_log = []

# Funktion zum Speichern der Chat-Logs
def save_chat_log(log_entry):
    # Füge Eintrag zum In-Memory-Log hinzu
    st.session_state.chat_history_log.append(log_entry)
    
    # Speichere in Datei
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_log.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_history_log, f, indent=2, ensure_ascii=False)
        f.flush()
    
    # Speichere in Pinecone
    save_chat_log_pinecone([log_entry])

# Funktion zum Formatieren des Gedankengangs
def format_thinking(thinking_text):
    if not thinking_text:
        return ""
    
    # Wenn es sich um Chain Output handelt
    if thinking_text.startswith("> Finished chain"):
        # Chain Output formatieren
        if "Chain Output:" in thinking_text:
            parts = thinking_text.split("Chain Output:", 1)
            header = parts[0].strip()
            content = parts[1].strip()
            
            # Versuche, das JSON in content zu formatieren
            try:
                # Wenn es JSON-ähnlich ist, formatieren wir es
                if content.startswith("{") and content.endswith("}"):
                    import json
                    parsed = json.loads(content)
                    
                    # Formatierung der einzelnen Teile
                    formatted_parts = []
                    
                    # Frage formatieren
                    if "question" in parsed:
                        formatted_parts.append(f"📝 **Frage:** {parsed['question']}")
                    
                    # Antwort formatieren
                    if "answer" in parsed:
                        formatted_parts.append(f"💬 **Antwort:** {parsed['answer']}")
                    
                    # Quellen formatieren
                    if "source_documents" in parsed:
                        docs = parsed["source_documents"]
                        docs_formatted = []
                        
                        formatted_parts.append("📚 **Quellen:**")
                        for i, doc in enumerate(docs):
                            if isinstance(doc, dict):
                                doc_content = doc.get("page_content", str(doc))
                            else:
                                doc_content = str(doc)
                            docs_formatted.append(f"  Dokument {i+1}:\n  {doc_content}")
                        
                        formatted_parts.append("\n".join(docs_formatted))
                    
                    # Andere Elemente
                    for key, value in parsed.items():
                        if key not in ["question", "answer", "source_documents"]:
                            formatted_parts.append(f"➤ **{key}:** {value}")
                    
                    # Alles zusammenfügen
                    formatted_content = "\n\n".join(formatted_parts)
                    return f"{header}\n\n{formatted_content}"
            except:
                # Bei Fehlern beim JSON-Parsen, minimal formatieren
                return f"{header}\nChain Output:\n{content}"
    
    # Generelle Formatierung mit Zeilenumbrüchen und Einrückungen
    formatted = ""
    lines = thinking_text.split("\n")
    current_section = ""
    
    for line in lines:
        # Abschnitte durch visuelle Trenner hervorheben
        if line.startswith("> Entering new") or line.startswith("> Finished"):
            if current_section:
                formatted += current_section + "\n\n"
                current_section = ""
            formatted += f"🔄 {line}\n"
        elif line.startswith("<think>"):
            if current_section:
                formatted += current_section + "\n\n"
                current_section = ""
            formatted += f"🧠 {line}\n"
        else:
            current_section += line + "\n"
    
    # Letzten Abschnitt hinzufügen
    if current_section:
        formatted += current_section
    
    return formatted.strip()

# Chat-Nachrichtenverlauf anzeigen
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("thinking") and message["role"] == "assistant":
                # Gedankengang formatiert anzeigen
                st.markdown("**Gedankengang des Chatbots:**")
                formatted_thinking = format_thinking(message["thinking"])
                st.markdown(f"```\n{formatted_thinking}\n```")

# Chat-Input
if prompt := st.chat_input("Wie kann ich Ihnen helfen?"):
    # Nutzernachricht anzeigen
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Nachricht zum Verlauf hinzufügen
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Intent und Stimmung analysieren
    intent, sentiment = analyze_intent_and_sentiment(prompt)
    
    # Antwort generieren
    with chat_container:
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            with st.spinner("Denke nach..."):
                forbidden_checks = ["delete file", "anything that is illegal", "anything related to other customers' data", "tracking numbers of all customers"]
                if any(fk in prompt.lower() for fk in forbidden_checks):
                    response = "Ich kann diese Anfrage leider nicht beantworten, da sie gegen unsere Richtlinien verstößt."
                    thinking = "Die Anfrage wurde als potenziell problematisch erkannt und abgelehnt."
                else:
                    # Hier rufen wir die RAG-Pipeline auf und speichern den Gedankengang
                    result = rag_pipeline_func(prompt)
                    response = result["reply"]
                    
                    # Gedankengang extrahieren - wenn vorhanden
                    thinking = ""
                    
                    # Chain Output integrieren, wenn vorhanden
                    if "source_documents" in result:
                        # Vereinfachen auf das reine Chain Output
                        thinking = f"> Finished chain.\nChain Output: {result}"
                    
                    # Bei offensiven Kommentaren besonders hervorheben
                    if intent == "offensive" or sentiment == "negative":
                        st.warning("Hinweis: Diese Nachricht enthält möglicherweise unangemessene Inhalte.")
                    
                    st.markdown(response)
                    st.session_state.thinking = thinking
                    
                    # Gedankengang direkt formatiert anzeigen
                    st.markdown("**Gedankengang des Chatbots:**")
                    formatted_thinking = format_thinking(thinking)
                    st.markdown(f"```\n{formatted_thinking}\n```")
    
    # Assistentenantwort zum Verlauf hinzufügen
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "thinking": st.session_state.thinking
    })
    
    # Log-Eintrag erstellen und speichern
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user": prompt,
        "assistant": response,
        "thinking": st.session_state.thinking,
        "intent": intent,
        "sentiment": sentiment,
        "session_id": st.session_state.session_id,
        "ticket_id": st.session_state.session_id,  # Für Kompatibilität mit pinecone_rag.py
        "ml_intent": "other"  # Platzhalter, kann durch ML-Vorhersage ersetzt werden
    }
    save_chat_log(log_entry)

# Trenner
st.markdown("---")

# Chat-Historie-Bereich
st.subheader("Chat-Historie")

# Tabs für verschiedene Ansichten
history_tab1, history_tab2 = st.tabs(["Aktuelle Sitzung", "Alle Gespräche"])

with history_tab1:
    # Aktuelle Sitzung filtern
    current_session_logs = [log for log in st.session_state.chat_history_log 
                        if log.get("session_id") == st.session_state.session_id]
    
    if current_session_logs:
        for log in current_session_logs:
            st.markdown(f"**Zeitstempel:** {log.get('timestamp')}")
            st.markdown(f"**Benutzer:** {log.get('user')}")
            st.markdown(f"**Assistent:** {log.get('assistant')}")
            st.markdown(f"**Intent:** {log.get('intent')}, **Stimmung:** {log.get('sentiment')}")
            
            if log.get("thinking"):
                # Gedankengang formatiert anzeigen
                st.markdown("**Gedankengang:**")
                formatted_thinking = format_thinking(log.get("thinking"))
                st.markdown(f"```\n{formatted_thinking}\n```")
            
            st.markdown("---")
    else:
        st.info("Keine Nachrichten in der aktuellen Sitzung.")

with history_tab2:
    # Alle Gespräche anzeigen
    if st.session_state.chat_history_log:
        # Sitzungen gruppieren
        sessions = {}
        for log in st.session_state.chat_history_log:
            session_id = log.get("session_id", "unbekannt")
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(log)
        
        # Sitzungen als Selectbox zur Auswahl anbieten
        session_options = []
        for session_id, logs in sessions.items():
            first_log = logs[0]
            first_time = datetime.datetime.fromisoformat(first_log.get("timestamp")).strftime("%d.%m.%Y %H:%M")
            session_options.append(f"Sitzung {session_id[:8]}... ({first_time}) - {len(logs)} Nachrichten")
        
        if session_options:
            selected_session = st.selectbox("Sitzung auswählen:", session_options)
            selected_idx = session_options.index(selected_session)
            selected_session_id = list(sessions.keys())[selected_idx]
            
            # Zeige Nachrichten der ausgewählten Sitzung
            for log in sessions[selected_session_id]:
                col1, col2 = st.columns([1, 4])
                with col1:
                    timestamp = datetime.datetime.fromisoformat(log.get("timestamp")).strftime("%H:%M:%S")
                    st.markdown(f"**{timestamp}**")
                with col2:
                    st.markdown(f"**Benutzer:** {log.get('user')}")
                    st.markdown(f"**Assistent:** {log.get('assistant')}")
                    
                    if log.get("thinking"):
                        # Gedankengang formatiert anzeigen
                        st.markdown("**Gedankengang:**")
                        formatted_thinking = format_thinking(log.get("thinking"))
                        st.markdown(f"```\n{formatted_thinking}\n```")
                
                st.markdown("---")
    else:
        st.info("Keine Chat-Historie vorhanden.")

# Systeminfo
with st.sidebar:
    st.subheader("System-Information")
    st.write("Streamlit Sitzungs-ID:", st.session_state.session_id)
    
    # Debug-Optionen
    with st.expander("Debug-Optionen", expanded=True):
        st.checkbox("Gedankengang automatisch anzeigen", value=True, key="auto_show_thinking", disabled=True,
                   help="Bei Aktivierung werden die Gedankengänge des Chatbots automatisch angezeigt")
        st.write("FAQ-Namespace: faq_2_json")
        st.write("Chat-Log-Namespace: chat_log")
    
    if st.button("Chat-Verlauf aktualisieren"):
        st.cache_data.clear()
        st.experimental_rerun()

# Footer
st.markdown("---") 