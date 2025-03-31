import os
import json
from pathlib import Path
from dotenv import load_dotenv
import logging
import sys
import socket
from typing import List, Dict
from datetime import datetime, timezone

# Lade Umgebungsvariablen
load_dotenv()

# Konfigurieren des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('db_sync')

# Pfad zur Chat-Log-Datei
filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_log.json")

# Supabase Client initialisieren (nur wenn beide Umgebungsvariablen gesetzt sind)
supabase_available = False
supabase = None

try:
    from supabase import create_client
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_TABLE = "conversation_history"
    
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            supabase_available = True
            logger.info("Supabase-Client initialisiert")
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des Supabase-Clients: {e}")
    else:
        logger.info("Keine Supabase-Credentials gefunden. Chatlogs werden nur lokal gespeichert.")
except ImportError:
    logger.warning("Supabase-Paket nicht installiert. Chatlogs werden nur lokal gespeichert.")
except Exception as e:
    logger.error(f"Fehler beim Laden der Umgebungsvariablen: {e}")

# Variable zum Verfolgen der hochgeladenen Logs
uploaded_entries = []

def check_internet_connection():
    """Prüft, ob eine Internetverbindung besteht."""
    try:
        # Versuche, eine Verbindung zu einem bekannten Server herzustellen
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def load_local_chat_log():
    """Lädt den lokalen Chat-Log aus der JSON-Datei."""
    if not Path(filepath).exists():
        return []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error("Fehler beim Parsen der Chat-Log-Datei")
        return []
    except Exception as e:
        logger.error(f"Fehler beim Laden der Chat-Log-Datei: {e}")
        return []

def upload_to_supabase(entries):
    """Lädt neue Chat-Log-Einträge in die Supabase-Datenbank hoch."""
    if not supabase_available:
        logger.info("Supabase nicht verfügbar, überspringe Upload")
        return False, "Supabase nicht verfügbar"
    
    if not check_internet_connection():
        return False, "Keine Internetverbindung verfügbar"
    
    try:
        # Füge jeden Eintrag in die Supabase-Tabelle ein
        for entry in entries:
            # Erstelle eine Kopie des Eintrags, um Änderungen zu vermeiden
            entry_data = entry.copy()
            
            # Entferne komplexe Strukturen oder konvertiere sie zu Strings
            if "source_documents" in entry_data:
                entry_data["source_documents"] = json.dumps(str(entry_data["source_documents"]))
            
            if "thinking" in entry_data and isinstance(entry_data["thinking"], dict):
                entry_data["thinking"] = json.dumps(entry_data["thinking"])
                
            # Format für Supabase anpassen
            data = {
                "timestamp": entry_data["timestamp"],
                "user_message": entry_data["user"],
                "assistant_reply": entry_data["assistant"],
                "intent": entry_data.get("intent", "other"),
                "sentiment": entry_data.get("sentiment", "neutral"),
                "session_id": entry_data.get("session_id", ""),
                "ml_intent": entry_data.get("ml_intent", "other"),
                "intent_conflict": entry_data.get("intent", "") != entry_data.get("ml_intent", ""),
                "thinking": entry_data.get("thinking", "")
            }
            
            # Füge den Eintrag zur Tabelle hinzu
            response = supabase.from_(SUPABASE_TABLE).upsert([data]).execute()
        
        return True, f"{len(entries)} Einträge hochgeladen"
    except Exception as e:
        logger.error(f"Fehler beim Hochladen in Supabase: {e}")
        return False, str(e)

def mark_as_uploaded(entries):
    """Markiert Einträge als hochgeladen."""
    global uploaded_entries
    entry_timestamps = [entry.get("timestamp") for entry in entries]
    uploaded_entries.extend(entry_timestamps)
    logger.debug(f"{len(entries)} Einträge als hochgeladen markiert")

def get_new_entries():
    """Holt neue Einträge, die noch nicht hochgeladen wurden."""
    chat_log = load_local_chat_log()
    new_entries = [
        entry for entry in chat_log 
        if entry.get("timestamp") not in uploaded_entries
    ]
    return new_entries

def batch_upload_db():
    """
    Hauptfunktion für den Scheduler. Lädt neue Chat-Log-Einträge in die Datenbank hoch.
    """
    try:
        # Überprüfe Internetverbindung, bevor wir weitermachen
        if not check_internet_connection():
            logger.debug("Keine Internetverbindung verfügbar, überspringe Upload")
            return
        
        # Keine Aktion erforderlich, wenn Supabase nicht konfiguriert ist
        if not supabase_available:
            return
        
        # Hole neue Einträge
        new_entries = get_new_entries()
        if not new_entries:
            return
        
        # Lade neue Einträge hoch
        success, message = upload_to_supabase(new_entries)
        if success:
            mark_as_uploaded(new_entries)
            logger.info(message)
        else:
            logger.warning(f"Upload fehlgeschlagen: {message}")
    except Exception as e:
        logger.error(f"Fehler im Batch-Upload: {e}")

if __name__ == "__main__":
    # Manueller Test des Uploads
    new_entries = get_new_entries()
    if new_entries:
        print(f"Gefundene neue Einträge: {len(new_entries)}")
        success, message = upload_to_supabase(new_entries)
        print(f"Upload-Ergebnis: {success}, {message}")
        if success:
            mark_as_uploaded(new_entries)
    else:
        print("Keine neuen Einträge gefunden")


