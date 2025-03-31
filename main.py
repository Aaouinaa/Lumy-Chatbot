import os
import subprocess
import threading
import logging
import time
from dotenv import load_dotenv

# Umgebungsvariablen laden
load_dotenv()

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def run_streamlit_app():
    """Startet die Streamlit-App"""
    try:
        logger.info("Starte Streamlit-App...")
        subprocess.run(["streamlit", "run", "streamlit_app.py"], check=True)
    except Exception as e:
        logger.error(f"Fehler beim Starten der Streamlit-App: {e}")

def run_supabase_sync():
    """Startet den Supabase-Sync-Prozess"""
    try:
        logger.info("Starte Supabase-Sync...")
        from logging_to_DB import start_db_sync
        start_db_sync()
    except Exception as e:
        logger.error(f"Fehler beim Starten des Supabase-Sync: {e}")

if __name__ == "__main__":
    # Threads für die verschiedenen Komponenten
    streamlit_thread = threading.Thread(target=run_streamlit_app)
    supabase_thread = threading.Thread(target=run_supabase_sync)
    
    # Threads als Daemon markieren, damit sie beendet werden, wenn das Hauptprogramm beendet wird
    streamlit_thread.daemon = True
    supabase_thread.daemon = True
    
    # Threads starten
    streamlit_thread.start()
    supabase_thread.start()
    
    # Warten, bis alle Threads beendet sind (wird nie passieren, da sie Daemon-Threads sind)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Programm wird beendet...")
