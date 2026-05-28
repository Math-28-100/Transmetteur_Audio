# =====================================================================
# TRANSCRIPTION AUDIO CONTINUE (VERSION FINALE - ANTI-COUPURES)
# =====================================================================

import sounddevice as sd
import numpy as np
import datetime
import json
import speech_recognition as sr
import queue
import threading
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
fs = 16000  # fréquence d'échantillonnage
channels = 1  # mono
dtype = 'int16'
chunk_duration = 6  # durée de chaque segment en secondes
overlap_duration = 2  # RECOUVREMENT : 2 secondes du bloc précédent sont répétées

# Sauvegarde JSON sur le Bureau
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
journal_file = os.path.join(desktop_path, "journal_audio_ia.json")

r = sr.Recognizer()
audio_buffer = queue.Queue()
file_lock = threading.Lock()

# -----------------------------
# INITIALISATION DU FICHIER JSON
# -----------------------------
if not os.path.exists(journal_file) or os.path.getsize(journal_file) == 0:
    with open(journal_file, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)


# -----------------------------
# NORMALISATION AUDIO
# -----------------------------
def normaliser_audio(audio_np, gain_max=3.0):
    rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
    if rms == 0:
        return audio_np
    gain = min(gain_max, 32767 / rms)
    audio_norm = audio_np * gain
    return np.clip(audio_norm, -32767, 32767).astype(np.int16)


# -----------------------------
# CALLBACK POUR INPUTSTREAM
# -----------------------------
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    audio_buffer.put(indata.copy())


# -----------------------------
# SAUVEGARDE EN TABLEAU JSON PROPRE
# -----------------------------
def sauvegarder_dans_json(texte, timestamp):
    with file_lock:
        try:
            with open(journal_file, "r", encoding="utf-8") as f:
                donnees = json.load(f)
                if not isinstance(donnees, list):
                    donnees = []
        except (json.JSONDecodeError, FileNotFoundError):
            donnees = []

        date_str, heure_str = timestamp.split(" ")

        entry = {
            "timestamp": timestamp,
            "metadonnees": {
                "date": date_str,
                "heure": heure_str,
                "source": "enregistrement_continu"
            },
            "texte": texte
        }

        # Évite les doublons exacts consécutifs à cause du recouvrement
        if donnees and donnees[-1]["texte"] == texte:
            return

        donnees.append(entry)

        with open(journal_file, "w", encoding="utf-8") as f:
            json.dump(donnees, f, ensure_ascii=False, indent=4)


# -----------------------------
# FONCTION DE TRANSCRIPTION AVEC RECOUVREMENT
# -----------------------------
def transcrire_continu():
    # Variables pour stocker la fin du bloc précédent
    samples_overlap = int(overlap_duration * fs)
    ancien_bloc_suffixe = np.zeros(samples_overlap, dtype=dtype)

    while True:
        audio_chunk = audio_buffer.get()
        audio_chunk_np = audio_chunk.flatten()

        # Fusion : on colle les 2 secondes de la fin du bloc d'avant AU DÉBUT du nouveau bloc
        audio_combine = np.concatenate((ancien_bloc_suffixe, audio_chunk_np))

        # On sauvegarde les 2 dernières secondes de ce bloc actuel pour le prochain tour
        ancien_bloc_suffixe = audio_chunk_np[-samples_overlap:]

        # Normalisation et traitement du bloc fusionné
        audio_combine = normaliser_audio(audio_combine)
        audio_data = sr.AudioData(audio_combine.tobytes(), fs, audio_combine.dtype.itemsize)

        try:
            texte_transcrit = r.recognize_google(audio_data, language="fr-FR")
            texte_clean = texte_transcrit.strip()

            if texte_clean and len(texte_clean) > 2:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] 📥 Capté : {texte_clean}")
                sauvegarder_dans_json(texte_clean, timestamp)

        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"Erreur service reconnaissance vocale : {e}")


# -----------------------------
# BOUCLE PRINCIPALE
# -----------------------------
try:
    threading.Thread(target=transcrire_continu, daemon=True).start()

    print("==========================================================")
    print("🎙️ MICRO ACTIVÉ - MODE ANTI-COUPURES INTÉGRÉ")
    print("==========================================================")
    print(f"Fichier JSON structuré créé sur le Bureau : {journal_file}")
    print("Parlez naturellement... (Ctrl+C pour arrêter)\n")

    with sd.InputStream(samplerate=fs, channels=channels, dtype=dtype,
                        blocksize=int(chunk_duration * fs),
                        callback=audio_callback):
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\n💾 Arrêt de l'écoute. Données préservées dans le JSON.")
except Exception as e:
    print("Erreur inattendue :", e)