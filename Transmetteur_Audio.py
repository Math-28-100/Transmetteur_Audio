# ==========================================================
# TRANSCRIPTION AUDIO CONTINUE OPTIMISÉE (JSON uniquement)
# ==========================================================

# pip install SpeechRecognition sounddevice numpy

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
fs = 16000               # fréquence d'échantillonnage
channels = 1             # mono
dtype = 'int16'
chunk_duration = 6       # durée en secondes de chaque segment (phrases complètes)
gain_max = 3.0           # amplification maximale voix faibles

# Sauvegarde JSON sur le Bureau
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
journal_file = os.path.join(desktop_path, "journal_audio.json")

r = sr.Recognizer()
audio_buffer = queue.Queue()

# -----------------------------
# NORMALISATION AUDIO
# -----------------------------
def normaliser_audio(audio_np, gain_max=gain_max):
    """
    Amplifie intelligemment le signal pour les voix faibles.
    """
    rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
    if rms == 0:
        return audio_np
    gain = min(gain_max, 32767 / rms)
    audio_norm = audio_np * gain
    return np.clip(audio_norm, -32767, 32767).astype(np.int16)

# -----------------------------
# CALLBACK POUR INPUTSTREAM
# -----------------------------
def audio_callback(indata, frames, time, status):
    """
    Ajoute chaque segment audio au buffer pour traitement.
    """
    if status:
        print(f"Status: {status}")
    audio_buffer.put(indata.copy())

# -----------------------------
# FONCTION DE TRANSCRIPTION
# -----------------------------
def transcrire_continu():
    """
    Prend les segments du buffer, normalise et envoie à Google pour transcription.
    Sauvegarde chaque transcription horodatée dans un fichier JSON.
    """
    while True:
        audio_chunk = audio_buffer.get()
        audio_chunk_np = audio_chunk.flatten()
        audio_chunk_np = normaliser_audio(audio_chunk_np)

        audio_data = sr.AudioData(audio_chunk_np.tobytes(), fs, audio_chunk_np.dtype.itemsize)
        try:
            texte_transcrit = r.recognize_google(audio_data, language="fr-FR")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {texte_transcrit}")

            # sauvegarde JSON
            entry = {"timestamp": timestamp, "texte": texte_transcrit}
            with open(journal_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except sr.UnknownValueError:
            print("[...] Silence ou incompréhensible")
        except sr.RequestError as e:
            print(f"Erreur service reconnaissance vocale : {e}")

# -----------------------------
# BOUCLE PRINCIPALE
# -----------------------------
try:
    # Thread pour transcription en parallèle
    threading.Thread(target=transcrire_continu, daemon=True).start()

    print("Micro activé. Parlez... (Ctrl+C pour arrêter)")
    print(f"Transcriptions sauvegardées dans : {journal_file}")

    with sd.InputStream(samplerate=fs, channels=channels, dtype=dtype,
                        blocksize=int(chunk_duration * fs),
                        callback=audio_callback):
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\nArrêt de l'écoute.")

except Exception as e:
    print("Erreur inattendue :", e)