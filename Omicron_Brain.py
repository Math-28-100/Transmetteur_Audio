# =====================================================================
# LE CERVEAU D'OMICRON - VRAIE VECTORISATION PROPRE ET SIMPLIFIÉE
# =====================================================================

import os
# Désactivation propre des avertissements de la communauté au démarrage
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
journal_file = os.path.join(desktop_path, "journal_audio_ia.json")

# Ta clé API Groq (Pense à la révoquer/générer une nouvelle si besoin)
GROQ_API_KEY = "Clé API Groq"

print("🧠 Chargement du modèle de vectorisation dans ton cerveau local...")
# Modèle de vectorisation hyper léger et ultra-performant pour le français
model_embedding = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------
# MOTEUR DE RECHERCHE VECTORIELLE (COSINE SIMILARITY)
# ---------------------------------------------------------------------
def recherche_vectorielle(question, limite=4):
    if not os.path.exists(journal_file):
        print("⚠️ Aucun journal audio trouvé sur le Bureau.")
        return ""

    with open(journal_file, "r", encoding="utf-8") as f:
        donnees = json.load(f)

    if not donnees:
        return ""

    # 1. On extrait toutes les phrases du journal
    phrases = [entree["texte"] for entree in donnees]

    # 2. VECTORISATION : On transforme tes souvenirs et ta question en nombres
    print("⚡ Analyse vectorielle de tes souvenirs en cours...")
    vecteurs_souvenirs = model_embedding.encode(phrases, convert_to_numpy=True)
    vecteur_question = model_embedding.encode(question, convert_to_numpy=True)

    # 3. MATHS : On calcule le "score de proximité" (produit scalaire normalisé)
    scores = np.dot(vecteurs_souvenirs, vecteur_question) / (
            np.linalg.norm(vecteurs_souvenirs, axis=1) * np.linalg.norm(vecteur_question)
    )

    # 4. On trie pour garder les meilleurs souvenirs
    index_meilleurs = np.argsort(scores)[::-1][:limite]

    # 5. Construction du contexte textuel pour l'IA
    contexte = ""
    for idx in index_meilleurs:
        # On ne prend le souvenir que s'il a un minimum de sens (score > 0.1)
        if scores[idx] > 0.1:
            entree = donnees[idx]
            meta = entree["metadonnees"]
            contexte += f"[{meta['date']} à {meta['heure']}] : {entree['texte']}\n"

    return contexte


# ---------------------------------------------------------------------
# INTERACTION AVEC L'API DU JUMEAU NUMÉRIQUE
# ---------------------------------------------------------------------
def discuter_avec_jumeau(question):
    # Récupération des souvenirs via notre moteur vectoriel maison
    contexte_souvenirs = recherche_vectorielle(question)

    # Prompt pour cloner ta personnalité
    system_prompt = f"""
    Tu es le jumeau numérique de l'utilisateur (son clone textuel parfait). Tu dois agir, penser et répondre exactement comme lui à plus de 90%.

    Voici les extraits de tes propres souvenirs et pensées vectorisés récemment en lien avec sa demande :
    {contexte_souvenirs if contexte_souvenirs else "Aucun souvenir spécifique trouvé."}

    CONSIGNES STRICTES :
    1. Utilise les faits présents dans tes souvenirs pour construire ta réponse.
    2. Adopte le style de l'utilisateur : s'il parle de manière décontractée, sois décontracté. S'il utilise des expressions courtes, fais de même.
    3. Parle obligatoirement à la première personne ("Je"). Tu ES lui.
    """

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",  # Version mise à jour à la place du modèle obsolète
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            resultat = response.json()
            return resultat["choices"][0]["message"]["content"]
        else:
            return f"❌ Erreur API ({response.status_code}) : {response.text}"
    except Exception as e:
        return f"❌ Impossible de joindre le serveur de l'IA : {e}"


# ---------------------------------------------------------------------
# BOUCLE PRINCIPALE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("==========================================================")
    print("🧠 OMICRON BRAIN V3.1 - VECTORISATION & API OK")
    print("==========================================================")

    if os.path.exists(journal_file):
        with open(journal_file, "r", encoding="utf-8") as f:
            nb_lignes = len(json.load(f))
        print(f"💾 Base connectée : {nb_lignes} souvenirs prêts à être vectorisés.")
    else:
        print("⚠️ En attente du fichier journal_audio_ia.json sur le Bureau...")

    print("\n🤖 Votre Jumeau Numérique est en ligne.")
    print("Discutez avec lui. (Tapez 'quitter' pour fermer)\n")

    while True:
        votre_message = input("Vous 👤 : ")
        if votre_message.lower() == "quitter":
            print("\n👋 Fermeture du cerveau Omicron.")
            break

        if not votre_message.strip():
            continue

        reponse_clone = discuter_avec_jumeau(votre_message)
        print(f"\nOmicron 🤖 : {reponse_clone}\n")