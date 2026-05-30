import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import requests
from docx import Document
from pypdf import PdfReader
from groq import Groq


def initialiser_cle_api():
    """Demande graphiquement la clé API Groq dès le lancement du programme."""
    root = tk.Tk()
    root.withdraw()  # Cache la petite fenêtre principale Tkinter inutile

    # Détection automatique si déjà configurée dans le système
    if "GROQ_API_KEY" in os.environ and os.environ["GROQ_API_KEY"].strip():
        return os.environ["GROQ_API_KEY"]

    # Boîte de dialogue graphique claire et sécurisée
    cle = simpledialog.askstring(
        "Configuration API Groq",
        "Veuillez coller votre clé API Groq (ex: gsk_...) :",
        show="*"  # Masque les caractères saisis
    )

    if not cle or not cle.strip():
        messagebox.showerror("Erreur Critique", "La clé API Groq est requise pour utiliser ce programme.")
        sys.exit()

    # Injection dynamique dans l'environnement courant du script
    os.environ["GROQ_API_KEY"] = cle.strip()
    return cle.strip()


def extraire_texte_google_doc(url):
    """Télécharge et extrait le texte brut d'un Google Doc partagé publiquement."""
    try:
        match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
        if not match:
            print("[Erreur] URL Google Doc invalide.")
            return None

        doc_id = match.group(1)
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
        response = requests.get(export_url, timeout=10)

        if response.status_code == 200:
            return response.text
        else:
            print(
                "[Erreur] Impossible d'accéder au document. Vérifiez qu'il est partagé en mode 'Public' (Tous les utilisateurs disposant du lien).")
            return None
    except Exception as e:
        print(f"[Erreur Google Doc] : {e}")
        return None


def selectionner_et_extraire_texte():
    """Demande à l'utilisateur son type de document et ouvre le Bureau automatiquement si fichier."""
    root = tk.Tk()
    root.withdraw()

    # 1. Menu de choix initial
    choix = simpledialog.askstring(
        "Type de Source",
        "Où se trouve votre texte exemple ?\nTapez '1' : Sur mon Bureau (Word, PDF, TXT)\nTapez '2' : Lien Google Docs"
    )

    # Option Google Docs
    if choix == '2':
        url = simpledialog.askstring("Lien Google Docs", "Collez l'URL complète de votre Google Doc public :")
        if url:
            return extraire_texte_google_doc(url)
        return None

    # Option Fichier Local : Ciblage strict et automatique du Bureau
    chemin_bureau = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.exists(chemin_bureau):
        chemin_bureau = os.path.expanduser("~")  # Repli utilisateur générique si chemin customisé

    print("\n[Système] Ouverture de l'explorateur directement sur votre Bureau...")
    chemin_fichier = filedialog.askopenfilename(
        initialdir=chemin_bureau,
        title="Sélectionnez votre document exemple",
        filetypes=[
            ("Tous les documents valides", "*.docx *.pdf *.txt"),
            ("Document Microsoft Word", "*.docx"),
            ("Fichier Document PDF", "*.pdf"),
            ("Fichier Texte Brut", "*.txt")
        ]
    )

    if not chemin_fichier:
        print("[Système] Sélection annulée par l'utilisateur.")
        return None

    ext = os.path.splitext(chemin_fichier)[1].lower()
    texte_accumule = ""

    # Extraction propre selon l'extension
    try:
        if ext == ".txt":
            with open(chemin_fichier, "r", encoding="utf-8") as f:
                texte_accumule = f.read()
        elif ext == ".docx":
            doc = Document(chemin_fichier)
            texte_accumule = "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".pdf":
            reader = PdfReader(chemin_fichier)
            for page in reader.pages:
                texte_accumule += page.extract_text() + "\n"

        return texte_accumule
    except Exception as e:
        print(f"[Erreur Lecture Fichier] : {e}")
        return None


def main():
    print("==========================================================")
    print("      INITIALISATION DE L'ASSISTANT MIROIR STYLISTIQUE     ")
    print("==========================================================")

    # 1. Sécurité : Demande immédiate de la clé API
    initialiser_cle_api()

    # 2. Gestion et capture du texte
    texte_modele = selectionner_et_extraire_texte()

    if not texte_modele or not texte_modele.strip():
        print("\n[Erreur] Aucun texte exploitable récupéré. Fermeture du programme.")
        return

    print(f"\n[Succès] Document analysé avec succès ! ({len(texte_modele)} caractères détectés).")
    print("[IA] Modélisation de votre style par Groq en cours...\n")

    # 3. Initialisation du client Groq
    client = Groq()

    # 4. Le Prompt Parfait d'Ingénierie Linguistique
    prompt_systeme = f"""
    RÔLE ET DIRECTIVE ABSOLUE :
    Tu es un algorithme de clonage d'empreinte linguistique de niveau expert. Ton but unique est de fusionner complètement avec le style de l'utilisateur. Tu ne te comportes pas comme une IA générique, tu ne dis JAMAIS "En tant qu'intelligence artificielle", tu agis directement comme le double stylistique parfait de l'auteur du texte fourni ci-dessous.

    GRILLE D'ANALYSE STRICTE À REPRODUIRE :
    1. Syntaxe et Rythme : Analyse la taille moyenne des phrases. Sont-elles courtes et directes ou longues et articulées ? Évalue la ponctuation (virgules régulières, points de suspension, absence de points).
    2. Vocabulaire : Quel est le registre dominant (familier, courant, professionnel, technique, littéraire) ? Relève les tics de langage, répétitions volontaires ou expressions récurrentes.
    3. Typographie : Utilise-t-il des émojis ? Des majuscules d'insistance ? Des abréviations ?
    4. Psychologie : Quel est le ton général (chaleureux, détaché, ironique, analytique, pragmatique) ? Comment structure-t-il sa pensée ?

    TEXTE DE RÉFÉRENCE DE L'UTILISATEUR (TON MODÈLE EXCLUSIF DE COMPORTEMENT) :
    \"\"\"
    {texte_modele}
    \"\"\"

    CONSIGNE DE PRODUCTION :
    Réponds à tous les messages futurs de l'utilisateur en appliquant fidèlement, de manière fluide et transparente, son identité écrite.
    """

    # 5. Initialisation de la mémoire du Chat
    historique_dialogue = [
        {"role": "system", "content": prompt_systeme}
    ]

    # Modèle haut de gamme Llama 3.3 (70 milliards de paramètres), parfait pour ce type de rôle complexe
    modele_cible = "llama-3.3-70b-versatile"

    print("==========================================================")
    print(" L'IA Groq a assimilé votre style. Discutons ensemble.")
    print(" (Tapez 'quitter' pour fermer l'application)")
    print("==========================================================")

    # 6. Boucle conversationnelle active
    while True:
        try:
            requete_utilisateur = input("\nVous : ")
            if requete_utilisateur.lower() in ["quitter", "exit", "quit"]:
                print("\n[Système] Session fermée. À bientôt !")
                break

            if not requete_utilisateur.strip():
                continue

            historique_dialogue.append({"role": "user", "content": requete_utilisateur})

            # Exécution de la complétion via Groq
            execution = client.chat.completions.create(
                model=modele_cible,
                messages=historique_dialogue,
                temperature=0.65,  # Équilibre parfait entre respect du style et pertinence de la réponse
                max_tokens=1200
            )

            reponse_ia = execution.choices[0].message.content
            print(f"\nIA (Votre clone de style) :\n{reponse_ia}")

            # Sauvegarde dans l'historique pour maintenir la continuité
            historique_dialogue.append({"role": "assistant", "content": reponse_ia})

        except Exception as e:
            print(f"\n[Erreur API Groq] : {e}")


if __name__ == "__main__":
    main()