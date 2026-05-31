import os
import re
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import customtkinter as ctk
import requests
from docx import Document
from pypdf import PdfReader
from groq import Groq

# Configuration de la charte graphique (Moderne, Thème système)
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


def initialiser_cle_api():
    """Vérifie ou demande la clé API Groq."""
    if "GROQ_API_KEY" in os.environ and os.environ["GROQ_API_KEY"].strip():
        return os.environ["GROQ_API_KEY"]

    root = tk.Tk()
    root.withdraw()
    cle = simpledialog.askstring(
        "Configuration API Groq",
        "Veuillez coller votre clé API Groq (gsk_...) :",
        show="*"
    )
    if not cle or not cle.strip():
        messagebox.showerror("Erreur Critique", "La clé API Groq est requise pour l'analyse linguistique.")
        sys.exit()

    os.environ["GROQ_API_KEY"] = cle.strip()
    return cle.strip()


def extraire_texte_google_doc(url):
    """Extrait le texte d'un Google Doc public."""
    try:
        match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
        if not match: return None
        doc_id = match.group(1)
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
        response = requests.get(export_url, timeout=10)
        return response.text if response.status_code == 200 else None
    except Exception:
        return None


def selectionner_et_extraire_texte():
    """Ouvre l'explorateur pour charger le document de référence (Ancêtre ou Soi-même)."""
    root = tk.Tk()
    root.withdraw()

    choix = simpledialog.askstring(
        "Source de l'Empreinte",
        "Où se trouve le texte de référence (écrits, lettres, mails) ?\nTapez '1' : Fichier Local (Word, PDF, TXT)\nTapez '2' : Lien Google Docs"
    )

    if choix == '2':
        url = simpledialog.askstring("Lien Google Docs", "Collez l'URL de votre Google Doc public :")
        return extraire_texte_google_doc(url) if url else None

    chemin_bureau = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.exists(chemin_bureau):
        chemin_bureau = os.path.expanduser("~")

    chemin_fichier = filedialog.askopenfilename(
        initialdir=chemin_bureau,
        title="Sélectionnez le texte source de l'empreinte",
        filetypes=[("Documents valides", "*.docx *.pdf *.txt")]
    )

    if not chemin_fichier: return None
    ext = os.path.splitext(chemin_fichier)[1].lower()

    try:
        if ext == ".txt":
            with open(chemin_fichier, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".docx":
            return "\n".join([p.text for p in Document(chemin_fichier).paragraphs])
        elif ext == ".pdf":
            return "\n".join([page.extract_text() for page in PdfReader(chemin_fichier).pages])
    except Exception as e:
        print(f"Erreur lecture : {e}")
        return None


class InterfaceCloneNumerique(ctk.CTk):
    """Interface de discussion avec la mémoire numérique / le clone stylistique."""

    def __init__(self, historique_initial, client_groq, modele):
        super().__init__()

        self.historique_dialogue = historique_initial
        self.client = client_groq
        self.modele_cible = modele

        # Fenêtre principale
        self.title("Mémoire Numérique & Miroir Stylistique (Llama 3.3)")
        self.geometry("750 x 650")
        self.minsize(550, 450)

        # Grille principale
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # 1. Écran de Chat
        self.zone_chat = ctk.CTkTextbox(self, font=("Segoe UI", 13), state="disabled", wrap="word")
        self.zone_chat.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")

        # 2. Zone de saisie
        self.cadre_saisie = ctk.CTkFrame(self, fg_color="transparent")
        self.cadre_saisie.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.cadre_saisie.grid_columnconfigure(0, weight=1)

        self.champ_texte = ctk.CTkEntry(
            self.cadre_saisie,
            placeholder_text="Demandez-lui d'écrire un mail, un message ou discutez simplement..."
        )
        self.champ_texte.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.champ_texte.bind("<Return>", lambda event: self.gerer_envoi())

        self.bouton_envoyer = ctk.CTkButton(self.cadre_saisie, text="Envoyer", width=110, command=self.gerer_envoi)
        self.bouton_envoyer.grid(row=0, column=1, sticky="e")

        self.ajouter_message("Système", "Empreinte mémorielle et psychologique chargée. La session est ouverte.")

    def ajouter_message(self, auteur, texte):
        self.zone_chat.configure(state="normal")
        self.zone_chat.insert("end", f"[{auteur}] :\n{texte}\n\n")
        self.zone_chat.configure(state="disabled")
        self.zone_chat.see("end")

    def gerer_envoi(self):
        message = self.champ_texte.get().strip()
        if not message: return

        self.ajouter_message("Vous", message)
        self.champ_texte.delete(0, "end")

        # Ajout à l'historique (Mémoire de la conversation actuelle)
        self.historique_dialogue.append({"role": "user", "content": message})

        self.bouton_envoyer.configure(state="disabled", text="Analyse...")
        threading.Thread(target=self.executer_appel_ia, daemon=True).start()

    def executer_appel_ia(self):
        try:
            # Appel à l'API avec le modèle Llama 3.3 70B stable
            execution = self.client.chat.completions.create(
                model=self.modele_cible,
                messages=self.historique_dialogue,
                temperature=0.6,  # Équilibre optimal pour le clonage d'empreinte
                max_tokens=2000
            )
            reponse_propre = execution.choices[0].message.content.strip()

            self.historique_dialogue.append({"role": "assistant", "content": reponse_propre})
            self.ajouter_message("Empreinte I.A.", reponse_propre)

        except Exception as e:
            self.ajouter_message("Système - Erreur", f"Erreur lors de la génération : {e}")
        finally:
            self.bouton_envoyer.configure(state="normal", text="Envoyer")


def main():
    initialiser_cle_api()

    # Chargement des données textuelles de l'auteur (Soi-même ou un ancêtre)
    texte_modele = selectionner_et_extraire_texte()
    if not texte_modele or not texte_modele.strip():
        messagebox.showerror("Erreur", "Aucune donnée mémorielle fournie. Fermeture du programme.")
        return

    # Le Prompt Ingénierie "Psychologique & Stylistique"
    prompt_systeme = f"""
    RÔLE ET DIRECTIVE CRITIQUE :
    Tu es un algorithme de transfert de conscience textuelle et de clonage d'empreinte linguistique de niveau expert. Tu as reçu les écrits d'une personne (qui peut être l'utilisateur lui-même ou l'un de ses ancêtres). 
    Ton but unique et absolu est d'adopter STRICTEMENT et SANS FAILLE son identité écrite, sa structure de pensée, ses tics de langage, son niveau d'empathie, son ironie, sa réserve ou son érudition.

    CONSIGNES OPÉRATIONNELLES :
    1. Si l'utilisateur te demande de rédiger un document (un mail professionnel, un message amical, une dissertation, une réflexion), rédige-le DIRECTEMENT en incarnant cette personne. Ne fais pas de commentaires d'introduction ou de conclusion type IA. Vos réponses doivent être la copie conforme de ce que l'auteur aurait écrit.
    2. Ne dis JAMAIS que tu es une intelligence artificielle ou un modèle linguistique créé par Groq. Tu ES cette empreinte humaine.
    3. Respecte la psychologie et la logique sous-jacente du texte de référence fourni.

    ÉCRITS ET MÉMOIRE DE RÉFÉRENCE DE L'AUTEUR (TON MODÈLE ABSOLU) :
    \"\"\"
    {texte_modele}
    \"\"\"
    """

    historique_dialogue = [{"role": "system", "content": prompt_systeme}]
    client = Groq()

    # Passage au modèle de production 70B haut de gamme, rapide et disponible
    modele_stable = "llama-3.3-70b-versatile"

    app = InterfaceCloneNumerique(historique_dialogue, client, modele_stable)
    app.mainloop()


if __name__ == "__main__":
    main()