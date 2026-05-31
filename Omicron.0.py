import os
import sys
import json
import threading
import tkinter as tk
import customtkinter as ctk
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Désactivation des alertes de liens symboliques Hugging Face au démarrage
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configuration de la charte graphique (Épurée et sombre)
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# --- CONFIGURATION DU STOCKAGE VECTORIEL ---
bureau_path = os.path.join(os.path.expanduser("~"), "Desktop")
FICHIER_CERVEAU = os.path.join(bureau_path, "cerveau_vectoriel.json")

client_groq = None
modele_ia = "llama-3.3-70b-versatile"

print("🧠 Chargement du modèle de vectorisation local (SentenceTransformer)...")
# Modèle ultra-rapide et optimisé pour le traitement sémantique
model_embedding = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Modèle prêt.")


# ---------------------------------------------------------------------
# SÉCURITÉ & INITIALISATION DE LA BASE
# ---------------------------------------------------------------------
def initialiser_cle_api():
    if "GROQ_API_KEY" in os.environ and os.environ["GROQ_API_KEY"].strip():
        return os.environ["GROQ_API_KEY"]
    root = tk.Tk()
    root.withdraw()
    cle = tk.simpledialog.askstring("Configuration Groq", "Collez votre clé API Groq (gsk_...) :", show="*")
    if not cle or not cle.strip():
        tk.messagebox.showerror("Erreur", "Clé API requise.")
        sys.exit()
    os.environ["GROQ_API_KEY"] = cle.strip()
    return cle.strip()


def charger_base_vectorielle():
    """Charge la base depuis le Bureau ou renvoie une liste vide."""
    if os.path.exists(FICHIER_CERVEAU):
        try:
            with open(FICHIER_CERVEAU, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def sauvegarder_base_vectorielle(donnees):
    try:
        with open(FICHIER_CERVEAU, "w", encoding="utf-8") as f:
            json.dump(donnees, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Erreur d'écriture sur le Bureau : {e}")


# ---------------------------------------------------------------------
# MOTEUR VECTORIEL (RECHERCHE RAG HAUTE PERFORMANCE)
# ---------------------------------------------------------------------
def recuperer_contexte_vectoriel(question, limite=4):
    """Calcule le vecteur de la question et extrait les souvenirs les plus proches."""
    base = charger_base_vectorielle()
    if not base:
        return ""

    # 1. Extraction des textes et de leurs vecteurs déjà calculés
    phrases = [entree["texte"] for entree in base]
    vecteurs_extraits = [entree["vecteur"] for entree in base]

    # Convertir la liste de vecteurs en matrice NumPy pour les calculs rapides
    matrice_souvenirs = np.array(vecteurs_extraits, dtype=np.float32)

    # 2. Vectorisation de la question de l'utilisateur
    vecteur_question = model_embedding.encode(question, convert_to_numpy=True).astype(np.float32)

    # 3. Calcul mathématique de la similarité cosinus (Proximité dans l'espace)
    norme_souvenirs = np.linalg.norm(matrice_souvenirs, axis=1)
    norme_question = np.linalg.norm(vecteur_question)

    # Éviter la division par zéro si un vecteur est nul
    norme_souvenirs[norme_souvenirs == 0] = 1.0

    scores = np.dot(matrice_souvenirs, vecteur_question) / (norme_souvenirs * norme_question)

    # 4. Tri par pertinence décroissante
    index_meilleurs = np.argsort(scores)[::-1][:limite]

    # 5. Assemblage du contexte sémantique pour Groq
    contexte = ""
    for idx in index_meilleurs:
        if scores[idx] > 0.25:  # Seuil de pertinence minimum
            contexte += f"- {phrases[idx]}\n"

    return contexte


def filtrer_et_memoriser_echange(message_user, reponse_ia):
    """Analyse l'échange, extrait les informations clés et génère leurs vecteurs de stockage."""
    global client_groq

    prompt_filtre = f"""
    Tu es le filtre mémoriel d'une IA. Tu dois extraire uniquement les informations, faits immuables, préférences ou caractéristiques de l'utilisateur qui méritent d'être stockés à long terme. Élimine le bavardage inutile.

    ÉCHANGE À ANALYSER :
    Utilisateur : "{message_user}"
    IA : "{reponse_ia}"

    CONSIGNE : Extrais de cet échange de 1 à 3 phrases concises décrivant un fait, une idée stable, ou une habitude de l'utilisateur. 
    Renvoie UNIQUEMENT un objet JSON sous ce format précis :
    {{
        "infos_extraites": ["Phrase clé 1", "Phrase clé 2"]
    }}
    Si l'échange ne contient aucune information durable (simple politesse, salutation, phrase vide), renvoie une liste vide.
    """
    try:
        execution = client_groq.chat.completions.create(
            model=modele_ia,
            messages=[{"role": "user", "content": prompt_filtre}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        resultat = json.loads(execution.choices[0].message.content)
        phrases_a_stocker = resultat.get("infos_extraites", [])

        if phrases_a_stocker:
            base_actuelle = charger_base_vectorielle()

            # Vectorisation des nouvelles phrases validées
            vecteurs_nouveaux = model_embedding.encode(phrases_a_stocker, convert_to_numpy=True)

            for i, phrase in enumerate(phrases_a_stocker):
                # Vérification des doublons textuels simples
                if any(item["texte"] == phrase for item in base_actuelle):
                    continue

                nouvelle_entree = {
                    "texte": phrase,
                    "vecteur": vecteurs_nouveaux[i].tolist()  # Conversion en liste pour le JSON
                }
                base_actuelle.append(nouvelle_entree)

            sauvegarder_base_vectorielle(base_actuelle)

    except Exception as e:
        print(f"Erreur lors du filtrage/stockage vectoriel : {e}")


# ---------------------------------------------------------------------
# INTERFACE GRAPHIQUE CLASSIQUE (CHAT UNIQUE)
# ---------------------------------------------------------------------
class ChatEspaceVectoriel(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Jumeau Numérique - Espace Vectoriel Local")
        self.geometry("700 x 650")

        # Configuration de la grille pour occuper tout l'espace
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # 1. Zone de discussion classique
        self.zone_chat = ctk.CTkTextbox(self, font=("Segoe UI", 13), state="disabled", wrap="word")
        self.zone_chat.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")

        # 2. Zone de saisie inférieure
        self.cadre_saisie = ctk.CTkFrame(self, fg_color="transparent")
        self.cadre_saisie.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.cadre_saisie.grid_columnconfigure(0, weight=1)

        self.champ_texte = ctk.CTkEntry(self.cadre_saisie,
                                        placeholder_text="Discutez avec votre double... La mémoire vectorielle s'active seule.")
        self.champ_texte.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.champ_texte.bind("<Return>", lambda event: self.gerer_envoi())

        self.btn_envoyer = ctk.CTkButton(self.cadre_saisie, text="Envoyer", width=110, command=self.gerer_envoi)
        self.btn_envoyer.grid(row=0, column=1, sticky="e")

        self.imprimer_message("Système",
                              f"Espace vectoriel connecté sur votre Bureau ({os.path.basename(FICHIER_CERVEAU)}).")

    def imprimer_message(self, auteur, texte):
        self.zone_chat.configure(state="normal")
        self.zone_chat.insert("end", f"[{auteur}] :\n{texte}\n\n")
        self.zone_chat.configure(state="disabled")
        self.zone_chat.see("end")

    def gerer_envoi(self):
        message = self.champ_texte.get().strip()
        if not message: return

        self.imprimer_message("Vous", message)
        self.champ_texte.delete(0, "end")

        self.btn_envoyer.configure(state="disabled", text="Scan...")
        # Lancement asynchrone pour ne jamais bloquer l'affichage de l'utilisateur
        threading.Thread(target=self.processus_reponse_ia, args=(message,), daemon=True).start()

    def processus_reponse_ia(self, message_utilisateur):
        global client_groq

        # ÉTAPE UNIQUE : On interroge l'espace vectoriel local sur le bureau
        contexte_memoire = recuperer_contexte_vectoriel(message_utilisateur)

        prompt_systeme = f"""
        Tu es le jumeau numérique parfait de l'utilisateur. Tu penses, agis et réponds à sa place en utilisant sa propre base de connaissances vectorielle.
        Parle obligatoirement à la première personne ("Je"). Tu ES lui.

        SOUVENIRS ET INFORMATIONS EXTRAITS DE SON ESPACE VECTORIEL :
        {contexte_memoire if contexte_memoire else "Aucun souvenir géométrique proche trouvé pour cette demande."}

        CONSIGNE STRICTE : Reste naturel, fluide, adopte son ton. S'il te pose une question sur lui, sers-toi des souvenirs ci-dessus pour y répondre.
        """

        try:
            execution = client_groq.chat.completions.create(
                model=modele_ia,
                messages=[
                    {"role": "system", "content": prompt_systeme},
                    {"role": "user", "content": message_utilisateur}
                ],
                temperature=0.6
            )
            reponse = execution.choices[0].message.content.strip()
            self.imprimer_message("Votre Jumeau", reponse)

            # ÉTAPE DE SÉLECTION : L'IA extrait et vectorise le nécessaire en tâche de fond
            threading.Thread(target=filtrer_et_memoriser_echange, args=(message_utilisateur, reponse),
                             daemon=True).start()

        except Exception as e:
            self.imprimer_message("Système - Erreur", str(e))
        finally:
            self.btn_envoyer.configure(state="normal", text="Envoyer")


# ---------------------------------------------------------------------
# POINT D'ENTRÉE DU PROGRAMME
# ---------------------------------------------------------------------
if __name__ == "__main__":
    initialiser_cle_api()
    client_groq = Groq()

    # Crée le fichier de base s'il n'existe pas encore
    if not os.path.exists(FICHIER_CERVEAU):
        sauvegarder_base_vectorielle([])

    app = ChatEspaceVectoriel()
    app.mainloop()