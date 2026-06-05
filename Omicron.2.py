import os
import json
import uuid
import threading
from datetime import datetime
import customtkinter as ctk
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from groq import Groq

# =====================================================================
# CONFIGURATION ABSOLUE & SECURITE
# =====================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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").strip()
MODEL_IA = "llama-3.3-70b-versatile"

# Dossier système localisé pour la persistance de Qdrant
DOSSIER_SYSTEME = os.path.join(os.path.expanduser("~"), ".omega_monolithe")
os.makedirs(DOSSIER_SYSTEME, exist_ok=True)
CHEMIN_QDRANT = os.path.join(DOSSIER_SYSTEME, "memoire_qdrant")


# =====================================================================
# MODULE 1 : LA MEMOIRE VECTORIELLE (QDRANT EMBEDDED)
# =====================================================================
class MemoireVectorielle:
    def __init__(self):
        print("[SYSTEM] Initialisation de la base vectorielle locale...")
        self.client = QdrantClient(path=CHEMIN_QDRANT)
        self.encodeur = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = "omega_archives"

        # Vérification et création stricte de la collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )
            print("[SYSTEM] Nouvelle matrice de stockage créée avec succès.")

    def rechercher_contexte(self, requete, limite=4):
        """Recherche sémantique corrigée pour éviter l'erreur d'attribut."""
        vecteur = self.encodeur.encode([requete]).tolist()[0]
        try:
            # Utilisation de query_points, la méthode universelle et moderne de Qdrant
            resultats = self.client.query_points(
                collection_name=self.collection,
                query=vecteur,
                limit=limite
            )
            # Filtrage sur le score (query_points retourne des objets avec 'score' et 'payload')
            return [r.payload for r in resultats.points if r.score > 0.45]
        except Exception as e:
            print(f"[ERREUR LECTURE MÉMOIRE] {e}")
            return []

    def graver_souvenir(self, fait, categorie):
        """Inscription immuable d'un fait dans la base avec UUID unique."""
        if not fait or len(fait.strip()) < 5:
            return

        vecteur = self.encodeur.encode([fait]).tolist()[0]
        id_unique = str(uuid.uuid4())
        horodatage = datetime.now().isoformat()

        payload = {
            "texte": fait,
            "categorie": categorie,
            "date": horodatage
        }

        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[models.PointStruct(id=id_unique, vector=vecteur, payload=payload)]
            )
            print(f"[ARCHIVAGE REUSSI] {categorie} : {fait}")
        except Exception as e:
            print(f"[ERREUR ECRITURE MÉMOIRE] {e}")

    def fermer(self):
        """Ferme proprement le client pour éviter les alertes de deallocator."""
        try:
            self.client.close()
        except:
            pass


# =====================================================================
# MODULE 2 : LE MOTEUR COGNITIF SEQUENTIEL
# =====================================================================
class MoteurCognitif:
    def __init__(self, memoire):
        self.llm = Groq(api_key=GROQ_API_KEY)
        self.memoire = memoire
        self.historique_session = []

    def generer_discussion(self, message_utilisateur):
        souvenirs = self.memoire.rechercher_contexte(message_utilisateur)
        contexte_str = "\n".join([f"- {s['texte']} (Catégorie: {s['categorie']})" for s in souvenirs if s])

        prompt_systeme = f"""
        Tu es OMEGA, l'alter-ego numérique de l'utilisateur. Tu es rationnel, précis et direct.

        SOUVENIRS ABSOLUS EXTRAITS DE TA BASE DE DONNÉES :
        {contexte_str if contexte_str else "Aucun souvenir spécifique lié à cette question."}

        Utilise ces souvenirs pour prouver que tu connais l'utilisateur, mais réponds de manière conversationnelle.
        """

        messages = [{"role": "system", "content": prompt_systeme}]
        messages.extend(self.historique_session)
        messages.append({"role": "user", "content": message_utilisateur})

        reponse = self.llm.chat.completions.create(
            model=MODEL_IA,
            messages=messages,
            temperature=0.5
        )
        texte_reponse = reponse.choices[0].message.content.strip()

        self.historique_session.append({"role": "user", "content": message_utilisateur})
        self.historique_session.append({"role": "assistant", "content": texte_reponse})
        if len(self.historique_session) > 8:
            self.historique_session = self.historique_session[-8:]

        return texte_reponse

    def analyser_et_archiver(self, message_utilisateur, reponse_ia):
        prompt_extraction = f"""
        Analyse l'échange suivant de manière clinique. Ton objectif est d'extraire de nouveaux faits immuables sur l'utilisateur (préférences, travail, famille, habitudes).
        Ignore les plaisanteries ou les informations temporaires.

        USER: "{message_utilisateur}"
        OMEGA: "{reponse_ia}"

        Renvoie STRICTEMENT un objet JSON avec cette structure (liste vide si rien n'est pertinent) :
        {{
            "faits": [
                {{"fait": "L'utilisateur aime le langage Python", "categorie": "Compétence technique"}}
            ]
        }}
        """

        try:
            reponse = self.llm.chat.completions.create(
                model=MODEL_IA,
                messages=[{"role": "user", "content": prompt_extraction}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            donnees = json.loads(reponse.choices[0].message.content)
            for item in donnees.get("faits", []):
                fait = item.get("fait")
                cat = item.get("categorie", "Général")
                if fait:
                    self.memoire.graver_souvenir(fait, cat)
        except Exception as e:
            print(f"[ECHEC EXTRACTION JSON] {e}")


# =====================================================================
# MODULE 3 : INTERFACE UTILISATEUR ET SYNCHRONISATION
# =====================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


class AppMonolithe(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("OMEGA - Architecture Séquentielle Stricte")
        self.geometry("900x700")

        self.boite_dialogue = ctk.CTkTextbox(self, font=("Consolas", 14), state="disabled", fg_color="#0F0F0F",
                                             text_color="#D4D4D4")
        self.boite_dialogue.pack(fill="both", expand=True, padx=20, pady=20)

        self.zone_saisie = ctk.CTkFrame(self, fg_color="transparent")
        self.zone_saisie.pack(fill="x", padx=20, pady=(0, 20))

        self.champ_texte = ctk.CTkEntry(self.zone_saisie, placeholder_text="Initialisation...", height=45,
                                        state="disabled")
        self.champ_texte.pack(side="left", fill="x", expand=True, padx=(0, 15))
        self.champ_texte.bind("<Return>", lambda e: self.enclencher_cycle())

        self.statut_label = ctk.CTkLabel(self.zone_saisie, text="HORS LIGNE", font=("Consolas", 12, "bold"),
                                         text_color="#FF0000", width=120)
        self.statut_label.pack(side="right")

        self.memoire = None
        self.moteur = None
        threading.Thread(target=self.demarrer_systemes, daemon=True).start()

        # Protocole de fermeture propre
        self.protocol("WM_DELETE_WINDOW", self.quitter_proprement)

    def demarrer_systemes(self):
        self.after(0, lambda: self.maj_chat("SYSTEM", "Connexion aux noyaux Qdrant et Groq..."))
        self.memoire = MemoireVectorielle()
        self.moteur = MoteurCognitif(self.memoire)
        self.after(0, self.activer_interface)

    def activer_interface(self):
        self.champ_texte.configure(state="normal", placeholder_text="Entrez votre message...")
        self.statut_label.configure(text="PRÊT", text_color="#00FF00")
        self.maj_chat("SYSTEM", "Séquence d'archivage garantie active.")

    def maj_chat(self, auteur, texte, couleur="#D4D4D4"):
        self.boite_dialogue.configure(state="normal")
        self.boite_dialogue.insert("end", f"[{auteur}] ", "auteur")
        self.boite_dialogue.insert("end", f"{texte}\n\n", "texte")
        if auteur == "VOUS":
            self.boite_dialogue.tag_config("auteur", foreground="#00A8FF")
        elif auteur == "OMEGA":
            self.boite_dialogue.tag_config("auteur", foreground="#FFB000")
        else:
            self.boite_dialogue.tag_config("auteur", foreground="#00FF00")
        self.boite_dialogue.tag_config("texte", foreground=couleur)
        self.boite_dialogue.configure(state="disabled")
        self.boite_dialogue.see("end")

    def modifier_statut(self, texte, couleur):
        self.statut_label.configure(text=texte, text_color=couleur)

    def enclencher_cycle(self):
        msg = self.champ_texte.get().strip()
        if not msg: return

        self.champ_texte.delete(0, "end")
        self.champ_texte.configure(state="disabled")
        self.maj_chat("VOUS", msg)

        threading.Thread(target=self.executer_sequence_stricte, args=(msg,), daemon=True).start()

    def executer_sequence_stricte(self, msg):
        # PHASE 1 : Raisonnement et Discussion
        self.after(0, lambda: self.modifier_statut("RÉFLEXION...", "#FFA500"))
        reponse = self.moteur.generer_discussion(msg)
        self.after(0, lambda: self.maj_chat("OMEGA", reponse))

        # PHASE 2 : Analyse et Archivage
        self.after(0, lambda: self.modifier_statut("ARCHIVAGE...", "#00A8FF"))
        self.moteur.analyser_et_archiver(msg, reponse)

        # FIN DU CYCLE
        self.after(0, lambda: self.modifier_statut("PRÊT", "#00FF00"))
        self.after(0, lambda: self.champ_texte.configure(state="normal"))
        self.after(0, lambda: self.champ_texte.focus())

    def quitter_proprement(self):
        if self.memoire:
            self.memoire.fermer()
        self.destroy()


if __name__ == "__main__":
    app = AppMonolithe()
    app.mainloop()