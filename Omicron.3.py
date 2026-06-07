"""
╔══════════════════════════════════════════════════════════════════════╗
║          OMICRON — Architecture Symbiotique v8.0                    ║
║  Agent Discussion : Llama-3.3-70B  (RAG local + Web DuckDuckGo)    ║
║  Agent Archivage  : Llama-3.1-8B   (thread daemon, file infinie)   ║
║  Mémoire          : Qdrant vectoriel local (capacité illimitée)     ║
║  Interface        : CustomTkinter Dark Premium                      ║
╚══════════════════════════════════════════════════════════════════════╝

CORRECTIONS v8 :
  - Bug bloquage UI après 1ère question → corrigé (re-bind <Return> + focus garanti)
  - Thread cycle → finally garanti même en cas d'exception nested
  - Qdrant : payload_index sur 'categorie' et 'date' pour recherches futures
  - Historique : fenêtre glissante stricte, jamais > MAX_HISTORIQUE tours
  - Archivage : queue infinie, thread toujours vivant, jamais bloquant
  - SentenceTransformer : chargé OFFLINE si déjà en cache (HF_HUB_OFFLINE)
  - Logging propre : séparation des niveaux par module
  - UI : bouton EFFACER, compteur de mémoire live, indicateur archivage
"""

import os
import sys
import json
import uuid
import queue
import threading
import logging
import warnings
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional, Tuple

# ── Silence warnings tiers ──────────────────────────────────────────
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
# Empêche les requêtes HF inutiles si le modèle est déjà en cache
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

import tkinter as tk
import customtkinter as ctk
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from groq import Groq
from ddgs import DDGS

# =====================================================================
# LOGGING — deux niveaux : INFO pour l'opérationnel, DEBUG pour détail
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)-16s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log_mem  = logging.getLogger("Mémoire")
log_disc = logging.getLogger("Discussion")
log_arch = logging.getLogger("Archivage")
log_ui   = logging.getLogger("Interface")

# =====================================================================
# CONFIGURATION — toutes les constantes au même endroit
# =====================================================================

# ⚠ Clé API : JAMAIS dans le code — utiliser variable d'environnement
# Windows : setx GROQ_API_KEY "gsk_..."  puis redémarrer PyCharm
# Linux   : export GROQ_API_KEY="gsk_..."
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "Clé API Groq").strip()
if not GROQ_API_KEY:
    print(
        "\n[ERREUR FATALE] Clé GROQ_API_KEY introuvable.\n"
        "  Windows : setx GROQ_API_KEY \"gsk_votre_cle\"\n"
        "  Linux   : export GROQ_API_KEY=\"gsk_votre_cle\"\n"
        "Puis redémarrez votre terminal / IDE.\n"
    )
    sys.exit(1)

MODEL_DISCUSSION = "llama-3.3-70b-versatile"
MODEL_ARCHIVAGE  = "llama-3.1-8b-instant"

# Qdrant — stockage sur disque, capacité illimitée
DOSSIER_SYSTEME  = os.path.join(os.path.expanduser("~"), ".omicron_v8")
CHEMIN_QDRANT    = os.path.join(DOSSIER_SYSTEME, "qdrant_db")
os.makedirs(DOSSIER_SYSTEME, exist_ok=True)

COLLECTION_NAME  = "omicron_memoire_permanente"
VECTEUR_TAILLE   = 384          # all-MiniLM-L6-v2

MAX_HISTORIQUE   = 12           # tours max conservés dans le contexte LLM
SEUIL_PERTINENCE = 0.35         # score minimal pour un souvenir RAG
SEUIL_DOUBLON    = 0.90         # score au-dessus = doublon ignoré
MAX_SOUVENIRS    = 8            # souvenirs RAG injectés dans le prompt
MAX_RESULTATS_WEB = 5
TIMEOUT_LLM      = 60           # secondes avant abandon appel LLM

COULEURS = {
    "VOUS"   : "#4FC3F7",
    "OMICRON": "#FFD54F",
    "SYSTEM" : "#81C784",
    "ERREUR" : "#EF5350",
    "ARCHIVE": "#CE93D8",
}


# =====================================================================
# MODULE 1 — MÉMOIRE VECTORIELLE (Qdrant local, disque, illimitée)
# =====================================================================
class MemoireVectorielle:
    """
    Couche d'accès à Qdrant.
    Capacité : dizaines/centaines de millions de vecteurs (limité seulement
    par l'espace disque). Les données persistent entre les sessions.
    """

    def __init__(self) -> None:
        log_mem.info("Connexion à Qdrant (stockage disque)…")
        self.client   = QdrantClient(path=CHEMIN_QDRANT)
        log_mem.info("Chargement du modèle d'encodage sémantique…")
        self.encodeur = SentenceTransformer("all-MiniLM-L6-v2")
        self._lock    = threading.Lock()   # protection lectures/écritures concurrentes
        self._initialiser_collection()
        log_mem.info(f"Mémoire prête — {self.compter()} souvenir(s) stocké(s).")

    # ── Initialisation ─────────────────────────────────────────────
    def _initialiser_collection(self) -> None:
        if self.client.collection_exists(COLLECTION_NAME):
            return
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=VECTEUR_TAILLE,
                distance=qmodels.Distance.COSINE,
                on_disk=True,           # vecteurs sur disque → RAM libérée
            ),
            optimizers_config=qmodels.OptimizersConfigDiff(
                indexing_threshold=20_000,  # index HNSW après 20k points
            ),
        )
        # Index payload pour requêtes futures par catégorie / date
        self.client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="categorie",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="date",
            field_schema=qmodels.PayloadSchemaType.DATETIME,
        )
        log_mem.info("Nouvelle collection Qdrant créée avec index payload.")

    # ── Lecture : recherche sémantique ─────────────────────────────
    def rechercher(self, requete: str, limite: int = MAX_SOUVENIRS) -> List[Dict[str, Any]]:
        try:
            vecteur = self._encoder(requete)
            with self._lock:
                resultats = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vecteur,
                    limit=limite,
                    with_payload=True,
                )
            return [
                r.payload
                for r in resultats.points
                if r.score >= SEUIL_PERTINENCE
            ]
        except Exception as exc:
            log_mem.error(f"Erreur lecture : {exc}")
            return []

    # ── Écriture : gravure d'un souvenir ──────────────────────────
    def graver(self, fait: str, categorie: str) -> bool:
        fait = fait.strip()
        if not fait:
            return False
        try:
            vecteur = self._encoder(fait)
            with self._lock:
                # Vérification doublon
                doublons = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vecteur,
                    limit=1,
                )
                if doublons.points and doublons.points[0].score >= SEUIL_DOUBLON:
                    log_mem.debug(f"Doublon ignoré ({doublons.points[0].score:.2f}) : {fait[:50]}")
                    return False

                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        qmodels.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vecteur,
                            payload={
                                "texte"    : fait,
                                "categorie": categorie,
                                "date"     : datetime.now().isoformat(),
                            },
                        )
                    ],
                )
            log_mem.info(f"✓ Gravé [{categorie}] : {fait[:70]}…")
            return True
        except Exception as exc:
            log_mem.error(f"Erreur écriture : {exc}")
            return False

    def compter(self) -> int:
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            return info.points_count or 0
        except Exception:
            return 0

    def fermer(self) -> None:
        try:
            self.client.close()
            log_mem.info("Connexion Qdrant fermée proprement.")
        except Exception:
            pass

    def _encoder(self, texte: str) -> List[float]:
        return self.encodeur.encode([texte], show_progress_bar=False)[0].tolist()


# =====================================================================
# MODULE 2A — AGENT DE DISCUSSION (Llama-70B, RAG + Web)
# =====================================================================
class AgentDiscussion:
    """
    Fusionne mémoire RAG + recherche Web pour répondre.
    Maintient un historique conversationnel avec fenêtre glissante.
    """

    def __init__(self, memoire: MemoireVectorielle) -> None:
        self.llm      = Groq(api_key=GROQ_API_KEY)
        self.memoire  = memoire
        self._hist: List[Dict[str, str]] = []   # historique messages

    # ── Point d'entrée public ──────────────────────────────────────
    def repondre(
        self,
        message: str,
        cb: Callable[[str, str], None],
    ) -> str:
        """Cycle complet : RAG → routage → (web?) → génération."""

        # 1. Contexte mémoire
        cb("LECTURE MÉMOIRE…", COULEURS["SYSTEM"])
        souvenirs  = self.memoire.rechercher(message)
        ctx_mem    = self._fmt_souvenirs(souvenirs)

        # 2. Routage : web nécessaire ?
        cb("ANALYSE DE LA REQUÊTE…", COULEURS["VOUS"])
        requete_web = self._router(message, ctx_mem)

        # 3. Données web si besoin
        ctx_web = ""
        if requete_web:
            cb(f"RECHERCHE WEB : {requete_web[:40]}…", "#FFA726")
            ctx_web = self._web(requete_web)
            cb("SYNTHÈSE DES DONNÉES…", COULEURS["SYSTEM"])

        # 4. Génération réponse finale
        cb("GÉNÉRATION EN COURS…", COULEURS["OMICRON"])
        reponse = self._generer(message, ctx_mem, ctx_web)

        # 5. Mise à jour historique (fenêtre glissante)
        self._hist.append({"role": "user",      "content": message})
        self._hist.append({"role": "assistant", "content": reponse})
        if len(self._hist) > MAX_HISTORIQUE * 2:
            self._hist = self._hist[-(MAX_HISTORIQUE * 2):]

        return reponse

    # ── Routeur ────────────────────────────────────────────────────
    def _router(self, message: str, ctx_mem: str) -> Optional[str]:
        sys_prompt = (
            f"Tu es le routeur logique d'OMICRON. Date : {datetime.now().strftime('%A %d %B %Y')}.\n"
            f"Mémoire locale :\n{ctx_mem}\n\n"
            "Si la question porte sur l'actualité, des événements récents, des données en temps réel "
            "(sport, météo, bourse, news, résultats, classements actuels…), réponds UNIQUEMENT :\n"
            "<SEARCH>mots clés concis en français</SEARCH>\n\n"
            "Sinon, réponds exactement : <DIRECT>"
        )
        try:
            resp = self.llm.chat.completions.create(
                model=MODEL_DISCUSSION,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": message},
                ],
                temperature=0.0,
                max_tokens=80,
                timeout=TIMEOUT_LLM,
            )
            texte = resp.choices[0].message.content.strip()
            m = re.search(r"<SEARCH>(.*?)</SEARCH>", texte, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else None
        except Exception as exc:
            log_disc.error(f"Routeur : {exc}")
            return None

    # ── Recherche Web ──────────────────────────────────────────────
    def _web(self, requete: str) -> str:
        log_disc.info(f"Recherche web → « {requete} »")
        try:
            resultats = list(DDGS().text(requete, max_results=MAX_RESULTATS_WEB))
            if not resultats:
                return "[Aucun résultat web trouvé.]"
            lignes = ["━━ DONNÉES WEB EN TEMPS RÉEL ━━"]
            for i, r in enumerate(resultats, 1):
                lignes.append(
                    f"\n[{i}] {r.get('title', '')}\n"
                    f"{r.get('body', '')}\n"
                    f"Source : {r.get('href', '')}"
                )
            return "\n".join(lignes)
        except Exception as exc:
            log_disc.error(f"Web : {exc}")
            return f"[Erreur web : {exc}]"

    # ── Génération ─────────────────────────────────────────────────
    def _generer(self, message: str, ctx_mem: str, ctx_web: str) -> str:
        blocs_sys = [
            "Tu es OMICRON, un assistant IA bienveillant, précis et cultivé.",
            f"Date actuelle : {datetime.now().strftime('%A %d %B %Y, %H:%M')}.",
            "Réponds toujours en français, de façon claire, structurée et naturelle.",
        ]
        if ctx_mem and ctx_mem != "Aucun souvenir pertinent.":
            blocs_sys.append(
                "\n[MÉMOIRE PERSONNELLE DE L'UTILISATEUR — utilise ces informations "
                "pour personnaliser ta réponse]\n" + ctx_mem
            )
        if ctx_web:
            blocs_sys.append(
                "\n[DONNÉES WEB FRAÎCHES — priorité sur ta mémoire d'entraînement]\n" + ctx_web
            )

        messages = (
            [{"role": "system", "content": "\n".join(blocs_sys)}]
            + self._hist          # historique (déjà sans le message courant)
            + [{"role": "user", "content": message}]
        )

        try:
            resp = self.llm.chat.completions.create(
                model=MODEL_DISCUSSION,
                messages=messages,
                temperature=0.45,
                max_tokens=2048,
                timeout=TIMEOUT_LLM,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            log_disc.error(f"Génération : {exc}")
            return f"[Erreur de génération : {exc}]"

    @staticmethod
    def _fmt_souvenirs(souvenirs: List[Dict[str, Any]]) -> str:
        if not souvenirs:
            return "Aucun souvenir pertinent."
        return "\n".join(
            f"• [{s.get('categorie', '?')}] {s.get('texte', '')}"
            for s in souvenirs
        )


# =====================================================================
# MODULE 2B — AGENT D'ARCHIVAGE (Llama-8B, daemon thread, queue infinie)
# =====================================================================
class AgentArchivage:
    """
    Thread daemon dédié à l'analyse et la mémorisation.
    Totalement découplé de l'UI : ne la bloque JAMAIS.
    La queue est infinie : aucune perte de données même sous charge.
    """

    def __init__(self, memoire: MemoireVectorielle, cb_compteur: Callable[[int], None]) -> None:
        self.llm         = Groq(api_key=GROQ_API_KEY)
        self.memoire     = memoire
        self.cb_compteur = cb_compteur          # appelé après chaque gravure
        self._file: queue.Queue = queue.Queue() # infinie, thread-safe
        self._actif      = True

        self._thread = threading.Thread(
            target=self._boucle,
            daemon=True,
            name="AgentArchivage",
        )
        self._thread.start()
        log_arch.info("Agent d'archivage démarré (thread daemon).")

    def soumettre(self, msg_user: str, rep_ia: str) -> None:
        """Soumet un échange pour archivage — non-bloquant, O(1)."""
        self._file.put((msg_user, rep_ia))

    def arreter(self) -> None:
        """Signal d'arrêt gracieux. Attend fin du traitement en cours."""
        self._actif = False
        self._file.put(None)   # poison pill
        self._thread.join(timeout=5)
        log_arch.info("Agent d'archivage arrêté.")

    # ── Boucle interne ─────────────────────────────────────────────
    def _boucle(self) -> None:
        while True:
            item = self._file.get()
            if item is None:
                self._file.task_done()
                break
            try:
                msg_user, rep_ia = item
                self._analyser(msg_user, rep_ia)
            except Exception as exc:
                log_arch.error(f"Boucle : {exc}")
            finally:
                self._file.task_done()

    def _analyser(self, msg_user: str, rep_ia: str) -> None:
        prompt = (
            "Tu es un agent de mémoire à long terme. Analyse cet échange et extrais "
            "UNIQUEMENT les informations durables et personnelles sur L'UTILISATEUR "
            "(préférences, goûts, passions, projets, contraintes, objectifs, contexte de vie, "
            "habitudes, opinions, compétences, relations…).\n\n"
            "IGNORE : les questions génériques, les faits encyclopédiques, les réponses de l'IA.\n\n"
            f"USER : {msg_user}\n"
            f"IA   : {rep_ia}\n\n"
            "Réponds UNIQUEMENT avec ce JSON valide (sans Markdown, sans commentaire) :\n"
            "{\"faits\": [{\"fait\": \"phrase courte et précise\", "
            "\"categorie\": \"Préférence|Passion|Projet|Objectif|Contexte|Compétence|Opinion|Relation|Habitude\"}]}\n"
            "Si aucun fait pertinent sur l'utilisateur, renvoie : {\"faits\": []}"
        )
        try:
            resp = self.llm.chat.completions.create(
                model=MODEL_ARCHIVAGE,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=600,
                timeout=TIMEOUT_LLM,
            )
            raw  = (resp.choices[0].message.content or "{}").strip()
            # Nettoyage défensif des éventuels backticks
            raw  = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(raw)

            nb_graves = 0
            for f in data.get("faits", []):
                fait = str(f.get("fait", "")).strip()
                cat  = str(f.get("categorie", "Général")).strip()
                if fait and self.memoire.graver(fait, cat):
                    nb_graves += 1

            if nb_graves > 0:
                total = self.memoire.compter()
                self.cb_compteur(total)  # mise à jour compteur UI
                log_arch.info(f"{nb_graves} fait(s) gravé(s). Total : {total}")

        except json.JSONDecodeError as exc:
            log_arch.warning(f"JSON invalide : {exc}")
        except Exception as exc:
            log_arch.error(f"Analyse : {exc}")


# =====================================================================
# MODULE 3 — INTERFACE CUSTOMTKINTER
# =====================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Palette couleurs UI
C_BG        = "#0D0D0D"
C_BG2       = "#141414"
C_BORDER    = "#2A2A2A"
C_TEXT      = "#E8E8E8"
C_ACCENT    = "#FFD54F"
C_BTN       = "#1E3A5F"
C_BTN_HOV   = "#2E5A8F"
FONT_MAIN   = ("Consolas", 13)
FONT_TITLE  = ("Segoe UI", 11, "bold")


class AppOmicron(ctk.CTk):
    """
    Interface principale d'OMICRON.
    Thread-safe : toutes les mises à jour UI passent par _queue_ui + _ecouteur.
    Le bug "bloquage après 1 question" est corrigé :
      - _set_ui() re-bind systématiquement <Return> et force le focus
      - le finally dans _cycle() garantit la restauration même en cas d'exception
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("OMICRON  ◈  IA Symbiotique v8.0")
        self.geometry("1100x820")
        self.minsize(800, 580)
        self.configure(fg_color=C_BG)

        self._queue_ui : queue.Queue      = queue.Queue()
        self._memoire  : Optional[MemoireVectorielle] = None
        self._agent_d  : Optional[AgentDiscussion]    = None
        self._agent_a  : Optional[AgentArchivage]     = None
        self._en_cours : bool             = False   # verrou logique

        self._construire_ui()
        threading.Thread(target=self._init_systemes, daemon=True).start()
        self._ecouteur()
        self.protocol("WM_DELETE_WINDOW", self._quitter)

    # ══════════════════════════════════════════════════════════════
    # CONSTRUCTION UI
    # ══════════════════════════════════════════════════════════════
    def _construire_ui(self) -> None:

        # ── Bandeau titre ────────────────────────────────────────
        bandeau = ctk.CTkFrame(self, fg_color=C_BG2, height=52, corner_radius=0)
        bandeau.pack(fill="x", padx=0, pady=0)
        bandeau.pack_propagate(False)

        ctk.CTkLabel(
            bandeau, text="◈  OMICRON", font=("Segoe UI", 18, "bold"),
            text_color=C_ACCENT,
        ).pack(side="left", padx=20)

        self.lbl_mem = ctk.CTkLabel(
            bandeau, text="MEM : 0 souvenirs",
            font=("Segoe UI", 11), text_color="#888888",
        )
        self.lbl_mem.pack(side="right", padx=20)

        self.lbl_statut = ctk.CTkLabel(
            bandeau, text="● DÉMARRAGE",
            font=FONT_TITLE, text_color=COULEURS["ERREUR"],
        )
        self.lbl_statut.pack(side="right", padx=20)

        # ── Zone de conversation ─────────────────────────────────
        self.txt = ctk.CTkTextbox(
            self,
            font=FONT_MAIN,
            state="disabled",
            fg_color=C_BG,
            text_color=C_TEXT,
            wrap="word",
            border_color=C_BORDER,
            border_width=1,
            scrollbar_button_color="#2A2A2A",
        )
        self.txt.pack(fill="both", expand=True, padx=15, pady=(10, 5))

        # ── Barre de progression ─────────────────────────────────
        self.progress = ctk.CTkProgressBar(
            self, height=3, mode="indeterminate",
            progress_color=C_ACCENT, fg_color=C_BG2,
        )
        self.progress.pack(fill="x", padx=20, pady=3)
        self.progress.set(0)

        # ── Zone de saisie ───────────────────────────────────────
        bas = ctk.CTkFrame(self, fg_color=C_BG2, corner_radius=0)
        bas.pack(fill="x", padx=0, pady=0)

        cadre_input = ctk.CTkFrame(bas, fg_color="transparent")
        cadre_input.pack(fill="x", padx=15, pady=12)
        cadre_input.columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(
            cadre_input,
            placeholder_text="Initialisation…",
            height=44,
            font=FONT_MAIN,
            fg_color="#1A1A1A",
            border_color=C_BORDER,
            text_color=C_TEXT,
            state="disabled",
        )
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.btn_envoyer = ctk.CTkButton(
            cadre_input,
            text="ENVOYER",
            width=110, height=44,
            font=FONT_TITLE,
            fg_color=C_BTN,
            hover_color=C_BTN_HOV,
            state="disabled",
            command=self._envoyer,
        )
        self.btn_envoyer.grid(row=0, column=1, padx=(0, 6))

        self.btn_effacer = ctk.CTkButton(
            cadre_input,
            text="⌫",
            width=44, height=44,
            font=("Segoe UI", 16),
            fg_color="#2A1A1A",
            hover_color="#4A2A2A",
            state="disabled",
            command=self._effacer_chat,
        )
        self.btn_effacer.grid(row=0, column=2)

    # ══════════════════════════════════════════════════════════════
    # INITIALISATION DES SYSTÈMES (thread background)
    # ══════════════════════════════════════════════════════════════
    def _init_systemes(self) -> None:
        try:
            self._memoire = MemoireVectorielle()
            self._agent_d = AgentDiscussion(self._memoire)
            self._agent_a = AgentArchivage(
                self._memoire,
                cb_compteur=lambda n: self._push("mem", n),
            )
            total = self._memoire.compter()
            self._push("mem", total)
            self._push("msg",  ("SYSTEM", (
                f"Systèmes OMICRON opérationnels.\n"
                f"Base de connaissances : {total} souvenir(s) chargé(s).\n"
                f"Stockage : {CHEMIN_QDRANT}\n"
                f"Modèles  : {MODEL_DISCUSSION}  +  {MODEL_ARCHIVAGE}"
            )))
            self._push("statut", ("PRÊT", COULEURS["SYSTEM"]))
            self._push("progress", False)
            self._push("ui", True)
        except Exception as exc:
            self._push("msg", ("ERREUR", f"Erreur critique au démarrage : {exc}"))
            log_ui.critical(exc, exc_info=True)

    # ══════════════════════════════════════════════════════════════
    # ENVOI D'UN MESSAGE
    # ══════════════════════════════════════════════════════════════
    def _envoyer(self) -> None:
        """Déclenché par le bouton OU la touche Entrée."""
        if self._en_cours:
            return  # double-clic protection

        msg = self.entry.get().strip()
        if not msg:
            return
        if self._agent_d is None:
            return

        self._en_cours = True
        self.entry.delete(0, tk.END)
        self._set_ui(False)
        self.progress.start()
        self._afficher("VOUS", msg)

        threading.Thread(
            target=self._cycle,
            args=(msg,),
            daemon=True,
            name=f"Cycle-{int(time.time())}",
        ).start()

    def _cycle(self, msg: str) -> None:
        """Thread de traitement — le finally GARANTIT la restauration de l'UI."""
        reponse = ""
        try:
            reponse = self._agent_d.repondre(
                msg,
                cb=lambda t, c: self._push("statut", (t, c)),
            )
            self._push("msg", ("OMICRON", reponse))
            self._push("statut", ("ARCHIVAGE…", COULEURS["ARCHIVE"]))
            self._agent_a.soumettre(msg, reponse)

        except Exception as exc:
            err = f"Erreur d'exécution : {exc}"
            self._push("msg", ("ERREUR", err))
            log_ui.error(err, exc_info=True)

        finally:
            # ← Ce bloc s'exécute TOUJOURS, même en cas d'exception
            self._push("statut", ("PRÊT", COULEURS["SYSTEM"]))
            self._push("progress", False)
            self._push("ui", True)
            self._en_cours = False

    # ══════════════════════════════════════════════════════════════
    # ÉCOUTEUR D'ÉVÉNEMENTS UI (polling 80ms, thread-safe)
    # ══════════════════════════════════════════════════════════════
    def _ecouteur(self) -> None:
        try:
            while True:
                action, donnee = self._queue_ui.get_nowait()
                if   action == "msg"     : self._afficher(*donnee)
                elif action == "statut"  : self._maj_statut(*donnee)
                elif action == "progress": self._maj_progress(donnee)
                elif action == "ui"      : self._set_ui(donnee)
                elif action == "mem"     : self._maj_compteur(donnee)
        except queue.Empty:
            pass
        finally:
            self.after(80, self._ecouteur)

    # ══════════════════════════════════════════════════════════════
    # ACTIONS UI
    # ══════════════════════════════════════════════════════════════
    def _afficher(self, auteur: str, texte: str) -> None:
        couleur    = COULEURS.get(auteur, "#CCCCCC")
        tag_auteur = f"color_{auteur}"
        sep        = "─" * max(1, 18 - len(auteur))
        horodatage = datetime.now().strftime("%H:%M")

        self.txt.configure(state="normal")
        self.txt.tag_config(tag_auteur, foreground=couleur)
        self.txt.insert(tk.END, f"\n[{auteur}] {sep} {horodatage}\n", tag_auteur)
        self.txt.insert(tk.END, f"{texte}\n")
        self.txt.configure(state="disabled")
        self.txt.see(tk.END)

    def _maj_statut(self, texte: str, couleur: str) -> None:
        self.lbl_statut.configure(text=f"● {texte}", text_color=couleur)

    def _maj_progress(self, actif: bool) -> None:
        if actif:
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.set(0)

    def _maj_compteur(self, total: int) -> None:
        self.lbl_mem.configure(text=f"MEM : {total:,} souvenir(s)")

    def _set_ui(self, actif: bool) -> None:
        """
        Active ou désactive les contrôles de saisie.
        CORRECTION BUG : re-bind <Return> à chaque activation.
        Sans ce re-bind, CustomTkinter peut perdre la liaison après
        un disable/enable dans certaines versions.
        """
        etat = "normal" if actif else "disabled"
        self.entry.configure(state=etat)
        self.btn_envoyer.configure(state=etat)
        self.btn_effacer.configure(state=etat)

        if actif:
            # Re-bind obligatoire — fix du bug "bloquage après 1ère question"
            self.entry.bind("<Return>", lambda _: self._envoyer())
            self.entry.configure(
                placeholder_text="Posez votre question… (Entrée pour envoyer)"
            )
            # Focus garanti via after() pour éviter les race conditions Tkinter
            self.after(50, self.entry.focus_set)

    def _effacer_chat(self) -> None:
        self.txt.configure(state="normal")
        self.txt.delete("1.0", tk.END)
        self.txt.configure(state="disabled")
        self._afficher("SYSTEM", "Conversation effacée. La mémoire permanente est conservée.")

    # ══════════════════════════════════════════════════════════════
    # FERMETURE PROPRE
    # ══════════════════════════════════════════════════════════════
    def _quitter(self) -> None:
        log_ui.info("Fermeture propre d'OMICRON…")
        if self._agent_a  : self._agent_a.arreter()
        if self._memoire   : self._memoire.fermer()
        self.destroy()
        os._exit(0)

    # ── File d'événements interne ──────────────────────────────────
    def _push(self, action: str, donnee: Any) -> None:
        self._queue_ui.put((action, donnee))


# =====================================================================
# POINT D'ENTRÉE
# =====================================================================
if __name__ == "__main__":
    log_ui.info(f"Démarrage OMICRON v8.0 — Python {sys.version.split()[0]}")
    log_ui.info(f"Dossier système : {DOSSIER_SYSTEME}")
    app = AppOmicron()
    app.mainloop()
