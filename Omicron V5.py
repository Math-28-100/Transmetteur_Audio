# =====================================================================
# OMICRON BRAIN V5.0 — EMPREINTE COGNITIVE + CLONAGE DE STYLE PARFAIT
# =====================================================================
# Usage :
#   python omicron_v5.py --fichier mon_texte.txt   (analyse + chat)
#   python omicron_v5.py                            (chat avec profil existant)
#
# Dépendances :
#   pip install requests numpy sentence-transformers rich
# =====================================================================

import os, sys, json, re, math, argparse, datetime, time
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

# Rich pour un terminal agréable (optionnel mais recommandé)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

import requests

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
GROQ_API_KEY   = "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"
GROQ_MODEL     = "llama-3.3-70b-versatile"
PROFILE_FILE   = os.path.join(os.path.expanduser("~"), "Desktop", "omicron_profile_v5.json")
HISTORY_FILE   = os.path.join(os.path.expanduser("~"), "Desktop", "omicron_history.json")
MAX_HISTORY    = 12   # nombre de tours gardés en contexte

MOTS_VIDES_FR = {
    "le","la","les","de","du","des","un","une","et","est","en","je","il",
    "elle","on","nous","vous","ils","elles","que","qui","ce","se","sa",
    "son","ses","mon","ma","mes","ton","ta","tes","par","pour","pas","ne",
    "avec","sur","dans","ou","si","au","aux","mais","car","donc","or","ni",
    "plus","très","tout","aussi","comme","ça","à","c'est","j'ai","j'","d'",
    "l'","qu'","être","avoir","faire","dire","aller","voir","vouloir","pouvoir",
    "alors","bien","même","après","avant","entre","sous","vers","sans","lors",
}

# ─────────────────────────────────────────────
# UTILITAIRES D'AFFICHAGE
# ─────────────────────────────────────────────
def print_header(titre):
    if RICH:
        console.rule(f"[bold cyan]{titre}[/bold cyan]")
    else:
        print(f"\n{'='*60}\n{titre}\n{'='*60}")

def print_info(msg):
    if RICH: console.print(f"[dim]▸ {msg}[/dim]")
    else: print(f"  {msg}")

def print_success(msg):
    if RICH: console.print(f"[bold green]✓[/bold green] {msg}")
    else: print(f"✓ {msg}")

def print_warn(msg):
    if RICH: console.print(f"[yellow]⚠[/yellow]  {msg}")
    else: print(f"⚠ {msg}")

def print_error(msg):
    if RICH: console.print(f"[red]✗[/red] {msg}")
    else: print(f"✗ {msg}")


# =====================================================================
# MODULE 1 — ANALYSE LINGUISTIQUE PROFONDE
# =====================================================================

class AnalyseurStyle:
    """
    Extrait une empreinte cognitive complète depuis un corpus textuel.
    Couvre : syntaxe, sémantique, ponctuation, émotion, rythme, rhétorique.
    """

    def __init__(self, texte_brut: str):
        self.texte = texte_brut
        self.mots  = re.findall(r"\b\w+\b", texte_brut.lower())
        self.phrases = self._segmenter_phrases(texte_brut)

    # ── Segmentation ──────────────────────────────────────────────────
    def _segmenter_phrases(self, texte: str) -> list[str]:
        phrases = re.split(r'(?<=[.!?…])\s+', texte)
        return [p.strip() for p in phrases if len(p.strip()) > 4]

    # ── 1. RYTHME ET STRUCTURE ─────────────────────────────────────────
    def analyser_rythme(self) -> dict:
        longueurs = [len(p.split()) for p in self.phrases]
        if not longueurs:
            return {}

        moy    = round(float(np.mean(longueurs)), 1)
        median = round(float(np.median(longueurs)), 1)
        ecart  = round(float(np.std(longueurs)), 1)

        # Variabilité du rythme
        if ecart < 3:
            variabilite = "rythme très régulier et uniforme — style prévisible"
        elif ecart < 7:
            variabilite = "rythme modérément varié"
        else:
            variabilite = "rythme très varié — alternance de phrases courtes et longues"

        # Distribution par taille
        tres_courtes = sum(1 for l in longueurs if l <= 5)
        courtes      = sum(1 for l in longueurs if 6 <= l <= 12)
        moyennes     = sum(1 for l in longueurs if 13 <= l <= 25)
        longues      = sum(1 for l in longueurs if l > 25)
        total        = len(longueurs)

        return {
            "longueur_moyenne_mots":  moy,
            "longueur_mediane_mots":  median,
            "ecart_type_longueur":    ecart,
            "variabilite_rythme":     variabilite,
            "distribution_phrases": {
                "très_courtes_≤5":   f"{round(tres_courtes/total*100)}%",
                "courtes_6-12":      f"{round(courtes/total*100)}%",
                "moyennes_13-25":    f"{round(moyennes/total*100)}%",
                "longues_>25":       f"{round(longues/total*100)}%",
            },
            "nb_phrases_total": total,
        }

    # ── 2. PONCTUATION ─────────────────────────────────────────────────
    def analyser_ponctuation(self) -> dict:
        t = self.texte
        n = max(len(self.phrases), 1)
        stats = {
            "points_suspension_par_phrase":  round(t.count("...") / n, 3),
            "exclamations_par_phrase":        round(t.count("!") / n, 3),
            "questions_par_phrase":           round(t.count("?") / n, 3),
            "virgules_par_phrase":            round(t.count(",") / n, 3),
            "tirets_par_phrase":              round((t.count(" — ")+t.count(" - ")) / n, 3),
            "parentheses_par_phrase":         round(t.count("(") / n, 3),
            "guillemets_par_phrase":          round((t.count('"')+t.count("«")) / n, 3),
        }

        notes = []
        if stats["points_suspension_par_phrase"] > 0.15:
            notes.append("utilise souvent '...' — suggère hésitation ou pensée inachevée")
        if stats["exclamations_par_phrase"] > 0.2:
            notes.append("expressif, nombreux '!' — ton énergique ou émotionnel")
        if stats["questions_par_phrase"] > 0.2:
            notes.append("style interrogatif dominant — remet en question, interpelle")
        if stats["virgules_par_phrase"] < 0.8:
            notes.append("peu de virgules — style direct, peu de subclauses")
        elif stats["virgules_par_phrase"] > 2.5:
            notes.append("beaucoup de virgules — pense par accumulation, listes internes")
        if stats["tirets_par_phrase"] > 0.1:
            notes.append("utilise les tirets pour des apartés ou précisions")
        if stats["parentheses_par_phrase"] > 0.1:
            notes.append("fréquentes parenthèses — pensée qui bifurque, commentaires internes")
        if not notes:
            notes.append("ponctuation sobre et conventionnelle")

        stats["observations"] = notes
        return stats

    # ── 3. MAJUSCULES ET ORTHOGRAPHE ──────────────────────────────────
    def analyser_majuscules(self) -> dict:
        mots_bruts = self.texte.split()
        debut_phrase = sum(1 for p in self.phrases if p and p[0].isupper())
        ratio_debut  = round(debut_phrase / max(len(self.phrases), 1), 2)

        tout_minus = sum(1 for m in mots_bruts if m.islower() and len(m) > 1)
        ratio_min  = round(tout_minus / max(len(mots_bruts), 1), 2)

        if ratio_debut < 0.4:
            style_maj = "n'utilise presque jamais les majuscules en début de phrase"
        elif ratio_debut < 0.75:
            style_maj = "utilise les majuscules de façon irrégulière"
        else:
            style_maj = "respecte systématiquement les majuscules en début de phrase"

        return {
            "ratio_debuts_phrase_majuscule": ratio_debut,
            "ratio_mots_tout_minuscule":     ratio_min,
            "description":                   style_maj,
        }

    # ── 4. VOCABULAIRE ET RICHESSE LEXICALE ───────────────────────────
    def analyser_vocabulaire(self) -> dict:
        mots_filtres = [m for m in self.mots if m not in MOTS_VIDES_FR and len(m) > 2]
        total = max(len(mots_filtres), 1)

        freq = Counter(mots_filtres)
        ttr  = round(len(freq) / total, 3)   # Type-Token Ratio

        if ttr > 0.75:
            richesse = "vocabulaire très riche et varié (peu de répétitions)"
        elif ttr > 0.5:
            richesse = "vocabulaire modéré"
        elif ttr > 0.3:
            richesse = "vocabulaire simple avec répétitions fréquentes"
        else:
            richesse = "vocabulaire très restreint — style oral/spontané"

        # Top mots significatifs
        top_mots = [m for m, _ in freq.most_common(30) if len(m) > 3]

        # Mots rares (hapax = apparus une seule fois)
        hapax = [m for m, c in freq.items() if c == 1 and len(m) > 4]
        ratio_hapax = round(len(hapax) / total, 3)

        # Longueur moyenne des mots
        long_mots = [len(m) for m in mots_filtres]
        moy_long_mot = round(float(np.mean(long_mots)), 1) if long_mots else 0

        return {
            "type_token_ratio":       ttr,
            "richesse_lexicale":      richesse,
            "top_mots_significatifs": top_mots[:20],
            "ratio_hapax":            ratio_hapax,
            "longueur_moyenne_mot":   moy_long_mot,
            "nb_mots_uniques":        len(freq),
            "nb_mots_total":          total,
        }

    # ── 5. TICS ET PATTERNS RÉCURRENTS ────────────────────────────────
    def analyser_tics(self) -> dict:
        mots_filtres = [m for m in self.mots if m not in MOTS_VIDES_FR and len(m) > 2]
        freq = Counter(mots_filtres)

        # Tics de langage (fréquence élevée)
        tics = [(m, c) for m, c in freq.most_common(15) if c >= 3]

        # Locutions / expressions multi-mots (bigrammes)
        bigrammes = []
        for i in range(len(self.mots) - 1):
            bg = f"{self.mots[i]} {self.mots[i+1]}"
            if self.mots[i] not in MOTS_VIDES_FR or self.mots[i+1] not in MOTS_VIDES_FR:
                bigrammes.append(bg)
        freq_bi = Counter(bigrammes)
        top_bi  = [(bg, c) for bg, c in freq_bi.most_common(10) if c >= 2]

        # Connecteurs logiques détectés
        connecteurs_map = {
            "cause":      ["parce que","car","puisque","vu que","étant donné"],
            "conséquence":["donc","ainsi","du coup","de ce fait","par conséquent"],
            "opposition": ["mais","cependant","pourtant","néanmoins","or","en revanche"],
            "addition":   ["et","aussi","de plus","en plus","par ailleurs","également"],
            "temps":      ["quand","lorsque","après","avant","pendant","dès que","puis"],
            "condition":  ["si","à condition","en cas","supposons","imagine"],
        }
        connecteurs_detectes = {}
        texte_lower = self.texte.lower()
        for categorie, mots_cat in connecteurs_map.items():
            count = sum(texte_lower.count(m) for m in mots_cat)
            if count > 0:
                connecteurs_detectes[categorie] = count

        return {
            "tics_langage":          [(m, c) for m, c in tics],
            "bigrammes_frequents":   top_bi,
            "connecteurs_logiques":  connecteurs_detectes,
        }

    # ── 6. TONALITÉ ÉMOTIONNELLE ───────────────────────────────────────
    def analyser_emotion(self) -> dict:
        lexique_positif = [
            "bien","super","génial","excellent","parfait","top","cool","adoré",
            "aime","magnifique","formidable","incroyable","réussi","content",
            "heureux","satisfait","merci","bravo","impressionnant","beau"
        ]
        lexique_negatif = [
            "pas","non","jamais","rien","nul","mauvais","problème","erreur",
            "difficile","impossible","échec","raté","catastrophe","horrible",
            "déçu","frustré","énervé","triste","inquiet","peur","dommage"
        ]
        lexique_incertitude = [
            "peut-être","probablement","je pense","je crois","semble","apparemment",
            "sans doute","sûrement","normalement","en principe","j'imagine","disons"
        ]
        lexique_intensite = [
            "vraiment","totalement","absolument","complètement","franchement",
            "honnêtement","clairement","évidemment","forcément","tellement"
        ]

        texte_lower = self.texte.lower()
        pos   = sum(texte_lower.count(m) for m in lexique_positif)
        neg   = sum(texte_lower.count(m) for m in lexique_negatif)
        incer = sum(texte_lower.count(m) for m in lexique_incertitude)
        inten = sum(texte_lower.count(m) for m in lexique_intensite)
        total = max(pos + neg, 1)

        ratio_pos = round(pos / total, 2)
        if ratio_pos > 0.65:
            tonalite = "globalement positive et enthousiaste"
        elif ratio_pos > 0.45:
            tonalite = "neutre ou équilibrée"
        else:
            tonalite = "plutôt négative ou critique"

        return {
            "score_positif":      pos,
            "score_negatif":      neg,
            "score_incertitude":  incer,
            "score_intensite":    inten,
            "tonalite_globale":   tonalite,
            "style_assertif":     incer < 3,
            "utilise_intensites": inten > 5,
        }

    # ── 7. STRUCTURE RHÉTORIQUE ────────────────────────────────────────
    def analyser_rhetorique(self) -> dict:
        texte_lower = self.texte.lower()

        # Questions rhétoriques (phrase interrogative sans attente de réponse)
        questions_tot = self.texte.count("?")

        # Répétitions de structure (anaphores)
        debuts = [p.split()[0].lower() if p.split() else "" for p in self.phrases]
        freq_debuts = Counter(debuts)
        anaphores = [(m, c) for m, c in freq_debuts.most_common(5) if c >= 3 and m not in MOTS_VIDES_FR]

        # Enumérations (virgules multiples dans une même phrase)
        phrases_enum = [p for p in self.phrases if p.count(",") >= 3]
        ratio_enum = round(len(phrases_enum) / max(len(self.phrases), 1), 3)

        # Passages directs (guillemets, dialogue)
        guillemets = self.texte.count('"') + self.texte.count("«")
        a_dialogues = guillemets > 4

        # Parenthèses et digressions
        parentheses = self.texte.count("(")

        # Métaphores / comparaisons
        comparaisons = sum(texte_lower.count(m) for m in ["comme","tel que","à l'image","ressemble","analogue"])

        return {
            "nb_questions_total":      questions_tot,
            "anaphores_detectees":     anaphores,
            "ratio_phrases_enumeration": ratio_enum,
            "utilise_dialogues_citations": a_dialogues,
            "nb_digressions_parentheses": parentheses,
            "nb_comparaisons":         comparaisons,
            "style_rhethorique_note":  self._note_rhetorique(anaphores, ratio_enum, comparaisons),
        }

    def _note_rhetorique(self, anaphores, ratio_enum, comparaisons) -> str:
        notes = []
        if anaphores:
            notes.append(f"répète souvent des débuts de phrase identiques (anaphore)")
        if ratio_enum > 0.1:
            notes.append("aime les énumérations et listes dans ses phrases")
        if comparaisons > 3:
            notes.append("fait souvent des comparaisons et métaphores")
        return "; ".join(notes) if notes else "rhétorique simple et directe"

    # ── 8. SYNTHÈSE GLOBALE ────────────────────────────────────────────
    def analyser_tout(self) -> dict:
        print_info("Analyse du rythme et de la structure...")
        rythme = self.analyser_rythme()

        print_info("Analyse de la ponctuation...")
        ponctuation = self.analyser_ponctuation()

        print_info("Analyse des majuscules...")
        majuscules = self.analyser_majuscules()

        print_info("Analyse du vocabulaire et richesse lexicale...")
        vocabulaire = self.analyser_vocabulaire()

        print_info("Extraction des tics et patterns récurrents...")
        tics = self.analyser_tics()

        print_info("Analyse de la tonalité émotionnelle...")
        emotion = self.analyser_emotion()

        print_info("Analyse rhétorique...")
        rhetorique = self.analyser_rhetorique()

        profil = {
            "version":     "5.0",
            "analyse_le":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "nb_mots_corpus": len(self.mots),
            "nb_phrases_corpus": len(self.phrases),
            "rythme":      rythme,
            "ponctuation": ponctuation,
            "majuscules":  majuscules,
            "vocabulaire": vocabulaire,
            "tics":        tics,
            "emotion":     emotion,
            "rhetorique":  rhetorique,
        }
        return profil


# =====================================================================
# MODULE 2 — GÉNÉRATION DU PROMPT DE STYLE
# =====================================================================

def construire_prompt_style(profil: dict) -> str:
    """
    Transforme le profil analytique en instructions précises pour le LLM.
    C'est le cœur du clonage : plus c'est précis, plus l'imitation est fidèle.
    """
    r  = profil.get("rythme", {})
    p  = profil.get("ponctuation", {})
    m  = profil.get("majuscules", {})
    v  = profil.get("vocabulaire", {})
    t  = profil.get("tics", {})
    e  = profil.get("emotion", {})
    rh = profil.get("rhetorique", {})

    lignes = []

    # — Rythme
    moy = r.get("longueur_moyenne_mots", 0)
    dist = r.get("distribution_phrases", {})
    lignes.append(f"RYTHME : Tes phrases font en moyenne {moy} mots. "
                  f"Distribution : {dist}. {r.get('variabilite_rythme','')}")

    # — Ponctuation
    obs_ponct = p.get("observations", [])
    if obs_ponct:
        lignes.append("PONCTUATION : " + " | ".join(obs_ponct))
    pts = p.get("points_suspension_par_phrase", 0)
    exc = p.get("exclamations_par_phrase", 0)
    qst = p.get("questions_par_phrase", 0)
    lignes.append(f"  → '...' : {pts}/phrase | '!' : {exc}/phrase | '?' : {qst}/phrase")

    # — Majuscules
    lignes.append(f"MAJUSCULES : {m.get('description','')}")

    # — Vocabulaire
    lignes.append(f"VOCABULAIRE : {v.get('richesse_lexicale','')} "
                  f"(TTR={v.get('type_token_ratio',0)}, "
                  f"longueur moy. mot={v.get('longueur_moyenne_mot',0)} car.)")
    top = v.get("top_mots_significatifs", [])[:12]
    if top:
        lignes.append(f"  → Mots les plus utilisés : {', '.join(top)}")

    # — Tics
    tics_list = t.get("tics_langage", [])
    if tics_list:
        tics_str = ", ".join(f'"{m_}" ({c}x)' for m_, c in tics_list[:8])
        lignes.append(f"TICS : {tics_str} — réutilise-les naturellement")
    bi = t.get("bigrammes_frequents", [])
    if bi:
        bi_str = ", ".join(f'"{bg}"' for bg, _ in bi[:5])
        lignes.append(f"EXPRESSIONS : {bi_str}")
    conn = t.get("connecteurs_logiques", {})
    if conn:
        conn_str = ", ".join(f"{k}({v})" for k, v in sorted(conn.items(), key=lambda x: -x[1])[:4])
        lignes.append(f"CONNECTEURS PRÉFÉRÉS : {conn_str}")

    # — Émotion
    lignes.append(f"TONALITÉ : {e.get('tonalite_globale','')}")
    if not e.get("style_assertif", True):
        lignes.append("  → Souvent incertain, utilise 'je pense', 'peut-être', 'sans doute'")
    if e.get("utilise_intensites", False):
        lignes.append("  → Renforce ses propos avec 'vraiment', 'totalement', 'franchement'")

    # — Rhétorique
    lignes.append(f"RHÉTORIQUE : {rh.get('style_rhethorique_note','')}")
    anaphores = rh.get("anaphores_detectees", [])
    if anaphores:
        ana_str = ", ".join(f'"{m_}"' for m_, _ in anaphores[:3])
        lignes.append(f"  → Commence souvent ses phrases par : {ana_str}")

    instructions = "\n".join(lignes)

    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMPREINTE DE STYLE — CLONAGE STRICT REQUIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{instructions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLES ABSOLUES DE CLONAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Reproduis EXACTEMENT la longueur moyenne de phrase ci-dessus.
2. Utilise la ponctuation avec les fréquences exactes indiquées.
3. Intègre naturellement les tics de langage et expressions listés.
4. Adopte la tonalité émotionnelle décrite.
5. INTERDIT : style "IA propre", formules de politesse génériques,
   phrases trop bien construites si ce n'est pas son style.
6. Parle à la première personne. Tu ES cet utilisateur.
7. Si tu t'éloignes du style, corrige-toi immédiatement.
"""


# =====================================================================
# MODULE 3 — APPEL API GROQ
# =====================================================================

def appel_groq(messages: list, system_prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "temperature": 0.8,
        "max_tokens": 1024,
        "top_p": 0.95,
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=20)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Erreur API ({r.status_code}) : {r.text}"
    except requests.Timeout:
        return "❌ Timeout Groq."
    except Exception as e:
        return f"❌ Erreur : {e}"


# =====================================================================
# MODULE 4 — ANALYSE GROQ APPROFONDIE DU STYLE (IA sur IA)
# =====================================================================

def analyse_groq_style(texte_brut: str) -> str:
    """
    Demande à Groq d'analyser le style de façon qualitative et nuancée,
    en complément de l'analyse algorithmique.
    """
    extrait = texte_brut[:3000]  # on limite pour ne pas exploser le contexte

    prompt_analyse = f"""Voici un texte écrit par un utilisateur. Ton rôle est d'analyser son style d'écriture de façon ULTRA-PRÉCISE.

TEXTE :
\"\"\"
{extrait}
\"\"\"

Analyse les dimensions suivantes et sois TRÈS précis et concret. Donne des exemples tirés du texte.

1. VOIX ET PERSONNALITÉ : Comment cet auteur "sonne" ? Quels traits de personnalité transparaissent ?
2. STRUCTURE DE PENSÉE : Comment organise-t-il ses idées ? Linéaire ? Par associations ? Par digressions ?
3. RAPPORT AU LECTEUR : S'adresse-t-il directement ? Suppose-t-il une connivence ? Pédagogue ou égal-à-égal ?
4. MARQUEURS STYLISTIQUES UNIQUES : Ce qui le distingue absolument des autres. Max 5 éléments très précis.
5. CE QU'IL NE FERAIT JAMAIS : Formulations, mots, structures qu'il n'utiliserait pas.
6. INSTRUCTIONS POUR L'IMITER : 5 règles pratiques et concrètes pour reproduire exactement son style.

Réponds de façon structurée et synthétique."""

    return appel_groq(
        [{"role": "user", "content": prompt_analyse}],
        "Tu es un expert en analyse stylistique et linguistique. Sois précis, concret, et utile."
    )


# =====================================================================
# MODULE 5 — GESTION DE L'HISTORIQUE DE CONVERSATION
# =====================================================================

def charger_historique() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def sauvegarder_historique(historique: list):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(historique[-MAX_HISTORY:], f, ensure_ascii=False, indent=2)

def ajouter_tour(historique: list, role: str, contenu: str) -> list:
    historique.append({"role": role, "content": contenu})
    return historique[-MAX_HISTORY:]


# =====================================================================
# MODULE 6 — AFFICHAGE DU PROFIL
# =====================================================================

def afficher_profil(profil: dict):
    if not RICH:
        print(json.dumps(profil, ensure_ascii=False, indent=2))
        return

    console.print()
    console.rule("[bold magenta]📊 EMPREINTE COGNITIVE DE L'UTILISATEUR[/bold magenta]")

    r  = profil.get("rythme", {})
    v  = profil.get("vocabulaire", {})
    e  = profil.get("emotion", {})
    t  = profil.get("tics", {})
    p  = profil.get("ponctuation", {})

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0,2))
    table.add_column("Dimension", style="dim", width=24)
    table.add_column("Analyse", style="white")

    table.add_row("Corpus analysé",
        f"{profil.get('nb_mots_corpus',0)} mots · {profil.get('nb_phrases_corpus',0)} phrases")
    table.add_row("Longueur moy. phrase",
        f"{r.get('longueur_moyenne_mots','?')} mots · {r.get('variabilite_rythme','')}")
    table.add_row("Vocabulaire",
        f"{v.get('richesse_lexicale','')} (TTR={v.get('type_token_ratio','')})")
    table.add_row("Tonalité",
        e.get("tonalite_globale","?"))
    table.add_row("Ponctuation",
        " | ".join(p.get("observations",[])))

    tics = t.get("tics_langage",[])
    if tics:
        tics_str = ", ".join(f'"{m_}"' for m_, _ in tics[:6])
        table.add_row("Tics détectés", tics_str)

    console.print(table)
    console.print()


# =====================================================================
# MAIN — ORCHESTRATION
# =====================================================================

def phase_analyse(fichier: str):
    print_header("PHASE 1 — ANALYSE DU STYLE")

    if not os.path.exists(fichier):
        print_error(f"Fichier introuvable : {fichier}")
        sys.exit(1)

    with open(fichier, "r", encoding="utf-8") as f:
        texte_brut = f.read()

    nb_mots = len(texte_brut.split())
    print_success(f"Fichier chargé : {nb_mots} mots")

    if nb_mots < 50:
        print_warn("Le corpus est très court. L'analyse sera moins fiable (idéal : 200+ mots).")

    # Analyse algorithmique
    print_header("Analyse algorithmique")
    analyseur = AnalyseurStyle(texte_brut)
    profil = analyseur.analyser_tout()

    # Analyse qualitative via Groq
    print_header("Analyse qualitative IA (Groq)")
    print_info("Interrogation du modèle de langage pour l'analyse stylistique nuancée...")
    analyse_qualitative = analyse_groq_style(texte_brut)
    profil["analyse_qualitative_ia"] = analyse_qualitative

    # Génération du prompt de style
    profil["prompt_style"] = construire_prompt_style(profil)

    # Sauvegarde
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(profil, f, ensure_ascii=False, indent=2)
    print_success(f"Profil sauvegardé : {PROFILE_FILE}")

    # Affichage
    afficher_profil(profil)

    if RICH:
        console.print(Panel(
            analyse_qualitative,
            title="[bold yellow]🧠 Analyse qualitative Groq[/bold yellow]",
            border_style="yellow"
        ))
    else:
        print("\n=== ANALYSE QUALITATIVE GROQ ===")
        print(analyse_qualitative)

    return profil


def phase_chat(profil: dict):
    print_header("PHASE 2 — CHAT CLONE")

    system_clone = f"""Tu es le clone parfait de l'utilisateur. Tu dois répondre EXACTEMENT comme lui.

{profil.get("prompt_style", "")}

ANALYSE QUALITATIVE DE SON STYLE (réalisée par IA) :
{profil.get("analyse_qualitative_ia", "Non disponible.")}

RAPPEL FINAL : Chaque réponse doit être indiscernable de ce que lui écrirait.
Pas de style IA. Pas de formules génériques. Sois lui."""

    historique = charger_historique()

    if RICH:
        console.print("\n[bold green]✓ Clone actif.[/bold green] [dim]Commandes : 'style', 'reset', 'quitter'[/dim]\n")
    else:
        print("\n✓ Clone actif. Commandes : 'style', 'reset', 'quitter'\n")

    while True:
        try:
            msg = input("Vous 👤 : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Fermeture.")
            break

        if not msg:
            continue

        if msg.lower() == "quitter":
            print("👋 Fermeture d'Omicron.")
            break

        if msg.lower() == "style":
            afficher_profil(profil)
            continue

        if msg.lower() == "reset":
            historique = []
            sauvegarder_historique(historique)
            print_success("Historique effacé.")
            continue

        historique = ajouter_tour(historique, "user", msg)
        reponse = appel_groq(historique, system_clone)
        historique = ajouter_tour(historique, "assistant", reponse)
        sauvegarder_historique(historique)

        if RICH:
            console.print(f"\n[bold cyan]Omicron 🤖[/bold cyan] : {reponse}\n")
        else:
            print(f"\nOmicron 🤖 : {reponse}\n")


def main():
    parser = argparse.ArgumentParser(description="Omicron Brain V5 — Clone stylistique IA")
    parser.add_argument("--fichier", "-f", type=str, help="Chemin vers le fichier texte à analyser")
    parser.add_argument("--chat-only", action="store_true", help="Passer directement au chat (profil existant)")
    args = parser.parse_args()

    print_header("OMICRON BRAIN V5.0 — EMPREINTE COGNITIVE")

    if args.chat_only or (not args.fichier and os.path.exists(PROFILE_FILE)):
        # Charger le profil existant
        if not os.path.exists(PROFILE_FILE):
            print_error("Aucun profil trouvé. Lance avec --fichier mon_texte.txt")
            sys.exit(1)
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            profil = json.load(f)
        print_success(f"Profil chargé (analysé le {profil.get('analyse_le','?')})")
        afficher_profil(profil)

    elif args.fichier:
        profil = phase_analyse(args.fichier)

    else:
        print_warn("Usage :")
        print("  python omicron_v5.py --fichier mon_texte.txt   (analyse + chat)")
        print("  python omicron_v5.py --chat-only               (chat avec profil existant)")
        sys.exit(0)

    phase_chat(profil)


if __name__ == "__main__":
    main()