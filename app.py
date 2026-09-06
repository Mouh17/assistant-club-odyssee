"""
Assistant Club Odyssée — version améliorée (RAG + reranking + interface chat)
À déployer sur Render.com (Web Service, Docker)
"""

import io
import os
import requests
import numpy as np
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import transformers
import torch

transformers.logging.set_verbosity_error()

# ----------------------------------------------------------------------
# 1. Configuration du modèle (CPU sur HF Spaces gratuit, sauf upgrade GPU)
# ----------------------------------------------------------------------
if torch.cuda.is_available():
    MODELE = "Qwen/Qwen2.5-1.5B-Instruct"
    DEVICE = 0
else:
    MODELE = "Qwen/Qwen2.5-0.5B-Instruct"
    DEVICE = -1

ROLE = ("Tu es l'assistant du Club Odyssée, une chaîne de salles de sport en France. "
        "Tu réponds aux questions des adhérents de façon précise, utile et directe.")

BASE_URL = "https://huggingface.co/datasets/Gui3/Atelier-Code-3-sep-25/resolve/main/"
FICHIERS = ["club-odyssee-paris.pdf", "club-odyssee-lyon.pdf", "club-odyssee-marseille.pdf",
            "club-odyssee-toulouse.pdf", "club-odyssee-lille.pdf", "club-odyssee-conditions-generales.pdf"]


# ----------------------------------------------------------------------
# 2. Chargement des documents (une seule fois au démarrage du Space)
# ----------------------------------------------------------------------
def charger_documents():
    textes = {}
    for f in FICHIERS:
        try:
            r = requests.get(BASE_URL + f, timeout=30)
            r.raise_for_status()
            pages = PdfReader(io.BytesIO(r.content)).pages
            textes[f] = "\n".join((p.extract_text() or "") for p in pages)
        except Exception as e:
            print(f"⚠️ Impossible de charger {f} : {e}")
    return textes


def decouper(texte, taille=500, chevauchement=80):
    """Découpe un texte en passages, en coupant de préférence en fin de phrase."""
    texte = " ".join(texte.split())
    passages, debut = [], 0
    while debut < len(texte):
        fin = min(debut + taille, len(texte))
        if fin < len(texte):
            coupe = texte.rfind(". ", debut + taille // 2, fin)
            if coupe != -1:
                fin = coupe + 1
        passages.append(texte[debut:fin].strip())
        if fin >= len(texte):
            break
        debut = max(fin - chevauchement, debut + 1)
    return [p for p in passages if p]


print("📚 Chargement des documents...")
TEXTES = charger_documents()

DOCUMENTS = []
for f, t in TEXTES.items():
    nom = f.replace(".pdf", "").replace("club-odyssee-", "Club Odyssée ").replace("-", " ").title()
    for i, p in enumerate(decouper(t), 1):
        DOCUMENTS.append({"titre": f"{nom} · passage {i}", "texte": p})
print(f"✅ {len(DOCUMENTS)} passages chargés.")


# ----------------------------------------------------------------------
# 3. Encodage + reranking (amélioration principale vs version atelier)
# ----------------------------------------------------------------------
print("🔎 Chargement des modèles d'embedding et de reranking...")
encodeur = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

textes_a_encoder = [f"{d['titre']} - {d['texte']}" for d in DOCUMENTS]
VECTEURS = encodeur.encode(textes_a_encoder, normalize_embeddings=True)
print(f"✅ {len(VECTEURS)} passages encodés.")


def chercher(question, k_large=8, k_final=3):
    """Recherche large par similarité, puis reranking précis avec un cross-encoder."""
    v_question = encodeur.encode(question, normalize_embeddings=True)
    similarites = VECTEURS @ v_question
    candidats_idx = np.argsort(-similarites)[:k_large]

    paires = [[question, DOCUMENTS[i]["texte"]] for i in candidats_idx]
    scores_rerank = reranker.predict(paires)

    ordre = np.argsort(-scores_rerank)[:k_final]
    resultats_idx = [candidats_idx[i] for i in ordre]

    return [(DOCUMENTS[i]["titre"], DOCUMENTS[i]["texte"], float(scores_rerank[ordre[j]]))
            for j, i in enumerate(resultats_idx)]


# ----------------------------------------------------------------------
# 4. Chargement du modèle génératif
# ----------------------------------------------------------------------
print("🧠 Chargement du modèle génératif (1 à 3 min)...")
generateur = pipeline("text-generation", model=MODELE, device=DEVICE)
generateur.tokenizer.clean_up_tokenization_spaces = False
generateur.model.generation_config.max_new_tokens = 200
generateur.model.generation_config.do_sample = False
generateur.model.generation_config.temperature = None
generateur.model.generation_config.top_p = None
generateur.model.generation_config.top_k = None
print("✅ Modèle chargé, l'assistant est prêt.")


def demander_au_modele(messages):
    sortie = generateur(messages)
    return sortie[0]["generated_text"][-1]["content"]


# ----------------------------------------------------------------------
# 5. Fonction principale, avec mémoire de conversation
# ----------------------------------------------------------------------
def repondre_chat(message, history):
    """history est fourni automatiquement par gr.ChatInterface (format messages)."""
    passages = chercher(message)
    contexte = "\n\n".join(f"### {t}\n{x}" for t, x, _ in passages)

    messages = [{"role": "system", "content": ROLE +
        " Tu réponds uniquement à partir des documents fournis. "
        "Si la réponse n'y figure pas, réponds exactement : « Je ne sais pas, il faut demander a l'accueil »"}]

    # On réinjecte les derniers échanges pour garder le fil de la conversation
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": f"Documents :\n{contexte}\n\nQuestion : {message}"})

    reponse = demander_au_modele(messages)
    sources = ", ".join(t for t, _, _ in passages)
    return f"{reponse}\n\n📎 *Sources : {sources}*"


# ----------------------------------------------------------------------
# 6. Interface — gr.ChatInterface au lieu de gr.Interface (vraie UI de chat)
# ----------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=repondre_chat,
    type="messages",
    title="🏋️ Assistant Club Odyssée",
    description=("Posez vos questions sur les clubs Odyssée (Paris, Lyon, Marseille, Toulouse, Lille). "
                  "Abonnements, équipements, horaires, résiliation..."),
    examples=[
        "Le club de Lyon a-t-il une piscine ?",
        "Combien coûte l'abonnement Premium à Marseille ?",
        "Quel est le préavis pour résilier mon abonnement ?",
    ],
    theme=gr.themes.Soft(primary_hue="orange"),
)

if __name__ == "__main__":
    # Render fournit le port à écouter via la variable d'environnement PORT.
    # server_name="0.0.0.0" est indispensable pour que Render puisse router le trafic.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
