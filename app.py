"""
Assistant Club Odyssée — RAG (recherche + génération) exposé via FastAPI
À déployer sur Render.com (Web Service, Docker) — plan Free (512 MB RAM)
"""

import io
import os
import gc
import requests
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from pypdf import PdfReader
from fastembed import TextEmbedding

# ----------------------------------------------------------------------
# 1. Client Groq — gratuit à vie, sans carte bancaire.
#    La clé API est lue depuis la variable d'environnement GROQ_API_KEY
#    (à définir sur Render, onglet "Environment" du service).
# ----------------------------------------------------------------------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODELE = "llama-3.3-70b-versatile"  # gratuit, rapide, bon niveau en français

ROLE = ("Tu es l'assistant du Club Odyssée, une chaîne de salles de sport en France. "
        "Tu réponds aux questions des adhérents de façon précise, utile et directe.")

BASE_URL = "https://huggingface.co/datasets/Gui3/Atelier-Code-3-sep-25/resolve/main/"
FICHIERS = ["club-odyssee-paris.pdf", "club-odyssee-lyon.pdf", "club-odyssee-marseille.pdf",
            "club-odyssee-toulouse.pdf", "club-odyssee-lille.pdf", "club-odyssee-conditions-generales.pdf"]


# ----------------------------------------------------------------------
# 2. Chargement des documents
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
del TEXTES
gc.collect()


# ----------------------------------------------------------------------
# 3. Encodage (embeddings légers via fastembed / onnxruntime — pas de torch)
#    On encode par petits lots pour limiter le pic de mémoire au démarrage.
# ----------------------------------------------------------------------
print("🔎 Chargement du modèle d'embedding...")
encodeur = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

textes_a_encoder = [f"{d['titre']} - {d['texte']}" for d in DOCUMENTS]
VECTEURS = np.array(list(encodeur.embed(textes_a_encoder, batch_size=8)))
VECTEURS = VECTEURS / np.linalg.norm(VECTEURS, axis=1, keepdims=True)
print(f"✅ {len(VECTEURS)} passages encodés.")
del textes_a_encoder
gc.collect()


def chercher(question, k=3):
    """Renvoie les k passages les plus proches de la question (similarité cosinus)."""
    v_question = np.array(list(encodeur.embed([question])))[0]
    v_question = v_question / np.linalg.norm(v_question)
    similarites = VECTEURS @ v_question
    indices = np.argsort(-similarites)[:k]
    return [(DOCUMENTS[i]["titre"], DOCUMENTS[i]["texte"], float(similarites[i]))
            for i in indices]


# ----------------------------------------------------------------------
# 4. Appel au modèle génératif via l'API Groq (gratuite, sans carte bancaire)
# ----------------------------------------------------------------------
def demander_au_modele(system_prompt, messages):
    reponse = client.chat.completions.create(
        model=MODELE,
        max_tokens=500,
        temperature=0,
        messages=[{"role": "system", "content": system_prompt}] + messages,
    )
    return reponse.choices[0].message.content


def repondre_chat(message, historique):
    """historique : liste de {"role": "user"|"assistant", "content": str}"""
    passages = chercher(message)
    contexte = "\n\n".join(f"### {t}\n{x}" for t, x, _ in passages)

    system_prompt = (ROLE +
        " Tu réponds uniquement à partir des documents fournis. "
        "Si la réponse n'y figure pas, réponds exactement : « Je ne sais pas, il faut demander a l'accueil »")

    messages = list(historique[-6:])
    messages.append({"role": "user", "content": f"Documents :\n{contexte}\n\nQuestion : {message}"})

    reponse = demander_au_modele(system_prompt, messages)
    sources = ", ".join(t for t, _, _ in passages)
    return reponse, sources


# ----------------------------------------------------------------------
# 5. API FastAPI + frontend statique custom
# ----------------------------------------------------------------------
app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    history: list = []


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    reponse, sources = repondre_chat(req.message, req.history)
    return {"response": reponse, "sources": sources}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def racine():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
