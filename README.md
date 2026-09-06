# Assistant Club Odyssée

Chatbot RAG répondant aux questions sur une chaîne fictive de salles de sport,
à partir de documents PDF (embeddings + reranking + génération avec Qwen2.5).

## Déploiement sur Render.com

1. Pousser ce dépôt sur GitHub.
2. Sur [render.com](https://render.com) → **New** → **Web Service**.
3. Connecter le dépôt GitHub.
4. Render détecte le `Dockerfile` automatiquement (ou utilise `render.yaml` si présent).
5. Plan : **Free**.
6. Déployer — l'URL publique est du type `https://ton-service.onrender.com`.

## Lancement en local

\`\`\`bash
pip install -r requirements.txt
python app.py
\`\`\`

L'app est alors accessible sur \`http://localhost:7860\`.
