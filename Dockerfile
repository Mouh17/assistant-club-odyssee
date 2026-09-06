FROM python:3.11-slim

# Dépendances système nécessaires pour torch / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render définit automatiquement la variable PORT ; app.py la lit pour se binder dessus.
EXPOSE 7860

CMD ["python", "app.py"]
