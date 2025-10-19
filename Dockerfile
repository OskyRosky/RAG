# Dockerfile
FROM python:3.12-slim

# Evitar preguntas interactivas y cache pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependencias del sistema (git para HF, build deps mínimos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos solo lo necesario para instalar deps
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiamos el resto del proyecto
COPY . .

# Streamlit en 8501
EXPOSE 8501

# Sugerencia: el índice y data se montarán como volúmenes
# OLLAMA_HOST vendrá por -e; por defecto, nada (si usas otro backend no hace falta)
ENV OLLAMA_HOST=""

# Arranque
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]