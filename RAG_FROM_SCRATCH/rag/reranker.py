"""
───────────────────────────────────────────────────────────────────────────────
 Módulo: rag/reranker.py
 Autor: Óscar Centeno Mora
 Descripción:
    Este módulo implementa una etapa opcional de *re-ranking* dentro del flujo RAG.
    Su objetivo es mejorar la precisión de recuperación reordenando los fragmentos
    (chunks) inicialmente obtenidos por similaridad, usando un modelo *Cross Encoder*.

 Contexto:
    En un pipeline RAG (Retrieve → Read):
       1. El retriever (ej. Chroma + embeddings) obtiene los documentos más parecidos.
       2. Este re-ranker evalúa de forma más profunda la relevancia semántica
          entre la pregunta y cada fragmento, priorizando los más relevantes.
       3. El LLM recibe los top-N documentos tras el re-rank, aumentando la exactitud.

 Modelo:
    "cross-encoder/ms-marco-MiniLM-L-6-v2" (≈60MB)
    - Entrenado en dataset MS MARCO (pregunta-respuesta).
    - Ofrece excelente balance entre velocidad y precisión.
    - Compatible con CPU (no requiere GPU obligatoriamente).

 Uso:
    from rag.reranker import rerank

    top_docs = rerank(query, retrieved_docs, top_n=8)

 Dependencias:
    - sentence-transformers
    - langchain_core.documents (para compatibilidad con objetos Document)

───────────────────────────────────────────────────────────────────────────────
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────────────────────────────
# Configuración del modelo de re-ranking
# ─────────────────────────────────────────────────────────────────────────────
# MiniLM (distilado de BERT) — rápido y robusto para tareas de QA y búsqueda.
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_ce = None  # caché global del modelo

def _get_model():
    """
    Carga el modelo CrossEncoder solo una vez en memoria.
    Retorna el modelo si ya está cargado (usa caché global).
    """
    global _ce
    if _ce is None:
        _ce = CrossEncoder(_MODEL_NAME)
    return _ce

# ─────────────────────────────────────────────────────────────────────────────
# Funciones principales de re-ranking
# ─────────────────────────────────────────────────────────────────────────────

def rerank(query: str, docs: List[Document], top_n: int) -> List[Document]:
    """
    Re-ordena los 'docs' según su relevancia con el 'query' usando CrossEncoder.

    Parámetros:
        query (str): Pregunta o texto de búsqueda.
        docs (List[Document]): Lista de documentos recuperados inicialmente.
        top_n (int): Número máximo de documentos a conservar tras re-ranking.

    Retorna:
        List[Document]: Lista ordenada (descendente) de los top_n más relevantes.
    """
    if not docs:
        return docs

    ce = _get_model()
    # Creamos pares (pregunta, documento)
    pairs = [(query, d.page_content) for d in docs]
    # El CrossEncoder predice un score de similitud semántica
    scores = ce.predict(pairs)  # valores mayores = más relevantes
    # Asociamos cada doc con su score y ordenamos
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Retornamos los documentos top_n más relevantes
    return [d for d, _ in scored[:max(1, top_n)]]

def rerank_with_scores(query: str, docs: List[Document], top_n: int) -> List[Tuple[Document, float]]:
    """
    Variante extendida: devuelve también los puntajes de relevancia.

    Retorna:
        List[Tuple[Document, float]] — [(documento, score_relevancia), ...]
    """
    if not docs:
        return []

    ce = _get_model()
    pairs = [(query, d.page_content) for d in docs]
    scores = ce.predict(pairs)
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:max(1, top_n)]