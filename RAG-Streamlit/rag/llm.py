"""
Etapa 4: Configuración del LLM (Ollama) para RAG.

Expone:
    get_llm(model: str, temperature: float) -> ChatOllama

Notas:
- Aquí solo definimos el modelo y parámetros conservadores (menos aleatoriedad).
- La regla de "no inventar si no está en el contexto" se impone en el PROMPT de qa.py.
"""

import os
from langchain_ollama import ChatOllama

DEFAULT_MODEL = os.getenv("RAG_MODEL", "llama3.3")
DEFAULT_TEMP = float(os.getenv("RAG_TEMP", "0.0"))

def get_llm(model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMP) -> ChatOllama:
    """
    Devuelve un LLM de Ollama con ajustes conservadores para minimizar alucinaciones.
    (La adherencia estricta al contexto se controla en el prompt de qa.py.)
    """
    return ChatOllama(
        model=model,
        temperature=temperature,   # 0.0 = determinista
        top_p=1.0,                 # sin recorte adicional de probas
        repeat_penalty=1.1,        # leve penalización a repeticiones
        # num_ctx=4096,            # opcional: aumentar contexto si tu modelo lo soporta
        # num_predict=512,         # opcional: límite de tokens de salida
    )