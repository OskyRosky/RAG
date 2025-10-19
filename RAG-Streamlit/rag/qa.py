# rag/qa.py
"""
───────────────────────────────────────────────────────────────────────────────
Etapa 5: QA (Retrieve → Read) robusto y simple
- Respuestas SIEMPRE en tercera persona impersonal.
- Usa SOLO el contexto recuperado (guardrails en el prompt).
- Presets de optimización para no pelear con parámetros:
    --mode fast      → rápido (k bajo / prefetch moderado / umbral estándar)
    --mode accurate  → mayor recall (k/prejfetch más altos / umbral más laxo)
- Permite overrides puntuales: --k, --prefetch, --threshold
- Sin re-ranker pesado: menor latencia y menos dependencias.
───────────────────────────────────────────────────────────────────────────────
Uso:
  python -m rag.qa --db chroma_db --collection trips_rag --model llama3.3 \
    --temp 0.0 --mode accurate --k 12 \
    --question "¿Dónde se almorzó el 16 de mayo de 2024 en Brasil?" \
    --show-sources
"""

import os
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple

# Silenciar avisos/verborrea de dependencias
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# LangChain / Chroma / Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Tu wrapper para LLM (Ollama u otro)
from rag.llm import get_llm


# ──────────────────────────────────────────────────────────────────────────────
# Config: modelo de embeddings
#   Elegido: paraphrase-multilingual-mpnet-base-v2
#   Razón: multilingüe, robusto, estable con versiones actuales de LangChain.
# ──────────────────────────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


# ──────────────────────────────────────────────────────────────────────────────
# Prompts (instrucciones del sistema y plantilla de usuario)
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un asistente de QA que SOLO usa el CONTEXTO.

REGLAS ESTRICTAS:
1) Usa EXCLUSIVAMENTE la información del CONTEXTO.
2) Si la respuesta NO aparece en el contexto, responde EXACTAMENTE:
   "Lo siento, no encuentro esa información en los documentos."
3) Responde SIEMPRE en tercera persona impersonal (con "se ..."; no uses “yo”, “me”, “nosotros”).
   - Si el contexto dice "Almorcé en X", escribe "Se almorzó en X".
4) Conserva literalmente fechas, nombres y lugares.
5) Da UNA sola oración clara (1–2 líneas), sin relleno, sin especulación.
"""

USER_PROMPT = """Pregunta: {question}

Contexto (fragmentos recuperados):
{context}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Carga de vectorstore (Chroma) con embeddings HF
# ──────────────────────────────────────────────────────────────────────────────
def load_vectorstore(db_dir: Path, collection: str) -> Chroma:
    """
    Conecta a la colección Chroma persistida en disco con el mismo modelo de
    embeddings que se usó al indexar (coherencia = mejor recall/precision).
    """
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},            # CPU para máxima compatibilidad
        encode_kwargs={"normalize_embeddings": True},  # normaliza vectores
    )
    return Chroma(
        persist_directory=str(db_dir),
        collection_name=collection,
        embedding_function=emb,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Retrieve: traemos candidatos con prefetch y filtramos por umbral
# ──────────────────────────────────────────────────────────────────────────────
def _similarity_with_scores(vs: Chroma, query: str, k: int) -> List[Tuple[Document, float]]:
    """
    Intenta devolver (doc, score) como relevancia normalizada [0..1].
    Si el backend solo expone distancia, la convierte a relevancia aprox.
    """
    try:
        return vs.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        docs_scores = vs.similarity_search_with_score(query, k=k)
        converted = []
        for doc, dist in docs_scores:
            try:
                rel = 1.0 / (1.0 + float(dist))  # mapeo distancia→“relevancia” aprox
            except Exception:
                rel = 0.0
            converted.append((doc, rel))
        return converted


def retrieve(vs: Chroma, query: str, k_final: int, threshold: float, prefetch: int) -> List[Document]:
    """
    1) Recupera top 'prefetch' (más ancho que k_final) para subir recall.
    2) Filtra por umbral de relevancia (threshold); si queda vacío, relaja.
    3) Corta a k_final y devuelve.
    """
    prefetch = max(prefetch, k_final)
    docs_scores = _similarity_with_scores(vs, query, k=prefetch)

    # Filtro por score; si se vacía, usamos sin filtrar (para no dejar en blanco)
    filtered = [(d, s) for (d, s) in docs_scores if s >= threshold]
    if not filtered:
        filtered = docs_scores

    docs = [d for (d, _) in filtered][:k_final]
    return docs


def format_context(docs: List[Document]) -> str:
    """Concatena fragmentos para el LLM con numeración simple."""
    if not docs:
        return "No hay fragmentos recuperados."
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Respuesta principal: arma prompt, invoca LLM y devuelve texto + fuentes
# ──────────────────────────────────────────────────────────────────────────────
def answer(
    question: str,
    db: str,
    collection: str,
    model: str,
    temp: float,
    k: int,
    prefetch: int,
    threshold: float,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Firma explícita (sin re-ranker): enfocada a velocidad y estabilidad.

    Devuelve:
        - respuesta (str) ya formateada según reglas del SYSTEM_PROMPT
        - lista de fuentes [(source, chunk_id), ...] para auditoría
    """
    vs = load_vectorstore(Path(db), collection)

    # Retrieve “ancho” con filtro suave
    docs = retrieve(vs, question, k_final=k, threshold=threshold, prefetch=prefetch)

    # Prompt + LLM
    context = format_context(docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]).format(question=question, context=context)

    llm = get_llm(model=model, temperature=temp)
    resp = llm.invoke(prompt)

    # Fuentes mínimas (archivo + id de chunk)
    srcs = [(d.metadata.get("source", "?"), d.metadata.get("chunk_id", "?")) for d in docs]
    return resp.content.strip(), srcs


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="QA (Retrieve→Read) con presets simples.")
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--collection", default="trips_rag")
    ap.add_argument("--model", default="llama3.3")
    ap.add_argument("--temp", type=float, default=0.0)

    # Presets de latencia/recall
    ap.add_argument("--mode", choices=["fast", "accurate"], default="accurate",
                    help="Preset: fast (rápido) | accurate (mayor recall)")

    # Overrides opcionales (si no los pasas, se aplican los del preset)
    ap.add_argument("--k", type=int, default=None, help="Docs finales a pasar al LLM")
    ap.add_argument("--prefetch", type=int, default=None, help="Docs a recuperar antes de filtrar")
    ap.add_argument("--threshold", type=float, default=None, help="Umbral de relevancia [0..1]")

    ap.add_argument("--question", required=True)
    ap.add_argument("--show-sources", action="store_true", help="Imprime fuentes recuperadas")
    args = ap.parse_args()

    # Presets (puedes sobreescribir con flags)
    if args.mode == "fast":
        k = 8 if args.k is None else args.k
        prefetch = 16 if args.prefetch is None else args.prefetch
        threshold = 0.30 if args.threshold is None else args.threshold
    else:  # accurate
        k = 12 if args.k is None else args.k
        prefetch = 36 if args.prefetch is None else args.prefetch
        threshold = 0.25 if args.threshold is None else args.threshold

    ans, srcs = answer(
        question=args.question,
        db=args.db,
        collection=args.collection,
        model=args.model,
        temp=args.temp,
        k=k,
        prefetch=prefetch,
        threshold=threshold,
    )

    print(ans)
    if args.show_sources:
        print("\nFuentes:")
        for s in srcs:
            print(" -", s)

if __name__ == "__main__":
    main()