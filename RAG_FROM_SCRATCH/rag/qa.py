# rag/qa.py
"""
Etapa 5: QA (Retrieve → Read) robusto y controlable por parámetros.

Diseño:
- Usa SOLAMENTE el CONTEXTO para responder (estilo “closed-book over retrieved context”).
- Tercera persona impersonal (“se …”).
- Recuperación en dos pasos: prefetch amplio → filtrado por umbral → (opcional) re-ranking cruzado.
- Sin dependencias frágiles: embeddings con langchain_community (evita langchain-huggingface).

CLI (ejemplos):
  python -m rag.qa --db chroma_db --collection trips_rag \
    --model llama3.3 --temp 0.0 \
    --k 12 --threshold 0.30 --prefetch 48 --use-rerank \
    --question "¿Dónde se cenó el 6 de agosto de 2024 en Bangkok, Tailandia?" \
    --show-sources
"""

# ── Ruido fuera ─────────────────────────────────────────────────────────────────
import os
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── LangChain / LLM / VectorStore ───────────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag.llm import get_llm

# Re-rank opcional (se carga si existe el módulo)
try:
    from rag.reranker import rerank as ce_rerank
    _HAS_RERANK = True
except Exception:
    _HAS_RERANK = False

# ── Modelo de embeddings (DEBE coincidir con el índice existente) ───────────────
# Si indexaste con bge-m3, deja este:
EMBED_MODEL = "BAAI/bge-m3"
# Si tu índice fue con MPNet multilingüe, cambia a:
# EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ── Prompt de sistema + usuario ────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un asistente de QA que SOLO usa el CONTEXTO.

REGLAS ESTRICTAS:
1) Usa EXCLUSIVAMENTE la información del CONTEXTO.
2) Si la respuesta NO aparece en el contexto, responde EXACTAMENTE:
   "Lo siento, no encuentro esa información en los documentos."
3) Responde SIEMPRE en tercera persona impersonal (con "se ..."; no uses “yo”, “me”, “nosotros”).
   - Si el contexto dice "Almorcé en X", escribe "Se almorzó en X".
4) Conserva literalmente fechas, nombres y lugares.
5) Da UNA sola oración clara (1–2 líneas), sin relleno ni especulación.
"""

USER_PROMPT = """Pregunta: {question}

Contexto (fragmentos recuperados):
{context}
"""

# ── Construcción de embeddings y VectorStore ───────────────────────────────────
def _build_embeddings() -> HuggingFaceEmbeddings:
    """
    Devuelve un embedder robusto y multilingüe, en CPU y con normalización L2.
    Debe ser el mismo modelo que se usó al indexar el Chroma.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def load_vectorstore(db_dir: Path, collection: str) -> Chroma:
    """
    Abre la colección Chroma existente con el embedder compatible.
    """
    emb = _build_embeddings()
    return Chroma(
        persist_directory=str(db_dir),
        collection_name=collection,
        embedding_function=emb,
    )

# ── Recuperación con compatibilidad de puntuaciones ────────────────────────────
def _similarity_with_scores(vs: Chroma, query: str, k: int) -> List[Tuple[Document, float]]:
    """
    Devuelve lista de (doc, score) donde score ≈ relevancia en [0..1] si es posible.
    Fallback: convierte distancia a “relevancia” ≈ 1/(1+dist).
    """
    try:
        return vs.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        docs_scores = vs.similarity_search_with_score(query, k=k)
        converted: List[Tuple[Document, float]] = []
        for doc, dist in docs_scores:
            try:
                rel = 1.0 / (1.0 + float(dist))
            except Exception:
                rel = 0.0
            converted.append((doc, rel))
        return converted

def retrieve(
    vs: Chroma,
    query: str,
    k_final: int = 10,
    threshold: float = 0.30,
    prefetch: int = 40,
) -> List[Document]:
    """
    Recuperación en dos etapas:
      1) Prefetch amplio (prefetch >= k_final) para mejorar el recall.
      2) Filtrado manual por umbral de relevancia (threshold).
         - Si el filtrado queda vacío, se relaja (se usan los prefetch sin filtrar).
      3) Corte a k_final.
    """
    prefetch = max(prefetch, k_final)
    docs_scores = _similarity_with_scores(vs, query, k=prefetch)
    filtered = [(d, s) for (d, s) in docs_scores if s >= threshold]
    if not filtered:
        filtered = docs_scores  # relajamos si filtra todo
    docs = [d for (d, _) in filtered][:k_final]
    return docs

# ── Formato del contexto y respuesta ───────────────────────────────────────────
def format_context(docs: List[Document]) -> str:
    """Concatena los contenidos con numeración para que el LLM cite mejor."""
    if not docs:
        return "No hay fragmentos recuperados."
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))

def answer(
    question: str,
    db: str,
    collection: str,
    model: str,
    temp: float,
    k: int,
    threshold: float,
    prefetch: int,
    use_rerank: bool,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Orquesta: carga VS → retrieve (prefetch + threshold) → (opcional) re-rank →
    formatea contexto → invoca LLM → devuelve (respuesta, fuentes).
    """
    # 1) Abrimos VectorStore
    vs = load_vectorstore(Path(db), collection)

    # 2) Recuperación inicial
    initial_docs = retrieve(
        vs,
        question,
        k_final=max(k, 8),                # asegura un mínimo para no quedar corto
        threshold=threshold,
        prefetch=max(prefetch, k * 4),    # prefetch generoso por defecto
    )

    # 3) Re-ranking cruzado (si está disponible y activado)
    docs = initial_docs
    if use_rerank and _HAS_RERANK and len(initial_docs) > 1:
        try:
            # Mantén al menos k; si el re-rank falla, conserva initial_docs
            docs = ce_rerank(question, initial_docs, top_n=max(k, len(initial_docs)))
            docs = docs[:k]
        except Exception:
            pass
    else:
        docs = docs[:k]

    # 4) Prompt y llamada al LLM
    context = format_context(docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]).format(question=question, context=context)

    llm = get_llm(model=model, temperature=temp)
    resp = llm.invoke(prompt)

    # 5) Fuentes útiles para auditoría
    srcs = [(d.metadata.get("source", "?"), d.metadata.get("chunk_id", "?")) for d in docs]
    return resp.content.strip(), srcs

# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--collection", default="trips_rag")
    ap.add_argument("--model", default="llama3.3")
    ap.add_argument("--temp", type=float, default=0.0)

    # Recuperación y ranking
    ap.add_argument("--k", type=int, default=12, help="Contexto final (nº de fragmentos)")
    ap.add_argument("--threshold", type=float, default=0.30, help="Umbral mínimo de relevancia (0..1 aprox)")
    ap.add_argument("--prefetch", type=int, default=48, help="Top-N inicial para recall antes del filtrado")
    ap.add_argument("--use-rerank", action="store_true", help="Activa re-ranking CrossEncoder si está disponible")

    # Entrada / salida
    ap.add_argument("--question", required=False)  # se valida abajo
    ap.add_argument("--show-sources", action="store_true", help="Imprime fuentes recuperadas")

    args = ap.parse_args()

    if not args.question:
        print("Falta --question")
        return

    ans, srcs = answer(
        question=args.question,
        db=args.db,
        collection=args.collection,
        model=args.model,
        temp=args.temp,
        k=args.k,
        threshold=args.threshold,
        prefetch=args.prefetch,
        use_rerank=args.use_rerank,
    )

    print(ans)
    if args.show_sources:
        print("\nFuentes:")
        for s in srcs:
            print(" -", s)

if __name__ == "__main__":
    main()