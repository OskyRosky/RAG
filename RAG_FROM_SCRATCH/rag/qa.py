"""
Etapa 5: QA (Retrieve -> Read) robusto y salida limpia.
- 3ª persona impersonal.
- Retriever con MMR (fetch_k amplio) para mejorar recall.
"""

import os
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple

# Silenciar barras/avisos
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rag.llm import get_llm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

# ---------- VectorStore ----------
def load_vectorstore(db_dir: Path, collection: str) -> Chroma:
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=str(db_dir),
        collection_name=collection,
        embedding_function=emb,
    )

def _similarity_with_scores(vs: Chroma, query: str, k: int):
    """Devuelve lista de (doc, score) en relevancia 0..1 si es posible."""
    try:
        return vs.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        docs_scores = vs.similarity_search_with_score(query, k=k)
        converted = []
        for doc, dist in docs_scores:
            try:
                rel = 1.0 / (1.0 + float(dist))
            except Exception:
                rel = 0.0
            converted.append((doc, rel))
        return converted

def retrieve(vs: Chroma, query: str, k: int = 10, threshold: float = 0.30) -> List[Document]:
    docs_scores = _similarity_with_scores(vs, query, k=k)
    filtered = [(d, s) for (d, s) in docs_scores if s >= threshold]
    if not filtered:
        filtered = docs_scores  # fallback
    return [d for (d, _) in filtered]

def format_context(docs: List[Document]) -> str:
    if not docs:
        return "No hay fragmentos recuperados."
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))

# ---------- QA ----------
def answer(question: str, db: str, collection: str, model: str, temp: float, k: int) -> Tuple[str, List[Tuple[str, str]]]:
    vs = load_vectorstore(Path(db), collection)
    docs = retrieve(vs, question, k=k, threshold=0.30)
    context = format_context(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]).format(question=question, context=context)

    llm = get_llm(model=model, temperature=temp)
    resp = llm.invoke(prompt)

    srcs = [(d.metadata.get("source","?"), d.metadata.get("chunk_id","?")) for d in docs]
    return resp.content.strip(), srcs

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--collection", default="trips_rag")
    ap.add_argument("--model", default="llama3.3")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--question", required=True)
    ap.add_argument("--show-sources", action="store_true", help="Imprime fuentes recuperadas")
    args = ap.parse_args()

    ans, srcs = answer(
        question=args.question,
        db=args.db,
        collection=args.collection,
        model=args.model,
        temp=args.temp,
        k=args.k,
    )

    print(ans)

    if args.show_sources:
        print("\nFuentes:")
        for s in srcs:
            print(" -", s)

if __name__ == "__main__":
    main()