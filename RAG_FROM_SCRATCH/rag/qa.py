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
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from rag.llm import get_llm

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

def load_vectorstore(db_dir: Path, collection: str) -> Chroma:
    emb = FastEmbedEmbeddings()
    return Chroma(
        persist_directory=str(db_dir),
        collection_name=collection,
        embedding_function=emb,
    )

def retrieve(vs: Chroma, query: str, k: int = 15) -> List[Document]:
    """
    Recupera usando MMR para mayor diversidad y recall.
    - k: número final de documentos a pasar al LLM.
    - fetch_k: candidatos iniciales (amplio para cubrir variantes).
    """
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 60},
    )
    return retriever.invoke(query)

def format_context(docs: List[Document]) -> str:
    if not docs:
        return "No hay fragmentos recuperados."
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))

def answer(question: str, db: str, collection: str, model: str, temp: float, k: int) -> Tuple[str, List[Tuple[str, str]]]:
    vs = load_vectorstore(Path(db), collection)
    docs = retrieve(vs, question, k=k)
    context = format_context(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]).format(question=question, context=context)

    llm = get_llm(model=model, temperature=temp)
    resp = llm.invoke(prompt)

    srcs = [(d.metadata.get("source","?"), d.metadata.get("chunk_id","?")) for d in docs]
    return resp.content.strip(), srcs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--collection", default="trips_rag")
    ap.add_argument("--model", default="llama3.3")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=15)  # default subido para más recall
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