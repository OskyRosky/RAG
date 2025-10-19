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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Iterable

# 🔇 Silenciar avisos
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag.llm import get_llm

# ⬇️ Cross-encoder opcional (solo si lo activas con --rerank-top)
try:
    from rag.reranker import rerank as ce_rerank
    _HAS_CE = True
except Exception:
    _HAS_CE = False

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

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

# ---------------------------
# Utilidades de preguntas
# ---------------------------

_COUNTRY_LIST = [
    "Japón","Francia","Australia","Sudáfrica","Brasil","Italia",
    "Estados Unidos","Tailandia","Canadá","Marruecos","Costa Rica"
]

def _strip_country(q: str) -> str:
    for c in _COUNTRY_LIST:
        q = re.sub(rf"\b{re.escape(c)}\b", "", q, flags=re.IGNORECASE)
    return " ".join(q.split())

def _strip_date(q: str) -> str:
    # elimina patrones tipo "el 10 de julio de 2024"
    q = re.sub(r"\b(el|del)?\s*\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b", " ", q, flags=re.IGNORECASE)
    # elimina solo "el 10 de julio"
    q = re.sub(r"\b(el|del)?\s*\d{1,2}\s+de\s+\w+\b", " ", q, flags=re.IGNORECASE)
    return " ".join(q.split())

def expand_queries(question: str) -> List[str]:
    """
    Multi-query ligero, sin LLM:
      - original
      - sin país
      - sin fechas explícitas
    Deja la semántica y aporta recall cuando hay ruido en la redacción.
    """
    variants = []
    base = question.strip()
    if base:
        variants.append(base)
        variants.append(_strip_country(base))
        variants.append(_strip_date(base))
    # de-duplicar preservando orden
    seen = set()
    uniq = []
    for v in variants:
        if v and v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq[:3]

# ---------------------------
# Vector store
# ---------------------------

def load_vectorstore(db_dir: Path, collection: str) -> Chroma:
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=str(db_dir),
        collection_name=collection,
        embedding_function=emb,
    )

# ---------------------------
# Retrieve con MMR y presets
# ---------------------------

def retrieve_union_mmr(
    vs: Chroma,
    queries: Iterable[str],
    k_final: int,
    fetch_k: int,
    lambda_mult: float = 0.5,
) -> List[Document]:
    """
    - Hace max_marginal_relevance_search por cada variante de la pregunta.
    - Une resultados (sin duplicar) conservando orden por aparición.
    - Corta a k_final.
    """
    bag: List[Document] = []
    seen_ids = set()
    for q in queries:
        try:
            docs = vs.max_marginal_relevance_search(
                q, k=min(k_final, fetch_k), fetch_k=fetch_k, lambda_mult=lambda_mult
            )
        except Exception:
            # fallback a similarity_search si MMR no está disponible
            docs = vs.similarity_search(q, k=min(k_final, fetch_k))
        for d in docs:
            doc_id = (d.metadata.get("source"), d.metadata.get("chunk_id"))
            if doc_id not in seen_ids:
                bag.append(d)
                seen_ids.add(doc_id)
            if len(bag) >= k_final:
                return bag
    return bag[:k_final]

def format_context(docs: List[Document]) -> str:
    if not docs:
        return "No hay fragmentos recuperados."
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))

# ---------------------------
# Presets
# ---------------------------

def get_preset(mode: str):
    """
    fast:    k=8,  fetch_k=24, MMR on (barato), cross-encoder OFF
    accurate:k=12, fetch_k=48, MMR on (barato), cross-encoder opcional
    """
    mode = (mode or "fast").lower()
    if mode == "accurate":
        return dict(k=12, fetch_k=48, lambda_mult=0.5, use_ce=False)
    # default fast
    return dict(k=8, fetch_k=24, lambda_mult=0.5, use_ce=False)

# ---------------------------
# QA principal
# ---------------------------

def answer(
    question: str,
    db: str,
    collection: str,
    model: str,
    temp: float,
    k: int = None,
    mode: str = "fast",
    rerank_top: int = 0,   # cross-encoder (opcional)
) -> Tuple[str, List[Tuple[str, str]]]:

    vs = load_vectorstore(Path(db), collection)

    # 1) Preset
    preset = get_preset(mode)
    k_final = k or preset["k"]
    fetch_k = preset["fetch_k"]
    lambda_mult = preset["lambda_mult"]

    # 2) Expandir consultas ligeras
    qlist = expand_queries(question)

    # 3) Retrieve + MMR (barato)
    docs = retrieve_union_mmr(vs, qlist, k_final=k_final, fetch_k=fetch_k, lambda_mult=lambda_mult)

    # 4) (Opcional) rerank cruzado si lo pides y está disponible
    if rerank_top and _HAS_CE and len(docs) > 1:
        docs = ce_rerank(question, docs, top_n=max(k_final, rerank_top))[:k_final]

    # 5) Prompt + LLM
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
    ap.add_argument("--k", type=int, default=None, help="Override de k (si no, usa preset)")
    ap.add_argument("--mode", choices=["fast","accurate"], default="fast")
    ap.add_argument("--rerank-top", type=int, default=0, help="N de docs a conservar tras cross-encoder (0=off)")
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
        mode=args.mode,
        rerank_top=args.rerank_top,
    )

    print(ans)
    if args.show_sources:
        print("\nFuentes:")
        for s in srcs:
            print(" -", s)

if __name__ == "__main__":
    main()