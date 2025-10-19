"""
Etapa 3: Embeddings + Vector Store (Chroma).

Uso (indexar desde cero):
  python -m rag.vectordb --in data/chunks.jsonl --db chroma_db --collection trips_rag --rebuild --stats

Uso (reusar índice existente y probar consulta):
  python -m rag.vectordb --db chroma_db --collection trips_rag --test "¿En qué restaurante almorcé el 16 de mayo de 2024 en Brasil?" --k 5

Notas:
- Usa FastEmbed (CPU-friendly, multilingüe).
- Persistencia en disco con Chroma (carpeta --db).
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---------- Embeddings ----------
def build_embeddings() -> HuggingFaceEmbeddings:
    # bge-m3 multilingüe, normalizado (mejor para cosine)
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# ---------- Utilidades ----------
def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    assert p.exists(), f"No existe el archivo: {p}"
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def to_documents(items: List[Dict[str, Any]]) -> List[Document]:
    docs = []
    for it in items:
        meta = it.get("metadata", {}) or {}
        # asegura metadatos mínimos útiles
        meta.setdefault("source", "chunks.txt")
        meta.setdefault("chunk_id", it.get("id"))
        docs.append(Document(page_content=it["text"], metadata=meta))
    return docs

# ---------- Construir / Cargar ----------
def build_chroma(db_dir: Path, collection: str, docs: List[Document]) -> Chroma:
    emb = build_embeddings()
    vs = Chroma(
        collection_name=collection,
        persist_directory=str(db_dir),
        embedding_function=emb,
    )
    # Limpieza por si existe (rebuild real lo hace main)
    ids = [f"{d.metadata.get('chunk_id','')}" for d in docs]
    try:
        vs.delete(ids=ids)
    except Exception:
        pass
    vs.add_documents(docs, ids=ids)
    # Chroma 0.4+ persiste automáticamente
    return vs

def get_vs(db_dir: Path, collection: str) -> Chroma:
    emb = build_embeddings()
    return Chroma(
        collection_name=collection,
        persist_directory=str(db_dir),
        embedding_function=emb,
    )

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", help="JSONL de chunks (para --rebuild)")
    ap.add_argument("--db", default="chroma_db", help="Directorio Chroma")
    ap.add_argument("--collection", default="trips_rag", help="Nombre de la colección")
    ap.add_argument("--rebuild", action="store_true", help="Reconstruir el índice")
    ap.add_argument("--stats", action="store_true", help="Imprime métricas")
    ap.add_argument("--test", help="Consulta de prueba (opcional)")
    ap.add_argument("--k", type=int, default=5, help="k para test")
    args = ap.parse_args()

    db_dir = Path(args.db)

    if args.rebuild:
        assert args.inp, "--in es requerido con --rebuild"
        items = load_jsonl(Path(args.inp))
        docs = to_documents(items)
        if args.stats:
            lens = [len(d.page_content) for d in docs]
            print(f"📦 Documentos a indexar: {len(docs)}")
            if lens:
                print(f"🔤 len chars → min {min(lens)}, median {sorted(lens)[len(lens)//2]}, mean {sum(lens)/len(lens):.1f}, max {max(lens)}")
        # borrar carpeta para rebuild real
        if db_dir.exists():
            import shutil
            shutil.rmtree(db_dir)
        _ = build_chroma(db_dir, args.collection, docs)
        print(f"✅ Índice creado y persistido en '{args.db}' (colección: {args.collection})")

    if args.test:
        vs = get_vs(db_dir, args.collection)
        query = args.test
        print(f"\n🔎 TEST QUERY (k={args.k}): {query}\n")
        # usa similarity con scores (cosine aprox)
        try:
            docs_scores = vs.similarity_search_with_relevance_scores(query, k=args.k)
            for i, (d, s) in enumerate(docs_scores, 1):
                src = d.metadata.get("source", "?")
                cid = d.metadata.get("chunk_id", "?")
                print(f"[{i}] score={s:.4f} | source={src} | chunk_id={cid}\n{d.page_content[:300]}...\n")
        except Exception:
            docs_scores = vs.similarity_search_with_score(query, k=args.k)
            for i, (d, s) in enumerate(docs_scores, 1):
                src = d.metadata.get("source", "?")
                cid = d.metadata.get("chunk_id", "?")
                # s aquí suele ser distancia; mostramos tal cual
                print(f"[{i}] score={s:.4f} | source={src} | chunk_id={cid}\n{d.page_content[:300]}...\n")

if __name__ == "__main__":
    main()