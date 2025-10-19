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
import json
import argparse
from pathlib import Path
from statistics import mean, median

# Silencio de barras/avisos
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def build_embeddings():
    # Embedding multilingüe, ligero y muy competente en español
    return FastEmbedEmbeddings(model_name="intfloat/multilingual-e5-small")


def load_jsonl(p: Path):
    assert p.exists(), f"No se encontró el archivo: {p}"
    docs = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("text", "").strip()
            meta = rec.get("metadata", {})
            docs.append(Document(page_content=text, metadata=meta))
    return docs


def build_chroma(db_dir: Path, collection: str, docs: list[Document]):
    emb = build_embeddings()
    vs = Chroma(
        persist_directory=str(db_dir),
        collection_name=collection,
        embedding_function=emb,
    )
    # Cargar docs desde cero
    if docs:
        vs.add_documents(docs)
        # Desde Chroma 0.4.x persiste automático, pero dejamos la llamada por compat
        try:
            vs.persist()
        except Exception:
            pass
    return vs


def get_vs(db_dir: Path, collection: str):
    emb = build_embeddings()
    return Chroma(
        persist_directory=str(db_dir),
        collection_name=collection,
        embedding_function=emb,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", help="JSONL de chunks")
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--collection", default="trips_rag")
    ap.add_argument("--rebuild", action="store_true", help="Borra y reconstruye el índice")
    ap.add_argument("--stats", action="store_true", help="Imprime estadísticas básicas")
    ap.add_argument("--test", help="Consulta de prueba al índice")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    db_dir = Path(args.db)

    if args.rebuild:
        # reconstrucción total
        if db_dir.exists():
            import shutil
            shutil.rmtree(db_dir)
        docs = load_jsonl(Path(args.inp))
        if args.stats and docs:
            lens = [len(d.page_content) for d in docs]
            print(f"📦 Documentos a indexar: {len(docs)}")
            print(f"🔤 len chars → min {min(lens)}, median {median(lens)}, mean {mean(lens):.1f}, max {max(lens)}")
        _ = build_chroma(db_dir, args.collection, docs)
        print(f"✅ Índice creado y persistido en '{args.db}' (colección: {args.collection})")

    # modo test o stats contra un índice existente
    vs = get_vs(db_dir, args.collection)

    if args.test:
        query = args.test
        try:
            results = vs.similarity_search_with_relevance_scores(query, k=args.k)
            # lista de (Document, score 0..1)
            print(f"\n🔎 TEST QUERY (k={args.k}): {query}\n")
            for i, (d, s) in enumerate(results, 1):
                src = d.metadata.get("source", "Trips.txt")
                cid = d.metadata.get("chunk_id", "?")
                prev = d.page_content[:300].replace("\n", " ")
                print(f"[{i}] score={s:.4f} | source={src} | chunk_id={cid}\n{prev}...\n")
        except Exception:
            # fallback para versiones antiguas
            results = vs.similarity_search_with_score(query, k=args.k)
            print(f"\n🔎 TEST QUERY (k={args.k}): {query}\n")
            for i, (d, dist) in enumerate(results, 1):
                # convertimos distancia a pseudo-relevancia
                try:
                    s = 1.0 / (1.0 + float(dist))
                except Exception:
                    s = 0.0
                src = d.metadata.get("source", "Trips.txt")
                cid = d.metadata.get("chunk_id", "?")
                prev = d.page_content[:300].replace("\n", " ")
                print(f"[{i}] score~={s:.4f} | source={src} | chunk_id={cid}\n{prev}...\n")


if __name__ == "__main__":
    main()