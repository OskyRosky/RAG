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

import argparse
import json
import shutil
from pathlib import Path
from statistics import mean, median

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings

def build_embeddings():
    # Modelo multilingüe, liviano y de buena calidad
    return FastEmbedEmbeddings(model_name="intfloat/multilingual-e5-small")


def load_chunks(jsonl_path: Path):
    assert jsonl_path.exists(), f"No se encontró: {jsonl_path}"
    docs = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            meta = obj.get("metadata") or {}
            meta.setdefault("source", "Trips.txt")
            meta.setdefault("chunk_id", obj.get("id"))
            docs.append(Document(page_content=text, metadata=meta))
    return docs

def build_stats(docs):
    lengths = [len(d.page_content) for d in docs] if docs else []
    return {
        "n": len(docs),
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "mean": mean(lengths) if lengths else 0.0,
        "median": median(lengths) if lengths else 0.0,
    }

def wipe_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)

def index_documents(docs, db_dir: Path, collection: str):
    embeddings = FastEmbedEmbeddings()
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(db_dir),
        collection_name=collection,
    )
    vs.persist()
    return vs

def load_vectorstore(db_dir: Path, collection: str):
    embeddings = FastEmbedEmbeddings()
    return Chroma(
        persist_directory=str(db_dir),
        collection_name=collection,
        embedding_function=embeddings,
    )

def test_query(vs: Chroma, query: str, k: int = 5):
    # Retorna documentos + puntajes de relevancia
    results = vs.similarity_search_with_relevance_scores(query, k=k)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", help="Ruta a data/chunks.jsonl (solo para indexar)")
    ap.add_argument("--db", dest="db", default="chroma_db", help="Directorio de la base Chroma")
    ap.add_argument("--collection", default="trips_rag", help="Nombre de colección")
    ap.add_argument("--rebuild", action="store_true", help="Eliminar y reconstruir el índice")
    ap.add_argument("--stats", action="store_true", help="Imprimir métricas de chunks al indexar")
    ap.add_argument("--test", dest="test_query", help="Consulta de prueba")
    ap.add_argument("--k", type=int, default=5, help="Top-k para prueba")
    args = ap.parse_args()

    db_dir = Path(args.db)

    if args.rebuild:
        assert args.inp, "--in es requerido cuando usas --rebuild"
        print("🧱 Reconstruyendo índice desde cero…")
        wipe_dir(db_dir)
        jsonl_path = Path(args.inp)
        docs = load_chunks(jsonl_path)
        if args.stats:
            s = build_stats(docs)
            print("📦 Documentos a indexar:", s["n"])
            print(f"🔤 len chars → min {s['min']}, median {s['median']:.1f}, mean {s['mean']:.1f}, max {s['max']}")
        vs = index_documents(docs, db_dir, args.collection)
        print(f"✅ Índice creado y persistido en '{db_dir}' (colección: {args.collection})")

    # Si no reconstruimos, cargamos (o también tras indexar, por si quieres probar inmediatamente)
    vs = load_vectorstore(db_dir, args.collection)

    if args.test_query:
        print(f"\n🔎 TEST QUERY (k={args.k}): {args.test_query}")
        results = test_query(vs, args.test_query, k=args.k)
        if not results:
            print("⚠️ Sin resultados.")
            return
        for i, (doc, score) in enumerate(results, 1):
            preview = doc.page_content[:240].replace("\n", " ")
            if len(doc.page_content) > 240:
                preview += "..."
            print(f"\n[{i}] score={score:.4f} | source={doc.metadata.get('source')} | chunk_id={doc.metadata.get('chunk_id')}")
            print(preview)

if __name__ == "__main__":
    main()
