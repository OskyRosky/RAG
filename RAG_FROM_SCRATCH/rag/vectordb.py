# rag/vectordb.py
"""
───────────────────────────────────────────────────────────────────────────────
 Módulo: VectorDB (Etapa 3 del pipeline RAG)
 Autor: Óscar Centeno Mora
 Descripción:
     Este módulo construye, administra y prueba una base vectorial (Chroma)
     a partir de los fragmentos de texto procesados (chunks.jsonl).

 Objetivos:
     1. Convertir los fragmentos en objetos Document de LangChain.
     2. Crear embeddings consistentes (mismo modelo para indexar y consultar).
     3. Persistir los vectores en un índice Chroma reutilizable.
     4. Permitir pruebas rápidas de búsqueda por similitud.

 Características:
     - Evita dependencias frágiles (usa langchain_community).
     - Multilingüe (soporta español, inglés y otros).
     - Compatibilidad total con el resto del pipeline RAG.
     - Incluye métricas estadísticas para análisis de calidad de los chunks.

 Ejemplos de uso:
     # Construcción del índice
     python -m rag.vectordb --in data/chunks.jsonl --db chroma_db \
         --collection trips_rag --rebuild --stats

     # Consulta de prueba
     python -m rag.vectordb --db chroma_db --collection trips_rag \
         --test "Aprazível 16 de mayo Brasil" --k 5

───────────────────────────────────────────────────────────────────────────────
"""

# ── Librerías base ─────────────────────────────────────────────────────────────
import json
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any

# Silencia warnings y deprecaciones visuales
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── LangChain / Chroma / Embeddings ────────────────────────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣  CONFIGURACIÓN DEL MODELO DE EMBEDDINGS
# ──────────────────────────────────────────────────────────────────────────────
def build_embeddings() -> HuggingFaceEmbeddings:
    """
    Construye un modelo de embeddings estable y multilingüe.

    Modelo usado:
        - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        - Funciona bien en CPU (sin requerir GPU).
        - Soporta normalización L2 (para evitar sesgos por magnitud).

    Retorna:
        HuggingFaceEmbeddings listo para usar en Chroma.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣  CARGA Y CONVERSIÓN DE LOS CHUNKS
# ──────────────────────────────────────────────────────────────────────────────
def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    """
    Lee un archivo JSONL (una línea = un JSON).
    Cada línea contiene: { "id": int, "text": str, "metadata": {...} }
    """
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]

def to_documents(items: List[Dict[str, Any]]) -> List[Document]:
    """
    Convierte una lista de diccionarios a objetos Document de LangChain.

    - Añade metadatos útiles para trazabilidad:
        • source: archivo original
        • chunk_id: identificador numérico
    """
    docs = []
    for it in items:
        meta = it.get("metadata", {}) or {}
        meta["source"] = meta.get("source", "chunks.txt")
        meta["chunk_id"] = it.get("id")
        docs.append(Document(page_content=it["text"], metadata=meta))
    return docs

# ──────────────────────────────────────────────────────────────────────────────
# 3️⃣  CONSTRUCCIÓN DEL ÍNDICE VECTORIAL (Chroma)
# ──────────────────────────────────────────────────────────────────────────────
def build_chroma(db_dir: Path, collection: str, docs: List[Document]) -> Chroma:
    """
    Construye (o reconstruye) un índice Chroma persistente.

    Parámetros:
        db_dir: ruta al directorio de la base vectorial (ej. chroma_db)
        collection: nombre de la colección
        docs: lista de documentos para indexar
    """
    emb = build_embeddings()

    # Inicialización del índice (vacío)
    vs = Chroma(
        persist_directory=str(db_dir),
        collection_name=collection,
        embedding_function=emb
    )

    # Limpieza preventiva (evita colecciones duplicadas)
    try:
        vs.delete_collection()
    except Exception:
        pass

    # Reconstrucción e inserción
    vs = Chroma(
        persist_directory=str(db_dir),
        collection_name=collection,
        embedding_function=emb
    )
    vs.add_documents(docs)

    # Persistencia local (en disco)
    try:
        vs.persist()
    except Exception:
        pass

    return vs

# ──────────────────────────────────────────────────────────────────────────────
# 4️⃣  CARGA DEL VECTORSTORE EXISTENTE
# ──────────────────────────────────────────────────────────────────────────────
def get_vs(db_dir: Path, collection: str) -> Chroma:
    """
    Carga una colección Chroma previamente construida.
    """
    emb = build_embeddings()
    return Chroma(
        persist_directory=str(db_dir),
        collection_name=collection,
        embedding_function=emb,
    )

# ──────────────────────────────────────────────────────────────────────────────
# 5️⃣  CLI PRINCIPAL: construcción, estadísticas y pruebas
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Gestor de índice vectorial Chroma (Etapa 3)")
    ap.add_argument("--in", dest="inp", help="Ruta al JSONL de chunks (ej. data/chunks.jsonl)")
    ap.add_argument("--db", default="chroma_db", help="Directorio de la base vectorial")
    ap.add_argument("--collection", default="trips_rag", help="Nombre de la colección")
    ap.add_argument("--rebuild", action="store_true", help="Reconstruye el índice desde cero")
    ap.add_argument("--stats", action="store_true", help="Muestra estadísticas de los chunks")
    ap.add_argument("--test", help="Consulta de prueba (opcional)")
    ap.add_argument("--k", type=int, default=5, help="Cantidad de resultados en test")
    args = ap.parse_args()

    db_dir = Path(args.db)

    # ── Construcción del índice ───────────────────────────────────────────────
    if args.rebuild:
        assert args.inp, "--in es requerido con --rebuild"
        items = read_jsonl(Path(args.inp))
        docs = to_documents(items)

        # Métricas descriptivas básicas
        if args.stats:
            lens = [len(d.page_content) for d in docs]
            print(f"📦 Documentos a indexar: {len(docs)}")
            if lens:
                print(
                    f"🔤 len chars → min {min(lens)}, "
                    f"median {sorted(lens)[len(lens)//2]}, "
                    f"mean {sum(lens)/len(lens):.1f}, "
                    f"max {max(lens)}"
                )

        _ = build_chroma(db_dir, args.collection, docs)
        print(f"✅ Índice creado y persistido en '{args.db}' (colección: {args.collection})")

    # ── Test de similitud (QA manual rápido) ───────────────────────────────────
    if args.test:
        vs = get_vs(db_dir, args.collection)
        print(f"\n🔎 TEST QUERY (k={args.k}): {args.test}\n")

        try:
            docs_scores = vs.similarity_search_with_relevance_scores(args.test, k=args.k)
            for i, (d, s) in enumerate(docs_scores, 1):
                print(
                    f"[{i}] score={s:.4f} | "
                    f"source={d.metadata.get('source')} | chunk_id={d.metadata.get('chunk_id')}\n"
                    f"{d.page_content[:260]}...\n"
                )
        except Exception:
            # Compatibilidad con versiones anteriores de LangChain
            docs_scores = vs.similarity_search_with_score(args.test, k=args.k)
            for i, (d, dist) in enumerate(docs_scores, 1):
                rel = 1.0 / (1.0 + float(dist))
                print(
                    f"[{i}] score≈{rel:.4f} | "
                    f"source={d.metadata.get('source')} | chunk_id={d.metadata.get('chunk_id')}\n"
                    f"{d.page_content[:260]}...\n"
                )

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()