# rag/eval.py
"""
───────────────────────────────────────────────────────────────────────────────
Evaluación (Etapa 6)
- Ejecuta 11 tests contra tu pipeline de QA.
- Normaliza respuestas (tildes, mayúsculas, signos) y compara con similitud
  difusa para evitar falsos negativos.
- Retrocompatible con distintas firmas de rag.qa.answer (inspección dinámica).
───────────────────────────────────────────────────────────────────────────────
Uso:
  python -m rag.eval --db chroma_db --collection trips_rag \
    --model llama3.3 --temp 0.0 --k 12 --rerank-top 8 --fuzzy 0.78
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import unicodedata
import re
import inspect
from difflib import SequenceMatcher
from typing import List, Tuple

from rag.qa import answer as qa_answer

# ──────────────────────────────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────────────────────────────
TESTS: List[Tuple[str, str]] = [
    ("Para el viaje a Tokio, Japón, del 15 al 25 de enero del 2024, el día 18 de enero. ¿Qué animal representa la estatua en Shibuya?", "Hachiko"),
    ("Para el viaje a Francia, del 10 al 20 de febrero del 2024, el día 14 de febrero. ¿Cómo se llaman los cruceros por el Sena?", "Bateaux Parisiens"),
    ("Para el viaje a Australia, del 5 al 15 de marzo del 2024, el día 5 de marzo. ¿Qué visitó?", "Ópera de Sídney"),
    ("Para el viaje a Sudáfrica, del 1 al 11 de abril del 2024, el día 8 de abril. ¿Tipo de excursión para ver ballenas y focas en la bahía?", "barco"),
    ("Para el viaje a Brasil, del 15 al 25 de mayo del 2024, el día 16 de mayo. ¿En qué restaurante almorzó?", "Aprazível"),
    ("Para el viaje a Italia, del 10 al 20 de junio del 2024, el día 14 de junio. ¿Qué visitó en la excursión a Tívoli?", "Villa Adriana y Villa d'Este"),
    ("Para el viaje a Estados Unidos, del 5 al 15 de julio del 2024, el día 10 de julio. ¿En qué restaurante de comida japonesa se cenó?", "Nobu"),
    ("Para el viaje a Tailandia, del 1 al 11 de agosto del 2024, el día 6 de agosto. ¿Dónde cenó?", "Issaya Siamese Club"),
    ("Para el viaje a Canadá, del 15 al 25 de septiembre del 2024, el día 18 de septiembre. ¿Qué visitó por la tarde?", "Mercado de Kensington"),
    ("Para el viaje a Marruecos, del 10 al 19 de octubre del 2024. ¿Qué plaza famosa visitó en Marrakech?", "Jemaa el-Fnaa"),
    ("Para el viaje a Sudáfrica, del 1 al 11 de abril del 2024, el día 10 de abril. ¿Qué marca de zapato compró?", "Lo siento"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Normalización y fuzzy
# ──────────────────────────────────────────────────────────────────────────────
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

def _strip_accents(s: str) -> str:
    nkfd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nkfd if not unicodedata.combining(ch))

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = _strip_accents(s)
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_negative_expected(exp: str) -> bool:
    return normalize_text(exp).startswith("lo siento")

def approx_match(answer: str, expected: str, fuzzy_threshold: float) -> bool:
    a = normalize_text(answer)
    e = normalize_text(expected)
    if not a or not e:
        return False
    if e in a:
        return True
    return SequenceMatcher(None, a, e).ratio() >= fuzzy_threshold

# ──────────────────────────────────────────────────────────────────────────────
# Llamada retrocompatible a qa.answer
# ──────────────────────────────────────────────────────────────────────────────
def call_qa(question: str, db: str, collection: str, model: str, temp: float,
            k: int, rerank_top: int):
    """
    Detecta qué parámetros acepta rag.qa.answer mediante inspect.signature
    y pasa únicamente los soportados. Así evitamos errores si la firma
    cambia entre ramas/versiones.
    """
    sig = inspect.signature(qa_answer)
    params = sig.parameters

    # Argumentos básicos siempre que existan en la firma:
    kwargs = {}
    for key, val in [
        ("question", question),
        ("db", db),
        ("collection", collection),
        ("model", model),
        ("temp", temp),
        ("k", k),
    ]:
        if key in params:
            kwargs[key] = val

    # Opcionales según la versión del qa.py:
    if "rerank_top" in params:
        kwargs["rerank_top"] = rerank_top
    if "use_rerank" in params:
        kwargs["use_rerank"] = rerank_top > 0
    if "threshold" in params:
        kwargs["threshold"] = 0.30
    if "prefetch" in params:
        kwargs["prefetch"] = max(40, k * 4)

    return qa_answer(**kwargs)

# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────
def run_eval(db: str, collection: str, model: str, temp: float, k: int,
             rerank_top: int, fuzzy_threshold: float):
    tp = fp = tn = fn = 0
    rows = []

    for idx, (q, expected) in enumerate(TESTS, 1):
        ans, _ = call_qa(
            question=q,
            db=db,
            collection=collection,
            model=model,
            temp=temp,
            k=k,
            rerank_top=rerank_top,
        )

        neg = is_negative_expected(expected)
        if neg:
            ok = "lo siento" in normalize_text(ans)
            if ok: tn += 1
            else:  fp += 1
        else:
            ok = approx_match(ans, expected, fuzzy_threshold)
            if ok: tp += 1
            else:  fn += 1

        mark = "✅" if ok else "❌"
        rows.append(f"{mark} {idx:02d}. {ans}")

    for r in rows:
        print(r)

    total = tp + fp + tn + fn
    acc = 100.0 * (tp + tn) / max(1, total)
    print("\n--- Tabla de Confusión ---")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Exactitud total: {acc:.2f}%")

# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Evaluación de 11 tests sobre el pipeline QA.")
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--collection", default="trips_rag")
    ap.add_argument("--model", default="llama3.3")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--rerank-top", type=int, default=8, help="Docs a conservar tras re-rank (0 = desactivado)")
    ap.add_argument("--fuzzy", type=float, default=0.78, help="Umbral de similitud difusa [0..1]")
    args = ap.parse_args()

    run_eval(
        db=args.db,
        collection=args.collection,
        model=args.model,
        temp=args.temp,
        k=args.k,
        rerank_top=args.rerank_top,
        fuzzy_threshold=args.fuzzy,
    )

if __name__ == "__main__":
    main()