"""
Etapa 6: Evaluación flexible (post-proceso).
- Imprime SOLO "respuesta + símbolo ✅/❌".
- Normaliza (acentos, mayúsculas, espacios) y usa similitud difusa.
- Sin tocar ingest/split/DB/QA.

Uso:
  python -m rag.eval --db chroma_db --collection trips_rag --model llama3.3 --temp 0.0 --k 15
Opcional:
  --csv results_eval.csv        # exporta resultados
  --threshold 0.75              # similitud difusa mínima (0..1)
  --verbose                     # muestra también el esperado si falla
"""

import argparse
import unicodedata
import re
import difflib
from typing import List, Tuple, Optional
from rag.qa import answer as qa_answer

# ------------ TESTS (edítalos a tu gusto) -------------
# Mantengo tus 11 cases recientes de control; puedes ampliarlos cuando quieras.
TESTS: List[Tuple[str, str]] = [
    # 1. Tokio
    ("Para el viaje a Tokio, Japón, del 15 al 25 de enero del 2024, el día 18 de enero. ¿Qué animal representa la estatua en Shibuya?", "Hachiko"),
    # 2. Francia
    ("Para el viaje a Francia, del 10 al 20 de febrero del 2024, el día 14 de febrero. ¿Cómo se llaman los cruceros por el Sena?", "Bateaux Parisiens"),
    # 3. Australia
    ("Para el viaje a Australia, del 10 al 20 de marzo del 2024, el día 10 de marzo. ¿En qué ciudad visitó la Ópera?", "Sídney"),
    # 4. Sudáfrica
    ("Para el viaje a Sudáfrica, del 1 al 15 de abril del 2024, el día 8 de abril. ¿Tipo de excursión para ver ballenas y focas en la bahía?", "Barco"),
    # 5. Brasil
    ("Para el viaje a Brasil, del 15 al 25 de mayo del 2024, el día 16 de mayo. ¿En qué restaurante almorzó?", "Aprazível"),
    # 6. Italia
    ("Para el viaje a Italia, del 10 al 20 de junio del 2024, el día 14 de junio. ¿Qué visitó en la excursión a Tívoli?", "Villa Adriana y Villa d'Este"),
    # 7. Estados Unidos
    ("Para el viaje a Estados Unidos, del 5 al 15 de julio del 2024, el día 10 de julio. ¿En qué restaurante de comida Japonesa cenaste?", "Nobu"),
    # 8. Tailandia
    ("Para el viaje a Tailandia, del 1 al 11 de agosto del 2024, el día 6 de agosto. ¿Dónde cenó?", "Issaya Siamese Club"),
    # 9. Canadá
    ("Para el viaje a Canadá, del 15 al 25 de septiembre del 2024, el día 18 de septiembre. ¿Qué visitó por la tarde?", "Mercado de Kensington"),
    # 10. Marruecos
    ("Para el viaje a Marruecos, del 15 al 20 de octubre del 2024, el día 15 de octubre. ¿Qué plaza famosa visitó en Marrakech?", "Jemaa el-Fna"),
    # 11. Negativa
    ("Para el viaje a Sudáfrica, del 1 al 15 de abril del 2024, el día 10 de abril. ¿Qué marca de zapato compró?", "Lo siento"),
]
# ------------------------------------------------------


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def normalize_text(s: str) -> str:
    s = s.strip()
    s = strip_accents(s)
    s = s.lower()
    s = re.sub(r"[“”\"'`´]", "", s)            # comillas varias
    s = re.sub(r"[^a-z0-9áéíóúñü\s]", " ", s)   # deja alfanum y espacios (tras de-accent ya son ascii)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fuzzy_match(answer: str, expected: str, threshold: float) -> bool:
    a = normalize_text(answer)
    e = normalize_text(expected)

    # Caso negativo explícito:
    if e.startswith("lo siento"):
        return a.startswith("lo siento")

    # Exacto o contiene
    if e in a:
        return True

    # Similaridad difusa
    ratio = difflib.SequenceMatcher(None, a, e).ratio()
    if ratio >= threshold:
        return True

    # Coincidencia por tokens clave (tolerante)
    etoks = [t for t in re.split(r"\s+", e) if len(t) >= 3]
    if etoks and all(t in a for t in etoks[:3]):  # primeras 3 “palabras clave”
        return True

    return False

def eval_once(
    db: str, collection: str, model: str, temp: float, k: int,
    threshold: float, verbose: bool, csv_path: Optional[str]
) -> None:
    TP = FP = TN = FN = 0
    rows = []

    for i, (q, expected) in enumerate(TESTS, 1):
        ans, _ = qa_answer(q, db=db, collection=collection, model=model, temp=temp, k=k)
        ok = fuzzy_match(ans, expected, threshold)

        if expected.lower().startswith("lo siento"):
            # esperábamos "no encontrado"
            if ok: TN += 1
            else:  FP += 1
        else:
            if ok: TP += 1
            else:  FN += 1

        symbol = "✅" if ok else "❌"
        print(f"{symbol} {i:02d}. {ans}")

        if csv_path:
            rows.append({"n": i, "question": q, "answer": ans, "expected": expected, "ok": ok})

    # Resumen
    total = TP + TN + FP + FN
    acc = (TP + TN) / total * 100 if total else 0.0
    print("\n--- Tabla de Confusión ---")
    print(f"TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    print(f"Exactitud total: {acc:.2f}%")

    if csv_path and rows:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["n", "question", "answer", "expected", "ok"])
            w.writeheader()
            w.writerows(rows)
        print(f"(guardado) CSV: {csv_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--collection", default="trips_rag")
    ap.add_argument("--model", default="llama3.3")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    eval_once(
        db=args.db,
        collection=args.collection,
        model=args.model,
        temp=args.temp,
        k=args.k,
        threshold=args.threshold,
        verbose=args.verbose,
        csv_path=args.csv,
    )

if __name__ == "__main__":
    main()