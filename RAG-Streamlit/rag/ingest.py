"""
Etapa 1: Ingesta de documentos PDF -> texto limpio.
Uso:
  python -m rag.ingest --in data/Trips.pdf --out data/Trips.txt --stats
"""

import re
import argparse
from pathlib import Path
import pdfplumber

def extract_pdf_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""
            # Normalización mínima
            t = re.sub(r'\s+', ' ', t).strip()
            pages.append(t)
    return "\n\n".join(pages)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Ruta al PDF")
    ap.add_argument("--out", dest="out", default="data/Trips.txt", help="Ruta de salida .txt")
    ap.add_argument("--stats", action="store_true", help="Imprime métricas básicas")
    args = ap.parse_args()

    pdf_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    assert pdf_path.exists(), f"No se encontró el PDF: {pdf_path}"

    text = extract_pdf_text(pdf_path)
    out_path.write_text(text, encoding="utf-8")

    if args.stats:
        pages_est = text.count("\n\n") + 1
        print("✅ Ingesta completada")
        print(f"📄 PDF: {pdf_path}")
        print(f"📝 TXT: {out_path}")
        print(f"🔢 Caracteres: {len(text):,}")
        print(f"📑 Páginas (aprox): {pages_est}")
        preview = text[:600]
        print("\n--- PREVIEW ---\n" + preview + ("\n... (cortado)" if len(text) > 600 else ""))

if __name__ == "__main__":
    main()