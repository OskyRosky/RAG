"""
Splitter avanzado: agrupa por día detectando fechas tipo '6 de agosto'
y agrega metadatos (fecha, ciudad, país, viaje).
"""

import re
import json
import argparse
from pathlib import Path
from statistics import mean, median

# -------------------------------
# Configuración de meses (ES)
# -------------------------------
MONTHS = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "setiembre": "09", "octubre": "10",
    "noviembre": "11", "diciembre": "12",
}

# -------------------------------
# Regex principales
# -------------------------------

# Bloques de viaje:
#   Viaje número X: <trip>. Fecha del viaje: <rango>
#   <cuerpo hasta el siguiente "Viaje número ...">
TRIP_BLOCK_RE = re.compile(
    r"(Viaje\s+número\s+\d+:\s*(?P<trip>.+?)\.\s*Fecha\s+del\s+viaje:\s*(?P<range>.+?))\s+(?P<body>.*?)(?=Viaje\s+número\s+\d+:|$)",
    flags=re.IGNORECASE | re.DOTALL,
)

# Fechas tipo: "6 de agosto" en cualquier parte del texto (no anclado a inicio)
DATE_ANY_RE = re.compile(
    r"(\d{1,2}\s+de\s+[a-záéíóúñ]+)",
    flags=re.IGNORECASE,
)

# -------------------------------
# Utilidades
# -------------------------------

def month_to_num(name: str) -> str:
    return MONTHS.get(name.lower(), "00")

def to_iso(day_month: str, fallback_year: int) -> str | None:
    m = re.match(r"^\s*(\d{1,2})\s+de\s+([a-záéíóúñ]+)", day_month, flags=re.IGNORECASE)
    if not m:
        return None
    d = int(m.group(1))
    mon = month_to_num(m.group(2))
    return f"{fallback_year}-{mon}-{d:02d}"

def country_from_trip(trip_name: str) -> str | None:
    # "Tokio, Japón" -> "Japón"
    parts = [p.strip() for p in trip_name.split(",")]
    return parts[-1] if parts else None

def year_from_range(range_text: str, default: int = 2024) -> int:
    # "... de mayo de 2024" -> 2024
    m = re.search(r"de\s+(\d{4})", range_text)
    return int(m.group(1)) if m else default

def split_days(body: str):
    """
    Corta el cuerpo del viaje en bloques por cada fecha detectada.
    Devuelve lista de (date_str_original, day_block) incluyendo desde esa fecha hasta la siguiente.
    """
    matches = [m for m in DATE_ANY_RE.finditer(body)]
    if not matches:
        return []

    segments = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        day_block = body[start:end].strip()
        date_str = m.group(1)  # ej. "6 de agosto"
        segments.append((date_str, day_block))
    return segments

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Ruta al TXT completo (todos los viajes).")
    ap.add_argument("--out", dest="out", required=True, help="Ruta de salida JSONL.")
    ap.add_argument("--min-len", type=int, default=120, help="Descarta chunks demasiado cortos.")
    ap.add_argument("--show", type=int, default=3, help="Cuántos ejemplos imprimir al final.")
    ap.add_argument("--stats", action="store_true", help="Imprime métricas.")
    args = ap.parse_args()

    text = Path(args.inp).read_text(encoding="utf-8")

    docs = []
    # Recorre cada viaje (bloque)
    for m in TRIP_BLOCK_RE.finditer(text):
        trip_name = m.group("trip").strip()
        trip_range = m.group("range").strip()
        body = m.group("body").strip()

        year = year_from_range(trip_range, default=2024)
        country = country_from_trip(trip_name)

        # Separa por días dentro del viaje
        day_segments = split_days(body)
        for date_str, day_block in day_segments:
            iso = to_iso(date_str, fallback_year=year)

            # Encabezado incluido EN EL TEXTO del chunk (ayuda al embedding/búsqueda)
            header = f"{date_str} de {year} — {trip_name} (País: {country})\n"
            chunk_text = (header + day_block).strip()

            if len(chunk_text) < args.min_len:
                continue

            docs.append({
                "text": chunk_text,
                "metadata": {
                    "date": iso,
                    "date_str": f"{date_str} de {year}",
                    "trip": trip_name,
                    "country": country,
                    "range": trip_range,
                }
            })

    # Persistimos
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            item = {"id": i, "text": d["text"], "metadata": d["metadata"]}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Stats / ejemplos
    if args.stats:
        lens = [len(d["text"]) for d in docs]
        print("✅ Splitter por fecha completado")
        print(f"📦 Total chunks: {len(docs)}")
        if lens:
            print(f"🔤 Promedio caracteres/chunk: {mean(lens):.1f}")
            print(f"📊 Mediana: {median(lens):.1f}")
            print(f"📈 Mín: {min(lens)}, Máx: {max(lens)}")

    if args.show > 0 and docs:
        print("\n--- EJEMPLOS ---")
        for i, d in enumerate(docs[:args.show], 1):
            preview = d["text"][:400]
            print(f"\n[{i}] {preview}{'...' if len(d['text'])>400 else ''}")

if __name__ == "__main__":
    main()