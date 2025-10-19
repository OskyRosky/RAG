# app/app.py
"""
UI mínima para tu RAG (Streamlit)
- Controles en la barra lateral (DB, colección, k, temperatura, modo).
- Llama a rag.qa.answer de forma retrocompatible (inspección de firma).
- Muestra respuesta y (opcional) fuentes recuperadas.

Ejecuta:
  streamlit run app/app.py
"""

import os
import inspect
import warnings
import streamlit as st

# Silencio de librerías verbosas
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Importa tu pipeline de QA
from rag.qa import answer as qa_answer  # <- usamos la misma función del proyecto


# --------- Helper: llamada retrocompatible a qa.answer ----------
def call_qa(question: str,
            db: str, collection: str,
            model: str, temp: float, k: int,
            mode: str, rerank_top: int):
    """
    Adapta la llamada a rag.qa.answer según los argumentos disponibles
    en tu versión (como hicimos en eval.py).
    Además aplica presets para 'mode' (fast / accurate / custom).
    """
    # Presets de rendimiento
    use_rerank = rerank_top > 0
    if mode == "fast":
        k = max(8, k)
        use_rerank = False
        threshold = 0.30
        prefetch = max(24, k * 2)
    elif mode == "accurate":
        k = max(12, k)
        use_rerank = True if rerank_top > 0 else False
        threshold = 0.30
        prefetch = max(40, k * 4)
    else:  # custom
        threshold = 0.30
        prefetch = max(36, k * 3)

    # Inspección de firma para no romper si cambian los args
    sig = inspect.signature(qa_answer)
    params = sig.parameters
    kwargs = {}

    base = {
        "question": question,
        "db": db,
        "collection": collection,
        "model": model,
        "temp": temp,
        "k": k,
    }
    for kname, val in base.items():
        if kname in params:
            kwargs[kname] = val

    optional = {
        "rerank_top": rerank_top,
        "use_rerank": use_rerank,
        "threshold": threshold,
        "prefetch": prefetch,
    }
    for kname, val in optional.items():
        if kname in params:
            kwargs[kname] = val

    return qa_answer(**kwargs)  # -> (texto, fuentes)


# ===================== UI =====================
st.set_page_config(page_title="RAG Demo", page_icon="🧭", layout="centered")
st.title("🧭 RAG de Viajes – Demo")

with st.sidebar:
    st.header("⚙️ Configuración")

    # Rutas por defecto coherentes con tu repo
    db_dir = st.text_input("Directorio Chroma DB", "chroma_db")
    collection = st.text_input("Colección", "trips_rag")

    # Modelo LLM (Ollama)
    model = st.text_input("Modelo (Ollama)", "llama3.3")
    temp = st.slider("Temperatura", 0.0, 1.0, 0.0, 0.1)

    # Recuperación
    mode = st.radio("Modo", ["fast", "accurate", "custom"], index=0, horizontal=True)
    k = st.slider("k (contexto)", 4, 20, 12, 1)

    # Re-rank opcional (si tu qa.py lo soporta)
    use_rer = st.checkbox("Usar re-rank (si disponible)", value=False)
    rerank_top = st.slider("Top re-rank", 0, 12, 8, 1, disabled=not use_rer)
    if not use_rer:
        rerank_top = 0

    show_sources = st.checkbox("Mostrar fuentes", value=True)

st.write("Escribe una pregunta basada en tus viajes. Ejemplos:")
st.caption("• ¿Dónde se almorzó el 16 de mayo de 2024 en Brasil?\n"
           "• ¿Dónde se cenó el 6 de agosto de 2024 en Bangkok, Tailandia?\n"
           "• ¿Qué se visitó por la tarde el 18 de septiembre de 2024 en Toronto, Canadá?")

question = st.text_input("Pregunta")

col1, col2 = st.columns([1, 3])
with col1:
    do_ask = st.button("Preguntar", type="primary", use_container_width=True)

if do_ask and question.strip():
    with st.spinner("Buscando en tu base y generando respuesta…"):
        try:
            ans, srcs = call_qa(
                question=question.strip(),
                db=db_dir,
                collection=collection,
                model=model,
                temp=temp,
                k=k,
                mode=mode,
                rerank_top=rerank_top,
            )
            st.success(ans)

            if show_sources and srcs:
                st.markdown("**Fuentes**")
                for (src, cid) in srcs:
                    st.code(f"{src} · chunk_id={cid}", language="text")
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
            st.info("Verifica que el índice exista (chroma_db) y que tu servidor Ollama esté activo si usas un LLM local.")
else:
    st.info("Introduce una pregunta y pulsa **Preguntar**.")