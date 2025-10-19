# app/app.py
"""
Streamlit UI para tu RAG (demo viajes)
- UI minimal: pregunta, modos (fast/accurate/custom), k, mostrar fuentes.
- Botón de 'copiar respuesta' como icono (SVG) con Clipboard API.
- Cachea el VectorStore para reducir latencia sin modificar rag/qa.py.
- Llamada retrocompatible a rag.qa.answer (inspecciona la firma y pasa solo args soportados).
"""

import sys
import pathlib
import json
import inspect
from pathlib import Path
from typing import Dict, Any, List, Tuple

import streamlit as st
from streamlit.components.v1 import html as st_html

# ──────────────────────────────────────────────────────────────────────────────
# Habilitar importación del paquete 'rag' desde la raíz del repo
# ──────────────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.qa import answer as qa_answer
import rag.qa as rag_qa

# ──────────────────────────────────────────────────────────────────────────────
# Cachear VectorStore/Embeddings SIN tocar rag/qa.py
# ──────────────────────────────────────────────────────────────────────────────
_original_loader = rag_qa.load_vectorstore

@st.cache_resource(show_spinner=False)
def _cached_vs(db_dir: str, collection: str):
    return _original_loader(Path(db_dir), collection)

rag_qa.load_vectorstore = _cached_vs

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades UI
# ──────────────────────────────────────────────────────────────────────────────
def render_copy_icon(text: str, key: str = "copy_icon_key"):
    """Botón de copiar (solo icono) con Clipboard API."""
    payload = json.dumps(text)
    st_html(f"""
    <div style="margin-top:8px;">
      <button id="copy-btn-{key}"
              title="Copiar respuesta"
              style="background:transparent;border:0;cursor:pointer;
                     display:inline-flex;align-items:center;gap:6px;">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
             viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4
                   a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
      </button>
      <span id="copied-{key}" style="display:none;margin-left:6px;color:#13c27a;
                               font-size:12px;">¡Copiado!</span>
    </div>
    <script>
      const btn = document.getElementById('copy-btn-{key}');
      const copied = document.getElementById('copied-{key}');
      btn.addEventListener('click', async () => {{
        try {{
          await navigator.clipboard.writeText({payload});
          copied.style.display = 'inline';
          setTimeout(() => copied.style.display='none', 1200);
        }} catch(e) {{ console.log(e); }}
      }});
    </script>
    """, height=40)

def call_qa(
    question: str,
    db: str,
    collection: str,
    model: str,
    k: int,
    mode: str,
) -> Tuple[str, List[Tuple[str, str]]]:
    """Construye kwargs según la firma de rag.qa.answer. Temp=0, sin re-rank."""
    sig = inspect.signature(qa_answer)
    params = sig.parameters

    if mode == "fast":
        threshold = 0.30
        prefetch = max(20, k * 2)
    elif mode == "accurate":
        threshold = 0.25
        prefetch = max(36, k * 3)
    else:
        threshold = 0.28
        prefetch = max(30, int(k * 2.5))

    kwargs: Dict[str, Any] = {}
    for key, val in [
        ("question", question),
        ("db", db),
        ("collection", collection),
        ("model", model),
        ("temp", 0.0),   # 🔒 siempre 0
        ("k", k),
    ]:
        if key in params:
            kwargs[key] = val

    if "threshold" in params:
        kwargs["threshold"] = float(threshold)
    if "prefetch" in params:
        kwargs["prefetch"] = int(prefetch)
    if "rerank_top" in params:
        kwargs["rerank_top"] = 0  # 🔒 sin re-rank

    return qa_answer(**kwargs)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG de Viajes — Demo", page_icon="🧭", layout="wide")

with st.sidebar:
    st.header("⚙️ Configuración")
    db_dir = st.text_input("Directorio Chroma DB", "chroma_db", key="db_dir")
    collection = st.text_input("Colección", "trips_rag", key="collection")
    model = st.text_input("Modelo (Ollama)", "llama3.3", key="model")

    st.markdown("**Modo**")
    mode = st.radio(
        label="Modo",
        options=["fast", "accurate", "custom"],
        index=0,
        key="mode_radio",
        label_visibility="collapsed",
    )
    k = st.slider("k (contexto)", 4, 20, 12, 1, key="k_slider")

    show_sources = st.checkbox("Mostrar fuentes", value=True, key="show_sources")

st.title("🚀 RAG de Viajes – Demo")
st.write("Escribe una pregunta basada en tus viajes. Ejemplos:")
st.markdown("• ¿Dónde se almorzó el 16 de mayo de 2024 en Brasil? • "
            "¿Dónde se cenó el 6 de agosto de 2024 en Bangkok, Tailandia? • "
            "¿Qué se visitó por la tarde el 18 de septiembre de 2024 en Toronto, Canadá?")

question = st.text_input("Pregunta", "", key="question_input")

col1, _ = st.columns([1, 3])
with col1:
    ask = st.button("Preguntar", type="primary", key="ask_btn")

if not question:
    st.info("Introduce una pregunta y pulsa **Preguntar**.")
elif ask:
    with st.spinner("Buscando en tus documentos…"):
        try:
            ans, srcs = call_qa(
                question=question,
                db=db_dir,
                collection=collection,
                model=model,
                k=k,
                mode=mode,
            )
        except Exception as e:
            st.error(f"Error al obtener respuesta: {e}")
            st.stop()

    st.success(ans)
    render_copy_icon(ans, key="copy_answer_icon")

    if show_sources and srcs:
        st.subheader("Fuentes")
        for s in srcs:
            st.code(f"{s[0]} · chunk_id={s[1]}", language="text")