# app/app.py
# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI para tu RAG (simple, sin temperatura ni re-rank):
# - Modos fast / accurate / custom
# - Pasa SIEMPRE threshold y prefetch (qa.answer los exige)
# - Historial de consultas, botón "Copiar", exportar a JSON
# ──────────────────────────────────────────────────────────────────────────────

import sys, pathlib, inspect, json, time
from datetime import datetime
import streamlit as st

# Hacemos import del paquete local "rag"
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.qa import answer as qa_answer  # pipeline principal

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def call_qa(question: str, db: str, collection: str, model: str,
            k: int, mode_choice: str, threshold_custom: float, prefetch_custom: int):
    """
    Llama a rag.qa.answer. Como tu qa exige 'prefetch' y 'threshold', SIEMPRE
    los calculamos según el modo y los pasamos. Temperatura fija a 0.0.
    """
    # presets por modo
    if mode_choice == "fast":
        threshold = 0.30
        prefetch = max(24, k * 2)    # rápido y razonable
    elif mode_choice == "accurate":
        threshold = 0.25
        prefetch = max(36, k * 3)    # un poco más ancho
    else:  # custom
        threshold = float(threshold_custom)
        prefetch  = int(prefetch_custom)

    # firma retrocompatible (por si cambiaste qa.py entre ramas)
    sig = inspect.signature(qa_answer)
    params = sig.parameters

    kwargs = {}
    base = [
        ("question", question),
        ("db", db),
        ("collection", collection),
        ("model", model),
        ("temp", 0.0),              # <- temperatura fija en 0 (cero alucinación)
        ("k", k),
        ("threshold", threshold),
        ("prefetch", prefetch),
    ]
    for key, val in base:
        if key in params:
            kwargs[key] = val

    # si tu qa expone "mode" lo llenamos; si no, no pasa nada
    if "mode" in params:
        kwargs["mode"] = mode_choice

    return qa_answer(**kwargs)


def copy_to_clipboard_js(text: str):
    escaped = json.dumps(text)
    return f"""
    <script>
    async function copyRagText() {{
      try {{
        await navigator.clipboard.writeText({escaped});
        const el = document.getElementById('copied-toast');
        if (el) {{ el.style.display = 'block'; setTimeout(()=> el.style.display='none', 1500); }}
      }} catch(err) {{ console.log('copy failed', err); }}
    }}
    </script>
    <div style="display:none" id="copied-toast">Copiado ✅</div>
    <button onclick="copyRagText()" style="
      background:#ea4b4b;color:white;border:0;border-radius:8px;padding:8px 12px;cursor:pointer;">
      Copiar respuesta
    </button>
    """


def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "mode_choice" not in st.session_state:
        st.session_state.mode_choice = "fast"
    if "threshold_custom" not in st.session_state:
        st.session_state.threshold_custom = 0.30
    if "prefetch_custom" not in st.session_state:
        st.session_state.prefetch_custom = 36


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG de Viajes — Demo", page_icon="🧭", layout="wide")
init_state()

with st.sidebar:
    st.header("⚙️ Configuración")

    db_dir = st.text_input("Directorio Chroma DB", "chroma_db")
    collection = st.text_input("Colección", "trips_rag")
    model = st.text_input("Modelo (Ollama)", "llama3.3")

    st.markdown("**Modo**")
    st.session_state.mode_choice = st.radio(
        "Modo", ["fast", "accurate", "custom"], index=["fast","accurate","custom"].index(st.session_state.mode_choice)
    )

    if st.session_state.mode_choice == "fast":
        k = st.slider("k (contexto)", 4, 16, 12, 1)

    elif st.session_state.mode_choice == "accurate":
        k = st.slider("k (contexto)", 8, 20, 12, 1)

    else:  # custom
        k = st.slider("k (contexto)", 4, 24, 12, 1)
        st.session_state.threshold_custom = st.slider("threshold", 0.0, 0.8, st.session_state.threshold_custom, 0.01)
        st.session_state.prefetch_custom  = st.slider("prefetch", 8, 96, st.session_state.prefetch_custom, 1)

    show_sources = st.checkbox("Mostrar fuentes", value=True)
    st.divider()

    st.subheader("🕘 Historial (resumen)")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-10:]):
            st.caption(f"{h['t']} • {h['q'][:48]}…")
    if st.button("🧹 Limpiar historial"):
        st.session_state.history.clear()
        st.success("Historial limpiado.")

st.title("🧭 RAG de Viajes – Demo")
st.write("Escribe una pregunta basada en tus viajes. Ejemplos:")
st.markdown("• ¿Dónde se almorzó el 16 de mayo de 2024 en Brasil? • ¿Dónde se cenó el 6 de agosto de 2024 en Bangkok, Tailandia? • ¿Qué se visitó por la tarde el 18 de septiembre de 2024 en Toronto, Canadá?")

tabs = st.tabs(["💬 Chat", "🕘 Historial"])

with tabs[0]:
    q = st.text_input("Pregunta", "")
    ask = st.button("Preguntar", type="primary")

    if ask and q.strip():
        t0 = time.time()
        ans, srcs = call_qa(
            question=q.strip(),
            db=db_dir,
            collection=collection,
            model=model,
            k=k,
            mode_choice=st.session_state.mode_choice,
            threshold_custom=st.session_state.threshold_custom,
            prefetch_custom=st.session_state.prefetch_custom,
        )
        dt = time.time() - t0

        st.session_state.history.append({
            "t": datetime.now().strftime("%H:%M:%S"),
            "q": q.strip(),
            "a": ans,
            "sources": srcs,
            "latency_s": round(dt, 3),
            "k": k,
            "mode": st.session_state.mode_choice,
        })

        st.success(ans)
        st.markdown(copy_to_clipboard_js(ans), unsafe_allow_html=True)

        if show_sources and srcs:
            st.subheader("Fuentes")
            for s in srcs:
                st.code(f"{s[0]} · chunk_id={s[1]}")

with tabs[1]:
    st.subheader("Historial detallado")
    if not st.session_state.history:
        st.info("Aún no hay entradas.")
    else:
        for i, h in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"{i:02d} · {h['t']} · {h['q'][:72]}…"):
                st.markdown(f"**Pregunta:** {h['q']}")
                st.markdown(f"**Respuesta:** {h['a']}")
                st.markdown(f"**Modo/k:** {h['mode']} / {h['k']}  •  **Latencia:** {h['latency_s']} s")
                if h.get("sources"):
                    st.markdown("**Fuentes:**")
                    for s in h["sources"]:
                        st.code(f"{s[0]} · chunk_id={s[1]}")

        export_bytes = json.dumps(st.session_state.history, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("📥 Exportar historial (JSON)", data=export_bytes,
                           file_name="historial_rag.json", mime="application/json")