# app/app.py
# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI para tu RAG:
# - Modos fast/accurate/custom
# - Historial de consultas (memoria de sesión)
# - Botón “Copiar respuesta” (JS simple) y exportar conversación (JSON)
# - Retrocompatible con distintas firmas de rag.qa.answer
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

def call_qa(question: str, db: str, collection: str, model: str, temp: float,
            k: int, use_rerank: bool, rerank_top: int):
    """
    Llama a rag.qa.answer pasando solo los argumentos que existan en la firma.
    Retrocompatible con versiones previas.
    """
    sig = inspect.signature(qa_answer)
    params = sig.parameters

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

    if "mode" in params:
        # Si tu qa.py soporta 'mode', respétalo según los toggles (fast/accurate/custom)
        kwargs["mode"] = st.session_state.get("mode_choice", "fast")

    # re-rank
    if "rerank_top" in params:
        kwargs["rerank_top"] = (rerank_top if use_rerank else 0)
    if "use_rerank" in params:
        kwargs["use_rerank"] = bool(use_rerank)

    # thresholds/prefetch si existen (custom)
    if kwargs.get("mode") == "custom":
        if "threshold" in params:
            kwargs["threshold"] = st.session_state.get("threshold_custom", 0.30)
        if "prefetch" in params:
            kwargs["prefetch"] = st.session_state.get("prefetch_custom", max(40, k * 4))

    return qa_answer(**kwargs)


def copy_to_clipboard_js(text: str):
    """
    Devuelve un HTML con JS para copiar 'text' al portapapeles.
    Streamlit ≥1.32 soporta st.html; si no, se ignora silenciosamente.
    """
    escaped = json.dumps(text)  # escapa correctamente
    return f"""
    <script>
    async function copyRagText() {{
        try {{
            await navigator.clipboard.writeText({escaped});
            const el = document.getElementById('copied-toast');
            if (el) {{ el.style.display = 'block'; setTimeout(()=> el.style.display='none', 1500); }}
        }} catch(err) {{
            console.log('copy failed', err);
        }}
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
        st.session_state.history = []  # lista de dicts {t, q, a, sources}
    if "mode_choice" not in st.session_state:
        st.session_state.mode_choice = "fast"
    if "threshold_custom" not in st.session_state:
        st.session_state.threshold_custom = 0.30
    if "prefetch_custom" not in st.session_state:
        st.session_state.prefetch_custom = 40


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
    temp = st.slider("Temperatura", 0.0, 1.0, 0.0, 0.01)

    st.markdown("**Modo**")
    colm1, colm2, colm3 = st.columns([1,1,1])
    with colm1:
        if st.radio(" ", ["fast", "accurate", "custom"], index=["fast","accurate","custom"].index(st.session_state.mode_choice), label_visibility="collapsed", key="mode_choice") == "fast":
            pass

    # Parámetros por modo
    if st.session_state.mode_choice == "fast":
        k = st.slider("k (contexto)", 4, 16, 12, 1)
        use_rerank = st.checkbox("Usar re-rank (si disponible)", value=False)
        rerank_top = st.slider("Top re-rank", 2, 12, 6, 1, disabled=not use_rerank)

    elif st.session_state.mode_choice == "accurate":
        k = st.slider("k (contexto)", 8, 20, 12, 1)
        use_rerank = st.checkbox("Usar re-rank (si disponible)", value=True)
        rerank_top = st.slider("Top re-rank", 4, 12, 8, 1, disabled=not use_rerank)

    else:  # custom
        k = st.slider("k (contexto)", 4, 24, 12, 1)
        st.session_state.threshold_custom = st.slider("threshold", 0.0, 0.8, st.session_state.threshold_custom, 0.01)
        st.session_state.prefetch_custom = st.slider("prefetch", 8, 96, st.session_state.prefetch_custom, 1)
        use_rerank = st.checkbox("Usar re-rank (si disponible)", value=False)
        rerank_top = st.slider("Top re-rank", 2, 12, 8, 1, disabled=not use_rerank)

    show_sources = st.checkbox("Mostrar fuentes", value=True)
    st.divider()

    # Historial en sidebar (resumen)
    st.subheader("🕘 Historial (resumen)")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[:10]), 1):
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
            temp=temp,
            k=k,
            use_rerank=use_rerank,
            rerank_top=rerank_top,
        )
        dt = time.time() - t0

        # Persistimos en historial
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

        # Copiar respuesta
        st.markdown(copy_to_clipboard_js(ans), unsafe_allow_html=True)

        # Fuentes
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

        # Exportar historial como JSON
        export_bytes = json.dumps(st.session_state.history, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("📥 Exportar historial (JSON)", data=export_bytes,
                           file_name="historial_rag.json", mime="application/json")