Proyecto educativo para construir un RAG paso a paso.
0) Base del proyecto (estructura, gitignore) ← **(listo)**
1) Ingesta PDF → texto
2) Splitter (chunking)
3) Embeddings + Vector store (Chroma)
4) LLM (Ollama) + prompt seguro
5) Cadena RAG (Retrieve→Read)
6) Evaluación rápida
7) Entradas (CLI) y UI (Streamlit)
- data/          Documentos fuente (e.g., Trips.pdf)
- chroma_db/     Índice vectorial persistente
- rag/           Núcleo del RAG por módulos
- app/           Entrypoints (CLI/UI)
- tests/         Pruebas
- notebooks/     Exploración opcional
