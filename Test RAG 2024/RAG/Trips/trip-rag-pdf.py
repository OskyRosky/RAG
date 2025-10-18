## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

##############################
##   1. Ingest PDF Files    ##
##############################

doc_path = "/Users/oskyroski/CGR/2024/LLM/Ollama/RAG/Trips/Trips.pdf"
model = "llama3.2"

# Local PDF file uploads
if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("Carga completada....")
else:
    print("Sube un archivo PDF")

    # Preview first page
content = data[0].page_content
# print(content[:100])

# ==== End of PDF Ingestion ====

####################################################################
##   2. Extract Text from PDF Files and split into small chunks   ##
####################################################################

# ==== Extract Text from PDF Files and Split into Small Chunks ====

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("División completada....")

# print(f"Number of chunks: {len(chunks)}")
# print(f"Example chunk: {chunks[0]}")

# ===== Add to vector database ===

####################################################################
##  3. Send the chunks to the embedding model   ##
####################################################################

import ollama

ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)
print("Agregado a la base de datos vectorial completado....")

####################################################################
##  # 4. Save the embeddings to a vector database   ##
####################################################################

## === Retrieval ===
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# set up our model to use
llm = ChatOllama(model=model)

# a simple technique to generate multiple questions from a single question and then retrieve documents
# based on those questions, getting the best of both worlds.
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Eres Mundi, una inteligencia artificial diseñada para ayudar a Óscar Javier Centeno Mora a recordar detalles específicos de sus viajes. Tu principal objetivo es buscar información en el documento proporcionado y responder de manera precisa y organizada, evitando cualquier tipo de suposición o invención.

Sigue estas reglas estrictas:
1. Responde únicamente con información contenida en el documento. Si no encuentras la respuesta, indícalo claramente y ofrece continuar con otra pregunta relevante.
2. Nunca inventes datos ni añadas información que no esté explícitamente mencionada en el documento.
3. Analiza cuidadosamente las secciones del documento para seleccionar la información más relevante.
4. Construye respuestas claras, precisas y organizadas. Si el documento tiene una estructura específica (secciones, fechas, categorías), referencia estos elementos en tu respuesta.
5. Siempre termina con una pregunta complementaria para ayudar a Óscar a recordar más detalles de sus experiencias.

Ejemplo:
Contexto: "El documento contiene información sobre un viaje de Óscar Javier Centeno Mora a Japón. Incluye detalles sobre las fechas, lugares visitados y comidas probadas."

Respuesta esperada: todo lo referente a los viajes de Òscar Javier Centeno Mora que tengas en la base de datos.

Recuerda, tu objetivo es ser un asistente confiable y preciso, ayudando a Óscar a explorar y revivir sus experiencias de viaje, basándote únicamente en el contenido del documento proporcionado.

Pregunta del usuario: {question}
"""
)

#####################################################################################
##  5. Perform similarity search on the vector database to find similar documents  ##
#####################################################################################

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)


# RAG prompt
template = """Responde a la pregunta según el contexto:
{context}
Pregunta: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

####################################################################
##  6. retrieve the similar documents and present them to the user
####################################################################

#res = chain.invoke(
#     input=("what are the main points as a business owner I should be aware of?",)
# )
# res = chain.invoke(input=("Fecha de viaje a Tokio, Japón.",))
res = chain.invoke(input=("Según los Viajes de Óscar Javier Centeno Mora en el 2024, cuando estuvo en Tokio Japón, qué dia fui a Shibuya y que hice.",))
# res = chain.invoke(input=("Según los Viajes de Óscar Javier Centeno Mora en el 2024, en que fecha y ciudad estuvo en Japón.",))

print(res)