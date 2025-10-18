# main.py

import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "/Users/oskyroski/CGR/2024/LLM/Ollama/RAG/Trips/Trips.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def create_vector_db(chunks):
    """Create a vector database from document chunks."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info("Vector database created.")
    return vector_db


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
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

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain"""
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

    logging.info("Chain created successfully.")
    return chain


def main():
    # Load and process the PDF document
    data = ingest_pdf(DOC_PATH)
    if data is None:
        return

    # Split the documents into chunks
    chunks = split_documents(data)

    # Create the vector database
    vector_db = create_vector_db(chunks)

    # Initialize the language model
    llm = ChatOllama(model=MODEL_NAME)

    # Create the retriever
    retriever = create_retriever(vector_db, llm)

    # Create the chain with preserved syntax
    chain = create_chain(retriever, llm)

    # Example query
    question = "Según fecha cronológica, en el 2024, cuál fue mi primer viaje."

    # Get the response
    res = chain.invoke(input=question)
    print("Response:")
    print(res)


if __name__ == "__main__":
    main()