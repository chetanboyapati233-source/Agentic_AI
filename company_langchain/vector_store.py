import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "./chroma_db"
DOCS_PATH = "./docs"

# LangChain manages the embedding model — same all-MiniLM-L6-v2 under the hood
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore() -> Chroma:
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def ingest_documents():
    documents = []

    # LangChain TextLoader reads each file
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(DOCS_PATH, filename))
            documents.extend(loader.load())

    # LangChain splitter — smarter than manual slicing, respects sentence boundaries
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # LangChain stores chunks + embeddings in ChromaDB in one call
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Ingested {len(chunks)} chunks into ChromaDB")

def search_documents(query: str, k: int = 3) -> str:
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

if __name__ == "__main__":
    ingest_documents()
