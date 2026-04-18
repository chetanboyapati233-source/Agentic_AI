import chromadb
import os

_client = chromadb.PersistentClient(path="./chroma_db")

def get_collection():
    return _client.get_or_create_collection("company_docs")

def ingest_documents():
    collection = get_collection()
    docs_dir = "./docs"
    documents = []
    ids = []

    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_dir, filename), "r") as f:
                content = f.read()
            chunks = [content[i:i+400] for i in range(0, len(content), 400)]
            for k, chunk in enumerate(chunks):
                if chunk.strip():
                    documents.append(chunk)
                    ids.append(f"{filename}_chunk{k}")

    collection.upsert(documents=documents, ids=ids)
    print(f"Ingested {len(documents)} chunks from {docs_dir}")

def search_documents(query: str, n_results: int = 3) -> str:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    if results["documents"][0]:
        return "\n\n---\n\n".join(results["documents"][0])
    return "No relevant documents found."

if __name__ == "__main__":
    ingest_documents()
