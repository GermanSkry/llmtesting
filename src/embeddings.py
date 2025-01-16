import nomic
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_nomic import NomicEmbeddings
from tqdm import tqdm

def llama_embeddings(all_docs, model="llama3.1"):
    # Initialize the OllamaEmbeddings with the specified model
    embedding_model = OllamaEmbeddings(model=model)
    
    embeddings = []

    for doc in tqdm(all_docs, desc="Estimating llama embeddings"):
        # Get the embedding for the chunk
        chunk = doc.page_content
        chunk_embedding = embedding_model.embed_query(chunk)
        embeddings.append(chunk_embedding)
    
    return embeddings

def nomic_embeddings(all_docs, model="nomic-embed-text-v1.5"):
    # Initialize the OllamaEmbeddings with the specified model
    embedding_model = NomicEmbeddings(model=model)
    embeddings = []

    for doc in tqdm(all_docs, desc="Estimating nomic embeddings"):
        # Get the embedding for the chunk
        chunk = doc.page_content
        chunk_embedding = embedding_model.embed_query(chunk)
        embeddings.append(chunk_embedding)
    
    return embeddings

if __name__ == "__main__":
    all_docs =[]
    result = nomic_embeddings(all_docs)
    print("\nAll files loaded successfully.")