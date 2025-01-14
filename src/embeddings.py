import nomic
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_nomic import NomicEmbeddings

def llama_embeddings(all_docs, model="llama3.1"):
    # Initialize the OllamaEmbeddings with the specified model
    embedding_model = OllamaEmbeddings(model=model)
    
    embeddings = []

    for chunk in all_docs:
        # Get the embedding for the chunk
        chunk_embedding = embedding_model.embed_query(chunk)
        embeddings.append(chunk_embedding)
    
    return embeddings

def nomic_embeddings(all_docs, model="nomic-embed-text-v1.5"):
    # Initialize the OllamaEmbeddings with the specified model
    embedding_model = NomicEmbeddings(model=model)
    embeddings = []

    for chunk in all_docs:
        # Get the embedding for the chunk
        chunk_embedding = embedding_model.embed_query(chunk)
        embeddings.append(chunk_embedding)
    
    return embeddings
