import faiss
import numpy as np
import pandas as pd
import pickle
from langchain.schema import Document
from typing import List
from langchain_nomic import NomicEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str, faiss_index, doc_metadata, embedding_model_name="nomic-embed-text-v1.5", generation_model_name="llama3.1", top_k=5):
    # Initialize the NomicEmbeddings model for embeddings
    embedding_model = NomicEmbeddings(model=embedding_model_name)
    
    # Embed the query text
    query_embedding = np.array(embedding_model.embed_query(query_text)).astype('float32').reshape(1, -1)
    
    # Check the dimensionality of the query embedding
    query_dim = query_embedding.shape[1]
    faiss_index_dim = faiss_index.d
    if query_dim != faiss_index_dim:
        raise ValueError(f"Dimensionality mismatch: query embedding dimension is {query_dim}, but index dimension is {index_dim}")
    
    # Search the FAISS index for the most similar documents
    D, I = faiss_index.search(query_embedding, top_k)
    
    # Retrieve the context based on similarity
    context = []
    sources = []
    for i in I[0]:
        context.append(doc_metadata[i].get('page_content', ''))
        sources.append(doc_metadata[i].get('source', ''))
    
    # Create the prompt based on the query text and context
    context_text = " ".join(context)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    #prompt = f"Query: {query_text}\nContext: {context_text}\nAnswer:"
    
    # Initialize the Ollama model for generating answers
    generation_model = OllamaLLM(model=generation_model_name)
    
    # Input the prompt to the Ollama model
    response = generation_model.invoke(prompt)
    #response = generation_model.generate(prompt)
    
    return response, context, sources