import os
import faiss
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple
from langchain.schema import Document
from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from uuid import uuid4
from typing import Tuple, List


output_dir = 'C:/Users/skrge/Documents/GitHub/llmtesting/output/'

def generate_chunk_id(doc: Document, current_page: int = None, current_page_part: int = 0) -> Tuple[str, int, int]:
    """
    Generate a unique ID for a chunk based on the Document's metadata.

    Args:
        doc (Document): A langchain Document object containing page content and metadata.
        current_page (int, optional): Current page number for PDF chunks. Default is None.
        current_page_part (int, optional): Current part number for the current page. Default is 0.

    Returns:
        Tuple[str, int, int]: Generated chunk ID, updated current_page, and updated current_page_part.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = doc.metadata

    # Extract only the file name from the full path
    file_name = os.path.basename(metadata.get("source", "unknown"))

    if "row" in metadata:  # For CSV files
        chunk_id = f"{file_name}_row{metadata['row']}_{current_time}"
    
    elif "page" in metadata:  # For PDF files
        page = metadata["page"]

        # Update page_part if current_page is the same, otherwise reset it
        if current_page == page:
            current_page_part += 1
        else:
            current_page = page
            current_page_part = 0

        chunk_id = f"{file_name}_page{page}:part{current_page_part}_{current_time}"
    
    else:
        chunk_id = f"{file_name}_{current_time}"  # Fallback for unknown formats

    return chunk_id, current_page, current_page_part

"""
Example usage:

current_page = None
current_page_part = 0
for doc in all_docs:
    chunk_id, current_page, current_page_part = generate_chunk_id(doc, current_page, current_page_part)
"""

def process_documents(all_docs: List[Document]) -> List[Document]:
    """
    Process a list of documents and generate chunk IDs with metadata.

    Args:
        all_docs (List[Document]): List of langchain Document objects.

    Returns:
        List[Document]: List of processed Document objects with chunk IDs.
    """
    current_page = None
    current_page_part = 0
    processed_docs = []

    for doc in all_docs:
        chunk_id, current_page, current_page_part = generate_chunk_id(doc, current_page, current_page_part)
        
        # Add chunk_id to the document's metadata
        new_metadata = doc.metadata.copy()
        new_metadata['chunk_id'] = chunk_id
        
        # Create a new Document with the updated metadata
        processed_doc = Document(
            metadata=new_metadata,
            page_content=doc.page_content
        )
        
        processed_docs.append(processed_doc)

    return processed_docs


def faiss_vector_store(processed_docs):
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

    # Create an empty FAISS index with the appropriate embedding dimension
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    # Initialize FAISS vector store with an in-memory document store
    vector_store = FAISS(
        embedding_function=embeddings,  # The function to generate embeddings
        index=index,  # The FAISS index to store vectors
        docstore=InMemoryDocstore(),  # Stores original documents
        index_to_docstore_id={}  # Mapping between FAISS index positions and document IDs
    )

    # Generate unique IDs for each document
    uuids = [str(uuid4()) for _ in range(len(processed_docs))]

    # Add documents to the vector store with generated UUIDs
    vector_store.add_documents(documents=processed_docs, ids=uuids)

    # Return the vector store with stored documents
    return vector_store

output_dir = r"C:\Users\skrge\Documents\GitHub\llmtesting\output"

def save_vector_store(vector_store, output_dir):
    #Saves the FAISS vector store to the specified directory.
    
    # Ensure the directory exists
    os.makedirs(output_dir , exist_ok=True)

    # Save the FAISS vector store
    vector_store.save_local(output_dir )
    print(f"FAISS index saved at: {output_dir }")

def load_vector_store(output_dir):

    #Loads a FAISS vector store from the specified directory.

    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

    # Check if the FAISS index exists before loading
    if not os.path.exists(os.path.join(output_dir, "index.faiss")):
        raise FileNotFoundError(f"No FAISS index found at: {output_dir}")

    # Load the FAISS vector store with safe pickle deserialization
    vector_store = FAISS.load_local(
        output_dir, 
        embeddings, 
        allow_dangerous_deserialization=True  # Allows pickle loading
    )

    print(f"FAISS index loaded from: {output_dir}")
    return vector_store

if __name__ == "__main__":
    proc_docs = process_documents(result) # result is the output of file_uploader.py
    vector_store = process_documents(proc_docs) 
    directory = input("Enter the path to the directory: ")
    save_vector_store(vector_store, directory)
    print("\nvectore store created and saved.")
