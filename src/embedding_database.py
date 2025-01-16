import os
import faiss
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from typing import Tuple
from langchain.schema import Document
from typing import Tuple, List, Dict

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

def create_vector_store(documents: List[Document], embeddings: List[List[float]]):
    # Convert embeddings to a numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Create a FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the index
    index.add(embeddings_array)
    
    # Store document metadata
    doc_metadata = [doc.metadata for doc in documents]
    
    return index, doc_metadata, embeddings_array

faiss_index_name = "faiss_index.bin"
doc_metadata_name = "doc_metadata.pkl"

def save_vector_store(faiss_index, doc_metadata, output_dir):
    #faiss_index_name = "faiss_index.bin"
    faiss_index_path = os.path.join(output_dir, faiss_index_name)
    #doc_metadata_name = "doc_metadata.pkl"
    metadata_path = os.path.join(output_dir, doc_metadata_name)
    # Save the FAISS index
    faiss.write_index(faiss_index, faiss_index_path)
    
    # Save the document metadata
    with open(metadata_path, 'wb') as f:
        pickle.dump(doc_metadata, f)
    
    print(f"Vector store saved to {faiss_index_path} and {metadata_path}")

def load_vector_store(output_dir):
    #faiss_index = "faiss_index.bin"
    faiss_index_path = os.path.join(output_dir, faiss_index_name)
    #doc_metadata = "doc_metadata.pkl"
    metadata_path = os.path.join(output_dir, doc_metadata_name)
    # Load the FAISS index
    faiss_index = faiss.read_index(faiss_index_path)
    
    # Load the document metadata
    with open(metadata_path, 'rb') as f:
        doc_metadata = pickle.load(f)
    
    print(f"Vector store loaded from {faiss_index_path} and {metadata_path}")
    return faiss_index, doc_metadata

"""
Example usage:
faiss_index_name = "faiss_index.bin" #names of file
doc_metadata_name = "doc_metadata.pkl" #names of file
save_vector_store(faiss_index, doc_metadata, output_dir)
faiss_index, doc_metadata = load_vector_store(output_dir)
"""

def create_dataframe(documents: List[Document], embeddings: np.ndarray):
    # Extract chunk_id and page_content from documents
    data = {
        'chunk_id': [doc.metadata['chunk_id'] for doc in documents],
        'page_content': [doc.page_content for doc in documents],
        'embeddings': list(embeddings)
    }
    
    # Create a DataFrame
    embedding_df = pd.DataFrame(data)
    return embedding_df

file_name = "embedding_df.csv"

def save_dataframe(embedding_df: pd.DataFrame, output_dir: str):
    # Ensure the directory exists
    output_path = os.path.join(output_dir, file_name)
    
    # Save the DataFrame as a CSV file
    os.makedirs(output_dir, exist_ok=True)
    embedding_df.to_csv(output_path , index=False)
    print(f"DataFrame saved to {output_path}")

def load_dataframe(input_dir: str) -> pd.DataFrame:
    # Load the DataFrame from a CSV file
    input_path = os.path.join(input_dir, file_name)
    os.makedirs(input_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    print(f"DataFrame loaded from {input_path}")
    return df

