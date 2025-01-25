import os
import re
from typing import List
from langchain.schema import Document
from langchain.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_csv_files(directory: str) -> List[Document]:
    """
    Load and return the content of all CSV files in the given directory.
    """
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory, file_name)
            loader = CSVLoader(file_path)
            documents.extend(loader.load())
    return documents

def split_docs(documents: List[Document], chunk_size: int = 400, chunk_overlap: int = 40) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents (List[Document]): List of Document objects to be split.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap size between chunks.

    Returns:
        List[Document]: List of split Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def remove_garbage_lines(text: str) -> str:
    """
    Removes lines that contain mostly numbers, standalone letters, or patterns like 'B = B', 'M = M'.
    """
    cleaned_lines = []
    
    for line in text.split("\n"):
        line = line.strip()
        
        # Skip lines that are mostly numbers, letters with =, or repeating patterns
        if re.match(r'^([\d\s]+|[A-Z]\s*=\s*[A-Z]\s*)+$', line):
            continue
        
        # Skip lines with excessive letter-number-symbol sequences (like slurB B B 0 B B)
        if re.search(r'(slurB|B\s*=\s*B|M\s*=\s*M|Y\s*=\s*Y|X\s*=\s*X|Z\s*=\s*Z)', line):
            continue
        
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

def load_pdf_files(directory: str) -> List[Document]:
    """
    Load and return the content of all PDF files in the given directory.
    """
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(directory, file_name)
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            
            for doc in pdf_docs:
                doc.page_content = remove_garbage_lines(doc.page_content)  # Clean extracted text

            documents.extend(pdf_docs)
    return documents

def upload_files(directory: str) -> List[Document]:
    """
    Upload all supported file types from a given directory, split PDF content into chunks, and return their content.
    """
    supported_loaders = {
        "csv": load_csv_files,
        "pdf": load_pdf_files
    }
    documents = []

    for ext, loader_func in supported_loaders.items():
        loaded_documents = loader_func(directory)
        if ext == "pdf":
            documents.extend(split_docs(loaded_documents))  # Split PDFs into chunks
        else:
            documents.extend(loaded_documents)
    
    return documents
