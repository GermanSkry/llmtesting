import os
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Union
from typing import Tuple
from langchain.schema import Document

directory = 'C:/Users/skrge/Documents/GitHub/llmtesting/data'


def load_csv_files(directory: str) -> List[str]:
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

"""
    Split documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents (List[Document]): List of Document objects to be split.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap size between chunks.

    Returns:
        List[Document]: List of split Document objects.
"""

def split_docs(documents: List[Document], chunk_size: int = 400, chunk_overlap: int = 40) -> List[Document]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def load_pdf_files(directory: str) -> List[str]:
    """
    Load and return the content of all PDF files in the given directory.
    """
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(directory, file_name)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            documents.extend(split_docs(documents)) 
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
        if ext == "pdf":
            pdf_documents = loader_func(directory)
            documents.extend(split_docs(pdf_documents))  # Split PDFs into chunks
        else:
            documents.extend(loader_func(directory))
    
    return documents

if __name__ == "__main__":
    directory = input("Enter the path to the directory: ")
    result = upload_files(directory)
    print("\nAll files loaded successfully.")