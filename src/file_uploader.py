import os
from langchain_community.document_loaders import TextLoader, JSONLoader, CSVLoader, PyPDFLoader
from typing import List, Union

directory = 'C:/Users/skrge/Documents/GitHub/llmtesting/data'

def load_txt_files(directory: str) -> List[str]:
    """
    Load and return the content of all TXT files in the given directory.
    """
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory, file_name)
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

def load_json_files(directory: str) -> List[str]:
    """
    Load and return the content of all JSON files in the given directory.
    """
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".json"):
            file_path = os.path.join(directory, file_name)
            loader = JSONLoader(file_path)
            documents.extend(loader.load())
    return documents

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
    return documents

def upload_files(directory: str) -> List[str]:
    """
    Upload all supported file types from a given directory and return their content.
    """
    supported_loaders = {
        "txt": load_txt_files,
        "json": load_json_files,
        "csv": load_csv_files,
        "pdf": load_pdf_files
    }
    documents = []

    for ext, loader_func in supported_loaders.items():
        documents.extend(loader_func(directory))

    return documents