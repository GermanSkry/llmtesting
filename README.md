# RAG-Based Book Project Documentation

## Introduction

This project implements a Retrieval-Augmented Generation (RAG) system to retrieve information from documents using LangChain and FAISS.  
The goal is to create different functions for book data to get content about them in various formats, leveraging a combination of document processing, embedding generation, and AI-driven retrieval.

## Technologies Used

- **Language Model**: Llama 3.1 (for generating responses)
- **Embeddings Model**: nomic-embed-text-v1.5 (for text vectorization)
- **Document Handling**: LangChain’s document loaders and text splitters
- **Vector Storage**: FAISS (for efficient similarity search)

## Project Workflow

### 1. File Uploading & Processing

- Supports **CSV** and **PDF** files
- Uses LangChain’s `CSVLoader` and `PyPDFLoader`
- Removes noise from text (e.g., lines with standalone numbers or repetitive patterns)
- Splits documents into chunks for better embedding performance

### 2. Embedding Generation

- Uses `nomic-embed-text-v1.5` to transform text chunks into vector embeddings
- Stores embeddings in a FAISS vector database

### 3. Vector Storage with FAISS

- Initializes an FAISS index
- Maps document chunks to embeddings and assigns unique IDs

## Notes

The project includes functions to load CSV and PDF files from a specified directory. PDFs undergo text cleaning and chunking before embedding.

