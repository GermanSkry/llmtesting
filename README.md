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

## RAG-process 

This project implements a **Retrieval-Augmented Generation (RAG)** system designed for extracting, classifying, and querying book information using **LangChain**, **FAISS**, and **Llama 3.1**. The system combines retrieval techniques with language model-based generation to provide detailed insights into books. The integration of **LLMs** allows for highly accurate content extraction, while **FAISS** is used for efficient similarity searches within document collections.

## RAG System Overview

The RAG framework uses a combination of **retrieval** and **generation** to answer user queries or extract information based on external knowledge sources (like documents or databases). In this project, the system leverages **LangChain**, **FAISS**, and **OllamaLLM** to process documents and generate responses.

1. **Document Retrieval**: The system uses FAISS to search a vector store for relevant documents based on a given query. These documents are then passed to an LLM for further processing and response generation.
   
2. **Generation via LLM**: After retrieving relevant documents, a language model (like **Llama 3.1**) generates answers or extracted data by processing the documents in context. This allows the system to provide context-aware responses based on the retrieved data.

## Role of Templates

Templates play a crucial role in guiding the LLM to produce structured and consistent output, ensuring that generated content is aligned with the expected format and context. The templates define the exact structure and format for inputs and outputs.

**Query templates** help format the user’s question and the context retrieved from the vector store to provide the LLM with a clear instruction set. These templates ensure that the system maintains consistency in how it structures its queries and responses. The LLM will generate an answer-based context, ensuring it is contextually relevant to the query.

## Preprocessing LLM Output into Structured Data
One of the key aspects of this project is the preprocessing of LLM output into a structured format for downstream use. After the LLM generates book details based on the extraction template, the output is parsed and converted into a Pandas DataFrame to allow for easy manipulation and analysis. The DataFrame structure ensures that data is consistent and easily queryable, enabling further analysis or downstream tasks (such as querying or visualizing book data). By transforming the LLM output into a structured DataFrame, it becomes much easier to query, manipulate, and visualize the data, which is essential for large datasets or tasks requiring further analysis.

## Notes
### Libraries version:
faiss == 1.9.0

numpy == 1.26.4

pandas == 2.2.1

langchain == 0.2.10

langchain_community == 0.2.9

tqdm == 4.66.4

langchain_ollama == 0.2.2

langchain_core == 0.3.29

### Nomic Model (API Upload)
https://docs.nomic.ai/atlas/introduction/quick-start
```python
import nomic
from nomic import atlas
nomic.login('NOMIC_API_KEY')
```
### Ollama info
https://ollama.com/library/llama3.1


