import re
import json
import tqdm
from typing import List
from collections import defaultdict
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import VectorStore



def query_rag_chat(query_text: str, vector_store):
    """
    Searches the FAISS vector store and generates a response using retrieved documents.
    """

    question_template = """
    Answer the {question} based only on the following context:

    {context}
    """
    # Perform similarity search in the existing vector store
    results = vector_store.similarity_search(query_text, k=10)

    # If no relevant documents are found, return a message
    if not results:
        print("No relevant context found.")
        return None, "No relevant context found.", []

    # Extract text from retrieved documents
    context = " ".join([doc.page_content for doc in results])
    # Extract document sources
    sources = [doc.metadata.get("chunk_id") for doc in results]

    # Format the prompt
    prompt_template = ChatPromptTemplate.from_template(question_template)
    prompt = prompt_template.format(context=context, question=query_text)

    # Initialize the Ollama model for generating answers
    generation_model = OllamaLLM(model="llama3.1")

    # Generate response
    response = generation_model.invoke(prompt)

    return response, sources

def extract_book_title(documents: List[Document]):
       
    book_title_template = """
    You are an expert in extracting book titles from structured lists.
    Please extract only the book titles and return in this format:  

    ["Full_title_1",
    "Full_title_2",
        ...,
    "Full_title_N"]

    Rules:
    - The book title always starts with an upper letter.
    - Titles can be placed on few lines in a row.
    - Preserve the entire title, including parts before and after `:` if present.
    - Preserve the full title, including subtitles and special characters.
    - Do NOT add any extra text, explanations, or formatting.

    Context:
    {context}

    """
    
    # Combine content from all documents
    context = " ".join([doc.page_content for doc in documents])
    
    # Format the prompt
    book_info_template = PromptTemplate(template=book_title_template, input_variables=["context"])
    prompt_title = book_info_template.format(context=context)
    
    # Initialize LLM
    generation_model = OllamaLLM(model="llama3.1")
    
    # Generate response
    response = generation_model.invoke(prompt_title)
 
    return response

def group_documents_by_source_and_page(documents):
    """
    Groups a list of langchain Document objects by their source and page.
    Args:
        documents (list): A list of langchain Document objects. 
    Returns:
        dict: A dictionary where keys are tuples (source, page), and values are lists of documents.
    """
    documents_grouped = defaultdict(lambda: defaultdict(list))

    for doc in documents:
        source = doc.metadata['source']
        page = doc.metadata['page']
        
        # Group documents by source and page
        documents_grouped[source][page].append(doc)

    # Flatten into a dictionary of (source, page) -> list of documents
    grouped_documents = {}
    for source, pages in documents_grouped.items():
        for page, docs in pages.items():
            grouped_documents[(source, page)] = docs

    return grouped_documents

def extract_titles_from_grouped_documents(documents: List[Document]):
    grouped_documents = group_documents_by_source_and_page(documents)
    
    all_titles = []
    
    # Apply extract_book_title to each group with progress bar
    for (source, page), docs in tqdm(grouped_documents.items(), desc="Processing Groups", unit="group"):
        titles = extract_book_title(docs)
        all_titles.append(titles)  # Add the titles of the current group to the unified list
    
    return all_titles

def extract_book_info(documents: List[Document], book_list: List[str], vector_store: VectorStore):
    extract_template = """Based on the provided document, extract the following details and return ONLY in valid JSON format:
    
        {{
            "Full_title": "<Full_title>",
            "City": "<City Name>",
            "Year": "<Year>",
            "ISBN": "<ISBN number>",
            "price": "<numeric price>",
            "book_shop_id": "<Only numeric Bookshop ID, if available>",
            "pages": "<Only number of pages>",
            "colour": "<Colour details>",
            "size": "<Size>",
            "language": "<Language>"
        }}
    

    Rules:
    - The book title always starts with an uppercase letter and may span multiple lines. Preserve the full title, including parts before and after `:` if present.
    - The City and Year are found in the publisher line, formatted as "Publisher, City Year".
    - The ISBN follows "ISBN" and follows the format: 'ISBN 978-XXXXXXXXXX'.
    - The price follows "Euro" and is always numeric, formatted as 'XX,XX'.
    - The book_shop_id follows "Idea Code" and is a 5-digit numeric value (omit if not available).
    - The number of pages is a 3-digit numeric value (e.g., 108).
    - Colour details include whether the book contains 'colour', 'bw', or both.
    - The size follows the format "XX x XX cm".
    - Language appears at the end of the book details.
    - Do NOT add any extra text, explanations, or formatting.

    Context:
    {context}
    """
    extracted_info = []
    generation_model = OllamaLLM(model="llama3.1")  # Initialize LLM once

    for book in tqdm(book_list, desc="Processing books", unit="book"):
        query = f"Find details about '{book}'"
        results = vector_store.similarity_search(query, k=3)
        context = " ".join([result.page_content for result in results])

        # If no relevant documents are found, skip processing
        if not context.strip():
            print(f"Warning: No relevant context found for book '{book}'")
            extracted_info.append(f'{{"Full_title": "{book}", "error": "No data found"}}')
            continue

        # Generate response
        prompt = extract_template.format(context=context)
        response = generation_model.invoke(prompt).strip()

        extracted_info.append(response)

    return extracted_info  # Return list of extracted book details in string format

def classify_books_from_docs(docs, genres: list) -> list:
    """
    Classifies books into genres based only on their title, author, and description.
    
    Args:
        docs (list): List of LangChain Document objects.
        genres (list): A predefined list of book genres.
    
    Returns:
        list: A list of dictionaries with book titles and assigned genres.
    """
    llm_model = OllamaLLM(model="llama3.1") 
    classified_books = []
    
    # Convert genres list into a single string
    genre_string = ", ".join(genres)

    # Process each book description and classify it
    for doc in tqdm(docs, desc="Classifying Books", unit="book"):
        # Extract relevant book info
        book_title, author, description = extract_relevant_info(doc.page_content) #extract_relevant_info in rag_preprocess.py

        # Step 1: Construct optimized LLM prompt
        prompt = f"""
        You are an expert book classifier. Assign the following book one or more genres from this list:
        {genre_string}

        Book Information:
        - Title: {book_title}
        - Author: {author}
        - Description: {description}

        - Return only a valid JSON list with genres. Example: ["Fiction", "Drama"]
        - Do NOT add any extra text, explanations, or formatting.
        """
        # Step 2: Get classification from LLM
        response = llm_model.invoke(prompt).strip()        
        classified_books.append(response)

    return classified_books
