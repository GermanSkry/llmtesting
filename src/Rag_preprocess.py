import re
import json
import pandas as pd
from typing import List, Union
from langchain.schema import Document

def combine_text_info(json_strings: List[str]) -> List[str]:
    combined_list = []
    
    for json_str in json_strings:
        try:
            # Parse the JSON string
            text_list = json.loads(json_str)
            # Extend the combined list with the parsed list
            combined_list.extend(text_list)
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {json_str}")
            continue
    
    return combined_list

def preprocess_field(value: Union[str, None], field_type: str) -> str:
    if not isinstance(value, str):
        return ""
    if field_type == "price":
        # Remove any non-numeric characters except for the decimal separator
        value = re.sub(r'[^\d,\.]', '', value)
        # Replace comma with dot if necessary
        value = value.replace(',', '.')
    elif field_type in ["ISBN", "Year", "book_shop_id", "pages"]:
        # Remove any non-numeric characters
        value = re.sub(r'[^\d]', '', value)
    return value

def clean_json_string(json_str: str) -> List[str]:
    # Remove leading/trailing non-JSON characters and split into individual JSON objects
    json_str = json_str.strip('```').strip()
    json_objects = re.findall(r'\{.*?\}', json_str, re.DOTALL)
    return json_objects

def create_dataframe_from_json_strings(json_strings: List[str]) -> pd.DataFrame:
    # List to store parsed JSON objects
    parsed_data = []
    
    # Parse each JSON string
    for json_str in json_strings:
        json_objects = clean_json_string(json_str)
        for obj_str in json_objects:
            try:
                book_info = json.loads(obj_str)
                # Preprocess relevant fields
                for field in ["price", "ISBN", "Year", "book_shop_id", "pages"]:
                    if field in book_info:
                        book_info[field] = preprocess_field(book_info[field], field)
                parsed_data.append(book_info)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {obj_str}")
                continue
    
    # Create DataFrame from parsed data
    df = pd.DataFrame(parsed_data)
    
    # Ensure the DataFrame has the desired columns
    desired_columns = ["Full_title", 
                       "City", 
                       "Year", 
                       "ISBN", 
                       "price", 
                       "book_shop_id", 
                       "pages", 
                       "colour", 
                       "size", 
                       "language"]
    df = df.reindex(columns=desired_columns)
    
    return df


def extract_books_from_docs(docs: list) -> list:
    """
    Extracts the book titles from a list of LangChain documents based on csv.
    
    Args:
        docs (list): List of LangChain documents.

    Returns:
        list: A list of book titles extracted from the documents.
    """
    return [re.search(r'Book:\s*(.*?)\n', doc.page_content).group(1).strip() 
            if re.search(r'Book:\s*(.*?)\n', doc.page_content) 
            else "Unknown Book" for doc in docs]

def extract_relevant_info(page_content):
    """
    Extracts the book title, author, and description from the given page_content.
    
    Args:
        page_content (str): The raw text content from a LangChain document.
    
    Returns:
        tuple: (book_title, author, description)
    """
    title_match = re.search(r'Book:\s*(.*?)\n', page_content)
    author_match = re.search(r'Author:\s*(.*?)\n', page_content)
    description_match = re.search(r'Description:\s*(.*?)\n', page_content)

    book_title = title_match.group(1).strip() if title_match else "Unknown Book"
    author = author_match.group(1).strip() if author_match else "Unknown Author"
    description = description_match.group(1).strip() if description_match else "No Description Available"

    return book_title, author, description

def classified_books_df(csv_short: List[Document], classified_books: List[str]) -> pd.DataFrame:
    """
    Extracts book data from documents and their corresponding genre strings, and creates a DataFrame.

    Args:
        csv_short (List[Document]): List of Document objects containing book information.
        classified_books (List[str]): List of genre strings corresponding to each document.

    Returns:
        pd.DataFrame: DataFrame containing extracted book data.
    """
    book_data = []
    
    for doc, genre_str in zip(csv_short, classified_books):
        title_match = re.search(r'Book:\s*(.*?)\n', doc.page_content)
        author_match = re.search(r'Author:\s*(.*?)\n', doc.page_content)
        desc_match = re.search(r'Description:\s*(.*?)\n', doc.page_content)
        
        book_title = title_match.group(1).strip() if title_match else "Unknown Book"
        author = author_match.group(1).strip() if author_match else "Unknown Author"
        description = desc_match.group(1).strip() if desc_match else "No Description Available"

        # Convert stringified list into an actual Python list
        try:
            genres = json.loads(genre_str)
        except json.JSONDecodeError:
            genres = ["Unknown"]

        book_data.append({"Title": book_title, "Author": author, "Description": description, "Genres": genres})

    # Create DataFrame
    books_df = pd.DataFrame(book_data)
    return books_df