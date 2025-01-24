import os
from src import *
from src import upload_files, process_documents, faiss_vector_store, query_rag_chat
from langchain_core.documents import Document  # Ensure you have this installed

def main():
    # Step 1: Ask user for the directory containing files
    directory = input("Enter the directory path for uploading files: ").strip()
    
    # Step 2: Upload files and create a list of Document objects
    docs: List[Document] = upload_files(directory)
    print(f"Uploaded {len(docs)} documents.")

    # Step 3: Process the documents (chunking, cleaning, etc.)
    proc_docs: List[Document] = process_documents(docs)
    print(f"Processed {len(proc_docs)} documents.")

    # Step 4: Generate FAISS vector store from processed documents
    new_vector_store = faiss_vector_store(proc_docs)
    print("Vector store created.")

    # Step 5: Allow user to query the RAG system
    while True:
        query_text = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query_text.lower() == "exit":
            print("Exiting RAG chat...")
            break

        response = query_rag_chat(query_text, new_vector_store)
        print(f"RAG Response: {response}")

if __name__ == "__main__":
    main()