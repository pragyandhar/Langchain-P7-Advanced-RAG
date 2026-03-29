from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

def load_file(file_path: str):
    # Extract extension
    ext = os.path.splitext(file_path)[-1].lower()

    # If that extension is .pdf or .txt
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')

    # Create the document object
    documents = loader.load()

    # Return the document object
    print(f"\nLoaded {len(documents)} document(s) from '{file_path}'")
    return documents
